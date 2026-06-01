#include "exx.h"

#include "abacus_symmetry.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "geometry.h"
#include "lapack_connector.h"
#include "libri_utils.h"
#include "matrix_m_parallel_utils.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "stl_io_helper.h"
#include "utils_blacs.h"
#include "vector3_order.h"
#include <fstream>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#ifdef LIBRPA_USE_LIBRI
#include <RI/physics/Exx.h>
#include <RI/ri/Filter_Atom.h>
#include <RI/ri/Cell_Nearest.h>
#else
#include "libri_stub.h"
#endif
#include "utils_io.h"

namespace LIBRPA
{

namespace
{

bool use_abacus_ibz_root_projection(const int n_target_kpoints,
                                    const int n_meanfield_kpoints)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return Params::use_abacus_gw_symmetry && ctx.available && !ctx.kstars.empty()
           && static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
           && ctx.kstars.size() == static_cast<std::size_t>(n_meanfield_kpoints)
           && n_target_kpoints == n_meanfield_kpoints;
}

bool force_root_dense_ks_projection()
{
    const char* env_value = std::getenv("LIBRPA_FORCE_ROOT_KS_PROJECTION");
    if (env_value == nullptr)
    {
        return false;
    }
    const std::string value(env_value);
    return !(value.empty() || value == "0" || value == "f" || value == "F"
             || value == "false" || value == "FALSE");
}

std::string format_debug_double(const double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string text = oss.str();
    std::replace(text.begin(), text.end(), '-', 'm');
    std::replace(text.begin(), text.end(), '.', 'p');
    return text;
}

std::vector<LIBRPA::AbacusKStarGridMappingEntry> build_abacus_full_k_mapping_for_restore(
    const std::vector<Vector3_Order<double>>& kfrac_list)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    if (!Params::use_abacus_gw_symmetry || !ctx.available || ctx.kstars.empty())
    {
        return {};
    }
    if (static_cast<int>(klist.size()) >= get_full_bz_kpoint_count())
    {
        return {};
    }
    if (ctx.kstars.size() != kfrac_list.size())
    {
        throw std::runtime_error(
            "ABACUS full-k restore mapping is inconsistent with the loaded IBZ k-point count");
    }
    return LIBRPA::build_abacus_kstar_grid_mapping(ctx, klist, kfrac_list, map_irk_ks);
}

bool complex_array_has_nonfinite(const std::complex<double>* data, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        const auto& value = data[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            return true;
        }
    }
    return false;
}

bool complex_matrix_has_nonfinite(const ComplexMatrix& mat)
{
    return complex_array_has_nonfinite(mat.c, mat.size);
}

bool exx_coulomb_uses_abacus_irreducible_sector_layout(
    const atpair_R_mat_t& coul_mat,
    const LIBRPA::AbacusSymmetryContext& symmetry_ctx)
{
    if (coul_mat.empty())
    {
        return false;
    }

    for (const auto& i_entry : coul_mat)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& j_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(j_entry.first);
            const auto sector_iter = symmetry_ctx.irreducible_sector.find({ir_I, ir_J});
            if (sector_iter == symmetry_ctx.irreducible_sector.end())
            {
                return false;
            }

            for (const auto& r_entry : j_entry.second)
            {
                const auto& R = r_entry.first;
                const abacus_R_t r_array{R.x, R.y, R.z};
                if (sector_iter->second.count(r_array) == 0)
                {
                    return false;
                }
            }
        }
    }

    return true;
}

template <typename T>
bool tensor_has_nonfinite(const RI::Tensor<T>& tensor)
{
    const auto size = static_cast<int>(tensor.get_shape_all());
    const auto* data = tensor.ptr();
    for (int i = 0; i < size; ++i)
    {
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (!std::isfinite(data[i].real()) || !std::isfinite(data[i].imag()))
            {
                return true;
            }
        }
        else
        {
            if (!std::isfinite(static_cast<double>(data[i])))
            {
                return true;
            }
        }
    }
    return false;
}

template <typename T>
double tensor_max_abs(const RI::Tensor<T>& tensor)
{
    const auto size = static_cast<int>(tensor.get_shape_all());
    const auto* data = tensor.ptr();
    double max_abs = 0.0;
    for (int i = 0; i < size; ++i)
    {
        max_abs = std::max(max_abs, static_cast<double>(std::abs(data[i])));
    }
    return max_abs;
}

template <typename T>
std::complex<double> tensor_sum(const RI::Tensor<T>& tensor)
{
    const auto size = static_cast<int>(tensor.get_shape_all());
    const auto* data = tensor.ptr();
    std::complex<double> sum(0.0, 0.0);
    for (int i = 0; i < size; ++i)
    {
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            sum += data[i];
        }
        else
        {
            sum += std::complex<double>(static_cast<double>(data[i]), 0.0);
        }
    }
    return sum;
}

template <typename T>
double tensor_frobenius_norm(const RI::Tensor<T>& tensor)
{
    const auto size = static_cast<int>(tensor.get_shape_all());
    const auto* data = tensor.ptr();
    double norm_sq = 0.0;
    for (int i = 0; i < size; ++i)
    {
        const double abs_value = static_cast<double>(std::abs(data[i]));
        norm_sq += abs_value * abs_value;
    }
    return std::sqrt(norm_sq);
}

template <typename T>
std::complex<double> tensor_trace(const RI::Tensor<T>& tensor)
{
    const auto shape = tensor.shape;
    const int nrows = static_cast<int>(shape.empty() ? 0 : shape[0]);
    const int ncols = static_cast<int>(shape.size() < 2 ? 0 : shape[1]);
    const int ndiag = std::min(nrows, ncols);
    const auto* data = tensor.ptr();
    std::complex<double> trace(0.0, 0.0);
    for (int i = 0; i < ndiag; ++i)
    {
        const int index = i * ncols + i;
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            trace += data[index];
        }
        else
        {
            trace += std::complex<double>(static_cast<double>(data[index]), 0.0);
        }
    }
    return trace;
}

template <typename T>
void dump_tensor_map_summary(
    const std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<T>>>& tensors,
    const std::string& file_path)
{
    if (!Params::debug || !LIBRPA::envs::mpi_comm_global_h.is_root())
    {
        return;
    }

    std::ofstream ofs(file_path);
    if (!ofs.good())
    {
        return;
    }

    ofs << std::scientific << std::setprecision(15);
    ofs << "# I J Rx Ry Rz rows cols nonfinite maxabs frob sum_re sum_im trace_re trace_im\n";
    for (const auto& i_entry : tensors)
    {
        const int iatom = i_entry.first;
        for (const auto& jr_entry : i_entry.second)
        {
            const int jatom = jr_entry.first.first;
            const auto& R = jr_entry.first.second;
            const auto& tensor = jr_entry.second;
            const auto shape = tensor.shape;
            const int nrows = static_cast<int>(shape.empty() ? 0 : shape[0]);
            const int ncols = static_cast<int>(shape.size() < 2 ? 0 : shape[1]);
            const auto sum = tensor_sum(tensor);
            const auto trace = tensor_trace(tensor);
            ofs << iatom << " " << jatom << " " << R[0] << " " << R[1] << " " << R[2] << " "
                << nrows << " " << ncols << " " << (tensor_has_nonfinite(tensor) ? 1 : 0) << " "
                << tensor_max_abs(tensor) << " " << tensor_frobenius_norm(tensor) << " "
                << sum.real() << " " << sum.imag() << " " << trace.real() << " "
                << trace.imag() << "\n";
        }
    }
}

void maybe_dump_hexx_ks_debug(const ComplexMatrix& hexx_ks,
                              const int ispin,
                              const int isoc1,
                              const int isoc2,
                              const int ik,
                              const Vector3_Order<double>& kfrac)
{
    if (!Params::debug || !LIBRPA::envs::mpi_comm_global_h.is_root())
    {
        return;
    }

    std::ostringstream tag;
    tag << "spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_ik" << ik << "_kx_"
        << format_debug_double(kfrac.x) << "_ky_" << format_debug_double(kfrac.y) << "_kz_"
        << format_debug_double(kfrac.z);
    print_complex_matrix_mm(hexx_ks,
                            Params::output_dir + "abacus_hexx_ks_" + tag.str() + ".mtx",
                            1e-14,
                            false);
}

void maybe_dump_hexx_ao_debug(const ComplexMatrix& hexx_ao,
                              const int ispin,
                              const int isoc1,
                              const int isoc2,
                              const int ik,
                              const Vector3_Order<double>& kfrac)
{
    if (!Params::debug || !LIBRPA::envs::mpi_comm_global_h.is_root())
    {
        return;
    }

    std::ostringstream tag;
    tag << "spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_ik" << ik << "_kx_"
        << format_debug_double(kfrac.x) << "_ky_" << format_debug_double(kfrac.y) << "_kz_"
        << format_debug_double(kfrac.z);
    print_complex_matrix_mm(hexx_ao,
                            Params::output_dir + "abacus_hexx_ao_" + tag.str() + ".mtx",
                            1e-14,
                            false);
}

std::map<std::pair<int, int>, std::set<std::array<int, 3>>>
convert_abacus_irreducible_sector_to_libri(
    const abacus_irreducible_sector_t& irreducible_sector,
    const std::array<int, 3>& period)
{
    auto canonicalize_r = [&period](const std::array<int, 3>& r) {
        auto centered_mod = [](const int value, const int cell_period) {
            if (cell_period <= 0)
            {
                return value;
            }
            return (value % cell_period + 3 * cell_period / 2) % cell_period - cell_period / 2;
        };
        return std::array<int, 3>{centered_mod(r[0], period[0]),
                                  centered_mod(r[1], period[1]),
                                  centered_mod(r[2], period[2])};
    };
    std::map<std::pair<int, int>, std::set<std::array<int, 3>>> libri_sector;
    for (const auto& pair_Rs : irreducible_sector)
    {
        const std::pair<int, int> atom_pair{static_cast<int>(pair_Rs.first.first),
                                            static_cast<int>(pair_Rs.first.second)};
        for (const auto& r : pair_Rs.second)
        {
            libri_sector[atom_pair].insert(canonicalize_r(r));
        }
    }
    return libri_sector;
}

template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_Atom_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_Atom_Symmetry(
        const TC& period,
        const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TA& A1,
                      const TAC& A2,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a1b0_a2b1:
            case RI::Label::ab_ab::a1b1_a2b0:
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b1_a2b0:
            case RI::Label::ab_ab::a1b1_a2b2:
            case RI::Label::ab_ab::a0b1_a2b2:
            case RI::Label::ab_ab::a1b0_a2b2:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TA& A2,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b2:
            case RI::Label::ab_ab::a0b1_a1b2:
            case RI::Label::ab_ab::a0b2_a1b0:
            case RI::Label::ab_ab::a0b2_a1b1:
                return !this->symmetry.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TAC& A2,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
            case RI::Label::ab_ab::a0b1_a1b0:
                return !this->symmetry.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry;
};

template <typename Tdata>
ComplexMatrix convert_libri_tensor_to_complex_matrix(const RI::Tensor<Tdata>& tensor,
                                                     const int nrows,
                                                     const int ncols)
{
    ComplexMatrix matrix(nrows, ncols);
    for (int row = 0; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                matrix(row, col) = tensor(row, col);
            }
            else
            {
                matrix(row, col) = std::complex<double>(tensor(row, col), 0.0);
            }
        }
    }
    return matrix;
}

} // namespace

Exx::Exx(const MeanField &mf, const vector<Vector3_Order<double>> &kfrac_list,
         const Vector3_Order<int> &period)
    : mf_(mf), kfrac_list_(kfrac_list), period_(period)
{
    is_rspace_build_ = false;
    is_kspace_built_ = false;
};

ComplexMatrix Exx::get_dmat_cplx_R_global(const int &ispin, const int &isoc1, const int &isoc2,
                                          const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    const bool use_abacus_symmetry = this->can_restore_dmat_from_abacus_symmetry();
    if (use_abacus_symmetry)
    {
        this->maybe_dump_restored_kspace_dmat_debug(ispin, isoc1, isoc2);
    }
    else
    {
        this->maybe_dump_full_kspace_dmat_debug(ispin, isoc1, isoc2);
    }
    auto dmat_cplx = use_abacus_symmetry
                         ? this->get_dmat_cplx_R_symmetry_restored(ispin, isoc1, isoc2, R)
                         : this->mf_.get_dmat_cplx_R(ispin, isoc1, isoc2, this->kfrac_list_, R);
    // renormalize to single spin channel
    if (!Params::use_soc) dmat_cplx *= 0.5 * nspins;
    this->maybe_dump_dmat_R_debug(use_abacus_symmetry ? "symmetry_restored" : "full_kgrid",
                                  ispin,
                                  isoc1,
                                  isoc2,
                                  R,
                                  dmat_cplx);

    return dmat_cplx;
}

bool Exx::can_restore_dmat_from_abacus_symmetry() const
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return ctx.available && ctx.has_ao_shell_layout() && !ctx.kstars.empty()
           && static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
           && ctx.kstars.size() == this->kfrac_list_.size()
           && this->mf_.get_n_kpoints() == static_cast<int>(ctx.kstars.size());
}

ComplexMatrix Exx::get_dmat_cplx_R_symmetry_restored(const int& ispin, const int& isoc1,
                                                     const int& isoc2,
                                                     const Vector3_Order<int>& R)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const auto kstar_grid_mapping = build_abacus_full_k_mapping_for_restore(this->kfrac_list_);
    ComplexMatrix dmat_cplx(this->mf_.get_n_aos(), this->mf_.get_n_aos());
    this->maybe_dump_abacus_kstar_debug();

    for (int ik_ibz = 0; ik_ibz < this->mf_.get_n_kpoints(); ++ik_ibz)
    {
        const auto& k_ibz = this->kfrac_list_[static_cast<std::size_t>(ik_ibz)];
        const auto& star = find_abacus_kstar_for_ibz_kpoint(ctx, k_ibz);
        const auto& mapping_entry = kstar_grid_mapping.at(static_cast<std::size_t>(ik_ibz));

        if (star.members.empty())
        {
            throw std::runtime_error("ABACUS k-star member list is empty");
        }
        if (mapping_entry.member_q_bz_keys.size() != star.members.size())
        {
            throw std::runtime_error("ABACUS full-k restore mapping is inconsistent with k-star members");
        }

        const double star_factor = 1.0 / static_cast<double>(star.members.size());
        const ComplexMatrix dmat_ibz = this->mf_.get_dmat_cplx(ispin, isoc1, isoc2, ik_ibz);
        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto k_bz_target_vec = latvec * mapping_entry.member_q_bz_keys[imember];
            const Vector3_Order<double> k_bz_target{
                k_bz_target_vec.x, k_bz_target_vec.y, k_bz_target_vec.z};
            // Always go through the ABACUS restore helper here. Even the identity member may
            // need a reciprocal-lattice gauge shift when LibRPA stores an equivalent full-k
            // representative outside the sidecar convention.
            const bool use_time_reversal = member.isym >= nsym_space;
            const ComplexMatrix dmat_member = rotate_abacus_kspace_matrix(
                ctx, member, dmat_ibz, atom_nw, k_ibz, coord_frac, use_time_reversal,
                &k_bz_target);

            const auto ang = -(k_bz_target * R) * TWO_PI;
            const auto kphase = std::complex<double>(std::cos(ang), std::sin(ang));
            dmat_cplx += (star_factor * kphase) * dmat_member;
        }
    }

    return dmat_cplx;
}

void Exx::maybe_dump_abacus_kstar_debug()
{
    if (!Params::debug || this->debug_kstar_dumped_ || !this->can_restore_dmat_from_abacus_symmetry())
    {
        return;
    }

    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    std::ofstream ofs(Params::output_dir + "abacus_kstars_debug.txt");
    if (!ofs.good())
    {
        return;
    }

    ofs << "# ABACUS k-star metadata used by LibRPA\n";
    ofs << "# star_index ik_ibz k_ibz_x k_ibz_y k_ibz_z member_index isym k_bz_x k_bz_y k_bz_z\n";
    for (std::size_t istar = 0; istar < ctx.kstars.size(); ++istar)
    {
        const auto& star = ctx.kstars[istar];
        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            ofs << star.star_index << " " << istar << " " << star.k_ibz.x << " " << star.k_ibz.y
                << " " << star.k_ibz.z << " " << imember << " " << member.isym << " "
                << member.k_bz.x << " " << member.k_bz.y << " " << member.k_bz.z << "\n";
            for (const auto& atom_rotation : member.atom_rotations)
            {
                ofs << "  atom " << atom_rotation.atom_from << " -> " << atom_rotation.atom_to
                    << " type=" << atom_rotation.atom_type << " lmax=" << atom_rotation.lmax
                    << "\n";
            }
        }
    }
    this->debug_kstar_dumped_ = true;
}

void Exx::maybe_dump_dmat_R_debug(const std::string& source_tag,
                                  const int& ispin,
                                  const int& isoc1,
                                  const int& isoc2,
                                  const Vector3_Order<int>& R,
                                  const ComplexMatrix& dmat_cplx)
{
    if (!Params::debug)
    {
        return;
    }

    std::ostringstream tag;
    tag << source_tag << "_spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_R_" << R.x
        << "_" << R.y << "_" << R.z;
    if (!this->debug_dmat_dump_tags_.insert(tag.str()).second)
    {
        return;
    }

    const std::string file_path = Params::output_dir + "abacus_dmat_" + tag.str() + ".mtx";
    print_complex_matrix_mm(dmat_cplx, file_path, 1e-14, false);
}

void Exx::maybe_dump_full_kspace_dmat_debug(const int& ispin, const int& isoc1, const int& isoc2)
{
    if (!Params::debug)
    {
        return;
    }

    for (int ik = 0; ik < this->mf_.get_n_kpoints(); ++ik)
    {
        const auto& kfrac = this->kfrac_list_[static_cast<std::size_t>(ik)];
        std::ostringstream tag;
        tag << "full_kgrid_spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_ik" << ik
            << "_kx_" << format_debug_double(kfrac.x) << "_ky_" << format_debug_double(kfrac.y)
            << "_kz_" << format_debug_double(kfrac.z);
        if (!this->debug_dmat_dump_tags_.insert(tag.str()).second)
        {
            continue;
        }
        const auto dmat_k = this->mf_.get_dmat_cplx(ispin, isoc1, isoc2, ik);
        print_complex_matrix_mm(
            dmat_k, Params::output_dir + "abacus_dmat_k_" + tag.str() + ".mtx", 1e-14, false);
    }
}

void Exx::maybe_dump_restored_kspace_dmat_debug(const int& ispin, const int& isoc1,
                                                const int& isoc2)
{
    if (!Params::debug || !this->can_restore_dmat_from_abacus_symmetry())
    {
        return;
    }

    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const auto kstar_grid_mapping = build_abacus_full_k_mapping_for_restore(this->kfrac_list_);
    for (int ik_ibz = 0; ik_ibz < this->mf_.get_n_kpoints(); ++ik_ibz)
    {
        const auto& k_ibz = this->kfrac_list_[static_cast<std::size_t>(ik_ibz)];
        const auto& star = find_abacus_kstar_for_ibz_kpoint(ctx, k_ibz);
        const auto& mapping_entry = kstar_grid_mapping.at(static_cast<std::size_t>(ik_ibz));
        const ComplexMatrix dmat_ibz = this->mf_.get_dmat_cplx(ispin, isoc1, isoc2, ik_ibz);

        std::ostringstream ibz_tag;
        ibz_tag << "ibz_spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_ik" << ik_ibz
                << "_kx_" << format_debug_double(k_ibz.x) << "_ky_" << format_debug_double(k_ibz.y)
                << "_kz_" << format_debug_double(k_ibz.z);
        if (this->debug_dmat_dump_tags_.insert(ibz_tag.str()).second)
        {
            print_complex_matrix_mm(
                dmat_ibz, Params::output_dir + "abacus_dmat_k_" + ibz_tag.str() + ".mtx", 1e-14, false);
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto k_bz_target_vec = latvec * mapping_entry.member_q_bz_keys[imember];
            const Vector3_Order<double> k_bz_target{
                k_bz_target_vec.x, k_bz_target_vec.y, k_bz_target_vec.z};
            const bool use_time_reversal = member.isym >= nsym_space;
            const ComplexMatrix dmat_member = rotate_abacus_kspace_matrix(
                ctx, member, dmat_ibz, atom_nw, k_ibz, coord_frac, use_time_reversal,
                &k_bz_target);

            std::ostringstream tag;
            tag << "restored_spin" << ispin << "_soc" << isoc1 << "_" << isoc2 << "_ikibz"
                << ik_ibz << "_member" << imember << "_isym" << member.isym << "_kx_"
                << format_debug_double(k_bz_target.x) << "_ky_" << format_debug_double(k_bz_target.y)
                << "_kz_" << format_debug_double(k_bz_target.z);
            if (!this->debug_dmat_dump_tags_.insert(tag.str()).second)
            {
                continue;
            }
            print_complex_matrix_mm(
                dmat_member, Params::output_dir + "abacus_dmat_k_" + tag.str() + ".mtx", 1e-14, false);
        }
    }
}

ComplexMatrix Exx::extract_dmat_cplx_R_IJblock(const ComplexMatrix &dmat_cplx, const atom_t &I,
                                               const atom_t &J)
{
    const auto I_num = atom_nw.at(I);
    const auto J_num = atom_nw.at(J);
    ComplexMatrix dmat_cplx_IJR(I_num, J_num);
    for (size_t i = 0; i != I_num; i++)
    {
        size_t i_glo = atom_iw_loc2glo(I, i);
        for (size_t j = 0; j != J_num; j++)
        {
            size_t j_glo = atom_iw_loc2glo(J, j);
            dmat_cplx_IJR(i, j) = dmat_cplx(i_glo, j_glo);
        }
    }
    return dmat_cplx_IJR;
}

void Exx::build_dmat_R(const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    const auto nsoc = this->mf_.get_n_soc();

    for (int is = 0; is != nspins; is++)
    {
        for (int isoc1 = 0; isoc1 != nsoc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 != nsoc; isoc2++)
            {
                auto dmat_cplx = this->get_dmat_cplx_R_global(is, isoc1, isoc2, R);
                for (int I = 0; I != natom; I++)
                {
                    for (int J = 0; J != natom; J++)
                    {
                        const auto dmat_cplx_IJR =
                            this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                        this->warn_dmat_IJR_nonzero_imag(dmat_cplx_IJR, is, I, J, R);
                        this->dmat[is][isoc1][isoc2][I][J][R] = std::make_shared<matrix>();
                        *(this->dmat[is][isoc1][isoc2][I][J][R]) = dmat_cplx_IJR.real();
                    }
                }
            }
        }
    }
}

void Exx::build_dmat_R(const atom_t &I, const atom_t &J, const Vector3_Order<int> &R)
{
    const auto nspins = this->mf_.get_n_spins();
    const auto nsoc = this->mf_.get_n_soc();

    for (int is = 0; is != nspins; is++)
    {
        for (int isoc1 = 0; isoc1 != nsoc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 != nsoc; isoc2++)
            {
                auto dmat_cplx = this->get_dmat_cplx_R_global(is, isoc1, isoc2, R);
                const auto dmat_cplx_IJR = this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                this->warn_dmat_IJR_nonzero_imag(dmat_cplx_IJR, is, I, J, R);
                this->dmat[is][isoc1][isoc2][I][J][R] = std::make_shared<matrix>();
                *(this->dmat[is][isoc1][isoc2][I][J][R]) = dmat_cplx_IJR.real();
            }
        }
    }
}

void Exx::warn_dmat_IJR_nonzero_imag(const ComplexMatrix &dmat_cplx, const int &ispin,
                                     const atom_t &I, const atom_t &J, const Vector3_Order<int> R)
{
    if (dmat_cplx.get_max_abs_imag() > 1e-2)
        utils::lib_printf(
            "Warning: complex-valued density matrix, spin %d IJR %zu %zu (%d, %d, %d)\n", ispin, I,
            J, R.x, R.y, R.z);
}

template <typename Tdata>
void Exx::build(const Cs_LRI &Cs, const vector<Vector3_Order<int>> &Rlist,
                const atpair_R_mat_t &coul_mat)
{
    using LIBRPA::envs::mpi_comm_global;
    using LIBRPA::envs::mpi_comm_global_h;

    assert(parallel_routing == ParallelRouting::LIBRI);

    if (this->is_rspace_build_)
    {
        return;
    }

    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_soc = this->mf_.get_n_soc();

#ifdef LIBRPA_USE_LIBRI
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf("Computing EXX orbital energy using LibRI\n");
        if (this->can_restore_dmat_from_abacus_symmetry())
        {
            utils::lib_printf("Restoring the EXX density matrix from ABACUS IBZ k-stars\n");
        }
    }
    mpi_comm_global_h.barrier();

    RI::Exx<int, int, 3, Tdata> exx_libri;
    map<int, std::array<double, 3>> atoms_pos;
    for (int i = 0; i != atom_mu.size(); i++)
        atoms_pos.insert(pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    std::array<double, 3> xa{latvec.e11, latvec.e12, latvec.e13};
    std::array<double, 3> ya{latvec.e21, latvec.e22, latvec.e23};
    std::array<double, 3> za{latvec.e31, latvec.e32, latvec.e33};
    std::array<std::array<double, 3>, 3> lat_array{xa, ya, za};
    std::array<int, 3> period_array{period_.x, period_.y, period_.z};
    exx_libri.set_parallel(mpi_comm_global, atoms_pos, lat_array, period_array);

    const auto& symmetry_ctx = LIBRPA::abacus_symmetry_ctx;
    const bool use_abacus_exx_symmetry =
        Params::use_abacus_exx_symmetry && symmetry_ctx.available
        && static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
        && symmetry_ctx.has_ao_shell_layout()
        && !symmetry_ctx.irreducible_sector.empty() && !symmetry_ctx.rspace_operations.empty()
        && symmetry_ctx.atom_to_type.size() == static_cast<std::size_t>(natom)
        && coord_frac.size() == static_cast<std::size_t>(natom);
    const auto libri_irreducible_sector =
        use_abacus_exx_symmetry
            ? convert_abacus_irreducible_sector_to_libri(symmetry_ctx.irreducible_sector,
                                                         period_array)
            : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    // Restore the validated EXX output-only symmetry filter. The internal LibRI contractions stay
    // unchanged, while the returned `Hs(R)` blocks are restricted to the ABACUS irreducible
    // sector and restored to the full AO real-space sector below.
    const bool use_libri_exx_symmetry_filter = use_abacus_exx_symmetry;
    abacus_rspace_sector_stars_t abacus_sector_stars;
    if (use_abacus_exx_symmetry)
    {
        if (mpi_comm_global_h.is_root())
        {
            utils::lib_printf("Reducing EXX real-space contractions with ABACUS irreducible sectors\n");
        }
        build_abacus_rspace_sector_stars(
            symmetry_ctx, coord_frac, period_, Rlist, abacus_sector_stars, nullptr);
        exx_libri.set_symmetry(false, {});
        if (use_libri_exx_symmetry_filter)
        {
            exx_libri.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_Atom_Symmetry<int, std::array<int, 3>, Tdata>>(
                    exx_libri.lri.period, libri_irreducible_sector);
        }
    }
    else
    {
        if (mpi_comm_global_h.is_root() && symmetry_ctx.available && Params::use_abacus_exx_symmetry)
        {
            utils::lib_printf(
                "ABACUS EXX real-space symmetry reduction is unavailable; falling back to the full sector\n");
        }
        if (mpi_comm_global_h.is_root() && !Params::use_abacus_exx_symmetry)
        {
            utils::lib_printf(
                "ABACUS EXX real-space symmetry reduction is disabled by input; keeping only the IBZ density-matrix restoration\n");
        }
        exx_libri.set_symmetry(false, {});
    }

    // Initialize Cs libRI container on each process
    // Note: we use different treatment in different routings
    //     R-tau routing:
    //         Each process has a full Cs copy.
    //         Thus in each process we only pass a few to LibRI container.
    //     atom-pair routing:
    //         Cs is already distributed across all processes.
    //         Pass the all Cs to libRI container.

    Profiler::start("build_real_space_exx_1", "Prepare C libRI object");
    envs::ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << "\n";
    // print_keys(envs::ofs_myid, Cs.data_libri);

    // TODO: template Cs_LRI
    if constexpr (std::is_same<Tdata, std::complex<double>>::value)
    {
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<Tdata>>> data_libri;
        for (const auto &I_JR_C : Cs.data_libri)
        {
            const auto I = I_JR_C.first;
            for (const auto &JR_C : I_JR_C.second)
            {
                const auto J = JR_C.first.first;
                const auto R = JR_C.first.second;
                const auto &C = JR_C.second;
                auto JR = std::pair<int, std::array<int, 3>>(J, R);
                data_libri[I][JR] = RI::Global_Func::convert<Tdata>(C);
            }
        }
        exx_libri.set_Cs(data_libri, Params::libri_exx_threshold_C);
        dump_tensor_map_summary(data_libri, Params::output_dir + "debug_exx_C_input_summary.txt");
    }
    else
    {
        exx_libri.set_Cs(Cs.data_libri, Params::libri_exx_threshold_C);
        dump_tensor_map_summary(Cs.data_libri, Params::output_dir + "debug_exx_C_input_summary.txt");
    }
    dump_tensor_map_summary(exx_libri.lri.data_pool.at("Cs_").Ds_ab,
                            Params::output_dir + "debug_exx_C_after_set_summary.txt");
    Profiler::stop("build_real_space_exx_1");
    envs::ofs_myid << "Finished setup Cs for EXX\n";
    std::flush(envs::ofs_myid);

    // initialize Coulomb matrix
    Profiler::start("build_real_space_exx_2", "Prepare V libRI object");
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> V_libri;
    Profiler::start("build_real_space_exx_2_1");
    atpair_R_mat_t exx_coul_mat_restored;
    const atpair_R_mat_t* exx_coul_mat_ptr = &coul_mat;
    const bool use_abacus_exx_coulomb_restore =
        use_abacus_exx_symmetry
        && exx_coulomb_uses_abacus_irreducible_sector_layout(coul_mat, symmetry_ctx);
    if (use_abacus_exx_coulomb_restore)
    {
        if (mpi_comm_global_h.is_root())
        {
            utils::lib_printf(
                "Restoring the ABACUS EXX auxiliary Coulomb blocks from the irreducible sector to the full real-space sector before LibRI contraction\n");
        }

        for (const auto& I_JRV : coul_mat)
        {
            const auto ir_I = I_JRV.first;
            for (const auto& J_RV : I_JRV.second)
            {
                const auto ir_J = J_RV.first;
                const auto sector_pair =
                    std::make_pair(static_cast<atom_t>(ir_I), static_cast<atom_t>(ir_J));
                const auto pair_iter = abacus_sector_stars.find(sector_pair);
                if (pair_iter == abacus_sector_stars.end())
                {
                    throw std::runtime_error(
                        "Failed to match an irreducible EXX Coulomb atom pair with the ABACUS restore map");
                }

                for (const auto& R_V : J_RV.second)
                {
                    const auto& ir_R = R_V.first;
                    const auto star_iter = pair_iter->second.find(ir_R);
                    if (star_iter == pair_iter->second.end())
                    {
                        throw std::runtime_error(
                            "Failed to match an irreducible EXX Coulomb real-space block with the ABACUS restore map");
                    }

                    const ComplexMatrix v_ir(*R_V.second);
                    for (const auto& restore_member : star_iter->second)
                    {
                        const ComplexMatrix v_full = rotate_abacus_abf_rspace_matrix(
                            symmetry_ctx,
                            restore_member.isym,
                            static_cast<atom_t>(ir_I),
                            static_cast<atom_t>(ir_J),
                            v_ir);
                        const matrix v_full_real = v_full.real();
                        auto& target_pair_map =
                            exx_coul_mat_restored[restore_member.full_atom_pair.first]
                                                 [restore_member.full_atom_pair.second];
                        if (target_pair_map.count(restore_member.full_R) != 0)
                        {
                            throw std::runtime_error(
                                "Duplicate full-sector EXX Coulomb block appears during ABACUS symmetry restore");
                        }
                        target_pair_map[restore_member.full_R] =
                            std::make_shared<matrix>(v_full_real);
                    }
                }
            }
        }
        exx_coul_mat_ptr = &exx_coul_mat_restored;
    }
    else if (use_abacus_exx_symmetry && mpi_comm_global_h.is_root())
    {
        utils::lib_printf(
            "ABACUS EXX Coulomb input already spans the full real-space sector; skip irreducible-sector restore\n");
    }
    const auto& exx_coul_mat = *exx_coul_mat_ptr;
    const bool use_symmetry_direct_vr_local_cache =
        use_abacus_exx_symmetry && exx_coul_mat_ptr == &coul_mat
        && LIBRPA::parallel_routing != LIBRPA::ParallelRouting::R_TAU
        && mpi_comm_global_h.nprocs > 1;
    const bool use_existing_local_vr_partition =
        !use_abacus_exx_symmetry && LIBRPA::parallel_routing != LIBRPA::ParallelRouting::R_TAU
        && mpi_comm_global_h.nprocs > 1;
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
    {
        // Full Coulomb case, have to re-distribute
        for (auto IJR : dispatch_vector_prod(get_atom_pair(exx_coul_mat), Rlist, mpi_comm_global_h.myid,
                                             mpi_comm_global_h.nprocs, true, true))
        {
            const auto I = IJR.first.first;
            const auto J = IJR.first.second;
            const auto R = IJR.second;
            const auto &VIJR = exx_coul_mat.at(I).at(J).at(R);
            // debug
            // printf("I J R %zu %zu %d %d %d, max(V) %f\n", I, J, R.x, R.y, R.z, VIJR->max());
            std::array<int, 3> Ra{R.x, R.y, R.z};
            std::valarray<Tdata> VIJR_va;
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                VIJR_va = std::valarray<std::complex<double>>(VIJR->size);
                for (size_t i = 0; i < VIJR->size; ++i)
                {
                    VIJR_va[i] = std::complex<double>(VIJR->c[i], 0.0);
                }
            }
            else
                VIJR_va = std::valarray<Tdata>(VIJR->c, VIJR->size);
            auto pv = std::make_shared<std::valarray<Tdata>>();
            *pv = VIJR_va;
            V_libri[I][{J, Ra}] = RI::Tensor<Tdata>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
        }
    }
    else
    {
        if (use_symmetry_direct_vr_local_cache || use_existing_local_vr_partition)
        {
            // For symmetry-restored EXX, `FT_Vq()` can leave a full local `V(R)` replica on every
            // rank. For symmetry-off LibRI runs, `FT_Vq()` keeps the real-space Coulomb blocks
            // partitioned across ranks already. In both cases, feed the currently available local
            // blocks directly into `V_libri` and avoid a second task dispatch that would either
            // duplicate or drop entries.
            for (const auto &I_JRV : exx_coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV : I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V : J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int, 3> Ra{R.x, R.y, R.z};
                        std::valarray<Tdata> VIJR_va;
                        if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                        {
                            VIJR_va = std::valarray<std::complex<double>>(V->size);
                            for (size_t i = 0; i < V->size; ++i)
                            {
                                VIJR_va[i] = std::complex<double>(V->c[i], 0.0);
                            }
                        }
                        else
                            VIJR_va = std::valarray<Tdata>(V->c, V->size);
                        auto pv = std::make_shared<std::valarray<Tdata>>();
                        *pv = VIJR_va;
                        V_libri[I][{J, Ra}] = RI::Tensor<Tdata>({size_t(V->nr), size_t(V->nc)}, pv);
                    }
                }
            }
        }
        else if (mpi_comm_global_h.nprocs > 1)
        {
            // `FT_Vq()` currently leaves the restored real-space Coulomb map replicated on each
            // MPI rank. Feed only a deterministic local subset into LibRI so `set_Vs()` does not
            // receive the full V(I,J,R) set from every rank at once.
            for (const auto &IJR :
                 dispatch_vector_prod(get_atom_pair(exx_coul_mat), Rlist, mpi_comm_global_h.myid,
                                      mpi_comm_global_h.nprocs, true, true))
            {
                const auto I = IJR.first.first;
                const auto J = IJR.first.second;
                const auto &R = IJR.second;
                if (exx_coul_mat.count(I) == 0 || exx_coul_mat.at(I).count(J) == 0
                    || exx_coul_mat.at(I).at(J).count(R) == 0)
                {
                    continue;
                }
                const auto &V = exx_coul_mat.at(I).at(J).at(R);
                std::array<int, 3> Ra{R.x, R.y, R.z};
                std::valarray<Tdata> VIJR_va;
                if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                {
                    VIJR_va = std::valarray<std::complex<double>>(V->size);
                    for (size_t i = 0; i < V->size; ++i)
                    {
                        VIJR_va[i] = std::complex<double>(V->c[i], 0.0);
                    }
                }
                else
                    VIJR_va = std::valarray<Tdata>(V->c, V->size);
                auto pv = std::make_shared<std::valarray<Tdata>>();
                *pv = VIJR_va;
                V_libri[I][{J, Ra}] = RI::Tensor<Tdata>({size_t(V->nr), size_t(V->nc)}, pv);
            }
        }
        else
        {
            for (const auto &I_JRV : exx_coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV : I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V : J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int, 3> Ra{R.x, R.y, R.z};
                        std::valarray<Tdata> VIJR_va;
                        if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                        {
                            VIJR_va = std::valarray<std::complex<double>>(V->size);
                            for (size_t i = 0; i < V->size; ++i)
                            {
                                VIJR_va[i] = std::complex<double>(V->c[i], 0.0);
                            }
                        }
                        else
                            VIJR_va = std::valarray<Tdata>(V->c, V->size);
                        auto pv = std::make_shared<std::valarray<Tdata>>();
                        *pv = VIJR_va;
                        V_libri[I][{J, Ra}] = RI::Tensor<Tdata>({size_t(V->nr), size_t(V->nc)}, pv);
                    }
                }
            }
        }
    }
    Profiler::cease("build_real_space_exx_2_1");
    envs::ofs_myid << "Number of V keys: " << get_num_keys(V_libri) << "\n";
    dump_tensor_map_summary(V_libri, Params::output_dir + "debug_exx_V_summary.txt");
    Profiler::start("build_real_space_exx_2_2");
    if (use_symmetry_direct_vr_local_cache)
    {
        exx_libri.lri.set_tensors_map2(
            V_libri,
            {RI::Label::ab::a0b0},
            {{"flag_comm", false}, {"threshold_filter", Params::libri_exx_threshold_V}},
            "Vs_");
        exx_libri.flag_finish.Vs = true;
    }
    else
        exx_libri.set_Vs(V_libri, Params::libri_exx_threshold_V);
    dump_tensor_map_summary(exx_libri.lri.data_pool.at("Vs_").Ds_ab,
                            Params::output_dir + "debug_exx_V_after_set_summary.txt");
    V_libri.clear();
    Profiler::cease("build_real_space_exx_2_2");
    Profiler::cease("build_real_space_exx_2");
    utils::lib_printf("Task %4d: V setup for EXX\n", mpi_comm_global_h.myid);
    // cout << V_libri << endl;

    // initialize density matrix
    vector<atpair_t> atpair_dmat;
    for (int I = 0; I < atom_nw.size(); I++)
        for (int J = 0; J < atom_nw.size(); J++) atpair_dmat.push_back({I, J});
    const auto dmat_IJRs_local = dispatch_vector_prod(atpair_dmat, Rlist, mpi_comm_global_h.myid,
                                                      mpi_comm_global_h.nprocs, true, true);

    for (auto isp = 0; isp != n_spins; isp++)
    {
        for (auto is1 = 0; is1 != n_soc; is1++)
        {
            for (auto is2 = 0; is2 != n_soc; is2++)
            {
                Profiler::start("build_real_space_exx_3", "Prepare DM libRI object");
                std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>
                    dmat_libri;
                for (const auto &R : Rlist)
                {
                    std::array<int, 3> Ra{R.x, R.y, R.z};
                    const auto dmat_cplx = this->get_dmat_cplx_R_global(isp, is1, is2, R);
                    for (const auto &IJR : dmat_IJRs_local)
                    {
                        if (IJR.second == R)
                        {
                            const auto &I = IJR.first.first;
                            const auto &J = IJR.first.second;
                            const auto dmat_IJR =
                                this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                            this->warn_dmat_IJR_nonzero_imag(dmat_IJR, isp, I, J, R);
                            std::valarray<Tdata> dmat_va;
                            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                                dmat_va = std::valarray<Tdata>(dmat_IJR.c, dmat_IJR.size);
                            else
                                dmat_va = std::valarray<Tdata>(dmat_IJR.real().c, dmat_IJR.size);
                            // const auto &n_I = atomic_basis_wfc.get_atom_nb(I);
                            // const auto &n_J = atomic_basis_wfc.get_atom_nb(J); 
                            // for (int i = 0; i!= n_I; i++) {
                            //     for(int j =0 ; j!= n_J; j++){
                                    
                            //         utils::lib_printf(
                            //         "dmat, spin, %d is1,is2,I,J,R,i,j %d %d %zu %zu (%d, %d, %d) %zu %zu = (%.10e,%.10e)\n", isp, is1, is2, I , J , R.x, R.y, R.z, i, j, dmat_IJR(i,j).real(),dmat_IJR(i,j).imag());
                            //     }
                            // }
                            
                            auto pdmat = std::make_shared<std::valarray<Tdata>>();
                            *pdmat = dmat_va;
                            dmat_libri[I][{J, Ra}] = RI::Tensor<Tdata>(
                                {size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);

                            // const auto &n_I = atomic_basis_wfc.get_atom_nb(I);
                            // const auto &n_J = atomic_basis_wfc.get_atom_nb(J); 
                            // for (int i = 0; i!= n_I; i++) {
                            //     for(int j =0 ; j!= n_J; j++){
                            //         utils::lib_printf(
                            //         "dmat, spin, %d is1,is2,I,J,R,i,j %d %d %zu %zu (%d, %d, %d) %zu %zu = %.10e\n", isp, is1, is2, I , J , R.x, R.y, R.z, i, j, dmat_libri[I][{J, Ra}](i,j));
                            //     }
                            // }
                            
                        }
                    }
                }
                envs::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri) << "\n";
                {
                    std::ostringstream oss;
                    oss << Params::output_dir << "debug_exx_D_summary_spin" << isp << "_soc"
                        << is1 << "_" << is2 << ".txt";
                    dump_tensor_map_summary(dmat_libri, oss.str());
                }
                // print_keys(envs::ofs_myid, dmat_libri);
                exx_libri.set_Ds(dmat_libri, Params::libri_exx_threshold_D);
                dump_tensor_map_summary(exx_libri.lri.data_pool.at("Ds_").Ds_ab,
                                        Params::output_dir
                                            + "debug_exx_D_after_set_summary_spin"
                                            + std::to_string(isp) + "_soc"
                                            + std::to_string(is1) + "_"
                                            + std::to_string(is2) + ".txt");
                Profiler::stop("build_real_space_exx_3");
                utils::lib_printf("Task %4d: DM setup for EXX\n", mpi_comm_global_h.myid);

                Profiler::start("build_real_space_exx_4", "Call libRI Hexx calculation");
                exx_libri.cal_Hs();
                Profiler::stop("build_real_space_exx_4");

                utils::lib_printf("Task %4d: cal_Hs elapsed time: %f\n", mpi_comm_global_h.myid,
                                  Profiler::get_wall_time_last("build_real_space_exx_4"));
                envs::ofs_myid << "Number of exx_libri.Hs keys: " << get_num_keys(exx_libri.Hs)
                               << "\n";
                {
                    std::ostringstream oss;
                    oss << Params::output_dir << "debug_exx_Hs_summary_spin" << isp << "_soc"
                        << is1 << "_" << is2 << ".txt";
                    dump_tensor_map_summary(exx_libri.Hs, oss.str());
                }
                // print_keys(envs::ofs_myid, exx_libri.Hs);
                // ofs_myid << "exx_libri.Hs:\n" << exx_libri.Hs << endl;

                auto store_exx_block_matrix = [&](const atom_t full_I,
                                                  const atom_t full_J,
                                                  const Vector3_Order<int>& full_R,
                                                  const ComplexMatrix& exx_block) {
                    const auto n_full_I = atomic_basis_wfc.get_atom_nb(full_I);
                    const auto n_full_J = atomic_basis_wfc.get_atom_nb(full_J);
                    if (exx_block.nr != n_full_I || exx_block.nc != n_full_J)
                    {
                        throw std::runtime_error(
                            "ABACUS EXX real-space restore produced an AO block with an inconsistent dimension");
                    }
                    if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                    {
                        Matz exx_temp(n_full_I, n_full_J, exx_block.c, MAJOR::ROW);
                        this->exx_cplx[isp][is1][is2][full_R][full_I][full_J] = exx_temp;
                    }
                    else
                    {
                        const auto exx_block_real = exx_block.real();
                        Matd exx_temp(n_full_I, n_full_J, exx_block_real.c, MAJOR::ROW);
                        this->exx[isp][is1][is2][full_R][full_I][full_J] = exx_temp;
                    }
                };

                if (use_libri_exx_symmetry_filter)
                {
                    utils::lib_printf(
                        "Restoring the symmetry-filtered LibRI EXX blocks to the full AO real-space sector\n");
                }
                for (const auto &I_JR_exx : exx_libri.Hs)
                {
                    const auto &I = I_JR_exx.first;
                    for (const auto &JR_exx : I_JR_exx.second)
                    {
                        const auto &J = JR_exx.first.first;
                        const auto &Ra = JR_exx.first.second;
                        const auto R = Vector3_Order<int>{Ra[0], Ra[1], Ra[2]};
                        const auto exx_block_ir =
                            convert_libri_tensor_to_complex_matrix(JR_exx.second,
                                                                   atomic_basis_wfc.get_atom_nb(I),
                                                                   atomic_basis_wfc.get_atom_nb(J));
                        if (complex_matrix_has_nonfinite(exx_block_ir))
                        {
                            std::ostringstream oss;
                            oss << "Non-finite EXX block appears directly in LibRI Hs at I=" << I
                                << " J=" << J << " R=(" << R.x << "," << R.y << "," << R.z
                                << ")";
                            throw std::runtime_error(oss.str());
                        }
                        if (use_libri_exx_symmetry_filter)
                        {
                            const auto sector_pair = std::make_pair(static_cast<atom_t>(I),
                                                                    static_cast<atom_t>(J));
                            const auto pair_iter = abacus_sector_stars.find(sector_pair);
                            if (pair_iter == abacus_sector_stars.end()
                                || pair_iter->second.count(R) == 0)
                            {
                                std::ostringstream oss;
                                oss << "Failed to match a symmetry-filtered EXX real-space block"
                                    << " with the ABACUS irreducible-sector restore map"
                                    << " for I=" << I << " J=" << J << " R=(" << R.x << ","
                                    << R.y << "," << R.z << ")";
                                throw std::runtime_error(oss.str());
                            }

                            for (const auto& restore_member : pair_iter->second.at(R))
                            {
                                const auto exx_block_full =
                                    rotate_abacus_rspace_matrix(symmetry_ctx,
                                                                restore_member.isym,
                                                                static_cast<atom_t>(I),
                                                                static_cast<atom_t>(J),
                                                                exx_block_ir);
                                if (complex_matrix_has_nonfinite(exx_block_full))
                                {
                                    std::ostringstream oss;
                                    oss << "Non-finite EXX real-space block appears after"
                                        << " restoring a symmetry-filtered LibRI Hs block"
                                        << " I=" << I << " J=" << J << " R=(" << R.x << ","
                                        << R.y << "," << R.z << "), isym="
                                        << restore_member.isym;
                                    throw std::runtime_error(oss.str());
                                }
                                store_exx_block_matrix(restore_member.full_atom_pair.first,
                                                       restore_member.full_atom_pair.second,
                                                       restore_member.full_R,
                                                       exx_block_full);
                            }
                            continue;
                        }

                        store_exx_block_matrix(static_cast<atom_t>(I),
                                               static_cast<atom_t>(J),
                                               R,
                                               exx_block_ir);
                    }
                }
            }
        }
    }
    // // debug, print the Hexx matrices
    // for (const auto& isp_IJkH: this->Hexx)
    // {
    //     const auto& isp = isp_IJkH.first;
    //     for (const auto& I_JkH: isp_IJkH.second)
    //     {
    //         const auto& I = I_JkH.first;
    //         for (const auto& J_kH: I_JkH.second)
    //         {
    //             const auto& J = J_kH.first;
    //             for (const auto& k_H: J_kH.second)
    //             {
    //                 cout << isp << " " << I << " " << J << " {" << k_H.first << "} whole size: "
    //                 << k_H.second->size << endl; print_complex_matrix("", *k_H.second);
    //             }
    //         }
    //     }
    // }

#else
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf(
            "Error: trying build EXX orbital energy with LibRI, but the program is not compiled "
            "against LibRI\n");
    }
    throw std::logic_error("compilation");
    mpi_comm_global_h.barrier();
#endif

    is_rspace_build_ = true;
}

/* void Exx::build(const Cs_LRI &Cs, const vector<Vector3_Order<int>> &Rlist,
                const atpair_R_mat_t &coul_mat)
{
    using LIBRPA::envs::mpi_comm_global;
    using LIBRPA::envs::mpi_comm_global_h;

    assert(parallel_routing == ParallelRouting::LIBRI);

    if (this->is_rspace_build_)
    {
        return;
    }

    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_soc = this->mf_.get_n_soc();

#ifdef LIBRPA_USE_LIBRI
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf("Computing EXX orbital energy using LibRI\n");
    }
    mpi_comm_global_h.barrier();

    RI::Exx<int, int, 3, double> exx_libri;
    map<int, std::array<double, 3>> atoms_pos;
    for (int i = 0; i != atom_mu.size(); i++)
        atoms_pos.insert(pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    std::array<double, 3> xa{latvec.e11, latvec.e12, latvec.e13};
    std::array<double, 3> ya{latvec.e21, latvec.e22, latvec.e23};
    std::array<double, 3> za{latvec.e31, latvec.e32, latvec.e33};
    std::array<std::array<double, 3>, 3> lat_array{xa, ya, za};
    std::array<int, 3> period_array{period_.x, period_.y, period_.z};
    exx_libri.set_parallel(mpi_comm_global, atoms_pos, lat_array, period_array);

    // Initialize Cs libRI container on each process
    // Note: we use different treatment in different routings
    //     R-tau routing:
    //         Each process has a full Cs copy.
    //         Thus in each process we only pass a few to LibRI container.
    //     atom-pair routing:
    //         Cs is already distributed across all processes.
    //         Pass the all Cs to libRI container.

    Profiler::start("build_real_space_exx_1", "Prepare C libRI object");
    envs::ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << "\n";
    // print_keys(envs::ofs_myid, Cs.data_libri);
    exx_libri.set_Cs(Cs.data_libri, Params::libri_exx_threshold_C);
    Profiler::stop("build_real_space_exx_1");
    envs::ofs_myid << "Finished setup Cs for EXX\n";
    std::flush(envs::ofs_myid);

    // initialize Coulomb matrix
    Profiler::start("build_real_space_exx_2", "Prepare V libRI object");
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> V_libri;
    Profiler::start("build_real_space_exx_2_1");
    const bool use_existing_local_vr_partition =
        LIBRPA::parallel_routing != LIBRPA::ParallelRouting::R_TAU && mpi_comm_global_h.nprocs > 1;
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
    {
        // Full Coulomb case, have to re-distribute
        for (auto IJR : dispatch_vector_prod(get_atom_pair(coul_mat), Rlist,
mpi_comm_global_h.myid, mpi_comm_global_h.nprocs, true, true))
        {
            const auto I = IJR.first.first;
            const auto J = IJR.first.second;
            const auto R = IJR.second;
            const auto &VIJR = coul_mat.at(I).at(J).at(R);
            // debug
            // printf("I J R %zu %zu %d %d %d, max(V) %f\n", I, J, R.x, R.y, R.z, VIJR->max());
            std::array<int, 3> Ra{R.x, R.y, R.z};
            std::valarray<double> VIJR_va(VIJR->c, VIJR->size);
            auto pv = std::make_shared<std::valarray<double>>();
            *pv = VIJR_va;
            V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
        }
    }
    else
    {
        if (use_existing_local_vr_partition)
        {
            // In LibRI atom-pair routing the incoming `coul_mat` is already partitioned by MPI
            // rank after the reciprocal-to-real transform. Re-dispatching the existing local map
            // would drop most entries, so pass the local blocks through directly.
            for (const auto &I_JRV : coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV : I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V : J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int, 3> Ra{R.x, R.y, R.z};
                        std::valarray<double> VIJR_va(V->c, V->size);
                        auto pv = std::make_shared<std::valarray<double>>();
                        *pv = VIJR_va;
                        V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(V->nr), size_t(V->nc)},
                                                                 pv);
                    }
                }
            }
        }
        else if (mpi_comm_global_h.nprocs > 1)
        {
            // `FT_Vq()` currently leaves the restored real-space Coulomb map replicated on each
            // MPI rank. Feed only a deterministic local subset into LibRI so `set_Vs()` does not
            // receive the full V(I,J,R) set from every rank at once.
            for (const auto &IJR :
                 dispatch_vector_prod(get_atom_pair(coul_mat), Rlist, mpi_comm_global_h.myid,
                                      mpi_comm_global_h.nprocs, true, true))
            {
                const auto I = IJR.first.first;
                const auto J = IJR.first.second;
                const auto &R = IJR.second;
                if (coul_mat.count(I) == 0 || coul_mat.at(I).count(J) == 0
                    || coul_mat.at(I).at(J).count(R) == 0)
                {
                    continue;
                }
                const auto &V = coul_mat.at(I).at(J).at(R);
                std::array<int, 3> Ra{R.x, R.y, R.z};
                std::valarray<double> VIJR_va(V->c, V->size);
                auto pv = std::make_shared<std::valarray<double>>();
                *pv = VIJR_va;
                V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(V->nr), size_t(V->nc)},
                                                         pv);
            }
        }
        else
        {
            for (const auto &I_JRV : coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV : I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V : J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int, 3> Ra{R.x, R.y, R.z};
                        std::valarray<double> VIJR_va(V->c, V->size);
                        auto pv = std::make_shared<std::valarray<double>>();
                        *pv = VIJR_va;
                        V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(V->nr), size_t(V->nc)},
                                                                 pv);
                    }
                }
            }
        }
    }
    Profiler::cease("build_real_space_exx_2_1");
    envs::ofs_myid << "Number of V keys: " << get_num_keys(V_libri) << "\n";
    Profiler::start("build_real_space_exx_2_2");
    exx_libri.set_Vs(V_libri, Params::libri_exx_threshold_V);
    V_libri.clear();
    Profiler::cease("build_real_space_exx_2_2");
    Profiler::cease("build_real_space_exx_2");
    utils::lib_printf("Task %4d: V setup for EXX\n", mpi_comm_global_h.myid);
    // cout << V_libri << endl;

    // initialize density matrix
    vector<atpair_t> atpair_dmat;
    for (int I = 0; I < atom_nw.size(); I++)
        for (int J = 0; J < atom_nw.size(); J++) atpair_dmat.push_back({I, J});
    const auto dmat_IJRs_local = dispatch_vector_prod(atpair_dmat, Rlist,
mpi_comm_global_h.myid, mpi_comm_global_h.nprocs, true, true);

    for (auto isp = 0; isp != n_spins; isp++)
    {
        for (auto is1 = 0; is1 != n_soc; is1++)
        {
            for (auto is2 = 0; is2 != n_soc; is2++)
            {
                Profiler::start("build_real_space_exx_3", "Prepare DM libRI object");
                std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>>
                    dmat_libri;
                for (const auto &R : Rlist)
                {
                    std::array<int, 3> Ra{R.x, R.y, R.z};
                    const auto dmat_cplx = this->get_dmat_cplx_R_global(isp, is1, is2, R);
                    for (const auto &IJR : dmat_IJRs_local)
                    {
                        if (IJR.second == R)
                        {
                            const auto &I = IJR.first.first;
                            const auto &J = IJR.first.second;
                            const auto dmat_IJR =
                                this->extract_dmat_cplx_R_IJblock(dmat_cplx, I, J);
                            this->warn_dmat_IJR_nonzero_imag(dmat_IJR, isp, I, J, R);
                            std::valarray<double> dmat_va(dmat_IJR.real().c, dmat_IJR.size);
                            auto pdmat = std::make_shared<std::valarray<double>>();
                            *pdmat = dmat_va;
                            dmat_libri[I][{J, Ra}] = RI::Tensor<double>(
                                {size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                        }
                    }
                }
                envs::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri) << "\n";
                // print_keys(envs::ofs_myid, dmat_libri);
                exx_libri.set_Ds(dmat_libri, Params::libri_exx_threshold_D);
                Profiler::stop("build_real_space_exx_3");
                utils::lib_printf("Task %4d: DM setup for EXX\n", mpi_comm_global_h.myid);

                Profiler::start("build_real_space_exx_4", "Call libRI Hexx calculation");
                exx_libri.cal_Hs();
                Profiler::stop("build_real_space_exx_4");

                utils::lib_printf("Task %4d: cal_Hs elapsed time: %f\n", mpi_comm_global_h.myid,
                                  Profiler::get_wall_time_last("build_real_space_exx_4"));
                envs::ofs_myid << "Number of exx_libri.Hs keys: " << get_num_keys(exx_libri.Hs)
                               << "\n";
                // print_keys(envs::ofs_myid, exx_libri.Hs);
                // ofs_myid << "exx_libri.Hs:\n" << exx_libri.Hs << endl;

                for (const auto &I_JR_exx : exx_libri.Hs)
                {
                    const auto &I = I_JR_exx.first;
                    const auto &n_I = atomic_basis_wfc.get_atom_nb(I);
                    for (const auto &JR_exx : I_JR_exx.second)
                    {
                        const auto &J = JR_exx.first.first;
                        const auto &n_J = atomic_basis_wfc.get_atom_nb(J);
                        const auto &Ra = JR_exx.first.second;
                        const auto R = Vector3_Order<int>{Ra[0], Ra[1], Ra[2]};
                        Matd exx_temp(n_I, n_J, JR_exx.second.ptr(), MAJOR::ROW);
                        this->exx[isp][is1][is2][R][I][J] = exx_temp;
                    }
                }
            }
        }
    }
    // debug, print the Hexx matrices
    // for (const auto& isp_IJkH: this->Hexx)
    // {
    //     const auto& isp = isp_IJkH.first;
    //     for (const auto& I_JkH: isp_IJkH.second)
    //     {
    //         const auto& I = I_JkH.first;
    //         for (const auto& J_kH: I_JkH.second)
    //         {
    //             const auto& J = J_kH.first;
    //             for (const auto& k_H: J_kH.second)
    //             {
    //                 cout << isp << " " << I << " " << J << " {" << k_H.first << "} whole
size: "
    //                 << k_H.second->size << endl; print_complex_matrix("", *k_H.second);
    //             }
    //         }
    //     }
    // }

#else
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf(
            "Error: trying build EXX orbital energy with LibRI, but the program is not compiled
" "against LibRI\n");
    }
    throw std::logic_error("compilation");
    mpi_comm_global_h.barrier();
#endif

    is_rspace_build_ = true;
} */

void Exx::build_KS(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_target,
                   const std::vector<Vector3_Order<double>> &kfrac_target)
{
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

    assert(this->is_rspace_build_);
    // Reset k-space matrices built from last call
    if (this->is_kspace_built_)
    {
        utils::lib_printf("Warning: reset EXX k-space matrices\n");
        this->reset_kspace();
    }

    const auto &n_aos = this->mf_.get_n_aos();
    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_bands = this->mf_.get_n_bands();
    const auto &n_soc = this->mf_.get_n_soc();
    const int n_target_kpoints = (!wfc_target.empty() && !wfc_target.front().empty())
                                     ? static_cast<int>(wfc_target.front().front().size())
                                     : 0;
    if (static_cast<int>(kfrac_target.size()) != n_target_kpoints && mpi_comm_global_h.is_root())
    {
        utils::lib_printf("Warning: EXX build_KS sees inconsistent k-point metadata: "
                          "kfrac_target=%zu, wfc_target=%d. Using the wavefunction count.\n",
                          kfrac_target.size(), n_target_kpoints);
    }

    // prepare scalapack array descriptors
    Array_Desc desc_nao_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband_fb(blacs_ctxt_global_h);

    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);
    const bool use_root_dense_projection =
        use_abacus_ibz_root_projection(n_target_kpoints, this->mf_.get_n_kpoints())
        || force_root_dense_ks_projection();
    Array_Desc desc_nao_nao_fb(blacs_ctxt_global_h);
    if (use_root_dense_projection)
    {
        desc_nao_nao_fb.init(n_aos, n_aos, n_aos, n_aos, 0, 0);
    }

    // local 2D-block submatrices
    auto Hexx_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<complex<double>>(desc_nband_nao, MAJOR::COL);
    auto Hexx_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);
    auto Hexx_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);

    const auto set_IJ_naonao = LIBRPA::utils::get_necessary_IJ_from_block_2D(
        atomic_basis_wfc, atomic_basis_wfc, desc_nao_nao);
    const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_naonao);

    for (int isp = 0; isp < n_spins; isp++)
    {
        for (int isoc1 = 0; isoc1 < n_soc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 < n_soc; isoc2++)
            {
                // collect necessary data
                Profiler::start("build_real_space_exx_5", "Collect Hexx IJ from world");
                map<Vector3_Order<int>, map<atom_t, map<atom_t, Matz>>> exx_is;
                if (Params::use_soc)
                {
                    if (this->exx_cplx.count(isp) && this->exx_cplx.at(isp).count(isoc1)
                        && this->exx_cplx.at(isp).at(isoc1).count(isoc2))
                    {
                        exx_is = this->exx_cplx.at(isp).at(isoc1).at(isoc2);
                    }
                }
                else
                {
                    if (this->exx.count(isp) && this->exx.at(isp).count(isoc1)
                        && this->exx.at(isp).at(isoc1).count(isoc2))
                    {
                        for (const auto &R_IJ_exx : this->exx.at(isp).at(isoc1).at(isoc2))
                        {
                            const auto R = R_IJ_exx.first;
                            for (const auto &I_J_exx : R_IJ_exx.second)
                            {
                                const auto I = I_J_exx.first;
                                for (const auto &J_exx : I_J_exx.second)
                                {
                                    const auto J = J_exx.first;
                                    exx_is[R][I][J] = J_exx.second.to_complex();
                                }
                            }
                        }
                    }
                }

                std::map<int, std::map<std::pair<int, std::array<int, 3>>,
                                       RI::Tensor<std::complex<double>>>>
                    exx_I_JR_local;
                for (const auto &R_IJ_exx : exx_is)
                {
                    const auto R = R_IJ_exx.first;
                    for (const auto &I_J_exx : R_IJ_exx.second)
                    {
                        const auto I = I_J_exx.first;
                        const auto &n_I = atomic_basis_wfc.get_atom_nb(I);
                        for (const auto &J_exx : I_J_exx.second)
                        {
                            const auto J = J_exx.first;
                            const auto &n_J = atomic_basis_wfc.get_atom_nb(J);
                            if (complex_array_has_nonfinite(J_exx.second.ptr(), J_exx.second.size()))
                            {
                                std::ostringstream oss;
                                oss << "Non-finite EXX real-space block appears before Fourier"
                                    << " at I=" << I << " J=" << J << " R=(" << R.x << ","
                                    << R.y << "," << R.z << ")";
                                throw std::runtime_error(oss.str());
                            }
                            const std::array<int, 3> Ra{R.x, R.y, R.z};
                            exx_I_JR_local[I][{J, Ra}] =
                                RI::Tensor<std::complex<double>>({n_I, n_J}, J_exx.second.sptr());
                        }
                    }
                }
                // Collect the IJ pair of Hs with all R for Fourier transform
                auto exx_I_JR = comm_map2_first(mpi_comm_global_h.comm, exx_I_JR_local,
                                                Iset_Jset.first, Iset_Jset.second);
                exx_I_JR_local.clear();

                std::map<int, std::map<std::pair<int, std::array<int, 3>>,
                                       RI::Tensor<std::complex<double>>>>
                    exx_I_JR_local_bvk;
                if (coord_frac.size() > 0)
                {
                    for (const auto &I_exxJR : exx_I_JR)
                    {
                        const auto &I = I_exxJR.first;
                        for (const auto &JR_exx : I_exxJR.second)
                        {
                            const auto &J = JR_exx.first.first;
                            const auto &R = JR_exx.first.second;

                            auto distsq = std::numeric_limits<double>::max();
                            Vector3<int> R_IJ;
                            std::array<int, 3> R_bvk;
                            for (int i = -1; i < 2; i++)
                            {
                                R_IJ.x = i * this->period_.x + R[0];
                                for (int j = -1; j < 2; j++)
                                {
                                    R_IJ.y = j * this->period_.y + R[1];
                                    for (int k = -1; k < 2; k++)
                                    {
                                        R_IJ.z = k * this->period_.z + R[2];
                                        const auto diff =
                                            (Vector3<double>(coord_frac[I][0], coord_frac[I][1],
                                                             coord_frac[I][2]) -
                                             Vector3<double>(coord_frac[J][0], coord_frac[J][1],
                                                             coord_frac[J][2]) -
                                             Vector3<double>(R_IJ.x, R_IJ.y, R_IJ.z)) *
                                            latvec;
                                        const auto norm2 = diff.norm2();
                                        if (norm2 < distsq)
                                        {
                                            distsq = norm2;
                                            R_bvk[0] = R_IJ.x;
                                            R_bvk[1] = R_IJ.y;
                                            R_bvk[2] = R_IJ.z;
                                        }
                                    }
                                }
                            }
                            exx_I_JR_local_bvk[I][{J, R_bvk}] = JR_exx.second;
                        }
                    }
                }

                Profiler::stop("build_real_space_exx_5");

                {
                    std::ostringstream oss;
                    oss << Params::output_dir << "debug_exx_HR_before_fourier_spin" << isp
                        << "_soc" << isoc1 << "_" << isoc2 << ".txt";
                    dump_tensor_map_summary(exx_I_JR, oss.str());
                }

                utils::lib_printf("Task %4d: tensor communicate elapsed time: %f\n",
                                  mpi_comm_global_h.myid,
                                  Profiler::get_wall_time_last("build_real_space_exx_5"));
                // cout << I_JallR_Hs << endl;

                for (int ik = 0; ik < n_target_kpoints; ik++)
                {
                    Hexx_nao_nao.zero_out();
                    Profiler::start("build_real_space_exx_6", "Hexx IJ -> 2D block");
                    const auto &kfrac = kfrac_target[ik];
                    const std::function<complex<double>(const int &,
                                                        const std::pair<int, std::array<int, 3>> &)>
                        fourier =
                            [kfrac](const int &I, const std::pair<int, std::array<int, 3>> &J_Ra)
                    {
                        const auto &Ra = J_Ra.second;
                        Vector3<double> R_IJ(Ra[0], Ra[1], Ra[2]);
                        const auto ang = (kfrac * R_IJ) * TWO_PI;
                        return complex<double>{std::cos(ang), std::sin(ang)};
                    };
                    const auto& exx_I_JR_active =
                        coord_frac.size() > 0 ? exx_I_JR_local_bvk : exx_I_JR;
                    collect_block_from_IJ_storage_tensor_transform(
                        Hexx_nao_nao, desc_nao_nao, atomic_basis_wfc, atomic_basis_wfc, fourier,
                        exx_I_JR_active);
                    if (complex_array_has_nonfinite(Hexx_nao_nao.ptr(), Hexx_nao_nao.size()))
                    {
                        std::ostringstream oss;
                        oss << "Non-finite EXX k-space AO matrix appears after Fourier transform"
                            << " at ik=" << ik << " k=(" << kfrac.x << "," << kfrac.y << ","
                            << kfrac.z << ")";
                        throw std::runtime_error(oss.str());
                    }
                    Profiler::stop("build_real_space_exx_6");
                    if (use_root_dense_projection)
                    {
                        Profiler::start("build_real_space_exx_7",
                                        "Rotate Hexx ij -> KS with root dense IBZ projection");
                        auto Hexx_nao_nao_fb =
                            init_local_mat<complex<double>>(desc_nao_nao_fb, MAJOR::COL);
                        ScalapackConnector::pgemr2d_f(
                            n_aos, n_aos, Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                            Hexx_nao_nao_fb.ptr(), 1, 1, desc_nao_nao_fb.desc,
                            desc_nao_nao_fb.ictxt());

                        ComplexMatrix Hexx_nband_nband_dense;
                        if (mpi_comm_global_h.is_root())
                        {
                            ComplexMatrix Hexx_nao_nao_dense(n_aos, n_aos);
                            for (int iao = 0; iao != n_aos; ++iao)
                            {
                                for (int jao = 0; jao != n_aos; ++jao)
                                {
                                    Hexx_nao_nao_dense(iao, jao) = Hexx_nao_nao_fb(iao, jao);
                                }
                            }
                            maybe_dump_hexx_ao_debug(
                                Hexx_nao_nao_dense, isp, isoc1, isoc2, ik, kfrac);

                            const auto& wfc_isp1_k = wfc_target[isp][isoc1][ik];
                            const auto& wfc_isp2_k = wfc_target[isp][isoc2][ik];
                            // The symmetry-on ABACUS starting point only stores IBZ KS states.
                            // Reproject the fully assembled AO matrix on the root rank using the
                            // dense IBZ eigenvectors instead of rebuilding fake ScaLAPACK blocks
                            // from possibly absent full-BZ wavefunction storage.
                            Hexx_nband_nband_dense =
                                (-1.0) * (conj(wfc_isp1_k) * Hexx_nao_nao_dense
                                          * transpose(wfc_isp2_k, false));
                            if (complex_matrix_has_nonfinite(Hexx_nband_nband_dense))
                            {
                                std::ostringstream oss;
                                oss << "Non-finite EXX KS matrix appears on the root projection path"
                                    << " before broadcast at ik=" << ik << " k=(" << kfrac.x << ","
                                    << kfrac.y << "," << kfrac.z << ")";
                                throw std::runtime_error(oss.str());
                            }
                        }
                        mpi_comm_global_h.broadcast_ComplexMatrix(Hexx_nband_nband_dense, 0);
                        if (complex_matrix_has_nonfinite(Hexx_nband_nband_dense))
                        {
                            std::ostringstream oss;
                            oss << "Non-finite EXX KS matrix appears on the root projection path"
                                << " after broadcast at ik=" << ik << " k=(" << kfrac.x << ","
                                << kfrac.y << "," << kfrac.z << ")";
                            throw std::runtime_error(oss.str());
                        }
                        Profiler::stop("build_real_space_exx_7");

                        Profiler::start("build_real_space_exx_8", "Collect Eexx to root process");
                        if (this->exx_is_ik_KS.count(isp) == 0 || this->exx_is_ik_KS[isp].count(ik) == 0)
                        {
                            this->exx_is_ik_KS[isp][ik] = Matz(
                                n_bands, n_bands, Hexx_nband_nband_dense.c, MAJOR::ROW, MAJOR::COL);
                        }
                        else
                        {
                            this->exx_is_ik_KS[isp][ik] += Matz(
                                n_bands, n_bands, Hexx_nband_nband_dense.c, MAJOR::ROW, MAJOR::COL);
                        }
                        for (int ib = 0; ib != n_bands; ib++)
                        {
                            this->Eexx[isp][ik][ib] += Hexx_nband_nband_dense(ib, ib).real();
                        }
                        maybe_dump_hexx_ks_debug(
                            Hexx_nband_nband_dense, isp, isoc1, isoc2, ik, kfrac);
                        Profiler::stop("build_real_space_exx_8");
                        continue;
                    }

                    // utils::lib_printf("%s\n", str(Hexx_nao_nao).c_str());
                    const auto &wfc_isp1_k = wfc_target[isp][isoc1][ik];
                    const auto &wfc_isp2_k = wfc_target[isp][isoc2][ik];
                    blacs_ctxt_global_h.barrier();
                    const auto wfc1_block =
                        get_local_mat(wfc_isp1_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                    const auto wfc2_block =
                        get_local_mat(wfc_isp2_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                    // utils::lib_printf("%s\n", str(wfc_block).c_str());
                    // utils::lib_printf("%s\n", desc_nao_nao.info_desc().c_str());
                    // utils::lib_printf("%s\n", desc_nband_nao.info_desc().c_str());
                    Profiler::start("build_real_space_exx_7", "Rotate Hexx ij -> KS");
                    ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                                wfc1_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc, 0.0,
                                                temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                    ScalapackConnector::pgemm_f(
                        'N', 'C', n_bands, n_bands, n_aos, -1.0, temp_nband_nao.ptr(), 1, 1,
                        desc_nband_nao.desc, wfc2_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                        Hexx_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                    Profiler::stop("build_real_space_exx_7");

                    // collect to master
                    Profiler::start("build_real_space_exx_8", "Collect Eexx to root process");
                    ScalapackConnector::pgemr2d_f(n_bands, n_bands, Hexx_nband_nband.ptr(), 1, 1,
                                                  desc_nband_nband.desc, Hexx_nband_nband_fb.ptr(),
                                                  1, 1, desc_nband_nband_fb.desc,
                                                  desc_nband_nband_fb.ictxt());
                    if (this->exx_is_ik_KS.count(isp) == 0 ||
                        this->exx_is_ik_KS[isp].count(ik) == 0)
                    {
                        this->exx_is_ik_KS[isp][ik] =
                            init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                    }

                    this->exx_is_ik_KS[isp][ik] += Hexx_nband_nband_fb.copy();
                    // cout << "Hexx_nband_nband_fb isp " << isp  << " ik " << ik << endl <<
                    // Hexx_nband_nband_fb;
                    if (blacs_ctxt_global_h.myid == 0)
                    {
                        for (int ib = 0; ib != n_bands; ib++){
                            this->Eexx[isp][ik][ib] += Hexx_nband_nband_fb(ib, ib).real();
                        ComplexMatrix hexx_ks_dense(n_bands, n_bands);
                        for (int ib = 0; ib != n_bands; ++ib)
                        {
                            for (int jb = 0; jb != n_bands; ++jb)
                            {
                                hexx_ks_dense(ib, jb) = Hexx_nband_nband_fb(ib, jb);
                            }
                        }
                        maybe_dump_hexx_ks_debug(hexx_ks_dense, isp, isoc1, isoc2, ik, kfrac);
                    }
                    Profiler::stop("build_real_space_exx_8");
                }
            }
        }
    }
}
void Exx::build_KS_kgrid0() { this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_); }
void Exx::build_KS_kgrid() { this->build_KS(this->mf_.get_eigenvectors(), this->kfrac_list_); }

void Exx::build_KS_band(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_band,
                        const std::vector<Vector3_Order<double>> &kfrac_band)
{
    this->build_KS(wfc_band, kfrac_band);
}

void Exx::reset_rspace()
{
    this->exx.clear();
    this->is_rspace_build_ = false;
}

void Exx::reset_kspace()
{
    this->exx_is_ik_KS.clear();
    this->Eexx.clear();
    this->is_kspace_built_ = false;
}

template void Exx::build<double>(const Cs_LRI &, const vector<Vector3_Order<int>> &,
                                 const atpair_R_mat_t &);
template void Exx::build<std::complex<double>>(const Cs_LRI &, const vector<Vector3_Order<int>> &,
                                               const atpair_R_mat_t &);

}  // namespace LIBRPA
