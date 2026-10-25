#include "gw_decay_bound.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>

#include "../gpu/la_connector.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/error.h"
#include "../utils/constants.h"
#include "epsilon.h"
#include "utils_atomic_basis_blacs.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#endif

namespace librpa_int
{
namespace
{
using Complex = std::complex<double>;
using AtomBlocks = atom_mapping<ComplexMatrix>::pair_t_old;
using TensorMap =
    std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<Complex>>>;

void collective_error(const MpiCommHandler &comm, const std::string &error)
{
    int failed = !error.empty();
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm.comm);
    if (failed)
        throw LIBRPA_RUNTIME_ERROR("GW body decay prepass: " +
                                   (error.empty() ? "another rank reported invalid input" : error));
}

using KKey = std::array<long long, 3>;
KKey kkey(const Vector3_Order<double> &x)
{
    constexpr long long scale = 100000000;
    const auto component = [](double a)
    {
        if (!std::isfinite(a)) throw LIBRPA_RUNTIME_ERROR("nonfinite k coordinate");
        const auto i = std::llround((a - std::floor(a)) * scale);
        return i % scale;
    };
    return {component(x.x), component(x.y), component(x.z)};
}

struct EnergyFrame
{
    Vector3_Order<double> k;
    int source;
};

struct EnergyBounds
{
    double gap;
    std::array<double, 2> tail{};
    std::map<Vector3_Order<double>, std::array<double, 2>> delta;
};

EnergyBounds energy_bounds(const Chi0 &chi, const std::vector<Vector3_Order<double>> &qs,
                           const int n_sigc, const int exclude_sigc)
{
    const auto &mf = chi.mf;
    const int nk = mf.get_n_kpoints();
    const int nb = mf.get_n_bands();
    const int full_nk = chi.pbc.get_n_cells_bvk();
    const bool reduced = nk != full_nk;
    if (mf.get_n_spins() > 1 && reduced)
        throw LIBRPA_RUNTIME_ERROR(
            "collinear multi-spin IBZ decay metadata requires explicit spin-channel sewing");
    const double spin_scale = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    std::vector<std::vector<std::vector<int>>> occupied(
        mf.get_n_spins(), std::vector<std::vector<int>>(nk, std::vector<int>(nb)));
    std::vector<EnergyFrame> frames;
    double eo = -std::numeric_limits<double>::infinity();
    double eu = std::numeric_limits<double>::infinity();
    for (int k = 0; k != nk; ++k)
    {
        double wk = 1.0 / nk;
        if (reduced)
        {
            const auto &star =
                find_symmetry_kstar_for_ibz_kpoint(chi.symmetry_context, chi.pbc.kfrac_list.at(k));
            wk = double(star.members.size()) / full_nk;
            for (const auto &member : star.members) frames.push_back({member.k_bz, k});
        }
        else
            frames.push_back({chi.pbc.kfrac_list.at(k), k});
        for (int s = 0; s != mf.get_n_spins(); ++s)
            for (int b = 0; b != nb; ++b)
            {
                const double e = mf.get_eigenvals()[s](k, b);
                const double f = spin_scale * mf.get_weight()[s](k, b) / wk;
                if (!std::isfinite(e) || !std::isfinite(f) ||
                    std::min(std::abs(f), std::abs(f - 1.0)) > 1e-8)
                    throw LIBRPA_RUNTIME_ERROR("finite integer occupations are required");
                occupied[s][k][b] = f > 0.5;
                if (f > 0.5)
                    eo = std::max(eo, e);
                else
                    eu = std::min(eu, e);
            }
    }
    const double mu = mf.get_efermi();
    if (!std::isfinite(mu) || !std::isfinite(eo) || !std::isfinite(eu) || !(eu > eo) ||
        mu < eo - 1e-10 || mu > eu + 1e-10)
        throw LIBRPA_RUNTIME_ERROR("chemical potential must be in the closed global gap");
    const int used_chi = chi.nbands_G < 0 ? nb : chi.nbands_G;
    const int used_sig = n_sigc < 0 ? nb : n_sigc;
    if (used_chi > nb || chi.n_bands_exclude < 0 || chi.n_bands_exclude > used_chi ||
        used_sig > nb || exclude_sigc < 0 || exclude_sigc > used_sig)
        throw LIBRPA_RUNTIME_ERROR("inconsistent retained band windows");
    struct RetainedExtrema
    {
        double occupied_min = std::numeric_limits<double>::infinity();
        double empty_max = -std::numeric_limits<double>::infinity();
    };
    std::vector<std::vector<RetainedExtrema>> extrema(
        mf.get_n_spins(), std::vector<RetainedExtrema>(nk));
    for (int s = 0; s != mf.get_n_spins(); ++s)
        for (int k = 0; k != nk; ++k)
            for (int b = chi.n_bands_exclude; b != used_chi; ++b)
            {
                const double e = mf.get_eigenvals()[s](k, b);
                if (occupied[s][k][b])
                    extrema[s][k].occupied_min = std::min(extrema[s][k].occupied_min, e);
                else
                    extrema[s][k].empty_max = std::max(extrema[s][k].empty_max, e);
            }
    EnergyBounds out{eu - eo};
    for (int s = 0; s != mf.get_n_spins(); ++s)
        for (int k = 0; k != nk; ++k)
            for (int b = exclude_sigc; b != used_sig; ++b)
            {
                const int side = occupied[s][k][b] ? 1 : 0;
                const double offset =
                    side ? mu - mf.get_eigenvals()[s](k, b) : mf.get_eigenvals()[s](k, b) - mu;
                out.tail[side] = std::max(out.tail[side], offset);
            }
    std::map<KKey, int> index;
    for (int i = 0; i != static_cast<int>(frames.size()); ++i)
        if (!index.emplace(kkey(frames[i].k), i).second)
            throw LIBRPA_RUNTIME_ERROR("duplicate restored full k frame");
    if (index.size() != static_cast<std::size_t>(full_nk))
        throw LIBRPA_RUNTIME_ERROR("restored energy frames do not cover full BZ");
    for (const auto &q : qs)
    {
        std::array<double, 2> maxima{};
        const Vector3_Order<double> qfrac{chi.pbc.latvec * q};
        for (const auto &frame : frames)
        {
            const int k = frame.source;
            const auto pos = index.find(kkey(frame.k + qfrac));
            if (pos == index.end()) throw LIBRPA_RUNTIME_ERROR("q does not close the k mesh");
            const int kp = frames[pos->second].source;
            for (int s = 0; s != mf.get_n_spins(); ++s)
            {
                // The validated global gap fixes every eligible pair's sign.
                // max_{o(k),u(k+q)} (e_u-e_o) = max_u e_u - min_o e_o;
                // the negative-time side reverses the two k frames, not spin.
                for (int side = 0; side != 2; ++side)
                {
                    const auto &from = extrema[s][side == 0 ? k : kp];
                    const auto &to = extrema[s][side == 0 ? kp : k];
                    if (std::isfinite(from.occupied_min) && std::isfinite(to.empty_max))
                    {
                        const double delta = to.empty_max - from.occupied_min;
                        maxima[side] = std::max(maxima[side], delta);
                    }
                }
            }
        }
        out.delta[q] = maxima;
    }
    return out;
}

TensorMap pack(const AtomBlocks &blocks, const Vector3_Order<double> &q, const AtomicBasis &basis)
{
    TensorMap out;
    for (const auto &[I, columns] : blocks)
        for (const auto &[J, matrix] : columns)
        {
            if (matrix.nr != int(basis[I]) || matrix.nc != int(basis[J]))
                throw LIBRPA_RUNTIME_ERROR("Gram atom block has incorrect dimensions");
            for (int i = 0; i != matrix.size; ++i)
                if (!std::isfinite(matrix.c[i].real()) || !std::isfinite(matrix.c[i].imag()))
                    throw LIBRPA_RUNTIME_ERROR("nonfinite atom matrix");
            auto storage = std::make_shared<std::valarray<Complex>>(matrix.c, matrix.size);
            out[I][{J, q.to_array()}] = RI::Tensor<Complex>({basis[I], basis[J]}, storage);
        }
    return out;
}

TensorMap pack_checked(const AtomBlocks &blocks, const Vector3_Order<double> &q,
                       const AtomicBasis &basis, const MpiCommHandler &comm)
{
    TensorMap tensors;
    std::string error;
    try
    {
        tensors = pack(blocks, q, basis);
    }
    catch (const std::exception &failure)
    {
        error = failure.what();
    }
    collective_error(comm, error);
    return tensors;
}

Matz collect(TensorMap &input, const Vector3_Order<double> &q, const AtomicBasis &basis,
             const ArrayDesc &desc, const MpiCommHandler &comm, const bool hermitian)
{
    std::set<int> rows, columns;
    const auto needed = get_necessary_IJ_from_block_2D(basis, basis, desc);
    for (const auto &[I, J] : needed)
    {
        rows.insert(hermitian ? std::min(I, J) : I);
        columns.insert(hermitian ? std::max(I, J) : J);
    }
#ifdef LIBRPA_USE_LIBRI
    const auto selected =
        RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm.comm, input, rows, columns);
    auto local = init_local_mat<Complex>(desc, MAJOR::COL);
    if (hermitian)
        collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(local, desc, basis, q.to_array(), true,
                                                             Complex{1.0}, selected, MAJOR::ROW);
    else
        collect_block_from_ordered_IJ_Tensor_sparse_zero_missing(local, desc, basis, q.to_array(),
                                                                 Complex{1.0}, selected);
    return local;
#else
    throw LIBRPA_RUNTIME_ERROR("GW body Gram requires LibRI");
#endif
}

void check_hermitian(const Matz &matrix, const ArrayDesc &desc, const MpiCommHandler &comm)
{
    auto dagger = init_local_mat<Complex>(desc, MAJOR::COL);
    ScalapackConnector::pgeadd_f('C', desc.m(), desc.n(), Complex{1.0}, matrix.ptr(), 1, 1,
                                 desc.desc, Complex{0.0}, dagger.ptr(), 1, 1, desc.desc);
    double error = 0.0, norm = 0.0;
    for (std::size_t i = 0; i != matrix.size(); ++i)
    {
        error = std::max(error, std::abs(matrix.ptr()[i] - dagger.ptr()[i]));
        norm = std::max(norm, std::abs(matrix.ptr()[i]));
    }
    double values[2]{error, norm};
    MPI_Allreduce(MPI_IN_PLACE, values, 2, MPI_DOUBLE, MPI_MAX, comm.comm);
    collective_error(comm, values[0] <= 1e-10 * std::max(1.0, values[1])
                               ? ""
                               : "non-Hermitian finite body matrix");
}
}  // namespace

GwBodyDecayBound compute_gw_body_decay_bound(
    Chi0 &chi, const Cs_LRI &Cs, const std::vector<atpair_t> &atpairs, const AtomicBasis &abf_Cs,
    std::map<Vector3_Order<double>, ComplexMatrix> &sinvS, const atpair_k_cplx_mat_t &coulomb,
    const BlacsCtxtHandler &blacs, const double threshold, const int n_sigc, const int exclude_sigc,
    const bool use_gpu, const bool use_elpa)
{
    const auto &comm = chi.comm_h;
    std::string error;
    const int loaded = chi.mf.get_n_bands();
    const int used_chi = chi.nbands_G < 0 ? loaded : chi.nbands_G;
    const int used_sig = n_sigc < 0 ? loaded : n_sigc;
    if (used_chi > loaded || chi.n_bands_exclude < 0 || chi.n_bands_exclude > used_chi ||
        used_sig > loaded || exclude_sigc < 0 || exclude_sigc > used_sig)
        error = "inconsistent retained band windows before Gram contraction";
    if (use_gpu || use_elpa)
        error = "body decay prepass currently supports the CPU ScaLAPACK backend only";
    // GF pruning need not preserve a finite PSD Lehmann representation.
    if (!chi.get_band_selection().empty() || chi.gf_threshold != 0 || chi.libri_threshold_G != 0)
        error = "body bound requires an unpruned finite GW band response";
    if (!std::isfinite(threshold)) error = "nonfinite Coulomb filtering threshold";
    collective_error(comm, error);
    GwBodyDecayBound result;
    result.grams = chi.build_transition_grams(Cs, atpairs, abf_Cs, sinvS, blacs);
    const auto &qs = result.grams.qpoints;
    EnergyBounds energy;
    try
    {
        energy = energy_bounds(chi, qs, n_sigc, exclude_sigc);
    }
    catch (const std::exception &failure)
    {
        error = failure.what();
    }
    collective_error(comm, error);
    result.gap = energy.gap;
    result.electron_tail = energy.tail;
    const int n = chi.atbasis_abf.nb_total;
    ArrayDesc desc(blacs);
    desc.init_square_blk_capped(n, n, 128, 0, 0);
    // Only auxiliary matrices are distributed/reconstructed. No Cs or WFC gather.
    std::map<Vector3_Order<double>, Matz> vq;
    const auto &view = chi.qpoint_view();
    const auto &source_qs =
        view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL ? view.representatives : qs;
    for (const auto &q : source_qs)
    {
        AtomBlocks blocks;
        int present = 0;
        try
        {
            for (const auto &[I, columns] : coulomb)
                for (const auto &[J, qvalues] : columns)
                {
                    const auto found = qvalues.find(q);
                    if (found == qvalues.end()) continue;
                    if (!found->second) throw LIBRPA_RUNTIME_ERROR("null dielectric Coulomb block");
                    blocks[I][J] = *found->second;
                    present = 1;
                }
        }
        catch (const std::exception &failure)
        {
            error = failure.what();
        }
        MPI_Allreduce(MPI_IN_PLACE, &present, 1, MPI_INT, MPI_MAX, comm.comm);
        if (!present) error = "missing full-q dielectric Coulomb (no implicit conjugation)";
        collective_error(comm, error);
        auto tensors = pack_checked(blocks, q, chi.atbasis_abf, comm);
        vq.emplace(q, collect(tensors, q, chi.atbasis_abf, desc, comm, true));
    }
    if (view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
        vq = restore_symmetry_dense_wq_map(vq, chi.pbc, view, chi.symmetry_context, chi.atbasis_abf,
                                           desc);
    for (const auto &q : qs)
    {
        auto &v = vq.at(q);
        check_hermitian(v, desc, comm);
        auto vectors = init_local_mat<Complex>(desc, MAJOR::COL);
        std::vector<double> eigen(n);
        std::size_t filtered = 0;
        // Integer power extracts the native eigenvectors before any fractional
        // power: meaningful retained negative channels must fail explicitly.
        auto discarded = LaConnector::power_hemat_la(v, desc, vectors, desc, filtered, eigen.data(),
                                                     1.0, threshold, use_gpu, use_elpa);
        discarded.clear();
        error.clear();
        for (int i = 0; i != n; ++i)
            if (!std::isfinite(eigen[i]) || (eigen[i] >= threshold && eigen[i] < 0.0))
                error = "dielectric Coulomb has a nonfinite or retained negative eigenvalue";
        collective_error(comm, error);
        auto scaled = vectors.copy();
        for (int j = 0; j != desc.n_loc(); ++j)
        {
            const double value = eigen[desc.indx_l2g_c(j)];
            const double factor = value < threshold ? 0.0 : std::sqrt(value);
            for (int i = 0; i != desc.m_loc(); ++i) scaled(i, j) *= factor;
        }
        auto sqrtv = init_local_mat<Complex>(desc, MAJOR::COL);
        ScalapackConnector::pgemm_f('N', 'C', n, n, n, Complex{1.0}, scaled.ptr(), 1, 1, desc.desc,
                                    vectors.ptr(), 1, 1, desc.desc, Complex{0.0}, sqrtv.ptr(), 1, 1,
                                    desc.desc);
        GwDecayQBound bound;
        bound.transition_max = energy.delta.at(q);
        bound.coulomb_min = eigen.front();
        bound.coulomb_filtered = filtered;
        for (int side = 0; side != 2; ++side)
        {
            const auto &blocks =
                side == 0 ? result.grams.positive.at(q) : result.grams.negative.at(q);
            auto tensors = pack_checked(blocks, q, chi.atbasis_abf, comm);
            auto gram = collect(tensors, q, chi.atbasis_abf, desc, comm, false);
            check_hermitian(gram, desc, comm);
            auto product = init_local_mat<Complex>(desc, MAJOR::COL);
            auto weighted = init_local_mat<Complex>(desc, MAJOR::COL);
            ScalapackConnector::pgemm_f('N', 'N', n, n, n, Complex{1.0}, sqrtv.ptr(), 1, 1,
                                        desc.desc, gram.ptr(), 1, 1, desc.desc, Complex{0.0},
                                        product.ptr(), 1, 1, desc.desc);
            ScalapackConnector::pgemm_f('N', 'N', n, n, n, Complex{1.0}, product.ptr(), 1, 1,
                                        desc.desc, sqrtv.ptr(), 1, 1, desc.desc, Complex{0.0},
                                        weighted.ptr(), 1, 1, desc.desc);
            check_hermitian(weighted, desc, comm);
            double trace = 0.0;
            for (int i = 0; i != desc.m_loc(); ++i)
                for (int j = 0; j != desc.n_loc(); ++j)
                    if (desc.indx_l2g_r(i) == desc.indx_l2g_c(j)) trace += weighted(i, j).real();
            MPI_Allreduce(MPI_IN_PLACE, &trace, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
            auto ignored = LaConnector::power_hemat_la(weighted, desc, vectors, desc, filtered,
                                                       eigen.data(), 1.0, 0.0, use_gpu, use_elpa);
            ignored.clear();
            collective_error(comm, eigen.front() >= -1e-10 && std::isfinite(eigen.back())
                                       ? ""
                                       : "signed transition Gram is not finite PSD");
            bound.coupling_max[side] = eigen.back();
            bound.coupling_trace[side] = trace;
            bound.pole_upper[side] = bound.transition_max[side] + eigen.back();
            result.upper =
                std::max(result.upper, result.electron_tail[side] + bound.pole_upper[side]);
        }
        result.q_bounds.emplace(q, bound);
        v.clear();
    }
    return result;
}

GwOpticalDecayBound augment_gw_body_decay_bound(
    GwBodyDecayBound body, OpticalTransitionGrams optical, const AtomicBasis &basis,
    const PeriodicBoundaryData &pbc, const MpiCommHandler &comm,
    const atpair_k_cplx_mat_t &coulomb, const BlacsCtxtHandler &blacs,
    const double threshold, const int option)
{
    GwOpticalDecayBound out;
    out.body = std::move(body);
    out.optical = std::move(optical);
    out.upper = out.body.upper;
    const int n = basis.nb_total;
    const double volume = std::abs(pbc.latvec.Det());
    const double head_factor = 4.0 * PI / volume;
    std::string error;
    if (n <= 0 || n > std::numeric_limits<int>::max() / 3 ||
        !std::isfinite(head_factor) || !(head_factor > 0.0) ||
        !std::isfinite(threshold) || (option != 3 && option != 53))
        error = "finite3D optical bound requires valid basis, volume and option3/53";
    if (option == 53)
    {
        bool has_s = false;
        if (basis.has_l_shells())
            for (int atom = 0; atom != int(basis.n_atoms); ++atom)
                for (const int l : basis.get_l_shells(atom)) has_s = has_s || l == 0;
        if (!has_s) error = "Gamma s-head optical bound requires auxiliary s-shell metadata";
    }
    for (int side = 0; side != 2; ++side)
    {
        if (out.optical.dipole[side].nr() != 3 || out.optical.dipole[side].nc() != 3 ||
            out.optical.charge_dipole[side].nr() != n ||
            out.optical.charge_dipole[side].nc() != 3)
            error = "optical endpoint Gram dimensions differ from the body basis";
        for (const auto *probe : {&out.optical.dipole[side], &out.optical.charge_dipole[side]})
            for (std::size_t i = 0; i != probe->size(); ++i)
                if (!std::isfinite(probe->ptr()[i].real()) ||
                    !std::isfinite(probe->ptr()[i].imag()))
                    error = "nonfinite optical endpoint Gram";
    }
    if (!std::isfinite(out.body.upper)) error = "nonfinite body bound before optical augmentation";
    auto gamma = std::find_if(out.body.grams.qpoints.begin(), out.body.grams.qpoints.end(),
                               [](const auto &q) { return q.norm() < 1e-12; });
    if (gamma == out.body.grams.qpoints.end()) error = "optical bound requires Gamma body Grams";
    collective_error(comm, error);
    const auto q = *gamma;
    ArrayDesc desc(blacs);
    desc.init_square_blk_capped(n, n, 128, 0, 0);
    AtomBlocks blocks;
    for (const auto &[I, columns] : coulomb)
        for (const auto &[J, values] : columns)
        {
            const auto found = values.find(q);
            if (found != values.end() && found->second) blocks[I][J] = *found->second;
        }
    auto tensors = pack_checked(blocks, q, basis, comm);
    auto v = collect(tensors, q, basis, desc, comm, true);
    check_hermitian(v, desc, comm);
    const auto raw_s_head = option == 53 ? compute_gamma_s_head_vector(basis, v, &desc)
                                          : std::vector<double>{};
    auto vectors = init_local_mat<Complex>(desc, MAJOR::COL);
    std::vector<double> eigen(n);
    std::size_t filtered = 0;
    auto scaled = LaConnector::power_hemat_la(v, desc, vectors, desc, filtered, eigen.data(),
                                              1.0, threshold, false, false);
    for (const auto value : eigen)
        if (!std::isfinite(value) || (value >= threshold && value < 0.0))
            error = "optical dielectric Coulomb has a retained negative/nonfinite eigenvalue";
    if (filtered == std::size_t(n)) error = "optical bound has no retained Coulomb mode";
    collective_error(comm, error);
    order_optical_coulomb_eigensystem(vectors, scaled, desc, eigen.data());
    out.coulomb_filtered = filtered;
    const int retained = n - filtered;
    // Rebuild sqrt(v) after the explicit optical eigensystem ordering.
    scaled = vectors.copy();
    for (int j = 0; j != desc.n_loc(); ++j)
    {
        const int global_j = desc.indx_l2g_c(j);
        const double factor = global_j < retained ? std::sqrt(eigen[global_j]) : 0.0;
        for (int i = 0; i != desc.m_loc(); ++i) scaled(i, j) *= factor;
    }
    auto sqrtv = init_local_mat<Complex>(desc, MAJOR::COL);
    ScalapackConnector::pgemm_f('N', 'C', n, n, n, Complex{1.0}, scaled.ptr(), 1, 1, desc.desc,
                                vectors.ptr(), 1, 1, desc.desc, Complex{}, sqrtv.ptr(), 1, 1,
                                desc.desc);
    out.head_direction.assign(n, Complex{});
    if (option == 3)
    {
        const int local_column = desc.indx_g2l_c(0);
        if (local_column >= 0)
            for (int i = 0; i != desc.m_loc(); ++i)
                out.head_direction[desc.indx_l2g_r(i)] = vectors(i, local_column);
    }
    else
    {
        std::vector<Complex> coefficients(n);
        for (int i = 0; i != desc.m_loc(); ++i)
            for (int j = 0; j != desc.n_loc(); ++j)
                if (desc.indx_l2g_c(j) < retained)
                    coefficients[desc.indx_l2g_c(j)] +=
                        std::conj(vectors(i, j)) * raw_s_head[desc.indx_l2g_r(i)];
        MPI_Allreduce(MPI_IN_PLACE, coefficients.data(), n, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                      comm.comm);
        for (int i = 0; i != desc.m_loc(); ++i)
            for (int j = 0; j != desc.n_loc(); ++j)
                if (desc.indx_l2g_c(j) < retained)
                    out.head_direction[desc.indx_l2g_r(i)] +=
                        vectors(i, j) * coefficients[desc.indx_l2g_c(j)];
    }
    MPI_Allreduce(MPI_IN_PLACE, out.head_direction.data(), n, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                  comm.comm);
    double norm_squared = 0.0;
    for (const auto z : out.head_direction) norm_squared += std::norm(z);
    collective_error(comm, std::isfinite(norm_squared) &&
                               norm_squared > 100.0 * std::numeric_limits<double>::epsilon()
                               ? "" : "optical head has negligible retained norm");
    for (auto &z : out.head_direction) z /= std::sqrt(norm_squared);
    // T sqrt(v)=(I-u*u^dagger)sqrt(v): F sqrt(v)=0, with no assumption
    // that the head projector commutes with sqrt(v).
    std::vector<Complex> head_sqrtv(n);
    for (int i = 0; i != desc.m_loc(); ++i)
        for (int j = 0; j != desc.n_loc(); ++j)
            head_sqrtv[desc.indx_l2g_c(j)] +=
                std::conj(out.head_direction[desc.indx_l2g_r(i)]) * sqrtv(i, j);
    MPI_Allreduce(MPI_IN_PLACE, head_sqrtv.data(), n, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm.comm);
    auto regular_factor = sqrtv.copy();
    for (int i = 0; i != desc.m_loc(); ++i)
        for (int j = 0; j != desc.n_loc(); ++j)
            regular_factor(i, j) -= out.head_direction[desc.indx_l2g_r(i)] *
                                     head_sqrtv[desc.indx_l2g_c(j)];
    out.augmented_desc = ArrayDesc(blacs);
    out.augmented_desc.init_square_blk_capped(n + 3, n + 3, 128, 0, 0);
    for (int side = 0; side != 2; ++side)
    {
        const auto &gram_blocks = side == 0 ? out.body.grams.positive.at(q)
                                           : out.body.grams.negative.at(q);
        auto gram_tensors = pack_checked(gram_blocks, q, basis, comm);
        auto gram = collect(gram_tensors, q, basis, desc, comm, false);
        auto temp = init_local_mat<Complex>(desc, MAJOR::COL);
        auto regular_gram = init_local_mat<Complex>(desc, MAJOR::COL);
        ScalapackConnector::pgemm_f('N', 'N', n, n, n, Complex{1.0}, regular_factor.ptr(), 1, 1,
                                    desc.desc, gram.ptr(), 1, 1, desc.desc, Complex{}, temp.ptr(),
                                    1, 1, desc.desc);
        ScalapackConnector::pgemm_f('N', 'C', n, n, n, Complex{1.0}, temp.ptr(), 1, 1, desc.desc,
                                    regular_factor.ptr(), 1, 1, desc.desc, Complex{},
                                    regular_gram.ptr(), 1, 1, desc.desc);
        std::vector<Complex> cross(n * 3);
        for (int i = 0; i != desc.m_loc(); ++i)
            for (int j = 0; j != desc.n_loc(); ++j)
                for (int a = 0; a != 3; ++a)
                    cross[3 * desc.indx_l2g_r(i) + a] += std::sqrt(head_factor) *
                        regular_factor(i, j) *
                        out.optical.charge_dipole[side](desc.indx_l2g_c(j), a);
        MPI_Allreduce(MPI_IN_PLACE, cross.data(), n * 3, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                      comm.comm);
        auto &enlarged = out.augmented_grams[side];
        enlarged = init_local_mat<Complex>(out.augmented_desc, MAJOR::COL);
        ScalapackConnector::pgemr2d_f(n, n, regular_gram.ptr(), 1, 1, desc.desc,
                                      enlarged.ptr(), 4, 4, out.augmented_desc.desc, blacs.ictxt);
        for (int i = 0; i != enlarged.nr(); ++i)
            for (int j = 0; j != enlarged.nc(); ++j)
            {
                const int row = out.augmented_desc.indx_l2g_r(i);
                const int column = out.augmented_desc.indx_l2g_c(j);
                if (row < 3 && column < 3)
                    enlarged(i, j) = head_factor * out.optical.dipole[side](row, column);
                else if (row >= 3 && column < 3)
                    enlarged(i, j) = cross[3 * (row - 3) + column];
                else if (row < 3 && column >= 3)
                    enlarged(i, j) = std::conj(cross[3 * (column - 3) + row]);
            }
        check_hermitian(enlarged, out.augmented_desc, comm);
        double trace = 0.0;
        for (int i = 0; i != enlarged.nr(); ++i)
            for (int j = 0; j != enlarged.nc(); ++j)
                if (out.augmented_desc.indx_l2g_r(i) == out.augmented_desc.indx_l2g_c(j))
                    trace += enlarged(i, j).real();
        MPI_Allreduce(MPI_IN_PLACE, &trace, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
        auto diagonalized = enlarged.copy();
        auto eigenvectors = init_local_mat<Complex>(out.augmented_desc, MAJOR::COL);
        std::vector<double> eigenvalues(n + 3);
        std::size_t unused = 0;
        auto ignored = LaConnector::power_hemat_la(
            diagonalized, out.augmented_desc, eigenvectors, out.augmented_desc, unused,
            eigenvalues.data(), 1.0, 0.0, false, false);
        ignored.clear();
        collective_error(comm, eigenvalues.front() >= -1e-10 &&
                                   std::isfinite(eigenvalues.back())
                                   ? "" : "optical augmented transition Gram is not finite PSD");
        out.coupling_max[side] = eigenvalues.back();
        out.coupling_trace[side] = trace;
        out.pole_upper[side] = out.body.q_bounds.at(q).transition_max[side] + eigenvalues.back();
        out.upper = std::max(out.upper, out.body.electron_tail[side] + out.pole_upper[side]);
    }
    return out;
}
}  // namespace librpa_int
