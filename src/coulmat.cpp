#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>

#include "abacus_symmetry.h"
#include "constants.h"
#include "coulmat.h"
#include "envs_mpi.h"
#include "geometry.h"
#include "parallel_mpi.h"
#include "params.h"
#include "pbc.h"
#include "utils_mpi_io.h"

namespace
{

bool are_equivalent_abacus_qpoints(const Vector3_Order<double>& lhs,
                                   const Vector3_Order<double>& rhs,
                                   const double tol = 1e-5);

enum class AbacusFtVqMode
{
    FullKGrid,
    LegacyIbzStarExpand,
    AbacusFullQRestore,
    AbacusIrreducibleSector,
};

std::string format_debug_double(const double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string text = oss.str();
    std::replace(text.begin(), text.end(), '-', 'm');
    std::replace(text.begin(), text.end(), '.', 'p');
    return text;
}

double canonicalize_mp_fractional_component(const double value,
                                            const int nk,
                                            const double tol = 1e-5)
{
    if (nk <= 0)
    {
        return value;
    }

    const double snapped = std::round(value * static_cast<double>(nk))
                           / static_cast<double>(nk);
    if (std::abs(value - snapped) < tol)
    {
        return snapped;
    }
    return value;
}

Vector3_Order<double> canonicalize_mp_fractional_qpoint(const Vector3_Order<double>& q_frac)
{
    return {canonicalize_mp_fractional_component(q_frac.x, kv_nmp[0]),
            canonicalize_mp_fractional_component(q_frac.y, kv_nmp[1]),
            canonicalize_mp_fractional_component(q_frac.z, kv_nmp[2])};
}

Vector3_Order<double> resolve_ft_q_fractional(
    const Vector3_Order<double>& q_internal,
    const std::map<Vector3_Order<double>, Vector3_Order<double>>* qfrac_lookup = nullptr)
{
    if (qfrac_lookup != nullptr)
    {
        const auto exact_iter = qfrac_lookup->find(q_internal);
        if (exact_iter != qfrac_lookup->end())
        {
            return canonicalize_mp_fractional_qpoint(exact_iter->second);
        }
        const auto matched_iter =
            std::find_if(qfrac_lookup->begin(), qfrac_lookup->end(),
                         [&q_internal](const auto& entry) {
                             return are_equivalent_abacus_qpoints(entry.first, q_internal);
                         });
        if (matched_iter != qfrac_lookup->end())
        {
            return canonicalize_mp_fractional_qpoint(matched_iter->second);
        }
    }

    for (std::size_t ik = 0; ik < klist.size(); ++ik)
    {
        if (are_equivalent_abacus_qpoints(klist[ik], q_internal))
        {
            return canonicalize_mp_fractional_qpoint(kfrac_list[ik]);
        }
    }

    return canonicalize_mp_fractional_qpoint(latvec * q_internal);
}

std::complex<double> build_ft_vq_phase(const Vector3_Order<double>& q_internal,
                                       const Vector3_Order<int>& R,
                                       const int n_k_points,
                                       const std::map<Vector3_Order<double>, Vector3_Order<double>>*
                                           qfrac_lookup = nullptr)
{
    const auto q_frac = resolve_ft_q_fractional(q_internal, qfrac_lookup);
    const double ang = -(q_frac * R) * TWO_PI;
    return std::complex<double>(std::cos(ang), std::sin(ang)) / double(n_k_points);
}

std::map<Vector3_Order<double>, Vector3_Order<double>> build_abacus_restored_qfrac_lookup(
    const LIBRPA::AbacusSymmetryContext& ctx)
{
    std::map<Vector3_Order<double>, Vector3_Order<double>> qfrac_lookup;
    const auto kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(ctx, klist, kfrac_list, map_irk_ks);
    for (const auto& mapping_entry : kstar_grid_mapping)
    {
        const auto& star = ctx.kstars.at(static_cast<std::size_t>(mapping_entry.star_list_index));
        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            qfrac_lookup[mapping_entry.member_q_bz_keys[imember]] =
                canonicalize_mp_fractional_qpoint(star.members[imember].k_bz);
        }
    }
    return qfrac_lookup;
}

bool are_equivalent_abacus_qpoints(const Vector3_Order<double>& lhs,
                                   const Vector3_Order<double>& rhs,
                                   const double tol)
{
    const auto same_component = [tol](const double lhs_component, const double rhs_component) {
        return std::abs((lhs_component - rhs_component) - std::round(lhs_component - rhs_component))
               < tol;
    };
    return same_component(lhs.x, rhs.x) && same_component(lhs.y, rhs.y)
           && same_component(lhs.z, rhs.z);
}

template <typename QMap>
typename QMap::const_iterator find_matching_abacus_qpoint(const QMap& q_map,
                                                          const Vector3_Order<double>& q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }

    return std::find_if(q_map.begin(), q_map.end(), [&q_target](const auto& entry) {
        return are_equivalent_abacus_qpoints(entry.first, q_target);
    });
}

std::string classify_ft_vq_debug_tag(const AbacusFtVqMode mode)
{
    if (mode == AbacusFtVqMode::AbacusIrreducibleSector)
    {
        return "abacus_irreducible_sector";
    }
    if (mode == AbacusFtVqMode::AbacusFullQRestore)
    {
        return "abacus_full_q_restore";
    }
    if (static_cast<int>(klist.size()) < get_full_bz_kpoint_count())
    {
        return "ibz_star_expand";
    }
    return "full_kgrid";
}

// Dump the Fourier-transformed Coulomb block so the symmetry-expanded `V(R)`
// can be compared directly against the no-symmetry full-k-grid reference.
void maybe_dump_ft_vq_debug_matrix(const ComplexMatrix& vr_cplx,
                                   const atom_t Mu,
                                   const atom_t Nu,
                                   const Vector3_Order<int>& R,
                                   const AbacusFtVqMode mode)
{
    if (!Params::debug)
    {
        return;
    }

    std::ostringstream file_name;
    file_name << Params::output_dir << "abacus_vr_" << classify_ft_vq_debug_tag(mode)
              << "_Mu_" << Mu << "_Nu_" << Nu
              << "_R_" << R.x << "_" << R.y << "_" << R.z
              << "_id_" << LIBRPA::envs::mpi_comm_global_h.myid << ".mtx";
    print_complex_matrix_mm(vr_cplx, file_name.str(), 1e-14, false);
}

// Dump the q-space Coulomb block that is fed into `FT_Vq` so the restored
// full-q operator can be compared directly against the symmetry-off reference.
void maybe_dump_qspace_vq_debug_matrix(const ComplexMatrix& vq_cplx,
                                       const atom_t Mu,
                                       const atom_t Nu,
                                       const Vector3_Order<double>& q_internal,
                                       const AbacusFtVqMode mode)
{
    if (!Params::debug)
    {
        return;
    }

    const auto q_frac = latvec * q_internal;
    std::ostringstream tag;
    tag << classify_ft_vq_debug_tag(mode)
        << "_Mu_" << Mu << "_Nu_" << Nu
        << "_qx_" << format_debug_double(q_frac.x)
        << "_qy_" << format_debug_double(q_frac.y)
        << "_qz_" << format_debug_double(q_frac.z)
        << "_id_" << LIBRPA::envs::mpi_comm_global_h.myid;
    static std::set<std::string> dumped_tags;
    if (!dumped_tags.insert(tag.str()).second)
    {
        return;
    }

    std::ostringstream file_name;
    file_name << Params::output_dir << "abacus_vq_" << tag.str() << ".mtx";
    print_complex_matrix_mm(vq_cplx, file_name.str(), 1e-14, false);
}

bool has_complete_abacus_abf_ibz_coverage(const atpair_k_cplx_mat_t& blocks_by_q_ibz,
                                          const std::map<atom_t, size_t>& atom_nabf)
{
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        for (std::size_t atom_j = atom_i; atom_j < atom_nabf.size(); ++atom_j)
        {
            const auto upper_iter = blocks_by_q_ibz.find(static_cast<atom_t>(atom_i));
            const bool has_upper =
                upper_iter != blocks_by_q_ibz.end()
                && upper_iter->second.count(static_cast<atom_t>(atom_j)) != 0;

            const auto lower_iter = blocks_by_q_ibz.find(static_cast<atom_t>(atom_j));
            const bool has_lower =
                lower_iter != blocks_by_q_ibz.end()
                && lower_iter->second.count(static_cast<atom_t>(atom_i)) != 0;

            if (!has_upper && !has_lower)
            {
                return false;
            }

            const auto& q_blocks =
                has_upper ? upper_iter->second.at(static_cast<atom_t>(atom_j))
                          : lower_iter->second.at(static_cast<atom_t>(atom_i));
            for (const auto& q_ibz : klist)
            {
                if (find_matching_abacus_qpoint(q_blocks, q_ibz) == q_blocks.end())
                {
                    return false;
                }
            }
        }
    }
    return true;
}

LIBRPA::abacus_atom_block_matrix_map_t collect_abacus_abf_ibz_blocks_for_q(
    const atpair_k_cplx_mat_t& blocks_by_q,
    const Vector3_Order<double>& q_ibz_internal)
{
    LIBRPA::abacus_atom_block_matrix_map_t blocks_ibz;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        const auto atom_i = atom_i_pair.first;
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const auto atom_j = atom_j_pair.first;
            const auto q_iter = find_matching_abacus_qpoint(atom_j_pair.second, q_ibz_internal);
            if (q_iter != atom_j_pair.second.end())
            {
                blocks_ibz[atom_i][atom_j] = *q_iter->second;
            }
        }
    }
    return blocks_ibz;
}

const LIBRPA::AbacusKStarMember& find_matching_abf_kstar_member(
    const LIBRPA::AbacusKStar& abf_star,
    const LIBRPA::AbacusKStarMember& ao_member)
{
    const auto matched = std::find_if(abf_star.members.begin(), abf_star.members.end(),
                                      [&ao_member](const LIBRPA::AbacusKStarMember& candidate) {
                                          return candidate.isym == ao_member.isym
                                                 && are_equivalent_abacus_qpoints(candidate.k_bz,
                                                                                  ao_member.k_bz);
                                      });
    if (matched == abf_star.members.end())
    {
        throw std::runtime_error(
            "Failed to match an ABF k-star member with the AO-side symmetry member");
    }
    return *matched;
}

atpair_k_cplx_mat_t restore_abacus_abf_full_qspace_operator(
    const atpair_k_cplx_mat_t& blocks_by_q_ibz,
    const std::map<atom_t, size_t>& atom_nabf)
{
    atpair_k_cplx_mat_t blocks_by_q_full;
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());
    const auto kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(ctx, klist, kfrac_list, map_irk_ks);

    for (const auto& star_mapping : kstar_grid_mapping)
    {
        const auto& star = ctx.kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        const LIBRPA::AbacusKStar* abf_star = nullptr;
        if (!ctx.abf_kstars.empty())
        {
            if (ctx.abf_kstars.size() != ctx.kstars.size())
            {
                throw std::runtime_error(
                    "ABF k-space symmetry sidecar count is inconsistent with symrot_k.txt");
            }
            abf_star = &ctx.abf_kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        }
        const auto q_ibz_internal = klist.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto blocks_ibz =
            collect_abacus_abf_ibz_blocks_for_q(blocks_by_q_ibz, q_ibz_internal);
        if (blocks_ibz.empty())
        {
            continue;
        }
        if (star.members.size() != star_mapping.member_q_bz_keys.size())
        {
            throw std::runtime_error(
                "ABACUS q-star mapping is inconsistent with the loaded full-q keys");
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& abf_member =
                (abf_star == nullptr) ? member : find_matching_abf_kstar_member(*abf_star, member);
            const bool use_time_reversal = member.isym >= nsym_space;
            LIBRPA::abacus_atom_block_matrix_map_t rotated_blocks;
            try
            {
                rotated_blocks = LIBRPA::rotate_abacus_abf_kspace_operator_blocks(
                    ctx, abf_member, blocks_ibz, atom_nabf, star.k_ibz, coord_frac, use_time_reversal);
            }
            catch (const std::exception& ex)
            {
                std::ostringstream oss;
                oss << "ABACUS bare-Coulomb q-star restore failed for star=" << star.star_index
                    << ", member=" << imember << ", isym=" << member.isym << ": "
                    << ex.what();
                throw std::runtime_error(oss.str());
            }
            for (const auto& atom_i_pair : rotated_blocks)
            {
                for (const auto& atom_j_pair : atom_i_pair.second)
                {
                    const auto& q_internal = star_mapping.member_q_bz_keys[imember];
                    blocks_by_q_full[atom_i_pair.first][atom_j_pair.first][q_internal] =
                        std::make_shared<ComplexMatrix>(atom_j_pair.second);
                }
            }
        }
    }

    return blocks_by_q_full;
}

LIBRPA::abacus_irreducible_sector_t filter_abacus_irreducible_sector_by_rlist(
    const LIBRPA::abacus_irreducible_sector_t& irreducible_sector,
    const std::vector<Vector3_Order<int>>& Rlist)
{
    LIBRPA::abacus_irreducible_sector_t filtered_sector;
    const std::set<Vector3_Order<int>> requested_rset(Rlist.begin(), Rlist.end());
    for (const auto& pair_Rs : irreducible_sector)
    {
        for (const auto& R_array : pair_Rs.second)
        {
            const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
            if (requested_rset.count(R) == 0)
            {
                continue;
            }
            filtered_sector[pair_Rs.first].insert(R_array);
        }
    }
    return filtered_sector;
}

std::set<std::pair<atom_t, atom_t>> build_abacus_irreducible_target_atom_pairs(
    const LIBRPA::abacus_irreducible_sector_t& irreducible_sector)
{
    std::set<std::pair<atom_t, atom_t>> target_atom_pairs;
    for (const auto& pair_Rs : irreducible_sector)
    {
        if (!pair_Rs.second.empty())
        {
            target_atom_pairs.insert(pair_Rs.first);
        }
    }
    return target_atom_pairs;
}

atpair_R_mat_t accumulate_abacus_abf_irreducible_sector_vr(
    const atpair_k_cplx_mat_t& blocks_by_q_ibz,
    const int n_k_points,
    const std::vector<Vector3_Order<int>>& Rlist,
    const std::map<atom_t, size_t>& atom_nabf)
{
    atpair_R_mat_t blocks_by_R_real;
    atpair_R_cplx_mat_t blocks_by_R_complex;
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    const auto filtered_sector = filter_abacus_irreducible_sector_by_rlist(ctx.irreducible_sector, Rlist);
    if (filtered_sector.empty())
    {
        return blocks_by_R_real;
    }

    const auto target_atom_pairs = build_abacus_irreducible_target_atom_pairs(filtered_sector);
    const auto kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(ctx, klist, kfrac_list, map_irk_ks);
    const int nsym_space = static_cast<int>(ctx.rspace_operations.size());

    for (const auto& pair_Rs : filtered_sector)
    {
        const auto atom_i = pair_Rs.first.first;
        const auto atom_j = pair_Rs.first.second;
        const int n_i = static_cast<int>(atom_nabf.at(atom_i));
        const int n_j = static_cast<int>(atom_nabf.at(atom_j));
        for (const auto& R_array : pair_Rs.second)
        {
            const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
            blocks_by_R_complex[atom_i][atom_j][R] = std::make_shared<ComplexMatrix>(n_i, n_j);
        }
    }

    for (const auto& star_mapping : kstar_grid_mapping)
    {
        const auto& star = ctx.kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        const LIBRPA::AbacusKStar* abf_star = nullptr;
        if (!ctx.abf_kstars.empty())
        {
            if (ctx.abf_kstars.size() != ctx.kstars.size())
            {
                throw std::runtime_error(
                    "ABF k-space symmetry sidecar count is inconsistent with symrot_k.txt");
            }
            abf_star = &ctx.abf_kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));
        }

        const auto q_ibz_internal = klist.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto blocks_ibz =
            collect_abacus_abf_ibz_blocks_for_q(blocks_by_q_ibz, q_ibz_internal);
        if (blocks_ibz.empty())
        {
            continue;
        }
        if (star.members.size() != star_mapping.member_q_bz_keys.size())
        {
            throw std::runtime_error(
                "ABACUS q-star mapping is inconsistent with the loaded full-q keys");
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& abf_member =
                (abf_star == nullptr) ? member : find_matching_abf_kstar_member(*abf_star, member);
            const bool use_time_reversal = member.isym >= nsym_space;
            LIBRPA::abacus_atom_block_matrix_map_t rotated_blocks;
            try
            {
                rotated_blocks = LIBRPA::rotate_abacus_abf_kspace_operator_blocks(
                    ctx, abf_member, blocks_ibz, atom_nabf, star.k_ibz, coord_frac, use_time_reversal,
                    &target_atom_pairs);
            }
            catch (const std::exception& ex)
            {
                std::ostringstream oss;
                oss << "ABACUS irreducible-sector FT failed for star=" << star.star_index
                    << ", member=" << imember << ", isym=" << member.isym << ": "
                    << ex.what();
                throw std::runtime_error(oss.str());
            }

            const auto& q_internal = star_mapping.member_q_bz_keys[imember];
            for (const auto& atom_i_pair : rotated_blocks)
            {
                for (const auto& atom_j_pair : atom_i_pair.second)
                {
                    const auto sector_iter =
                        filtered_sector.find({atom_i_pair.first, atom_j_pair.first});
                    if (sector_iter == filtered_sector.end())
                    {
                        continue;
                    }

                    maybe_dump_qspace_vq_debug_matrix(
                        atom_j_pair.second, atom_i_pair.first, atom_j_pair.first, q_internal,
                        AbacusFtVqMode::AbacusIrreducibleSector);
                    for (const auto& R_array : sector_iter->second)
                    {
                        const Vector3_Order<int> R{R_array[0], R_array[1], R_array[2]};
                        const auto phase = build_ft_vq_phase(q_internal, R, n_k_points);
                        *blocks_by_R_complex.at(atom_i_pair.first).at(atom_j_pair.first).at(R) +=
                            atom_j_pair.second * phase;
                    }
                }
            }
        }
    }

    for (const auto& atom_i_pair : blocks_by_R_complex)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            for (const auto& R_block : atom_j_pair.second)
            {
                blocks_by_R_real[atom_i_pair.first][atom_j_pair.first][R_block.first] =
                    std::make_shared<matrix>(R_block.second->real());
                maybe_dump_ft_vq_debug_matrix(
                    *R_block.second, atom_i_pair.first, atom_j_pair.first, R_block.first,
                    AbacusFtVqMode::AbacusIrreducibleSector);
            }
        }
    }

    return blocks_by_R_real;
}

} // namespace

atpair_R_mat_t
FT_Vq(const atpair_k_cplx_mat_t &coulmat_k, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist, bool return_ordered_atom_pair)
{
    atpair_R_mat_t coulmat_R;
    const auto& symmetry_ctx = LIBRPA::abacus_symmetry_ctx;
    const bool can_use_abacus_full_q_restore =
        Params::use_abacus_gw_symmetry
        && symmetry_ctx.available
        && symmetry_ctx.has_abf_shell_layout()
        && !symmetry_ctx.kstars.empty()
        && symmetry_ctx.kstars.size() == kfrac_list.size()
        && !map_irk_ks.empty()
        && atom_mu.size() == symmetry_ctx.atom_to_type.size()
        && coord_frac.size() == atom_mu.size()
        && static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
        && has_complete_abacus_abf_ibz_coverage(coulmat_k, atom_mu);
    const bool use_abacus_irreducible_sector_ft =
        can_use_abacus_full_q_restore
        // The irreducible-sector q -> R accumulation is the common ABACUS symmetry path for
        // both EXX and GW. Do not gate the GW Coulomb transform on the EXX-only switch.
        && (Params::use_abacus_exx_symmetry || Params::use_abacus_gw_symmetry)
        && symmetry_ctx.has_ao_shell_layout()
        && !symmetry_ctx.irreducible_sector.empty()
        && !symmetry_ctx.rspace_operations.empty()
        && LIBRPA::parallel_routing == LIBRPA::ParallelRouting::LIBRI;
    const bool use_abacus_full_q_restore =
        can_use_abacus_full_q_restore && !use_abacus_irreducible_sector_ft;
    if (use_abacus_irreducible_sector_ft)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS EXX symmetry accumulates irreducible-sector `V(R)` directly from IBZ q-stars\n");
        return accumulate_abacus_abf_irreducible_sector_vr(coulmat_k, n_k_points, Rlist, atom_mu);
    }

    const auto debug_mode = use_abacus_full_q_restore ? AbacusFtVqMode::AbacusFullQRestore
                            : (static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
                                   ? AbacusFtVqMode::LegacyIbzStarExpand
                                   : AbacusFtVqMode::FullKGrid);
    const auto coulmat_k_effective =
        use_abacus_full_q_restore ? restore_abacus_abf_full_qspace_operator(coulmat_k, atom_mu)
                                  : coulmat_k;
    const auto restored_qfrac_lookup =
        use_abacus_full_q_restore ? build_abacus_restored_qfrac_lookup(symmetry_ctx)
                                  : std::map<Vector3_Order<double>, Vector3_Order<double>>{};
    if (use_abacus_full_q_restore)
    {
        LIBRPA::utils::lib_printf_root(
            "ABACUS GW symmetry restores the full ABF q-star before `FT_Vq`\n");
    }

    for (auto R: Rlist)
    {
        auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        auto iR = std::distance(Rlist.cbegin(), iteR);
        for (const auto &Mu_NuqV: coulmat_k_effective)
        {
            const auto Mu = Mu_NuqV.first;
            const int n_mu = atom_mu[Mu];
            for (const auto &Nu_qV: Mu_NuqV.second)
            {
                const auto Nu = Nu_qV.first;
                const int n_nu = atom_mu[Nu];
                for (const auto& q_V : Nu_qV.second)
                {
                    maybe_dump_qspace_vq_debug_matrix(
                        *q_V.second, Mu, Nu, q_V.first, debug_mode);
                }
                coulmat_R[Mu][Nu][R] = make_shared<matrix>();
                // a temporary complex matrix to save the transformed matrix
                ComplexMatrix VR_cplx(n_mu, n_nu);
                for (const auto &q_V: Nu_qV.second)
                {
                    auto q = q_V.first;
                    if (use_abacus_full_q_restore)
                    {
                        const complex<double> kphase =
                            build_ft_vq_phase(q, R, n_k_points, &restored_qfrac_lookup);
                        VR_cplx += (*q_V.second) * kphase;
                    }
                    else
                    {
                        for (auto q_bz: map_irk_ks[q])
                        {
                            const complex<double> kphase =
                                build_ft_vq_phase(q_bz, R, n_k_points);
                            // Legacy fallback: this branch is only formally correct when the
                            // full-q star reduces to {q, -q}. The general ABACUS restore path
                            // above should be used whenever the complete IBZ q-mesh is available.
                            if (q_bz == q)
                            {
                                VR_cplx += (*q_V.second) * kphase;
                            }
                            else
                            {
                                VR_cplx += conj(*q_V.second) * kphase;
                            }
                        }
                    }
                    // minyez debug: check hermicity of Vq
                    // if (iR == 0)
                    // {
                    //     int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
                    //     sprintf(fn, "Vq_Mu_%zu_Nu_%zu_iq_%d.mtx", Mu, Nu, iq);
                    //     print_complex_matrix_mm(*q_V.second, fn);
                    // }
                    // end minyez debug
                }
                *coulmat_R[Mu][Nu][R] = VR_cplx.real();
                maybe_dump_ft_vq_debug_matrix(VR_cplx, Mu, Nu, R, debug_mode);
                // debug print
                // sprintf(fn, "VR_cplx_Mu_%zu_Nu_%zu_iR_%zu.mtx", Mu, Nu, iR);
                // print_complex_matrix_mm(VR_cplx, fn);
                // sprintf(fn, "VR_Mu_%zu_Nu_%zu_iR_%zu.mtx", Mu, Nu, iR);
                // print_matrix_mm(*coulmat_R[Mu][Nu][R], fn);

                // when ordered atom pair is requested, check whether it is available in the original map
                if (!use_abacus_full_q_restore && return_ordered_atom_pair && Mu != Nu
                    && (coulmat_k.count(Nu) == 0 || coulmat_k.at(Nu).count(Mu) == 0))
                {
                    coulmat_R[Nu][Mu][R] = make_shared<matrix>();
                    ComplexMatrix VR_cplx(n_nu, n_mu);
                    for (const auto &q_V: Nu_qV.second)
                    {
                        auto q = q_V.first;
                        for (auto q_bz: map_irk_ks[q])
                        {
                            const complex<double> kphase =
                                build_ft_vq_phase(q_bz, R, n_k_points);
                            if (q_bz == q)
                            {
                                VR_cplx += transpose(*q_V.second, true) * kphase;
                            }
                            else
                            {
                                VR_cplx += transpose(*q_V.second, false) * kphase;
                            }
                        }
                    }
                    *coulmat_R[Nu][Mu][R] = VR_cplx.real();
                    maybe_dump_ft_vq_debug_matrix(VR_cplx, Nu, Mu, R, debug_mode);
                }
            }
        }
    }
    // myz debug: check the imaginary part of the coulomb matrix
    // char fn[80];
    /* for (const auto & Mu_NuRV: VR) */
    /* { */
    /*     auto Mu = Mu_NuRV.first; */
    /*     const int n_mu = atom_mu[Mu]; */
    /*     for (const auto & Nu_RV: Mu_NuRV.second) */
    /*     { */
    /*         auto Nu = Nu_RV.first; */
    /*         const int n_nu = atom_mu[Nu]; */
    /*         for (const auto & R_V: Nu_RV.second) */
    /*         { */
    /*             auto R = R_V.first; */
    /*             auto &V = R_V.second; */
    /*             auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R); */
    /*             auto iR = std::distance(Rlist.cbegin(), iteR); */
    /*             sprintf(fn, "VR_Mu_%zu_Nu_%zu_iR_%zu.mtx", Mu, Nu, iR); */
    /*             print_complex_matrix_mm(*V, fn); */
    /*         } */
    /*     } */
    /* } */
    return coulmat_R;
}
