#include "abacus_symmetry.h"
#include "pbc.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace
{

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream ofs(path);
    assert(ofs.good());
    ofs << content;
}

std::string build_identity_shell_block(const int l)
{
    const int nm = 2 * l + 1;
    std::ostringstream oss;
    for (int row = 0; row < nm; ++row)
    {
        for (int col = 0; col < nm; ++col)
        {
            oss << (row == col ? "(1,0)" : "(0,0)");
        }
        oss << "\n";
    }
    return oss.str();
}

std::string build_symrot_abf_k_file()
{
    std::ostringstream oss;
    oss << "ABF shell layouts:\n"
        << "# Each line stores one auxiliary-basis candidate for one atom type.\n"
        << "# shell_counts[l] is the number of auxiliary radial shells at angular momentum l.\n"
        << "type 1 label B nao 4 lmax 1 shell_counts 1 1\n"
        << "type 1 label B nao 9 lmax 2 shell_counts 1 1 1\n"
        << "End ABF shell layouts\n"
        << "Number of IBZ k-points (k stars): 1\n"
        << "Format:\n"
        << "dummy\n"
        << "Star 1 of IBZ k-point (0.000000 0.000000 0.000000):\n"
        << "0\n"
        << "(0.000000 0.000000 0.000000)\n"
        << "atom 1 -> 1 of type 1 with Lmax= 2\n";
    for (int l = 0; l <= 2; ++l)
    {
        oss << build_identity_shell_block(l);
    }
    return oss.str();
}

std::string build_symrot_k_file()
{
    std::ostringstream oss;
    oss << "AO shell layouts:\n"
        << "# Each line stores one AO-basis layout for one atom type.\n"
        << "# shell_counts[l] is the number of AO radial shells at angular momentum l.\n"
        << "type 1 label B nao 4 lmax 1 shell_counts 1 1\n"
        << "End AO shell layouts\n"
        << "Number of IBZ k-points (k stars): 1\n"
        << "Format:\n"
        << "dummy\n"
        << "Star 1 of IBZ k-point (0.000000 0.000000 0.000000):\n"
        << "0\n"
        << "(0.000000 0.000000 0.000000)\n"
        << "atom 1 -> 1 of type 1 with Lmax= 1\n";
    for (int l = 0; l <= 1; ++l)
    {
        oss << build_identity_shell_block(l);
    }
    return oss.str();
}

} // namespace

int main()
{
    const std::string test_dir = "abacus_symmetry_test_tmp";
    std::system(("rm -rf " + test_dir).c_str());
    std::system(("mkdir -p " + test_dir).c_str());

    write_file(test_dir + "/irreducible_sector.txt",
               "atompair (0, 0), R = (0, 0, 0)\n"
               "atompair (0, 1), R = (1, 0, 0)\n");

    write_file(test_dir + "/symrot_R.txt",
               "Number of irreducible sector: 2\n"
               "Lmax of AOs: 1\n"
               "Lmax of ABFs: 1\n"
               "Format:\n"
               "dummy\n"
               "0\n"
               "1 0 0\n"
               "0 1 0\n"
               "0 0 1\n"
               "(0.000000 0.000000 0.000000)\n"
               "(1,0)\n"
               "(1,0)(0,0)(0,0)\n"
               "(0,0)(1,0)(0,0)\n"
               "(0,0)(0,0)(1,0)\n");

    write_file(test_dir + "/symrot_k.txt", build_symrot_k_file());

    write_file(test_dir + "/symrot_abf_k.txt", build_symrot_abf_k_file());

    LIBRPA::AbacusSymmetryContext ctx;
    const bool loaded = LIBRPA::load_abacus_symmetry_context(test_dir, ctx, nullptr);
    assert(loaded);
    assert(ctx.available);
    assert(ctx.count_irreducible_pairs() == 2);
    assert(ctx.count_irreducible_blocks() == 2);
    assert(ctx.rspace_operations.size() == 1);
    assert(ctx.kstars.size() == 1);
    assert(ctx.abf_kstars.size() == 1);
    assert(ctx.count_kstar_members() == 1);
    assert(ctx.ao_lmax == 1);
    assert(ctx.abf_lmax == 1);
    assert(ctx.rspace_operations.front().shell_rotations.at(1).nr == 3);
    assert(ctx.kstars.front().members.front().atom_rotations.front().shell_rotations.at(1).nc == 3);
    assert(ctx.abf_kstars.front().members.front().atom_rotations.front().lmax == 2);
    assert(ctx.abf_kstars.front().members.front().atom_rotations.front().shell_rotations.at(2).nr == 5);
    assert(ctx.has_ao_shell_layout());
    assert(ctx.ao_type_layouts.size() == 1);
    assert(ctx.count_atoms_with_layout() == 1);
    assert(ctx.get_ao_type_layout(0).nao == 4);
    assert(ctx.get_ao_type_layout(0).shell_counts.size() == 2);
    assert(ctx.get_ao_type_layout(0).shell_counts[0] == 1);
    assert(ctx.get_ao_type_layout(0).shell_counts[1] == 1);
    assert(ctx.atom_to_type.at(0) == 0);
    assert(ctx.has_abf_shell_layout());
    assert(ctx.count_abf_layout_candidates() == 2);
    assert(ctx.find_abf_type_layout(0, 4).nao == 4);
    assert(ctx.find_abf_type_layout(0, 9).nao == 9);
    assert(ctx.find_abf_type_layout(0, 9).shell_counts.size() == 3);
    assert(ctx.find_abf_type_layout(0, 9).shell_counts[2] == 1);

    LIBRPA::AbacusKStar closure_star;
    closure_star.star_index = 1;
    closure_star.k_ibz = {0.0, 0.0, 0.0};
    LIBRPA::AbacusKStarMember closure_member;
    closure_member.isym = 1;
    closure_member.k_bz = {0.0, 0.0, 0.0};
    closure_member.atom_rotations = {
        LIBRPA::AbacusKAtomRotation{0, 0, 0, 0, {}},
        LIBRPA::AbacusKAtomRotation{1, 1, 0, 0, {}},
        LIBRPA::AbacusKAtomRotation{2, 3, 0, 0, {}},
        LIBRPA::AbacusKAtomRotation{3, 2, 0, 0, {}}};
    closure_star.members.push_back(closure_member);

    const auto closure_pairs = LIBRPA::build_abacus_upper_atom_pair_closure(
        closure_star, {{0, 2}});
    assert(closure_pairs.size() == 2);
    assert(closure_pairs.count({0, 2}) == 1);
    assert(closure_pairs.count({0, 3}) == 1);

    const auto full_rotation = LIBRPA::build_abacus_ao_rotation_matrix(
        ctx, 0, ctx.rspace_operations.front().shell_rotations);
    assert(full_rotation.nr == 4);
    assert(full_rotation.nc == 4);
    assert(full_rotation(0, 0) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(1, 1) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(2, 2) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(3, 3) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(0, 1) == std::complex<double>(0.0, 0.0));

    latvec.Identity();
    const std::array<std::array<double, 3>, 3> identity_rotation{
        {{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    const auto abf_identity_rotation = LIBRPA::build_abacus_abf_rotation_matrix(
        ctx, 0, 9, ctx.rspace_operations.front().shell_rotations, identity_rotation);
    assert(abf_identity_rotation.nr == 9);
    assert(abf_identity_rotation.nc == 9);
    assert(abf_identity_rotation(0, 0) == std::complex<double>(1.0, 0.0));
    assert(abf_identity_rotation(8, 8) == std::complex<double>(1.0, 0.0));
    assert(abf_identity_rotation(5, 7) == std::complex<double>(0.0, 0.0));

    std::map<int, ComplexMatrix> s_only_rotation;
    s_only_rotation[0] = ComplexMatrix(1, 1);
    s_only_rotation[0](0, 0) = std::complex<double>(1.0, 0.0);
    const std::array<std::array<double, 3>, 3> c41_rotation{
        {{{0.0, 1.0, 0.0}}, {{-1.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    const auto abf_c41_rotation =
        LIBRPA::build_abacus_abf_rotation_matrix(ctx, 0, 4, s_only_rotation, c41_rotation);
    assert(abf_c41_rotation.nr == 4);
    assert(abf_c41_rotation.nc == 4);
    assert(abf_c41_rotation(0, 0) == std::complex<double>(1.0, 0.0));
    assert(abf_c41_rotation(1, 1) == std::complex<double>(1.0, 0.0));
    assert(abf_c41_rotation(1, 2) == std::complex<double>(0.0, 0.0));
    assert(abf_c41_rotation(2, 3) == std::complex<double>(-1.0, 0.0));
    assert(abf_c41_rotation(3, 2) == std::complex<double>(1.0, 0.0));

    ComplexMatrix dmat_ibz(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        dmat_ibz(i, i) = std::complex<double>(10.0 + i, 0.0);
        for (int j = i + 1; j < 4; ++j)
        {
            const std::complex<double> value(10.0 * i + j, i - j);
            dmat_ibz(i, j) = value;
            dmat_ibz(j, i) = std::conj(value);
        }
    }

    const auto rotated_dmat =
        LIBRPA::rotate_abacus_kspace_matrix(ctx,
                                            ctx.kstars.front().members.front(),
                                            dmat_ibz,
                                            std::map<atom_t, size_t>{{0, 4}},
                                            Vector3_Order<double>{0.0, 0.0, 0.0},
                                            std::map<atom_t, std::array<double, 3>>{{0, {0.0, 0.0, 0.0}}},
                                            false);
    assert(rotated_dmat.nr == 4);
    assert(rotated_dmat.nc == 4);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            assert(rotated_dmat(i, j) == dmat_ibz(i, j));
        }
    }

    auto tr_member = ctx.kstars.front().members.front();
    tr_member.isym = static_cast<int>(ctx.rspace_operations.size());
    const auto rotated_dmat_tr =
        LIBRPA::rotate_abacus_kspace_matrix(ctx,
                                            tr_member,
                                            dmat_ibz,
                                            std::map<atom_t, size_t>{{0, 4}},
                                            Vector3_Order<double>{0.0, 0.0, 0.0},
                                            std::map<atom_t, std::array<double, 3>>{{0, {0.0, 0.0, 0.0}}},
                                            true);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            assert(rotated_dmat_tr(i, j) == std::conj(dmat_ibz(i, j)));
        }
    }

    LIBRPA::AbacusSymmetryContext phase_ctx;
    phase_ctx.available = true;
    phase_ctx.ao_shell_layout_available = true;
    phase_ctx.ao_lmax = 0;
    phase_ctx.ao_type_layouts = {LIBRPA::AbacusAOTypeLayout{"X", "X.orb", {1}, 1}};
    phase_ctx.atom_to_type = {{0, 0}, {1, 0}};

    LIBRPA::AbacusSymmetryOperation phase_identity;
    phase_identity.isym = 0;
    phase_identity.rotation = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    phase_identity.translation = {0.0, 0.0, 0.0};
    phase_identity.shell_rotations[0] = ComplexMatrix(1, 1);
    phase_identity.shell_rotations[0](0, 0) = std::complex<double>(1.0, 0.0);

    LIBRPA::AbacusSymmetryOperation phase_inversion = phase_identity;
    phase_inversion.isym = 1;
    phase_inversion.rotation = {{{{-1.0, 0.0, 0.0}}, {{0.0, -1.0, 0.0}}, {{0.0, 0.0, -1.0}}}};
    phase_ctx.rspace_operations = {phase_identity, phase_inversion};

    LIBRPA::AbacusKStarMember phase_member;
    phase_member.isym = 1;
    phase_member.k_bz = {0.0, 0.0, 0.0};

    LIBRPA::AbacusKAtomRotation phase_atom0;
    phase_atom0.atom_from = 0;
    phase_atom0.atom_to = 0;
    phase_atom0.atom_type = 0;
    phase_atom0.lmax = 0;
    phase_atom0.shell_rotations[0] = ComplexMatrix(1, 1);
    phase_atom0.shell_rotations[0](0, 0) = std::complex<double>(1.0, 0.0);

    LIBRPA::AbacusKAtomRotation phase_atom1 = phase_atom0;
    phase_atom1.atom_from = 1;
    phase_atom1.atom_to = 1;
    phase_member.atom_rotations = {phase_atom0, phase_atom1};

    ComplexMatrix phase_dmat_ibz(2, 2);
    phase_dmat_ibz(0, 0) = std::complex<double>(2.0, 0.0);
    phase_dmat_ibz(1, 1) = std::complex<double>(3.0, 0.0);
    phase_dmat_ibz(0, 1) = std::complex<double>(1.25, -0.5);
    phase_dmat_ibz(1, 0) = std::conj(phase_dmat_ibz(0, 1));

    const auto phase_rotated_dmat = LIBRPA::rotate_abacus_kspace_matrix(
        phase_ctx,
        phase_member,
        phase_dmat_ibz,
        std::map<atom_t, size_t>{{0, 1}, {1, 1}},
        Vector3_Order<double>{0.5, 0.0, 0.0},
        std::map<atom_t, std::array<double, 3>>{{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}},
        false);
    assert(std::abs(phase_rotated_dmat(0, 0) - phase_dmat_ibz(0, 0)) < 1e-12);
    assert(std::abs(phase_rotated_dmat(1, 1) - phase_dmat_ibz(1, 1)) < 1e-12);
    assert(std::abs(phase_rotated_dmat(0, 1) + phase_dmat_ibz(0, 1)) < 1e-12);
    assert(std::abs(phase_rotated_dmat(1, 0) + phase_dmat_ibz(1, 0)) < 1e-12);

    LIBRPA::AbacusSymmetryContext rspace_ctx;
    rspace_ctx.available = true;
    rspace_ctx.ao_shell_layout_available = true;
    rspace_ctx.ao_lmax = ctx.ao_lmax;
    rspace_ctx.abf_lmax = ctx.abf_lmax;
    rspace_ctx.ao_type_layouts = ctx.ao_type_layouts;
    rspace_ctx.atom_to_type = {{0, 0}, {1, 0}};
    rspace_ctx.irreducible_sector[{0, 0}].insert({0, 0, 0});
    rspace_ctx.irreducible_sector[{0, 1}].insert({0, 0, 0});

    LIBRPA::AbacusSymmetryOperation op_identity = ctx.rspace_operations.front();
    op_identity.isym = 0;
    op_identity.translation = {0.0, 0.0, 0.0};
    LIBRPA::AbacusSymmetryOperation op_shift = op_identity;
    op_shift.isym = 1;
    op_shift.translation = {0.5, 0.5, 0.5};
    rspace_ctx.rspace_operations = {op_identity, op_shift};

    const std::map<atom_t, std::array<double, 3>> coord_frac_test{
        {0, {0.0, 0.0, 0.0}},
        {1, {0.5, 0.5, 0.5}},
    };
    const Vector3_Order<int> period{1, 1, 1};
    const std::vector<Vector3_Order<int>> Rlist{{0, 0, 0}};
    LIBRPA::abacus_rspace_sector_stars_t sector_stars;
    LIBRPA::build_abacus_rspace_sector_stars(
        rspace_ctx, coord_frac_test, period, Rlist, sector_stars, nullptr);
    assert(sector_stars.size() == 2);
    const atpair_t pair_00{0, 0};
    const atpair_t pair_01{0, 1};
    const atpair_t pair_11{1, 1};
    const atpair_t pair_10{1, 0};
    const Vector3_Order<int> R0{0, 0, 0};
    assert(!sector_stars.at(pair_00).at(R0).empty());
    assert(!sector_stars.at(pair_01).at(R0).empty());

    const std::map<atom_t, std::array<double, 3>> coord_frac_test_noisy{
        {0, {0.0, 0.0, 0.0}},
        {1, {0.500002, 0.500002, 0.500002}},
    };
    LIBRPA::abacus_rspace_sector_stars_t sector_stars_noisy;
    LIBRPA::build_abacus_rspace_sector_stars(
        rspace_ctx, coord_frac_test_noisy, period, Rlist, sector_stars_noisy, nullptr);
    assert(!sector_stars_noisy.at(pair_00).at(R0).empty());
    assert(!sector_stars_noisy.at(pair_01).at(R0).empty());

    LIBRPA::AbacusSymmetryContext inverse_ctx = rspace_ctx;
    inverse_ctx.irreducible_sector.clear();
    inverse_ctx.irreducible_sector[pair_00].insert({1, 0, 0});
    LIBRPA::AbacusSymmetryOperation op_swap = op_identity;
    op_swap.isym = 1;
    op_swap.rotation = {{{{0.0, 1.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    inverse_ctx.rspace_operations = {op_identity, op_swap};
    const std::vector<Vector3_Order<int>> Rlist_inverse{{1, 0, 0}, {0, 1, 0}};
    LIBRPA::abacus_rspace_sector_stars_t inverse_sector_stars;
    LIBRPA::build_abacus_rspace_sector_stars(
        inverse_ctx, coord_frac_test_noisy, Vector3_Order<int>{3, 3, 1}, Rlist_inverse,
        inverse_sector_stars, nullptr);
    const Vector3_Order<int> Rx{1, 0, 0};
    const Vector3_Order<int> Ry{0, 1, 0};
    assert(inverse_sector_stars.at(pair_00).at(Rx).size() == 2);
    assert(inverse_sector_stars.at(pair_00).at(Rx)[0].full_R == Rx);
    assert(inverse_sector_stars.at(pair_00).at(Rx)[1].full_R == Ry);

    ComplexMatrix hr_source(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            hr_source(i, j) = std::complex<double>(100.0 + 10.0 * i + j, 0.0);
        }
    }
    const auto hr_rotated =
        LIBRPA::rotate_abacus_rspace_matrix(rspace_ctx, 1, 0, 1, hr_source);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            assert(hr_rotated(i, j) == hr_source(i, j));
        }
    }

    latvec.Identity();
    G.Identity();
    klist.clear();
    kfrac_list.clear();
    map_irk_ks.clear();

    LIBRPA::AbacusSymmetryContext kmap_ctx;
    kmap_ctx.available = true;

    LIBRPA::AbacusKStar gamma_star;
    gamma_star.star_index = 0;
    gamma_star.k_ibz = {0.0, 0.0, 0.0};
    gamma_star.members.push_back(LIBRPA::AbacusKStarMember{0, {-0.5, 0.0, 0.0}, {}});
    gamma_star.members.push_back(LIBRPA::AbacusKStarMember{0, {0.0, 0.0, 0.0}, {}});

    LIBRPA::AbacusKStar edge_star;
    edge_star.star_index = 1;
    edge_star.k_ibz = {0.5, 0.0, 0.0};
    edge_star.members.push_back(LIBRPA::AbacusKStarMember{0, {0.0, 0.5, 0.0}, {}});
    edge_star.members.push_back(LIBRPA::AbacusKStarMember{0, {0.5, 0.0, 0.0}, {}});

    // Keep the ABACUS star list deliberately out of the LibRPA IBZ order so the new helper
    // has to build the explicit iq->star mapping instead of relying on positional coincidence.
    kmap_ctx.kstars = {gamma_star, edge_star};

    klist.push_back({0.5, 0.0, 0.0});
    klist.push_back({0.0, 0.0, 0.0});
    kfrac_list = klist;
    map_irk_ks[klist[0]] = {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}};
    map_irk_ks[klist[1]] = {{0.0, 0.0, 0.0}, {-0.5, 0.0, 0.0}};

    const auto kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(kmap_ctx, klist, kfrac_list, map_irk_ks);
    const Vector3_Order<double> q_edge_member0{0.0, 0.5, 0.0};
    const Vector3_Order<double> q_edge_member1{0.5, 0.0, 0.0};
    const Vector3_Order<double> q_gamma_member0{-0.5, 0.0, 0.0};
    const Vector3_Order<double> q_gamma_member1{0.0, 0.0, 0.0};
    assert(kstar_grid_mapping.size() == 2);
    assert(kstar_grid_mapping[0].iq_ibz == 0);
    assert(kstar_grid_mapping[0].star_list_index == 1);
    assert(kstar_grid_mapping[0].member_q_bz_keys.size() == 2);
    assert(kstar_grid_mapping[0].member_q_bz_keys[0] == q_edge_member0);
    assert(kstar_grid_mapping[0].member_q_bz_keys[1] == q_edge_member1);
    assert(kstar_grid_mapping[1].iq_ibz == 1);
    assert(kstar_grid_mapping[1].star_list_index == 0);
    assert(kstar_grid_mapping[1].member_q_bz_keys[0] == q_gamma_member0);
    assert(kstar_grid_mapping[1].member_q_bz_keys[1] == q_gamma_member1);

    // If LibRPA's rebuilt full-star list chooses a different representative set from the
    // ABACUS sidecar, the mapping helper must still trust the sidecar coordinates instead of
    // aborting. This mirrors the production path where ABACUS exports the authoritative k-star.
    map_irk_ks.clear();
    map_irk_ks[klist[0]] = {{0.5, 0.0, 0.0}, {0.25, 0.25, 0.0}};
    map_irk_ks[klist[1]] = {{0.0, 0.0, 0.0}, {-0.5, 0.0, 0.0}};
    const auto fallback_kstar_grid_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(kmap_ctx, klist, kfrac_list, map_irk_ks);
    assert(fallback_kstar_grid_mapping[0].member_q_bz_keys[0] == q_edge_member0);
    assert(fallback_kstar_grid_mapping[0].member_q_bz_keys[1] == q_edge_member1);

    // The sidecar->internal conversion must use the same row-vector convention as `klist`.
    // A non-symmetric reciprocal matrix catches accidental use of the column-vector formula.
    G = Matrix3(1.0, 2.0, 0.0,
                3.0, 4.0, 0.0,
                0.0, 0.0, 1.0);
    klist.clear();
    kfrac_list.clear();
    map_irk_ks.clear();
    klist.push_back({0.25 * G.e11 + 0.5 * G.e21, 0.25 * G.e12 + 0.5 * G.e22, 0.0});
    kfrac_list.push_back({0.25, 0.5, 0.0});
    map_irk_ks[klist[0]] = {klist[0]};
    LIBRPA::AbacusKStar skew_star;
    skew_star.star_index = 0;
    skew_star.k_ibz = {0.25, 0.5, 0.0};
    skew_star.members.push_back(LIBRPA::AbacusKStarMember{0, {0.25, 0.5, 0.0}, {}});
    LIBRPA::AbacusSymmetryContext skew_ctx;
    skew_ctx.available = true;
    skew_ctx.kstars = {skew_star};
    const auto skew_mapping =
        LIBRPA::build_abacus_kstar_grid_mapping(skew_ctx, klist, kfrac_list, map_irk_ks);
    assert(skew_mapping[0].member_q_bz_keys[0] == klist[0]);
    G.Identity();
    klist.clear();
    kfrac_list.clear();
    map_irk_ks.clear();
    klist.push_back({0.5, 0.0, 0.0});
    klist.push_back({0.0, 0.0, 0.0});
    kfrac_list = klist;
    map_irk_ks[klist[0]] = {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}};
    map_irk_ks[klist[1]] = {{0.0, 0.0, 0.0}, {-0.5, 0.0, 0.0}};

    // LibRPA can label an IBZ q-point by any symmetry-equivalent full-star member. The ABACUS
    // sidecar stores only one representative per star, so the matcher must also recognize member
    // coordinates and not only the canonical `k_ibz` entry.
    const auto& matched_edge_star_from_member =
        LIBRPA::find_abacus_kstar_for_kpoint(kmap_ctx.kstars, {0.0, 0.5, 0.0}, "test k-stars");
    assert(matched_edge_star_from_member.star_index == 1);

    const auto edge_members =
        LIBRPA::build_abacus_full_kpoint_member_list(kmap_ctx, {{0.0, 0.5, 0.0}, {0.0, 0.0, 0.0}});
    assert(edge_members.size() == 4);
    assert(edge_members[0].star_list_index == 1);
    assert(edge_members[1].star_list_index == 1);
    assert(edge_members[2].star_list_index == 0);
    assert(edge_members[3].star_list_index == 0);

    LIBRPA::AbacusSymmetryContext block_symm_ctx;
    block_symm_ctx.available = true;
    block_symm_ctx.abf_shell_layout_available = true;
    block_symm_ctx.abf_lmax = 0;
    block_symm_ctx.abf_type_layout_candidates = {
        {LIBRPA::AbacusAOTypeLayout{"X", "X.abf", {1}, 1}}
    };
    block_symm_ctx.atom_to_type = {{0, 0}};

    LIBRPA::AbacusSymmetryOperation block_identity;
    block_identity.isym = 0;
    block_identity.rotation = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    block_identity.translation = {0.0, 0.0, 0.0};
    block_identity.shell_rotations[0] = ComplexMatrix(1, 1);
    block_identity.shell_rotations[0](0, 0) = std::complex<double>(1.0, 0.0);
    block_symm_ctx.rspace_operations = {block_identity};

    LIBRPA::AbacusKAtomRotation block_atom_rotation;
    block_atom_rotation.atom_from = 0;
    block_atom_rotation.atom_to = 0;
    block_atom_rotation.atom_type = 0;
    block_atom_rotation.lmax = 0;
    block_atom_rotation.shell_rotations[0] = ComplexMatrix(1, 1);
    block_atom_rotation.shell_rotations[0](0, 0) = std::complex<double>(1.0, 0.0);

    LIBRPA::AbacusKStarMember block_spatial_member;
    block_spatial_member.isym = 0;
    block_spatial_member.k_bz = {0.0, 0.0, 0.0};
    block_spatial_member.atom_rotations = {block_atom_rotation};

    LIBRPA::AbacusKStarMember block_time_reversal_member = block_spatial_member;
    block_time_reversal_member.isym = 1;

    LIBRPA::AbacusKStar block_star;
    block_star.star_index = 0;
    block_star.k_ibz = {0.0, 0.0, 0.0};
    block_star.members = {block_spatial_member, block_time_reversal_member};
    block_symm_ctx.kstars = {block_star};

    LIBRPA::abacus_atom_block_matrix_map_t block_input;
    block_input[0][0] = ComplexMatrix(1, 1);
    block_input[0][0](0, 0) = std::complex<double>(1.25, 0.75);

    const std::map<atom_t, size_t> atom_nabf{{0, 1}};
    const std::map<atom_t, std::array<double, 3>> block_coord_frac{{0, {0.0, 0.0, 0.0}}};
    const auto block_symmetrized = LIBRPA::symmetrize_abacus_abf_ibz_kspace_operator_blocks(
        block_symm_ctx, {0.0, 0.0, 0.0}, block_input, atom_nabf, block_coord_frac);

    ComplexMatrix block_matrix_input(1, 1);
    block_matrix_input(0, 0) = block_input.at(0).at(0)(0, 0);
    const auto matrix_symmetrized = LIBRPA::symmetrize_abacus_abf_ibz_kspace_operator_matrix(
        block_symm_ctx, {0.0, 0.0, 0.0}, block_matrix_input, atom_nabf, block_coord_frac);

    assert(std::abs(block_symmetrized.at(0).at(0)(0, 0) - block_input.at(0).at(0)(0, 0)) < 1e-12);
    assert(std::abs(matrix_symmetrized(0, 0) - block_matrix_input(0, 0)) < 1e-12);
    assert(std::abs(block_symmetrized.at(0).at(0)(0, 0) - matrix_symmetrized(0, 0)) < 1e-12);

    LIBRPA::AbacusSymmetryContext abf_gauge_ctx;
    abf_gauge_ctx.available = true;
    abf_gauge_ctx.abf_shell_layout_available = true;
    abf_gauge_ctx.abf_lmax = 0;
    abf_gauge_ctx.abf_type_layout_candidates = {
        {LIBRPA::AbacusAOTypeLayout{"X", "X.abf", {1}, 1}}
    };
    abf_gauge_ctx.atom_to_type = {{0, 0}, {1, 0}};
    abf_gauge_ctx.rspace_operations = {block_identity};

    LIBRPA::AbacusKAtomRotation abf_atom0 = block_atom_rotation;
    abf_atom0.atom_from = 0;
    abf_atom0.atom_to = 0;
    LIBRPA::AbacusKAtomRotation abf_atom1 = block_atom_rotation;
    abf_atom1.atom_from = 1;
    abf_atom1.atom_to = 1;

    LIBRPA::AbacusKStarMember abf_exact_member;
    abf_exact_member.isym = 0;
    abf_exact_member.k_bz = {0.5, 0.0, 0.0};
    abf_exact_member.atom_rotations = {abf_atom0, abf_atom1};

    LIBRPA::AbacusKStarMember abf_equiv_member = abf_exact_member;
    abf_equiv_member.k_bz = {-0.5, 0.0, 0.0};

    LIBRPA::AbacusKStar abf_gauge_star;
    abf_gauge_star.star_index = 0;
    abf_gauge_star.k_ibz = {0.5, 0.0, 0.0};
    abf_gauge_star.members = {abf_exact_member, abf_equiv_member};
    abf_gauge_ctx.kstars = {abf_gauge_star};

    LIBRPA::abacus_atom_block_matrix_map_t abf_gauge_input;
    abf_gauge_input[0][1] = ComplexMatrix(1, 1);
    abf_gauge_input[0][1](0, 0) = std::complex<double>(1.0, 0.0);
    abf_gauge_input[1][0] = ComplexMatrix(1, 1);
    abf_gauge_input[1][0](0, 0) = std::complex<double>(1.0, 0.0);

    const std::map<atom_t, size_t> abf_gauge_atom_nabf{{0, 1}, {1, 1}};
    const std::map<atom_t, std::array<double, 3>> abf_gauge_coord_frac{
        {0, {0.0, 0.0, 0.0}},
        {1, {0.25, 0.0, 0.0}},
    };
    const auto abf_gauge_symmetrized = LIBRPA::symmetrize_abacus_abf_ibz_kspace_operator_blocks(
        abf_gauge_ctx, {0.5, 0.0, 0.0}, abf_gauge_input, abf_gauge_atom_nabf,
        abf_gauge_coord_frac);
    const std::complex<double> expected_abf_gauge_average(0.5, -0.5);
    assert(std::abs(abf_gauge_symmetrized.at(0).at(1)(0, 0) - expected_abf_gauge_average) < 1e-12);
    assert(std::abs(abf_gauge_symmetrized.at(1).at(0)(0, 0)
                    - std::conj(expected_abf_gauge_average))
           < 1e-12);

    LIBRPA::AbacusSymmetryContext opposite_k_ctx;
    opposite_k_ctx.available = true;
    opposite_k_ctx.ao_shell_layout_available = true;
    opposite_k_ctx.ao_lmax = 1;
    opposite_k_ctx.ao_type_layouts = {
        LIBRPA::AbacusAOTypeLayout{"X", "X.orb", {0, 1}, 3}
    };
    opposite_k_ctx.atom_to_type = {{0, 0}};

    LIBRPA::AbacusSymmetryOperation opposite_identity;
    opposite_identity.isym = 0;
    opposite_identity.rotation = {{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    opposite_identity.translation = {0.0, 0.0, 0.0};
    opposite_k_ctx.rspace_operations.push_back(opposite_identity);

    LIBRPA::AbacusSymmetryOperation opposite_inversion;
    opposite_inversion.isym = 1;
    opposite_inversion.rotation = {{{{-1.0, 0.0, 0.0}}, {{0.0, -1.0, 0.0}}, {{0.0, 0.0, -1.0}}}};
    opposite_inversion.translation = {0.0, 0.0, 0.0};
    opposite_k_ctx.rspace_operations.push_back(opposite_inversion);

    LIBRPA::AbacusKAtomRotation opposite_atom_rotation;
    opposite_atom_rotation.atom_from = 0;
    opposite_atom_rotation.atom_to = 0;
    opposite_atom_rotation.atom_type = 0;
    opposite_atom_rotation.lmax = 1;
    opposite_atom_rotation.shell_rotations[1] = ComplexMatrix(3, 3);
    for (int index = 0; index < 3; ++index)
    {
        opposite_atom_rotation.shell_rotations[1](index, index) = std::complex<double>(-1.0, 0.0);
    }

    LIBRPA::AbacusKStarMember opposite_member;
    opposite_member.isym = 1;
    opposite_member.k_bz = {-0.25, -0.25, 0.0};
    opposite_member.atom_rotations = {opposite_atom_rotation};

    const std::map<atom_t, size_t> opposite_atom_nw{{0, 3}};
    const std::map<atom_t, std::array<double, 3>> opposite_coord_frac{{0, {0.0, 0.0, 0.0}}};
    ComplexMatrix opposite_matrix_ibz(3, 3);
    opposite_matrix_ibz(0, 0) = std::complex<double>(1.0, 2.0);
    opposite_matrix_ibz(0, 1) = std::complex<double>(0.2, -0.4);
    opposite_matrix_ibz(0, 2) = std::complex<double>(-0.1, 0.3);
    opposite_matrix_ibz(1, 0) = std::complex<double>(-0.7, 0.5);
    opposite_matrix_ibz(1, 1) = std::complex<double>(0.9, -0.6);
    opposite_matrix_ibz(1, 2) = std::complex<double>(0.8, 0.1);
    opposite_matrix_ibz(2, 0) = std::complex<double>(-0.4, -0.2);
    opposite_matrix_ibz(2, 1) = std::complex<double>(0.6, 0.7);
    opposite_matrix_ibz(2, 2) = std::complex<double>(-0.3, 0.9);

    const auto rotated_opposite_matrix = LIBRPA::rotate_abacus_kspace_matrix(
        opposite_k_ctx, opposite_member, opposite_matrix_ibz, opposite_atom_nw, {0.25, 0.25, 0.0},
        opposite_coord_frac, false);
    const auto conjugated_opposite_matrix = conj(opposite_matrix_ibz);

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            assert(std::abs(rotated_opposite_matrix(row, col) - opposite_matrix_ibz(row, col)) < 1e-12);
        }
    }
    assert(std::abs(rotated_opposite_matrix(0, 0) - conjugated_opposite_matrix(0, 0)) > 1e-6);

    std::system(("rm -rf " + test_dir).c_str());
    return 0;
}
