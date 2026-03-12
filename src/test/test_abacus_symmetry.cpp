#include "abacus_symmetry.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>

namespace
{

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream ofs(path);
    assert(ofs.good());
    ofs << content;
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

    write_file(test_dir + "/symrot_k.txt",
               "Number of IBZ k-points (k stars): 1\n"
               "Format:\n"
               "dummy\n"
               "Star 1 of IBZ k-point (0.000000 0.000000 0.000000):\n"
               "0\n"
               "(0.000000 0.000000 0.000000)\n"
               "atom 1 -> 1 of type 1 with Lmax= 1\n"
               "(1,0)\n"
               "(1,0)(0,0)(0,0)\n"
               "(0,0)(1,0)(0,0)\n"
               "(0,0)(0,0)(1,0)\n");

    write_file(test_dir + "/INPUT",
               "INPUT_PARAMETERS\n"
               "orbital_dir ./\n");

    write_file(test_dir + "/STRU",
               "ATOMIC_SPECIES\n"
               "B 1.0 B.upf\n"
               "\n"
               "NUMERICAL_ORBITAL\n"
               "B.orb\n"
               "\n"
               "LATTICE_CONSTANT\n"
               "1.0\n"
               "\n"
               "LATTICE_VECTORS\n"
               "1 0 0\n"
               "0 1 0\n"
               "0 0 1\n"
               "\n"
               "ATOMIC_POSITIONS\n"
               "Direct\n"
               "B\n"
               "0.0\n"
               "1\n"
               "0.0 0.0 0.0 0 0 0\n");

    write_file(test_dir + "/B.orb",
               "Element                     B\n"
               "Lmax                        1\n"
               "Number of Sorbital-->       1\n"
               "Number of Porbital-->       1\n"
               "SUMMARY  END\n");

    LIBRPA::AbacusSymmetryContext ctx;
    const bool loaded = LIBRPA::load_abacus_symmetry_context(test_dir, ctx, nullptr);
    assert(loaded);
    assert(ctx.available);
    assert(ctx.count_irreducible_pairs() == 2);
    assert(ctx.count_irreducible_blocks() == 2);
    assert(ctx.rspace_operations.size() == 1);
    assert(ctx.kstars.size() == 1);
    assert(ctx.count_kstar_members() == 1);
    assert(ctx.ao_lmax == 1);
    assert(ctx.abf_lmax == 1);
    assert(ctx.rspace_operations.front().shell_rotations.at(1).nr == 3);
    assert(ctx.kstars.front().members.front().atom_rotations.front().shell_rotations.at(1).nc == 3);
    assert(ctx.has_ao_shell_layout());
    assert(ctx.ao_type_layouts.size() == 1);
    assert(ctx.count_atoms_with_layout() == 1);
    assert(ctx.get_ao_type_layout(0).nao == 4);
    assert(ctx.get_ao_type_layout(0).shell_counts.size() == 2);
    assert(ctx.get_ao_type_layout(0).shell_counts[0] == 1);
    assert(ctx.get_ao_type_layout(0).shell_counts[1] == 1);
    assert(ctx.atom_to_type.at(0) == 0);

    const auto full_rotation = LIBRPA::build_abacus_ao_rotation_matrix(
        ctx, 0, ctx.rspace_operations.front().shell_rotations);
    assert(full_rotation.nr == 4);
    assert(full_rotation.nc == 4);
    assert(full_rotation(0, 0) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(1, 1) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(2, 2) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(3, 3) == std::complex<double>(1.0, 0.0));
    assert(full_rotation(0, 1) == std::complex<double>(0.0, 0.0));

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

    const auto rotated_dmat_tr =
        LIBRPA::rotate_abacus_kspace_matrix(ctx,
                                            ctx.kstars.front().members.front(),
                                            dmat_ibz,
                                            std::map<atom_t, size_t>{{0, 4}},
                                            Vector3_Order<double>{0.0, 0.0, 0.0},
                                            std::map<atom_t, std::array<double, 3>>{{0, {0.0, 0.0, 0.0}}},
                                            true);
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            assert(rotated_dmat_tr(i, j) == dmat_ibz(i, j));
        }
    }

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
    assert(sector_stars.at(pair_00).at(R0).size() == 2);
    assert(sector_stars.at(pair_01).at(R0).size() == 2);
    assert(sector_stars.at(pair_00).at(R0)[0].full_atom_pair == pair_00);
    assert(sector_stars.at(pair_00).at(R0)[1].full_atom_pair == pair_11);
    assert(sector_stars.at(pair_01).at(R0)[0].full_atom_pair == pair_01);
    assert(sector_stars.at(pair_01).at(R0)[1].full_atom_pair == pair_10);

    const std::map<atom_t, std::array<double, 3>> coord_frac_test_noisy{
        {0, {0.0, 0.0, 0.0}},
        {1, {0.500002, 0.500002, 0.500002}},
    };
    LIBRPA::abacus_rspace_sector_stars_t sector_stars_noisy;
    LIBRPA::build_abacus_rspace_sector_stars(
        rspace_ctx, coord_frac_test_noisy, period, Rlist, sector_stars_noisy, nullptr);
    assert(sector_stars_noisy.at(pair_00).at(R0).size() == 2);
    assert(sector_stars_noisy.at(pair_01).at(R0).size() == 2);

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

    std::system(("rm -rf " + test_dir).c_str());
    return 0;
}
