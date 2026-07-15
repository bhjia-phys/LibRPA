#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/hartree_density.h"
#include "../qsgw/occupation.h"

#include "../core/symmetry_context.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

using librpa_int::ComplexMatrix;
using librpa_int::MeanField;
using librpa_int::SpeciesBasisLayout;
using librpa_int::SymmetryContext;
using librpa_int::Vector3_Order;
using librpa_int::cplxdb;
using librpa_int::qsgw::PeriodicOperatorRMap;
using librpa_int::qsgw::build_total_density_rspace;
using librpa_int::qsgw::build_total_density_delta_rspace;
using librpa_int::qsgw::build_total_density_delta_rspace_symmetry;
using librpa_int::qsgw::build_total_density_rspace_symmetry;
using librpa_int::qsgw::project_periodic_operator_to_fixed_basis;
using librpa_int::qsgw::physical_electron_count;
using librpa_int::qsgw::reconstruct_weighted_full_grid_density;
using librpa_int::qsgw::total_density_k;
using librpa_int::qsgw::update_qsgw_occupations;
using librpa_int::qsgw::validate_canonical_bvk_grid;

namespace
{

void assert_close(const cplxdb actual, const cplxdb expected,
                  const double tolerance = 1.0e-12)
{
    assert(std::abs(actual - expected) < tolerance);
}

template <typename Function>
void assert_throws(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    assert(threw);
}

void set_unit_wfc(MeanField& meanfield, const int spin, const int kpoint)
{
    ComplexMatrix wfc(1, 1);
    wfc(0, 0) = 1.0;
    meanfield.get_eigenvectors()[spin][0][kpoint] = wfc;
}

void test_scalar_and_two_spin_total_density_are_identical()
{
    MeanField scalar(1, 1, 1, 1, 1);
    scalar.get_weight()[0](0, 0) = 2.0;
    set_unit_wfc(scalar, 0, 0);

    MeanField collinear(2, 1, 1, 1, 1);
    collinear.get_weight()[0](0, 0) = 1.0;
    collinear.get_weight()[1](0, 0) = 1.0;
    set_unit_wfc(collinear, 0, 0);
    set_unit_wfc(collinear, 1, 0);

    const auto scalar_density = total_density_k(scalar, 0);
    const auto collinear_density = total_density_k(collinear, 0);
    assert_close(scalar_density(0, 0), 2.0);
    assert_close(collinear_density(0, 0), 2.0);
    assert_close(scalar_density(0, 0), collinear_density(0, 0));
}

void test_scalar_and_spinor_total_density_are_identical()
{
    MeanField scalar(1, 1, 1, 1, 1);
    scalar.get_weight()[0](0, 0) = 2.0;
    set_unit_wfc(scalar, 0, 0);

    MeanField spinor(1, 1, 2, 1, 2);
    spinor.get_weight()[0](0, 0) = 1.0;
    spinor.get_weight()[0](0, 1) = 1.0;
    ComplexMatrix up(2, 1);
    ComplexMatrix down(2, 1);
    up(0, 0) = 1.0;
    down(1, 0) = 1.0;
    spinor.get_eigenvectors()[0][0][0] = up;
    spinor.get_eigenvectors()[0][1][0] = down;

    const auto scalar_density = total_density_k(scalar, 0);
    const auto spinor_density = total_density_k(spinor, 0);
    assert_close(spinor_density(0, 0), 2.0);
    assert_close(scalar_density(0, 0), spinor_density(0, 0));
}

void test_equivalent_one_and_two_kpoint_grids_have_same_rspace_density()
{
    MeanField gamma(1, 1, 1, 1, 1);
    gamma.get_weight()[0](0, 0) = 2.0;
    set_unit_wfc(gamma, 0, 0);
    const std::vector<Vector3_Order<double>> gamma_k{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> gamma_r{{0, 0, 0}};

    MeanField k2(1, 2, 1, 1, 1);
    k2.get_weight()[0](0, 0) = 1.0;
    k2.get_weight()[0](1, 0) = 1.0;
    set_unit_wfc(k2, 0, 0);
    set_unit_wfc(k2, 0, 1);
    const std::vector<Vector3_Order<double>> k2_k{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> k2_r{
        {0, 0, 0},
        {1, 0, 0},
    };

    const auto density_gamma = build_total_density_rspace(gamma, gamma_k, gamma_r);
    const auto density_k2 = build_total_density_rspace(k2, k2_k, k2_r);
    assert_close(density_gamma.at(gamma_r[0])(0, 0), 2.0);
    assert_close(density_k2.at(k2_r[0])(0, 0), 2.0);
    assert_close(density_k2.at(k2_r[1])(0, 0), 0.0);
}

void test_ibz_kstar_restoration_matches_explicit_full_bvk_density()
{
    SymmetryContext context;
    context.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    context.atom_to_type[0] = 0;
    context.input_coord_frac[0] = {0.0, 0.0, 0.0};
    librpa_int::SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    context.rspace_operations.push_back(identity_operation);
    context.rsh_rotations.emplace_back();
    context.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    context.rsh_rotations.back()[0](0, 0) = 1.0;

    librpa_int::SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = 1.0;

    librpa_int::SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    context.kstars.push_back(star);

    MeanField ibz(1, 1, 1, 1, 1);
    ibz.get_weight()[0](0, 0) = 2.0;
    set_unit_wfc(ibz, 0, 0);
    const std::vector<Vector3_Order<double>> ibz_k{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> r_grid{{0, 0, 0}, {1, 0, 0}};
    const std::map<librpa_int::atom_t, std::size_t> atom_nw{{0, 1}};
    const auto restored = build_total_density_rspace_symmetry(
        context, wfc_layouts, ibz, ibz_k, r_grid, atom_nw);
    const auto zero_delta = build_total_density_delta_rspace_symmetry(
        context, wfc_layouts, ibz, ibz, ibz_k, r_grid, atom_nw);

    MeanField full(1, 2, 1, 1, 1);
    full.get_weight()[0](0, 0) = 1.0;
    full.get_weight()[0](1, 0) = 1.0;
    set_unit_wfc(full, 0, 0);
    set_unit_wfc(full, 0, 1);
    const std::vector<Vector3_Order<double>> full_k{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const auto explicit_full = build_total_density_rspace(full, full_k, r_grid);

    for (const auto& r : r_grid)
    {
        assert_close(restored.at(r)(0, 0), explicit_full.at(r)(0, 0));
        assert_close(zero_delta.at(r)(0, 0), 0.0);
    }
    assert_close(restored.at(r_grid[0])(0, 0), 2.0);
    assert_close(restored.at(r_grid[1])(0, 0), 0.0);
}

void test_density_delta_is_live_minus_reference_and_iteration_zero_is_exactly_zero()
{
    MeanField reference(1, 1, 1, 2, 1);
    reference.get_weight()[0](0, 0) = 2.0;
    ComplexMatrix reference_wfc(1, 2);
    reference_wfc(0, 0) = 1.0;
    reference.get_eigenvectors()[0][0][0] = reference_wfc;
    const MeanField reference_before = reference;

    const std::vector<Vector3_Order<double>> kpoints{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> r_grid{{0, 0, 0}};
    const auto zero_delta = build_total_density_delta_rspace(
        reference, reference, kpoints, r_grid);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            assert_close(zero_delta.at(r_grid[0])(row, column), 0.0);
        }
    }

    MeanField live = reference;
    live.get_eigenvectors()[0][0][0].zero_out();
    live.get_eigenvectors()[0][0][0](0, 1) = 1.0;
    const auto delta = build_total_density_delta_rspace(
        live, reference, kpoints, r_grid);
    assert_close(delta.at(r_grid[0])(0, 0), -2.0);
    assert_close(delta.at(r_grid[0])(1, 1), 2.0);
    assert_close(delta.at(r_grid[0])(0, 1), 0.0);
    assert_close(delta.at(r_grid[0])(1, 0), 0.0);
    assert_close(delta.at(r_grid[0])(0, 0) +
                     delta.at(r_grid[0])(1, 1),
                 0.0);

    const auto reference_after = build_total_density_rspace(
        reference, kpoints, r_grid);
    const auto reference_expected = build_total_density_rspace(
        reference_before, kpoints, r_grid);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            assert_close(reference_after.at(r_grid[0])(row, column),
                         reference_expected.at(r_grid[0])(row, column));
        }
    }
}

void test_normalized_ks0_reference_starts_with_exactly_zero_density_delta()
{
    MeanField producer(1, 2, 2, 2, 1);
    producer.get_weight()[0].zero_out();
    producer.get_weight()[0](0, 0) = 0.5;
    producer.get_weight()[0](0, 1) = 0.5;
    producer.get_eigenvals()[0](0, 0) = -1.0;
    producer.get_eigenvals()[0](0, 1) = 2.0;
    producer.get_eigenvals()[0](1, 0) = -2.0;
    producer.get_eigenvals()[0](1, 1) = 3.0;
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        ComplexMatrix wfc(2, 2);
        wfc(0, 0) = 1.0;
        wfc(1, 1) = 1.0;
        producer.get_eigenvectors()[0][0][kpoint] = wfc;
    }

    const std::vector<double> weights{0.25, 0.75};
    const double electrons = physical_electron_count(producer, weights);
    MeanField normalized_reference = producer;
    const auto occupations = update_qsgw_occupations(
        normalized_reference, producer, weights, electrons);
    const MeanField live = normalized_reference;

    assert_close(electrons, 1.0);
    assert_close(occupations.electron_count, electrons);
    assert(std::abs(normalized_reference.get_weight()[0](0, 0) -
                    producer.get_weight()[0](0, 0)) > 1.0e-12);

    const std::vector<Vector3_Order<double>> kpoints{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> r_grid{
        {0, 0, 0},
        {1, 0, 0},
    };
    const auto zero_delta = build_total_density_delta_rspace(
        live, normalized_reference, kpoints, r_grid);
    for (const auto& translation : r_grid)
    {
        for (int row = 0; row < 2; ++row)
        {
            for (int column = 0; column < 2; ++column)
            {
                assert_close(
                    zero_delta.at(translation)(row, column), 0.0);
            }
        }
    }
}

void test_rspace_to_full_grid_reconstruction_has_exactly_one_inverse_nk()
{
    std::map<Vector3_Order<int>, ComplexMatrix> density_r;
    ComplexMatrix r0(1, 1);
    ComplexMatrix r1(1, 1);
    r0(0, 0) = 2.0;
    r1(0, 0) = 0.0;
    density_r[{0, 0, 0}] = r0;
    density_r[{1, 0, 0}] = r1;

    const std::vector<Vector3_Order<double>> full_k{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const auto density_k = reconstruct_weighted_full_grid_density(density_r, full_k);
    assert_close(density_k.at(0)(0, 0), 1.0);
    assert_close(density_k.at(1)(0, 0), 1.0);
}

void test_rspace_to_full_grid_reconstruction_uses_positive_fourier_phase()
{
    std::map<Vector3_Order<int>, ComplexMatrix> density_r;
    for (int r = 0; r < 4; ++r)
    {
        density_r[{r, 0, 0}] = ComplexMatrix(1, 1);
    }
    density_r.at({0, 0, 0})(0, 0) = 4.0;
    density_r.at({1, 0, 0})(0, 0) = cplxdb(0.0, 1.0);
    density_r.at({2, 0, 0})(0, 0) = 0.0;
    density_r.at({3, 0, 0})(0, 0) = cplxdb(0.0, -1.0);

    const std::vector<Vector3_Order<double>> full_k{
        {0.0, 0.0, 0.0},
        {0.25, 0.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.75, 0.0, 0.0},
    };
    const auto density_k = reconstruct_weighted_full_grid_density(density_r, full_k);
    assert_close(density_k.at(0)(0, 0), 1.0);
    assert_close(density_k.at(1)(0, 0), 0.5);
    assert_close(density_k.at(2)(0, 0), 1.0);
    assert_close(density_k.at(3)(0, 0), 1.5);
}

void test_canonical_bvk_validation_rejects_missing_and_duplicate_points()
{
    const Vector3_Order<int> period{2, 1, 1};
    const std::vector<Vector3_Order<double>> valid{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    validate_canonical_bvk_grid(period, valid, 1.0e-10);
    validate_canonical_bvk_grid(
        period, {{0.0, 0.0, 0.0}, {-0.5, 0.0, 0.0}}, 1.0e-10);

    const Vector3_Order<int> period_2d{2, 2, 1};
    validate_canonical_bvk_grid(
        period_2d,
        {{0.5, 0.5, 0.0}, {0.0, 0.0, 0.0},
         {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}},
        1.0e-10);

    assert_throws([&] {
        validate_canonical_bvk_grid(period, {{0.0, 0.0, 0.0}}, 1.0e-10);
    });
    assert_throws([&] {
        validate_canonical_bvk_grid(
            period, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, 1.0e-10);
    });
    assert_throws([&] {
        validate_canonical_bvk_grid(
            period, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}, 1.0e-10);
    });
    assert_throws([&] {
        validate_canonical_bvk_grid(
            period, {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}}, 1.0e-10);
    });
}

void test_grid_and_band_projection_return_independent_maps()
{
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, ComplexMatrix>> operator_r;
    ComplexMatrix h0(1, 1);
    ComplexMatrix h1(1, 1);
    h0(0, 0) = 2.0;
    h1(0, 0) = 0.5;
    operator_r[0][{0, {0, 0, 0}}] = h0;
    operator_r[0][{0, {1, 0, 0}}] = h1;

    MeanField grid_reference(1, 1, 1, 1, 1);
    set_unit_wfc(grid_reference, 0, 0);
    MeanField band_reference(1, 2, 1, 1, 1);
    set_unit_wfc(band_reference, 0, 0);
    set_unit_wfc(band_reference, 0, 1);

    const auto grid = project_periodic_operator_to_fixed_basis(
        operator_r, grid_reference, {{0.0, 0.0, 0.0}});
    const auto band = project_periodic_operator_to_fixed_basis(
        operator_r, band_reference,
        {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}});

    assert_close(grid.at(0).at(0)(0, 0), 2.5);
    assert_close(band.at(0).at(0)(0, 0), 2.5);
    assert_close(band.at(0).at(1)(0, 0), 1.5);
    assert(grid.at(0).size() == 1);
    assert(band.at(0).size() == 2);
}

void test_periodic_operator_projection_is_hermitianized()
{
    PeriodicOperatorRMap operator_r;
    ComplexMatrix block(2, 2);
    block(0, 0) = 1.0;
    block(1, 1) = 2.0;
    block(0, 1) = {0.4, 0.3};
    block(1, 0) = {-0.2, 0.1};
    operator_r[0][{0, {0, 0, 0}}] = block;

    MeanField reference(1, 1, 2, 2, 1);
    ComplexMatrix identity(2, 2);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    reference.get_eigenvectors()[0][0][0] = identity;
    const auto projected = project_periodic_operator_to_fixed_basis(
        operator_r, reference, {{0.0, 0.0, 0.0}});
    const auto& value = projected.at(0).at(0);
    assert_close(value(0, 1), {0.1, 0.1});
    assert_close(value(1, 0), {0.1, -0.1});
}

void test_projection_uses_positive_fourier_phase_and_sums_spinors()
{
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, ComplexMatrix>>
        phase_operator;
    for (int r = 0; r < 4; ++r)
    {
        phase_operator[0][{0, {r, 0, 0}}] = ComplexMatrix(1, 1);
    }
    phase_operator[0][{0, {0, 0, 0}}](0, 0) = 4.0;
    phase_operator[0][{0, {1, 0, 0}}](0, 0) = cplxdb(0.0, 1.0);
    phase_operator[0][{0, {2, 0, 0}}](0, 0) = 0.0;
    phase_operator[0][{0, {3, 0, 0}}](0, 0) = cplxdb(0.0, -1.0);

    MeanField phase_reference(1, 1, 1, 1, 1);
    set_unit_wfc(phase_reference, 0, 0);
    const auto phase_projected = project_periodic_operator_to_fixed_basis(
        phase_operator, phase_reference, {{0.25, 0.0, 0.0}});
    assert_close(phase_projected.at(0).at(0)(0, 0), 2.0);

    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, ComplexMatrix>>
        scalar_operator;
    scalar_operator[0][{0, {0, 0, 0}}] = ComplexMatrix(1, 1);
    scalar_operator[0][{0, {0, 0, 0}}](0, 0) = 2.0;

    MeanField spinor_reference(1, 1, 1, 1, 2);
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    ComplexMatrix up(1, 1);
    ComplexMatrix down(1, 1);
    up(0, 0) = inv_sqrt_two;
    down(0, 0) = cplxdb(0.0, inv_sqrt_two);
    spinor_reference.get_eigenvectors()[0][0][0] = up;
    spinor_reference.get_eigenvectors()[0][1][0] = down;
    const auto spinor_projected = project_periodic_operator_to_fixed_basis(
        scalar_operator, spinor_reference, {{0.0, 0.0, 0.0}});
    assert_close(spinor_projected.at(0).at(0)(0, 0), 2.0);
}

} // namespace

int main()
{
    test_scalar_and_two_spin_total_density_are_identical();
    test_scalar_and_spinor_total_density_are_identical();
    test_equivalent_one_and_two_kpoint_grids_have_same_rspace_density();
    test_ibz_kstar_restoration_matches_explicit_full_bvk_density();
    test_density_delta_is_live_minus_reference_and_iteration_zero_is_exactly_zero();
    test_normalized_ks0_reference_starts_with_exactly_zero_density_delta();
    test_rspace_to_full_grid_reconstruction_has_exactly_one_inverse_nk();
    test_rspace_to_full_grid_reconstruction_uses_positive_fourier_phase();
    test_canonical_bvk_validation_rejects_missing_and_duplicate_points();
    test_grid_and_band_projection_return_independent_maps();
    test_periodic_operator_projection_is_hermitianized();
    test_projection_uses_positive_fourier_phase_and_sums_spinors();
    std::cout << "test_qsgw_hartree_density: all tests passed\n";
    return 0;
}
