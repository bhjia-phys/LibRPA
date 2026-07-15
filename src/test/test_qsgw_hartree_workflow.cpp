#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/hartree_workflow.h"

#include "../core/pbc.h"

#include <cassert>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>
#include <vector>

using librpa_int::AtomicBasis;
using librpa_int::AtomPairBvKRemap;
using librpa_int::ComplexMatrix;
using librpa_int::Cs_LRI;
using librpa_int::Matrix3;
using librpa_int::Matz;
using librpa_int::Vector3;
using librpa_int::Vector3_Order;
using librpa_int::atom_t;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::cplxdb;
using librpa_int::matrix;
using librpa_int::qsgw::HartreeDkMap;
using librpa_int::qsgw::HartreeKNormalization;
using librpa_int::qsgw::HartreeStaticData;
using librpa_int::qsgw::build_hartree_c_k;
using librpa_int::qsgw::build_hartree_delta_fixed_basis;
using librpa_int::qsgw::build_hartree_static_data;
using librpa_int::qsgw::build_hartree_v_q0;
using librpa_int::qsgw::inverse_fourier_hartree_operator;
using librpa_int::qsgw::materialize_hartree_coefficients;
using librpa_int::qsgw::split_weighted_density_by_atom;

namespace
{

void assert_close(const cplxdb actual, const cplxdb expected,
                  const double tolerance = 1.0e-13)
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

std::shared_ptr<matrix> real_matrix(const int rows, const int columns,
                                    const std::vector<double>& values)
{
    assert(static_cast<int>(values.size()) == rows * columns);
    auto result = std::make_shared<matrix>(rows, columns, true);
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            (*result)(row, column) = values[row * columns + column];
        }
    }
    return result;
}

void test_real_space_ri_coefficients_are_fourier_transformed()
{
    AtomicBasis wfc;
    AtomicBasis auxiliary;
    wfc.set(std::vector<std::size_t>{1});
    auxiliary.set(std::vector<std::size_t>{1});
    Cs_LRI coefficients;
    coefficients.use_libri = false;
    coefficients.data_IJR[0][0][{0, 0, 0}] = real_matrix(1, 1, {2.0});
    coefficients.data_IJR[0][0][{1, 0, 0}] = real_matrix(1, 1, {0.5});

    const std::vector<Vector3_Order<double>> kpoints{
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const auto transformed = build_hartree_c_k(
        coefficients, wfc, auxiliary, kpoints);

    assert_close(transformed.at(0).at(0).at(0)(0, 0), 2.5);
    assert_close(transformed.at(0).at(0).at(1)(0, 0), 1.5);
}

void test_libri_coefficients_are_materialized_without_mutating_the_source()
{
    AtomicBasis wfc;
    AtomicBasis auxiliary;
    wfc.set(std::vector<std::size_t>{2, 1});
    auxiliary.set(std::vector<std::size_t>{1, 2});
    Cs_LRI coefficients;
    coefficients.use_libri = true;
    auto data = std::make_shared<std::valarray<double>>(2);
    (*data)[0] = 1.25;
    (*data)[1] = -0.5;
    coefficients.data_libri[0][{1, {1, -1, 0}}] =
        RI::Tensor<double>({1, 2, 1}, data);

    const Cs_LRI materialized = materialize_hartree_coefficients(
        coefficients, wfc, auxiliary);
    const auto& block = materialized.data_IJR.at(0).at(1).at({1, -1, 0});
    assert(!materialized.use_libri);
    assert(block->nr == 2);
    assert(block->nc == 1);
    assert_close((*block)(0, 0), 1.25);
    assert_close((*block)(1, 0), -0.5);
    assert(coefficients.use_libri);
    assert(coefficients.data_IJR.empty());
    assert(coefficients.data_libri.size() == 1);
}

void test_bare_coulomb_is_projected_to_a_complete_hermitian_operator()
{
    AtomicBasis auxiliary;
    auxiliary.set(std::vector<std::size_t>{1, 2});
    atpair_k_cplx_mat_t coulomb;
    auto diagonal0 = std::make_shared<ComplexMatrix>(1, 1);
    (*diagonal0)(0, 0) = 2.0;
    coulomb[0][0][{0.0, 0.0, 0.0}] = diagonal0;
    auto cross = std::make_shared<ComplexMatrix>(1, 2);
    (*cross)(0, 0) = {0.5, 0.25};
    (*cross)(0, 1) = {-0.2, 0.4};
    coulomb[0][1][{0.0, 0.0, 0.0}] = cross;
    auto reverse_cross = std::make_shared<ComplexMatrix>(2, 1);
    (*reverse_cross)(0, 0) = {0.7, -0.25};
    (*reverse_cross)(1, 0) = {-0.2, -0.2};
    coulomb[1][0][{0.0, 0.0, 0.0}] = reverse_cross;
    auto diagonal1 = std::make_shared<ComplexMatrix>(2, 2);
    (*diagonal1)(0, 0) = 3.0;
    (*diagonal1)(1, 1) = 4.0;
    (*diagonal1)(0, 1) = {0.1, -0.3};
    (*diagonal1)(1, 0) = {0.1, 0.3};
    coulomb[1][1][{0.0, 0.0, 0.0}] = diagonal1;

    const auto completed = build_hartree_v_q0(coulomb, auxiliary, 1.0e-12);
    assert_close(completed.at(0).at(1)(0, 0), {0.6, 0.25});
    assert_close(completed.at(0).at(1)(0, 1), {-0.2, 0.3});
    assert_close(completed.at(1).at(0)(0, 0), {0.6, -0.25});
    assert_close(completed.at(1).at(0)(1, 0), {-0.2, -0.3});
}

void test_density_blocking_and_inverse_fourier_have_one_inverse_grid_factor()
{
    const std::map<int, int> atom_sizes{{0, 1}, {1, 1}};
    std::map<int, ComplexMatrix> density;
    density[0] = ComplexMatrix(2, 2);
    density[0](0, 0) = 1.0;
    density[0](0, 1) = {2.0, 0.5};
    density[0](1, 0) = {2.0, -0.5};
    density[0](1, 1) = 3.0;
    const auto blocked = split_weighted_density_by_atom(density, atom_sizes);
    assert_close(blocked.at(0).at(1).at(0)(0, 0), {2.0, 0.5});
    assert_close(blocked.at(1).at(0).at(0)(0, 0), {2.0, -0.5});

    HartreeDkMap operator_k;
    for (int atom_i = 0; atom_i < 2; ++atom_i)
    {
        for (int atom_j = 0; atom_j < 2; ++atom_j)
        {
            operator_k[atom_i][atom_j][0] = ComplexMatrix(1, 1);
            operator_k[atom_i][atom_j][1] = ComplexMatrix(1, 1);
            operator_k[atom_i][atom_j][0](0, 0) = 6.0;
            operator_k[atom_i][atom_j][1](0, 0) = 2.0;
        }
    }
    const std::vector<Vector3_Order<double>> kpoints{
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> translations{
        {0, 0, 0}, {1, 0, 0}};
    const auto operator_r = inverse_fourier_hartree_operator(
        operator_k, kpoints, translations);
    assert_close(operator_r.at(0).at({0, {0, 0, 0}})(0, 0), 4.0);
    assert_close(operator_r.at(0).at({0, {1, 0, 0}})(0, 0), 2.0);
}

void test_inverse_fourier_uses_atom_pair_nearest_bvk_cells()
{
    HartreeDkMap operator_k;
    const std::vector<Vector3_Order<double>> kpoints{
        {0.0, 0.0, 0.0},
        {1.0 / 3.0, 0.0, 0.0},
        {2.0 / 3.0, 0.0, 0.0},
    };
    for (int kpoint = 0; kpoint < 3; ++kpoint)
    {
        operator_k[0][1][kpoint] = ComplexMatrix(1, 1);
        const double angle = 2.0 * std::acos(-1.0) * kpoints[kpoint].x;
        operator_k[0][1][kpoint](0, 0) =
            cplxdb(std::cos(angle), std::sin(angle));
    }

    const std::vector<Vector3_Order<int>> translations{
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    const std::map<atom_t, Vector3<double>> coordinates{
        {0, {0.1, 0.0, 0.0}},
        {1, {0.9, 0.0, 0.0}},
    };
    const AtomPairBvKRemap<atom_t> remap(
        coordinates, translations, {3, 1, 1}, Matrix3{}, 0);

    const auto operator_r = inverse_fourier_hartree_operator(
        operator_k, kpoints, translations, &remap);
    const auto& row = operator_r.at(0);
    assert(row.count({1, {-2, 0, 0}}) == 1);
    assert(row.count({1, {1, 0, 0}}) == 0);
    assert_close(row.at({1, {-2, 0, 0}})(0, 0), 1.0);
}

void test_distributed_libri_coefficients_without_full_hartree_copy_are_rejected()
{
    AtomicBasis basis;
    basis.set(std::vector<std::size_t>{1});
    Cs_LRI coefficients;
    coefficients.use_libri = true;
    assert_throws([&] {
        build_hartree_c_k(
            coefficients, basis, basis, {{0.0, 0.0, 0.0}});
    });
}

void test_libri_coefficients_with_full_hartree_copy_are_supported()
{
    AtomicBasis basis;
    basis.set(std::vector<std::size_t>{1});
    Cs_LRI coefficients;
    coefficients.use_libri = true;
    coefficients.data_IJR[0][0][{0, 0, 0}] =
        real_matrix(1, 1, {2.0});

    const auto transformed = build_hartree_c_k(
        coefficients, basis, basis, {{0.0, 0.0, 0.0}});
    assert_close(transformed.at(0).at(0).at(0)(0, 0), 2.0);
}

void test_end_to_end_delta_is_projected_to_an_independent_target_basis()
{
    librpa_int::MeanField reference(1, 1, 1, 1, 1);
    reference.get_eigenvectors()[0][0][0] = ComplexMatrix(1, 1);
    reference.get_eigenvectors()[0][0][0](0, 0) = 1.0;
    reference.get_weight()[0](0, 0) = 1.0;
    librpa_int::MeanField live = reference;
    live.get_weight()[0](0, 0) = 1.5;

    librpa_int::MeanField band_reference(1, 2, 2, 1, 1);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        band_reference.get_eigenvectors()[0][0][kpoint] =
            ComplexMatrix(2, 1);
        band_reference.get_eigenvectors()[0][0][kpoint](0, 0) = 1.0;
        band_reference.get_eigenvectors()[0][0][kpoint](1, 0) = 2.0;
    }

    HartreeStaticData static_data;
    static_data.c_k[0][0][0] = ComplexMatrix(1, 1);
    static_data.c_k[0][0][0](0, 0) = 1.0;
    static_data.v_q0[0][0] = ComplexMatrix(1, 1);
    static_data.v_q0[0][0](0, 0) = 3.0;
    static_data.atom_ao_sizes = {{0, 1}};
    static_data.period = {1, 1, 1};
    static_data.full_kpoints = {{0.0, 0.0, 0.0}};
    static_data.translations = {{0, 0, 0}};

    const auto projected = build_hartree_delta_fixed_basis(
        static_data, live, reference, {{0.0, 0.0, 0.0}},
        band_reference,
        {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}});
    assert(projected.at(0).size() == 2);
    assert(projected.at(0).at(0).nr() == 2);
    assert_close(projected.at(0).at(0)(0, 0), 6.0);
    assert_close(projected.at(0).at(0)(0, 1), 12.0);
    assert_close(projected.at(0).at(0)(1, 1), 24.0);
}

void test_workflow_propagates_explicit_legacy_k_normalization()
{
    librpa_int::MeanField reference(1, 2, 1, 1, 1);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        reference.get_eigenvectors()[0][0][kpoint] = ComplexMatrix(1, 1);
        reference.get_eigenvectors()[0][0][kpoint](0, 0) = 1.0;
        reference.get_weight()[0](kpoint, 0) = 0.5;
    }
    librpa_int::MeanField live = reference;
    live.get_weight()[0](0, 0) = 0.75;
    live.get_weight()[0](1, 0) = 0.75;

    HartreeStaticData corrected;
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        corrected.c_k[0][0][kpoint] = ComplexMatrix(1, 1);
        corrected.c_k[0][0][kpoint](0, 0) = 1.0;
    }
    corrected.v_q0[0][0] = ComplexMatrix(1, 1);
    corrected.v_q0[0][0](0, 0) = 3.0;
    corrected.atom_ao_sizes = {{0, 1}};
    corrected.period = {2, 1, 1};
    corrected.full_kpoints = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    corrected.translations = {{0, 0, 0}, {1, 0, 0}};

    HartreeStaticData legacy = corrected;
    legacy.normalization =
        HartreeKNormalization::legacy_extra_inverse_nk;
    const auto corrected_projection = build_hartree_delta_fixed_basis(
        corrected, live, reference, corrected.full_kpoints,
        reference, corrected.full_kpoints);
    const auto legacy_projection = build_hartree_delta_fixed_basis(
        legacy, live, reference, legacy.full_kpoints,
        reference, legacy.full_kpoints);

    assert(std::abs(corrected_projection.at(0).at(0)(0, 0)) > 1.0e-12);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        assert_close(
            legacy_projection.at(0).at(kpoint)(0, 0),
            0.5 * corrected_projection.at(0).at(kpoint)(0, 0));
    }
}

void test_static_builder_binds_complete_full_bz_input()
{
    AtomicBasis wavefunction;
    AtomicBasis auxiliary;
    wavefunction.set(std::vector<std::size_t>{1});
    auxiliary.set(std::vector<std::size_t>{1});

    Cs_LRI coefficients;
    coefficients.use_libri = false;
    coefficients.data_IJR[0][0][{0, 0, 0}] =
        real_matrix(1, 1, {2.0});

    atpair_k_cplx_mat_t coulomb;
    auto q0 = std::make_shared<ComplexMatrix>(1, 1);
    (*q0)(0, 0) = 3.0;
    coulomb[0][0][{0.0, 0.0, 0.0}] = q0;

    librpa_int::PeriodicBoundaryData pbc;
    pbc.set_period(2, 1, 1);
    pbc.kfrac_list_full = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    const AtomPairBvKRemap<atom_t> remap;
    const HartreeStaticData result = build_hartree_static_data(
        coefficients, coulomb, wavefunction, auxiliary, pbc, remap,
        HartreeKNormalization::legacy_extra_inverse_nk, 1.0e-12);

    assert(result.period == Vector3_Order<int>(2, 1, 1));
    assert(result.full_kpoints.size() == 2);
    assert(result.translations.size() == 2);
    assert((result.atom_ao_sizes == std::map<int, int>{{0, 1}}));
    assert(result.normalization ==
           HartreeKNormalization::legacy_extra_inverse_nk);
    assert_close(result.c_k.at(0).at(0).at(0)(0, 0), 2.0);
    assert_close(result.c_k.at(0).at(0).at(1)(0, 0), 2.0);
    assert_close(result.v_q0.at(0).at(0)(0, 0), 3.0);
}

void test_static_builder_rejects_an_incomplete_full_bz_grid()
{
    AtomicBasis basis;
    basis.set(std::vector<std::size_t>{1});
    Cs_LRI coefficients;
    coefficients.use_libri = false;
    coefficients.data_IJR[0][0][{0, 0, 0}] =
        real_matrix(1, 1, {1.0});

    atpair_k_cplx_mat_t coulomb;
    auto q0 = std::make_shared<ComplexMatrix>(1, 1);
    (*q0)(0, 0) = 1.0;
    coulomb[0][0][{0.0, 0.0, 0.0}] = q0;

    librpa_int::PeriodicBoundaryData pbc;
    pbc.set_period(2, 1, 1);
    pbc.kfrac_list_full = {{0.0, 0.0, 0.0}};
    const AtomPairBvKRemap<atom_t> remap;

    assert_throws([&] {
        build_hartree_static_data(
            coefficients, coulomb, basis, basis, pbc, remap,
            HartreeKNormalization::weighted_occupations, 1.0e-12);
    });
}

} // namespace

int main()
{
    test_real_space_ri_coefficients_are_fourier_transformed();
    test_libri_coefficients_are_materialized_without_mutating_the_source();
    test_bare_coulomb_is_projected_to_a_complete_hermitian_operator();
    test_density_blocking_and_inverse_fourier_have_one_inverse_grid_factor();
    test_inverse_fourier_uses_atom_pair_nearest_bvk_cells();
    test_distributed_libri_coefficients_without_full_hartree_copy_are_rejected();
    test_libri_coefficients_with_full_hartree_copy_are_supported();
    test_end_to_end_delta_is_projected_to_an_independent_target_basis();
    test_workflow_propagates_explicit_legacy_k_normalization();
    test_static_builder_binds_complete_full_bz_input();
    test_static_builder_rejects_an_incomplete_full_bz_grid();
    std::cout << "test_qsgw_hartree_workflow: all tests passed\n";
    return 0;
}
