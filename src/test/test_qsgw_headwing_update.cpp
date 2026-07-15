#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/headwing_update.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

using librpa_int::ComplexMatrix;
using librpa_int::MeanField;
using librpa_int::Matz;
using librpa_int::Vector3_Order;
using librpa_int::cplxdb;
using librpa_int::qsgw::IndependentHeadwingUpdateResult;
using librpa_int::qsgw::SpinKMatrixMap;
using librpa_int::qsgw::VelocityMatrix;
using librpa_int::qsgw::update_independent_headwing_state;

namespace
{

constexpr double tolerance = 2.0e-11;

ComplexMatrix identity_wfc()
{
    ComplexMatrix result(2, 2);
    result(0, 0) = 1.0;
    result(1, 1) = 1.0;
    return result;
}

MeanField make_reference()
{
    MeanField result(1, 2, 2, 2, 1);
    result.get_eigenvectors()[0][0][0] = identity_wfc();
    result.get_eigenvectors()[0][0][1] = identity_wfc();
    result.get_eigenvals()[0](0, 0) = -0.8;
    result.get_eigenvals()[0](0, 1) = 0.7;
    result.get_eigenvals()[0](1, 0) = -0.6;
    result.get_eigenvals()[0](1, 1) = 0.9;
    result.get_weight()[0].zero_out();
    result.get_weight()[0](0, 0) = 1.0;
    result.get_weight()[0](1, 0) = 1.0;
    return result;
}

VelocityMatrix make_reference_velocity()
{
    VelocityMatrix result(1);
    result[0].resize(2);
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        result[0][kpoint].assign(3, ComplexMatrix(2, 2));
        for (int direction = 0; direction < 3; ++direction)
        {
            result[0][kpoint][direction](0, 0) =
                1.0 + 0.2 * kpoint + 0.1 * direction;
            result[0][kpoint][direction](1, 1) =
                2.0 + 0.3 * kpoint + 0.1 * direction;
        }
    }
    return result;
}

Matz hermitian(const double d0, const double d1,
               const cplxdb off_diagonal)
{
    Matz result(2, 2);
    result(0, 0) = d0;
    result(0, 1) = off_diagonal;
    result(1, 0) = std::conj(off_diagonal);
    result(1, 1) = d1;
    return result;
}

void assert_close(const double actual, const double expected,
                  const double threshold = tolerance)
{
    assert(std::abs(actual - expected) < threshold);
}

void assert_matrix_close(const Matz& actual, const Matz& expected)
{
    assert(actual.nr() == expected.nr());
    assert(actual.nc() == expected.nc());
    for (int row = 0; row < actual.nr(); ++row)
    {
        for (int column = 0; column < actual.nc(); ++column)
            assert(std::abs(actual(row, column) - expected(row, column)) <
                   tolerance);
    }
}

void assert_reference_unchanged(const MeanField& actual,
                                const MeanField& snapshot)
{
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        for (int band = 0; band < 2; ++band)
        {
            assert_close(actual.get_eigenvals()[0](kpoint, band),
                         snapshot.get_eigenvals()[0](kpoint, band));
            assert_close(actual.get_weight()[0](kpoint, band),
                         snapshot.get_weight()[0](kpoint, band));
        }
        const auto& actual_wfc =
            actual.get_eigenvectors().at(0).at(0).at(kpoint);
        const auto& snapshot_wfc =
            snapshot.get_eigenvectors().at(0).at(0).at(kpoint);
        for (int row = 0; row < 2; ++row)
        {
            for (int column = 0; column < 2; ++column)
                assert(std::abs(actual_wfc(row, column) -
                                snapshot_wfc(row, column)) < tolerance);
        }
    }
}

void test_live_independent_grid_updates_every_coupled_quantity()
{
    const MeanField source_reference = make_reference();
    const MeanField target_reference = make_reference();
    const MeanField target_snapshot = target_reference;
    MeanField target_live = target_reference;
    const VelocityMatrix reference_velocity = make_reference_velocity();
    VelocityMatrix live_velocity = reference_velocity;

    SpinKMatrixMap source_hamiltonian;
    source_hamiltonian[0][0] = hermitian(-1.0, 1.0, {0.2, 0.0});
    source_hamiltonian[0][1] = hermitian(-0.5, 0.8, {-0.1, 0.0});
    const std::vector<Vector3_Order<double>> kpoints = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> cells = {
        {0, 0, 0}, {1, 0, 0}};

    const IndependentHeadwingUpdateResult result =
        update_independent_headwing_state(
            source_hamiltonian, source_reference, kpoints, cells,
            target_live, target_reference, kpoints,
            reference_velocity, live_velocity, {0.5, 0.5}, 2.0);

    assert_matrix_close(result.projected_hamiltonian.at(0).at(0),
                        source_hamiltonian.at(0).at(0));
    assert_matrix_close(result.projected_hamiltonian.at(0).at(1),
                        source_hamiltonian.at(0).at(1));
    assert(result.maximum_source_roundtrip_relative_error < tolerance);
    assert(result.maximum_target_hermiticity_error < tolerance);
    assert(result.maximum_target_relative_hermiticity_error < tolerance);
    assert(result.maximum_repaired_target_hermiticity_error < tolerance);
    assert_close(target_live.get_eigenvals()[0](0, 0),
                 -std::sqrt(1.04));
    assert_close(target_live.get_eigenvals()[0](0, 1),
                 std::sqrt(1.04));
    assert(result.occupations.electron_count == 2.0);
    assert_close(target_live.get_weight()[0](0, 0), 1.0);
    assert_close(target_live.get_weight()[0](1, 0), 1.0);
    assert(std::abs(target_live.get_eigenvectors().at(0).at(0).at(0)(0, 1)) >
           1.0e-3);
    assert(std::abs(live_velocity.at(0).at(0).at(0)(0, 1)) > 1.0e-3);
    assert_reference_unchanged(target_reference, target_snapshot);
}

void test_failed_projection_is_transactional()
{
    const MeanField source_reference = make_reference();
    const MeanField target_reference = make_reference();
    MeanField target_live = target_reference;
    target_live.get_eigenvals()[0](0, 0) = -9.0;
    const MeanField live_snapshot = target_live;
    const VelocityMatrix reference_velocity = make_reference_velocity();
    VelocityMatrix live_velocity = reference_velocity;
    live_velocity[0][0][0](0, 0) = 9.0;
    const VelocityMatrix velocity_snapshot = live_velocity;

    SpinKMatrixMap invalid;
    invalid[0][0] = hermitian(-1.0, 1.0, {0.2, 0.0});
    invalid[0][1] = hermitian(-0.5, 0.8, {-0.1, 0.0});
    invalid[0][0](1, 0) = 0.4;
    const std::vector<Vector3_Order<double>> kpoints = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> cells = {
        {0, 0, 0}, {1, 0, 0}};

    bool threw = false;
    try
    {
        update_independent_headwing_state(
            invalid, source_reference, kpoints, cells,
            target_live, target_reference, kpoints,
            reference_velocity, live_velocity, {0.5, 0.5}, 2.0);
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    assert(threw);
    assert_close(target_live.get_eigenvals()[0](0, 0),
                 live_snapshot.get_eigenvals()[0](0, 0));
    assert(std::abs(live_velocity[0][0][0](0, 0) -
                    velocity_snapshot[0][0][0](0, 0)) < tolerance);
}

} // namespace

int main()
{
    test_live_independent_grid_updates_every_coupled_quantity();
    test_failed_projection_is_transactional();
    std::cout << "test_qsgw_headwing_update: all tests passed\n";
    return 0;
}
