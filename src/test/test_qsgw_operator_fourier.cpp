#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../qsgw/operator_fourier.h"

#include "../core/atomic_basis.h"
#include "../core/symmetry_context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using librpa_int::ComplexMatrix;
using librpa_int::MeanField;
using librpa_int::Matz;
using librpa_int::AtomicBasis;
using librpa_int::SymmetryContext;
using librpa_int::SymmetryKAtomRotation;
using librpa_int::SymmetryKStar;
using librpa_int::SymmetryOperation;
using librpa_int::Vector3_Order;
using librpa_int::conj;
using librpa_int::cplxdb;
using librpa_int::transpose;
using librpa_int::qsgw::OperatorFourierResult;
using librpa_int::qsgw::OperatorFourierOptions;
using librpa_int::qsgw::SpinKMatrixMap;
using librpa_int::qsgw::interpolate_fixed_basis_operator;
using librpa_int::qsgw::interpolate_symmetry_reduced_fixed_basis_operator;

namespace
{

constexpr double tolerance = 2.0e-11;

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

template <typename Function>
std::string exception_message(Function&& function)
{
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    assert(false);
    return {};
}

void assert_matrix_close(const Matz& actual,
                         const Matz& expected,
                         const double threshold = tolerance)
{
    assert(actual.nr() == expected.nr());
    assert(actual.nc() == expected.nc());
    double difference = 0.0;
    double scale = 0.0;
    for (int row = 0; row < actual.nr(); ++row)
    {
        for (int column = 0; column < actual.nc(); ++column)
        {
            difference += std::norm(actual(row, column) -
                                    expected(row, column));
            scale += std::norm(expected(row, column));
        }
    }
    assert(std::sqrt(difference / std::max(1.0, scale)) < threshold);
}

ComplexMatrix to_complex_matrix(const Matz& input)
{
    ComplexMatrix output(input.nr(), input.nc());
    for (int row = 0; row < input.nr(); ++row)
    {
        for (int column = 0; column < input.nc(); ++column)
        {
            output(row, column) = input(row, column);
        }
    }
    return output;
}

MeanField make_reference(const std::vector<Matz>& wavefunctions)
{
    MeanField result(1, static_cast<int>(wavefunctions.size()), 2, 2, 1);
    for (std::size_t kpoint = 0; kpoint < wavefunctions.size(); ++kpoint)
    {
        result.get_eigenvectors()[0][0][static_cast<int>(kpoint)] =
            to_complex_matrix(wavefunctions[kpoint]);
    }
    return result;
}

Matz make_hermitian(const double d0,
                    const double d1,
                    const cplxdb off_diagonal)
{
    Matz result(2, 2);
    result(0, 0) = d0;
    result(0, 1) = off_diagonal;
    result(1, 0) = std::conj(off_diagonal);
    result(1, 1) = d1;
    return result;
}

Matz project_to_state_basis(const Matz& ao_operator,
                            const Matz& wavefunctions)
{
    return conj(wavefunctions) * ao_operator * transpose(wavefunctions);
}

SymmetryContext make_time_reversal_kstar_context()
{
    SymmetryContext context;
    context.set_available();
    context.basis_convention = {
        -1, 0, LIBRPA_ANGULAR_ORDER_NATURAL,
        LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M};
    context.atom_to_type[0] = 0;
    context.input_coord_frac[0] = {0.0, 0.0, 0.0};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    context.rspace_operations.push_back(identity_operation);

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.25, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.25, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].time_reversal = true;
    star.members[1].k_bz = {-0.25, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    context.kstars.push_back(star);
    return context;
}

void test_symmetry_reduced_operator_matches_explicit_full_grid()
{
    Matz identity(2, 2);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const MeanField reduced_reference = make_reference({identity});
    const MeanField full_reference = make_reference({identity, identity});

    const Matz ao_ibz = make_hermitian(1.2, 2.4, {0.35, -0.2});
    SpinKMatrixMap reduced_operator;
    reduced_operator[0][0] = ao_ibz;
    SpinKMatrixMap explicit_full_operator;
    explicit_full_operator[0][0] = ao_ibz;
    explicit_full_operator[0][1] = conj(ao_ibz);

    const std::vector<Vector3_Order<double>> reduced_kpoints = {
        {0.25, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> full_kpoints = {
        {0.25, 0.0, 0.0}, {0.75, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> cells = {
        {0, 0, 0}, {1, 0, 0}};

    AtomicBasis basis(std::vector<std::size_t>{2});
    basis.set_l_shells({{0, 0}});
    const SymmetryContext context = make_time_reversal_kstar_context();

    const OperatorFourierResult expected = interpolate_fixed_basis_operator(
        explicit_full_operator, full_reference, full_kpoints, cells,
        full_reference, full_kpoints);
    const OperatorFourierResult actual =
        interpolate_symmetry_reduced_fixed_basis_operator(
            reduced_operator, reduced_reference, reduced_kpoints,
            full_kpoints, cells, full_reference, full_kpoints,
            context, basis);

    assert_matrix_close(actual.target.at(0).at(0),
                        expected.target.at(0).at(0));
    assert_matrix_close(actual.target.at(0).at(1),
                        expected.target.at(0).at(1));
    assert_matrix_close(actual.real_space_ao.at(0).at({0, 0, 0}),
                        expected.real_space_ao.at(0).at({0, 0, 0}));
    assert_matrix_close(actual.real_space_ao.at(0).at({1, 0, 0}),
                        expected.real_space_ao.at(0).at({1, 0, 0}));
    assert(actual.maximum_fourier_orthogonality_residual < tolerance);
    assert(actual.maximum_source_roundtrip_relative_error < tolerance);
}

void test_complete_grid_round_trip_and_target_gauge_projection()
{
    Matz source_c0(2, 2);
    source_c0(0, 0) = 1.0;
    source_c0(0, 1) = cplxdb(0.1, 0.2);
    source_c0(1, 0) = cplxdb(-0.05, 0.1);
    source_c0(1, 1) = 1.1;

    Matz source_c1(2, 2);
    source_c1(0, 0) = cplxdb(0.9, 0.1);
    source_c1(0, 1) = cplxdb(-0.2, 0.05);
    source_c1(1, 0) = cplxdb(0.1, -0.15);
    source_c1(1, 1) = cplxdb(1.0, -0.1);

    const Matz ao_k0 = make_hermitian(1.0, 2.0, {0.3, 0.2});
    const Matz ao_k1 = make_hermitian(1.4, 2.6, {-0.1, 0.35});
    const MeanField source = make_reference({source_c0, source_c1});

    SpinKMatrixMap source_operator;
    source_operator[0][0] = project_to_state_basis(ao_k0, source_c0);
    source_operator[0][1] = project_to_state_basis(ao_k1, source_c1);

    Matz target_c_at_k1(2, 2);
    target_c_at_k1(0, 0) = cplxdb(0.8, -0.2);
    target_c_at_k1(0, 1) = cplxdb(0.15, 0.1);
    target_c_at_k1(1, 0) = cplxdb(-0.1, 0.05);
    target_c_at_k1(1, 1) = cplxdb(1.2, 0.15);

    Matz target_c_at_k0(2, 2);
    target_c_at_k0(0, 0) = cplxdb(1.1, 0.0);
    target_c_at_k0(0, 1) = cplxdb(0.0, -0.1);
    target_c_at_k0(1, 0) = cplxdb(0.2, 0.05);
    target_c_at_k0(1, 1) = cplxdb(0.95, 0.0);
    const MeanField target =
        make_reference({target_c_at_k1, target_c_at_k0});

    const std::vector<Vector3_Order<double>> source_kpoints = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> cells = {
        {0, 0, 0}, {1, 0, 0}};
    const std::vector<Vector3_Order<double>> target_kpoints = {
        {0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    const OperatorFourierResult result = interpolate_fixed_basis_operator(
        source_operator, source, source_kpoints, cells,
        target, target_kpoints);

    assert_matrix_close(
        result.target.at(0).at(0),
        project_to_state_basis(ao_k1, target_c_at_k1));
    assert_matrix_close(
        result.target.at(0).at(1),
        project_to_state_basis(ao_k0, target_c_at_k0));
    Matz expected_r0(2, 2);
    Matz expected_r1(2, 2);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            expected_r0(row, column) =
                0.5 * (ao_k0(row, column) + ao_k1(row, column));
            expected_r1(row, column) =
                0.5 * (ao_k0(row, column) - ao_k1(row, column));
        }
    }
    assert_matrix_close(
        result.real_space_ao.at(0).at({0, 0, 0}), expected_r0);
    assert_matrix_close(
        result.real_space_ao.at(0).at({1, 0, 0}), expected_r1);
    assert(result.maximum_basis_inverse_residual < tolerance);
    assert(result.maximum_source_roundtrip_relative_error < tolerance);
    assert(result.maximum_target_hermiticity_error < tolerance);

    SpinKMatrixMap exact_hermitian_source = source_operator;
    for (auto& [spin, by_kpoint] : exact_hermitian_source)
    {
        (void)spin;
        for (auto& [kpoint, matrix] : by_kpoint)
        {
            (void)kpoint;
            for (int row = 0; row < matrix.nr(); ++row)
            {
                matrix(row, row) = matrix(row, row).real();
                for (int column = row + 1; column < matrix.nc(); ++column)
                {
                    matrix(column, row) = std::conj(matrix(row, column));
                }
            }
        }
    }
    OperatorFourierOptions repair_options;
    repair_options.hermiticity_tolerance = 1.0e-20;
    repair_options.relative_hermiticity_tolerance = 1.0e-10;
    const OperatorFourierResult repaired = interpolate_fixed_basis_operator(
        exact_hermitian_source, source, source_kpoints, cells,
        target, target_kpoints, repair_options);
    assert(repaired.maximum_target_hermiticity_error >
           repair_options.hermiticity_tolerance);
    assert(repaired.maximum_target_relative_hermiticity_error <
           repair_options.relative_hermiticity_tolerance);
    assert(repaired.maximum_repaired_target_hermiticity_error <=
           repair_options.hermiticity_tolerance);

    OperatorFourierOptions strict_options = repair_options;
    strict_options.relative_hermiticity_tolerance = 1.0e-20;
    const std::string message = exception_message([&] {
        interpolate_fixed_basis_operator(
            exact_hermitian_source, source, source_kpoints, cells,
            target, target_kpoints, strict_options);
    });
    assert(message.find("max_abs=") != std::string::npos);
    assert(message.find("relative_frobenius=") != std::string::npos);
}

void test_invalid_basis_and_fourier_contracts_are_rejected()
{
    Matz identity(2, 2);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const MeanField valid = make_reference({identity, identity});
    SpinKMatrixMap hamiltonian;
    hamiltonian[0][0] = make_hermitian(1.0, 2.0, {0.1, 0.0});
    hamiltonian[0][1] = make_hermitian(1.5, 2.5, {0.2, 0.0});
    const std::vector<Vector3_Order<double>> kpoints = {
        {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> cells = {
        {0, 0, 0}, {1, 0, 0}};

    MeanField incomplete(1, 2, 1, 2, 1);
    incomplete.get_eigenvectors()[0][0][0] = ComplexMatrix(1, 2);
    incomplete.get_eigenvectors()[0][0][1] = ComplexMatrix(1, 2);
    assert_throws([&] {
        interpolate_fixed_basis_operator(
            hamiltonian, incomplete, kpoints, cells, valid, kpoints);
    });

    MeanField singular = valid;
    for (int kpoint = 0; kpoint < 2; ++kpoint)
    {
        auto& block = singular.get_eigenvectors()[0][0][kpoint];
        block(1, 0) = block(0, 0);
        block(1, 1) = block(0, 1);
    }
    assert_throws([&] {
        interpolate_fixed_basis_operator(
            hamiltonian, singular, kpoints, cells, valid, kpoints);
    });

    assert_throws([&] {
        interpolate_fixed_basis_operator(
            hamiltonian, valid, kpoints,
            std::vector<Vector3_Order<int>>{{0, 0, 0}},
            valid, kpoints);
    });

    const std::vector<Vector3_Order<double>> duplicate_kpoints = {
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    assert_throws([&] {
        interpolate_fixed_basis_operator(
            hamiltonian, valid, duplicate_kpoints, cells,
            valid, kpoints);
    });

    SpinKMatrixMap nonhermitian = hamiltonian;
    nonhermitian[0][0](1, 0) = cplxdb(0.4, 0.0);
    assert_throws([&] {
        interpolate_fixed_basis_operator(
            nonhermitian, valid, kpoints, cells, valid, kpoints);
    });

    MeanField nonfinite = valid;
    nonfinite.get_eigenvectors()[0][0][0](0, 0) =
        cplxdb(std::numeric_limits<double>::quiet_NaN(), 0.0);
    assert_throws([&] {
        interpolate_fixed_basis_operator(
            hamiltonian, nonfinite, kpoints, cells, valid, kpoints);
    });
}

} // namespace

int main()
{
    test_complete_grid_round_trip_and_target_gauge_projection();
    test_symmetry_reduced_operator_matches_explicit_full_grid();
    test_invalid_basis_and_fourier_contracts_are_rejected();
    std::cout << "test_qsgw_operator_fourier: all tests passed\n";
    return 0;
}
