#include "operator_fourier.h"

#include "../math/complexmatrix.h"
#include "../utils/constants.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace librpa_int
{
namespace qsgw
{
namespace
{

bool is_finite(const cplxdb value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

double frobenius_norm(const Matz& matrix)
{
    double sum = 0.0;
    for (int row = 0; row < matrix.nr(); ++row)
    {
        for (int column = 0; column < matrix.nc(); ++column)
        {
            sum += std::norm(matrix(row, column));
        }
    }
    return std::sqrt(sum);
}

double relative_frobenius_difference(const Matz& actual,
                                     const Matz& expected)
{
    if (actual.nr() != expected.nr() || actual.nc() != expected.nc())
    {
        throw std::invalid_argument(
            "QSGW operator Fourier comparison has inconsistent shapes");
    }
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
    return std::sqrt(difference / std::max(1.0, scale));
}

double maximum_hermiticity_error(const Matz& matrix)
{
    if (matrix.nr() != matrix.nc())
    {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (int row = 0; row < matrix.nr(); ++row)
    {
        for (int column = 0; column < matrix.nc(); ++column)
        {
            result = std::max(
                result,
                std::abs(matrix(row, column) -
                         std::conj(matrix(column, row))));
        }
    }
    return result;
}

double relative_frobenius_hermiticity_error(const Matz& matrix)
{
    if (matrix.nr() != matrix.nc())
    {
        return std::numeric_limits<double>::infinity();
    }
    double difference_squared = 0.0;
    for (int row = 0; row < matrix.nr(); ++row)
    {
        for (int column = 0; column < matrix.nc(); ++column)
        {
            difference_squared += std::norm(
                matrix(row, column) -
                std::conj(matrix(column, row)));
        }
    }
    const double scale = frobenius_norm(matrix);
    if (scale == 0.0)
    {
        return 0.0;
    }
    return std::sqrt(difference_squared) / scale;
}

void require_finite_matrix(const Matz& matrix, const char* label)
{
    for (int row = 0; row < matrix.nr(); ++row)
    {
        for (int column = 0; column < matrix.nc(); ++column)
        {
            if (!is_finite(matrix(row, column)))
            {
                throw std::invalid_argument(
                    std::string("QSGW ") + label +
                    " contains non-finite data");
            }
        }
    }
}

void require_options(const OperatorFourierOptions& options)
{
    const double values[] = {
        options.basis_inverse_tolerance,
        options.maximum_basis_condition_estimate,
        options.fourier_orthogonality_tolerance,
        options.source_roundtrip_tolerance,
        options.hermiticity_tolerance,
        options.relative_hermiticity_tolerance};
    for (const double value : values)
    {
        if (!(value > 0.0) || !std::isfinite(value))
        {
            throw std::invalid_argument(
                "QSGW operator Fourier tolerances must be finite and positive");
        }
    }
}

void require_complete_reference(
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints,
    const char* label)
{
    if (!reference.initialized() ||
        reference.get_n_kpoints() != static_cast<int>(kpoints.size()))
    {
        throw std::invalid_argument(
            std::string("QSGW ") + label +
            " reference and k-point list have inconsistent shapes");
    }
    const int dimension =
        reference.get_n_aos() * reference.get_n_spinor();
    if (reference.get_n_bands() != dimension)
    {
        throw std::invalid_argument(
            std::string("QSGW ") + label +
            " reference is not a complete square AO basis");
    }
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int spinor = 0;
             spinor < reference.get_n_spinor(); ++spinor)
        {
            for (int kpoint = 0;
                 kpoint < reference.get_n_kpoints(); ++kpoint)
            {
                const ComplexMatrix* block =
                    reference.find_wfc(spin, spinor, kpoint);
                if (block == nullptr ||
                    block->nr != reference.get_n_bands() ||
                    block->nc != reference.get_n_aos())
                {
                    throw std::invalid_argument(
                        std::string("QSGW ") + label +
                        " reference wavefunction map is incomplete");
                }
                for (int index = 0; index < block->size; ++index)
                {
                    if (!is_finite(block->c[index]))
                    {
                        throw std::invalid_argument(
                            std::string("QSGW ") + label +
                            " reference wavefunction contains non-finite data");
                    }
                }
            }
        }
    }
}

void require_compatible_references(const MeanField& source,
                                   const MeanField& target)
{
    if (source.get_n_spins() != target.get_n_spins() ||
        source.get_n_bands() != target.get_n_bands() ||
        source.get_n_aos() * source.get_n_spinor() !=
            target.get_n_aos() * target.get_n_spinor())
    {
        throw std::invalid_argument(
            "QSGW source and target operator bases are incompatible");
    }
}

Matz collect_wavefunction_rows(const MeanField& reference,
                               const int spin,
                               const int kpoint)
{
    const int dimension = reference.get_n_bands();
    Matz result(dimension, dimension);
    for (int band = 0; band < dimension; ++band)
    {
        for (int spinor = 0;
             spinor < reference.get_n_spinor(); ++spinor)
        {
            const ComplexMatrix& block =
                reference.get_eigenvectors().at(spin).at(spinor).at(kpoint);
            for (int ao = 0; ao < reference.get_n_aos(); ++ao)
            {
                result(band, ao * reference.get_n_spinor() + spinor) =
                    block(band, ao);
            }
        }
    }
    return result;
}

ComplexMatrix to_complex_matrix(const Matz& input)
{
    ComplexMatrix result(input.nr(), input.nc());
    for (int row = 0; row < input.nr(); ++row)
    {
        for (int column = 0; column < input.nc(); ++column)
        {
            result(row, column) = input(row, column);
        }
    }
    return result;
}

Matz to_matz(const ComplexMatrix& input)
{
    Matz result(input.nr, input.nc);
    for (int row = 0; row < input.nr; ++row)
    {
        for (int column = 0; column < input.nc; ++column)
        {
            result(row, column) = input(row, column);
        }
    }
    return result;
}

Matz checked_inverse(const Matz& matrix,
                     const OperatorFourierOptions& options,
                     OperatorFourierResult& diagnostics)
{
    Matz result = matrix.copy();
    const int dimension = matrix.nr();
    std::vector<int> pivots(static_cast<std::size_t>(dimension));
    std::vector<cplxdb> work(static_cast<std::size_t>(
        std::max(1, dimension)));
    int info = 0;
    if (result.is_row_major())
    {
        LapackConnector::getrf(
            dimension, dimension, result.ptr(), dimension,
            pivots.data(), info);
    }
    else
    {
        LapackConnector::getrf_f(
            dimension, dimension, result.ptr(), dimension,
            pivots.data(), info);
    }
    if (info != 0)
    {
        throw std::invalid_argument(
            "QSGW fixed reference wavefunction matrix is singular");
    }
    if (result.is_row_major())
    {
        LapackConnector::getri(
            dimension, result.ptr(), dimension, pivots.data(),
            work.data(), static_cast<int>(work.size()), info);
    }
    else
    {
        LapackConnector::getri_f(
            dimension, result.ptr(), dimension, pivots.data(),
            work.data(), static_cast<int>(work.size()), info);
    }
    if (info != 0)
    {
        throw std::invalid_argument(
            "QSGW fixed reference wavefunction matrix inversion failed");
    }
    require_finite_matrix(result, "basis inverse");

    const Matz product = matrix * result;
    double residual_squared = 0.0;
    for (int row = 0; row < product.nr(); ++row)
    {
        for (int column = 0; column < product.nc(); ++column)
        {
            const cplxdb expected = row == column ? 1.0 : 0.0;
            residual_squared += std::norm(product(row, column) - expected);
        }
    }
    const double residual =
        std::sqrt(residual_squared / std::max(1, product.nr()));
    const double condition_estimate =
        frobenius_norm(matrix) * frobenius_norm(result);
    diagnostics.maximum_basis_inverse_residual = std::max(
        diagnostics.maximum_basis_inverse_residual, residual);
    diagnostics.maximum_basis_condition_estimate = std::max(
        diagnostics.maximum_basis_condition_estimate,
        condition_estimate);
    if (!std::isfinite(residual) ||
        residual > options.basis_inverse_tolerance ||
        !std::isfinite(condition_estimate) ||
        condition_estimate > options.maximum_basis_condition_estimate)
    {
        throw std::invalid_argument(
            "QSGW fixed reference wavefunction matrix is singular or ill-conditioned");
    }
    return result;
}

cplxdb phase(const Vector3_Order<double>& kpoint,
             const Vector3_Order<int>& cell,
             const double sign)
{
    const double angle = sign * TWO_PI *
        (kpoint.x * static_cast<double>(cell.x) +
         kpoint.y * static_cast<double>(cell.y) +
         kpoint.z * static_cast<double>(cell.z));
    return {std::cos(angle), std::sin(angle)};
}

void add_scaled(Matz& destination,
                const Matz& source,
                const cplxdb factor)
{
    if (destination.nr() != source.nr() ||
        destination.nc() != source.nc())
    {
        throw std::invalid_argument(
            "QSGW operator Fourier accumulation has inconsistent shapes");
    }
    for (int row = 0; row < destination.nr(); ++row)
    {
        for (int column = 0; column < destination.nc(); ++column)
        {
            destination(row, column) += factor * source(row, column);
        }
    }
}

double validate_fourier_grid(
    const std::vector<Vector3_Order<double>>& kpoints,
    const std::vector<Vector3_Order<int>>& cells,
    const double tolerance)
{
    if (kpoints.empty() || kpoints.size() != cells.size())
    {
        throw std::invalid_argument(
            "QSGW operator Fourier requires equal nonzero k-point and BvK-cell counts");
    }
    double maximum_residual = 0.0;
    const double inverse_count = 1.0 / static_cast<double>(kpoints.size());
    for (std::size_t left = 0; left < kpoints.size(); ++left)
    {
        for (std::size_t right = 0; right < kpoints.size(); ++right)
        {
            cplxdb overlap = 0.0;
            for (const auto& cell : cells)
            {
                const Vector3_Order<double> difference{
                    kpoints[left].x - kpoints[right].x,
                    kpoints[left].y - kpoints[right].y,
                    kpoints[left].z - kpoints[right].z};
                overlap += phase(difference, cell, 1.0) * inverse_count;
            }
            const cplxdb expected = left == right ? 1.0 : 0.0;
            maximum_residual = std::max(
                maximum_residual, std::abs(overlap - expected));
        }
    }
    if (!std::isfinite(maximum_residual) ||
        maximum_residual > tolerance)
    {
        throw std::invalid_argument(
            "QSGW source k points and BvK cells do not form a complete discrete Fourier grid");
    }
    return maximum_residual;
}

void require_source_operator(const SpinKMatrixMap& source_operator,
                             const MeanField& reference,
                             const double hermiticity_tolerance)
{
    if (static_cast<int>(source_operator.size()) !=
        reference.get_n_spins())
    {
        throw std::invalid_argument(
            "QSGW source operator spin map is incomplete");
    }
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        const auto spin_it = source_operator.find(spin);
        if (spin_it == source_operator.end() ||
            static_cast<int>(spin_it->second.size()) !=
                reference.get_n_kpoints())
        {
            throw std::invalid_argument(
                "QSGW source operator k-point map is incomplete");
        }
        for (int kpoint = 0;
             kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            const auto operator_it = spin_it->second.find(kpoint);
            if (operator_it == spin_it->second.end() ||
                operator_it->second.nr() != reference.get_n_bands() ||
                operator_it->second.nc() != reference.get_n_bands())
            {
                throw std::invalid_argument(
                    "QSGW source operator has an invalid shape");
            }
            require_finite_matrix(operator_it->second, "source operator");
            if (maximum_hermiticity_error(operator_it->second) >
                hermiticity_tolerance)
            {
                throw std::invalid_argument(
                    "QSGW source operator is not Hermitian");
            }
        }
    }
}

void project_to_hermitian(Matz& matrix)
{
    if (matrix.nr() != matrix.nc())
    {
        throw std::invalid_argument(
            "QSGW Hermitian projection requires a square matrix");
    }
    for (int row = 0; row < matrix.nr(); ++row)
    {
        matrix(row, row) = matrix(row, row).real();
        for (int column = row + 1; column < matrix.nc(); ++column)
        {
            const cplxdb average = 0.5 *
                (matrix(row, column) +
                 std::conj(matrix(column, row)));
            matrix(row, column) = average;
            matrix(column, row) = std::conj(average);
        }
    }
}

SpinKMatrixMap lift_source_operator_to_ao(
    const SpinKMatrixMap& source_operator,
    const MeanField& source_reference,
    const OperatorFourierOptions& options,
    OperatorFourierResult& diagnostics)
{
    SpinKMatrixMap source_ao;
    for (int spin = 0; spin < source_reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0;
             kpoint < source_reference.get_n_kpoints(); ++kpoint)
        {
            const Matz wavefunctions = collect_wavefunction_rows(
                source_reference, spin, kpoint);
            const Matz inverse_wavefunctions = checked_inverse(
                wavefunctions, options, diagnostics);
            const Matz ao_operator =
                conj(inverse_wavefunctions) *
                source_operator.at(spin).at(kpoint) *
                transpose(inverse_wavefunctions);
            require_finite_matrix(ao_operator, "lifted AO operator");

            const Matz roundtrip =
                conj(wavefunctions) * ao_operator *
                transpose(wavefunctions);
            const double roundtrip_error = relative_frobenius_difference(
                roundtrip, source_operator.at(spin).at(kpoint));
            diagnostics.maximum_source_roundtrip_relative_error = std::max(
                diagnostics.maximum_source_roundtrip_relative_error,
                roundtrip_error);
            if (!std::isfinite(roundtrip_error) ||
                roundtrip_error > options.source_roundtrip_tolerance)
            {
                throw std::invalid_argument(
                    "QSGW state-to-AO operator lift failed its source-grid round trip");
            }
            source_ao[spin][kpoint] = ao_operator;
        }
    }
    return source_ao;
}

void require_complete_ao_grid(const SpinKMatrixMap& source_ao,
                              const int n_spins,
                              const int n_kpoints,
                              const int dimension,
                              const double hermiticity_tolerance)
{
    if (static_cast<int>(source_ao.size()) != n_spins)
    {
        throw std::invalid_argument(
            "QSGW lifted AO operator spin map is incomplete");
    }
    for (int spin = 0; spin < n_spins; ++spin)
    {
        const auto spin_it = source_ao.find(spin);
        if (spin_it == source_ao.end() ||
            static_cast<int>(spin_it->second.size()) != n_kpoints)
        {
            throw std::invalid_argument(
                "QSGW lifted AO operator k-point map is incomplete");
        }
        for (int kpoint = 0; kpoint < n_kpoints; ++kpoint)
        {
            const auto matrix_it = spin_it->second.find(kpoint);
            if (matrix_it == spin_it->second.end() ||
                matrix_it->second.nr() != dimension ||
                matrix_it->second.nc() != dimension)
            {
                throw std::invalid_argument(
                    "QSGW lifted AO operator has an invalid shape");
            }
            require_finite_matrix(matrix_it->second, "lifted AO operator");
            if (maximum_hermiticity_error(matrix_it->second) >
                hermiticity_tolerance)
            {
                throw std::invalid_argument(
                    "QSGW lifted AO operator is not Hermitian");
            }
        }
    }
}

OperatorFourierResult interpolate_ao_operator(
    const SpinKMatrixMap& source_ao,
    const int n_spins,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const OperatorFourierOptions& options,
    OperatorFourierResult result)
{
    const int dimension = target_reference.get_n_bands();
    require_complete_ao_grid(
        source_ao, n_spins, static_cast<int>(source_kpoints.size()),
        dimension, options.hermiticity_tolerance);
    result.maximum_fourier_orthogonality_residual =
        validate_fourier_grid(
            source_kpoints, real_space_cells,
            options.fourier_orthogonality_tolerance);

    const double inverse_kpoint_count =
        1.0 / static_cast<double>(source_kpoints.size());
    std::map<int, std::vector<Matz>> real_space;
    for (int spin = 0; spin < n_spins; ++spin)
    {
        auto& spin_real_space = real_space[spin];
        spin_real_space.reserve(real_space_cells.size());
        for (const auto& cell : real_space_cells)
        {
            Matz cell_operator(dimension, dimension);
            for (int kpoint = 0;
                 kpoint < static_cast<int>(source_kpoints.size());
                 ++kpoint)
            {
                add_scaled(
                    cell_operator, source_ao.at(spin).at(kpoint),
                    phase(source_kpoints[static_cast<std::size_t>(kpoint)],
                          cell, -1.0) * inverse_kpoint_count);
            }
            require_finite_matrix(cell_operator, "real-space operator");
            spin_real_space.push_back(std::move(cell_operator));
        }

        for (int target_kpoint = 0;
             target_kpoint < target_reference.get_n_kpoints();
             ++target_kpoint)
        {
            Matz target_ao(dimension, dimension);
            for (std::size_t cell = 0;
                 cell < real_space_cells.size(); ++cell)
            {
                add_scaled(
                    target_ao, spin_real_space[cell],
                    phase(target_kpoints[static_cast<std::size_t>(target_kpoint)],
                          real_space_cells[cell], 1.0));
            }
            const Matz target_wavefunctions = collect_wavefunction_rows(
                target_reference, spin, target_kpoint);
            Matz target_operator =
                conj(target_wavefunctions) * target_ao *
                transpose(target_wavefunctions);
            require_finite_matrix(target_operator, "target operator");
            const double hermiticity_error =
                maximum_hermiticity_error(target_operator);
            const double relative_hermiticity_error =
                relative_frobenius_hermiticity_error(target_operator);
            result.maximum_target_hermiticity_error = std::max(
                result.maximum_target_hermiticity_error,
                hermiticity_error);
            result.maximum_target_relative_hermiticity_error = std::max(
                result.maximum_target_relative_hermiticity_error,
                relative_hermiticity_error);
            if (!std::isfinite(hermiticity_error) ||
                !std::isfinite(relative_hermiticity_error) ||
                (hermiticity_error > options.hermiticity_tolerance &&
                 relative_hermiticity_error >
                     options.relative_hermiticity_tolerance))
            {
                std::ostringstream message;
                message << std::scientific << std::setprecision(17)
                        << "QSGW Fourier-interpolated target operator is not Hermitian"
                        << " (max_abs=" << hermiticity_error
                        << ", relative_frobenius="
                        << relative_hermiticity_error
                        << ", absolute_tolerance="
                        << options.hermiticity_tolerance
                        << ", relative_tolerance="
                        << options.relative_hermiticity_tolerance
                        << ")";
                throw std::invalid_argument(message.str());
            }
            project_to_hermitian(target_operator);
            const double repaired_hermiticity_error =
                maximum_hermiticity_error(target_operator);
            result.maximum_repaired_target_hermiticity_error = std::max(
                result.maximum_repaired_target_hermiticity_error,
                repaired_hermiticity_error);
            if (!std::isfinite(repaired_hermiticity_error) ||
                repaired_hermiticity_error > options.hermiticity_tolerance)
            {
                throw std::invalid_argument(
                    "QSGW target operator remains non-Hermitian after numerical projection");
            }
            result.target[spin][target_kpoint] =
                std::move(target_operator);
        }
    }
    return result;
}

} // namespace

OperatorFourierResult interpolate_fixed_basis_operator(
    const SpinKMatrixMap& source_operator,
    const MeanField& source_reference,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const OperatorFourierOptions& options)
{
    require_options(options);
    require_complete_reference(
        source_reference, source_kpoints, "source");
    require_complete_reference(
        target_reference, target_kpoints, "target");
    require_compatible_references(source_reference, target_reference);
    require_source_operator(
        source_operator, source_reference, options.hermiticity_tolerance);

    OperatorFourierResult result;
    const SpinKMatrixMap source_ao = lift_source_operator_to_ao(
        source_operator, source_reference, options, result);
    return interpolate_ao_operator(
        source_ao, source_reference.get_n_spins(), source_kpoints,
        real_space_cells, target_reference, target_kpoints,
        options, std::move(result));
}

OperatorFourierResult interpolate_symmetry_reduced_fixed_basis_operator(
    const SpinKMatrixMap& source_operator,
    const MeanField& source_reference,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<double>>& full_source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const SymmetryContext& symmetry_context,
    const AtomicBasis& source_basis,
    const OperatorFourierOptions& options)
{
    require_options(options);
    require_complete_reference(
        source_reference, source_kpoints, "source");
    require_complete_reference(
        target_reference, target_kpoints, "target");
    require_compatible_references(source_reference, target_reference);
    require_source_operator(
        source_operator, source_reference, options.hermiticity_tolerance);

    if (source_reference.get_n_spinor() != 1 ||
        !symmetry_context.available || symmetry_context.kstars.empty() ||
        full_source_kpoints.size() <= source_kpoints.size() ||
        symmetry_context.kstars.size() != source_kpoints.size() ||
        symmetry_context.count_kstar_members() != full_source_kpoints.size())
    {
        throw std::invalid_argument(
            "QSGW symmetry-reduced operator Fourier has an inconsistent k-star contract");
    }
    if (!source_basis.initialized() || !source_basis.has_l_shells() ||
        source_basis.nb_total !=
            static_cast<std::size_t>(source_reference.get_n_aos()))
    {
        throw std::invalid_argument(
            "QSGW symmetry-reduced operator Fourier requires the complete AO shell layout");
    }

    const auto atom_nw = source_basis.get_atom_nb_map();
    const auto layouts = source_basis.build_species_basis_layouts(
        symmetry_context.atom_to_type);
    if (layouts.empty() ||
        !symmetry_species_layouts_match_atom_counts(
            layouts, symmetry_context.atom_to_type, atom_nw))
    {
        throw std::invalid_argument(
            "QSGW symmetry-reduced operator Fourier AO layout does not match the symmetry context");
    }
    const auto member_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(
            symmetry_context, full_source_kpoints);
    if (member_targets.size() != symmetry_context.kstars.size())
    {
        throw std::invalid_argument(
            "QSGW symmetry-reduced operator Fourier cannot map k-star members to the full grid");
    }

    OperatorFourierResult result;
    const SpinKMatrixMap reduced_ao = lift_source_operator_to_ao(
        source_operator, source_reference, options, result);
    SpinKMatrixMap full_ao;
    for (int spin = 0; spin < source_reference.get_n_spins(); ++spin)
    {
        std::vector<bool> used_full_kpoints(full_source_kpoints.size(), false);
        std::vector<bool> used_stars(symmetry_context.kstars.size(), false);
        for (int source_kpoint = 0;
             source_kpoint < source_reference.get_n_kpoints();
             ++source_kpoint)
        {
            const auto& k_ibz =
                source_kpoints[static_cast<std::size_t>(source_kpoint)];
            const auto& star = find_symmetry_kstar_for_ibz_kpoint(
                symmetry_context, k_ibz);
            const std::size_t star_index = static_cast<std::size_t>(
                &star - symmetry_context.kstars.data());
            if (star_index >= symmetry_context.kstars.size() ||
                used_stars[star_index] ||
                member_targets[star_index].size() != star.members.size())
            {
                throw std::invalid_argument(
                    "QSGW symmetry-reduced operator Fourier has an ambiguous k-star mapping");
            }
            used_stars[star_index] = true;

            const ComplexMatrix ao_ibz = to_complex_matrix(
                reduced_ao.at(spin).at(source_kpoint));
            for (std::size_t member_index = 0;
                 member_index < star.members.size(); ++member_index)
            {
                const auto& member = star.members[member_index];
                const auto& target_k = member_targets[star_index][member_index];
                const FoldedKPoint folded = fold_fractional_kpoint_to_targets(
                    target_k, full_source_kpoints);
                if (folded.target_k_index < 0 ||
                    folded.target_k_index >=
                        static_cast<int>(full_source_kpoints.size()) ||
                    used_full_kpoints[static_cast<std::size_t>(
                        folded.target_k_index)])
                {
                    throw std::invalid_argument(
                        "QSGW symmetry-reduced operator Fourier produced a duplicate full-grid k point");
                }
                used_full_kpoints[static_cast<std::size_t>(
                    folded.target_k_index)] = true;
                const ComplexMatrix rotated = rotate_symmetry_kspace_matrix(
                    symmetry_context, layouts, member, ao_ibz, atom_nw,
                    k_ibz, member.time_reversal, &target_k);
                full_ao[spin][folded.target_k_index] = to_matz(rotated);
            }
        }
        if (std::find(used_stars.begin(), used_stars.end(), false) !=
                used_stars.end() ||
            std::find(used_full_kpoints.begin(), used_full_kpoints.end(),
                      false) != used_full_kpoints.end())
        {
            throw std::invalid_argument(
                "QSGW symmetry-reduced operator Fourier did not cover the complete full grid");
        }
    }

    return interpolate_ao_operator(
        full_ao, source_reference.get_n_spins(), full_source_kpoints,
        real_space_cells, target_reference, target_kpoints,
        options, std::move(result));
}

} // namespace qsgw
} // namespace librpa_int
