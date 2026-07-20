#include "headwing_update.h"

#include <stdexcept>
#include <utility>

namespace librpa_int
{
namespace qsgw
{
namespace
{

void require_distinct_headwing_state(
    MeanField& target_live,
    const MeanField& target_reference,
    const VelocityMatrix& target_reference_velocity,
    VelocityMatrix& target_live_velocity)
{
    if (&target_live == &target_reference)
    {
        throw std::invalid_argument(
            "QSGW independent head-wing live and reference mean fields must be distinct");
    }
    if (&target_live_velocity == &target_reference_velocity)
    {
        throw std::invalid_argument(
            "QSGW independent head-wing live and reference velocities must be distinct");
    }
}

IndependentHeadwingUpdateResult apply_headwing_projection(
    OperatorFourierResult projection,
    MeanField& target_live,
    const MeanField& target_reference,
    const VelocityMatrix& target_reference_velocity,
    VelocityMatrix& target_live_velocity,
    const std::vector<double>& target_kpoint_weights,
    const double electron_count,
    const OccupationSettings& occupation_settings)
{
    MeanField next_live = target_live;
    VelocityMatrix next_velocity = target_live_velocity;
    FixedBasisDiagonalizationResult diagonalization =
        diagonalize_in_reference_basis(
            next_live, target_reference, projection.target,
            &target_reference_velocity, &next_velocity);
    OccupationResult occupations = update_qsgw_occupations(
        next_live, target_reference, target_kpoint_weights,
        electron_count, occupation_settings);

    target_live = std::move(next_live);
    target_live_velocity = std::move(next_velocity);

    IndependentHeadwingUpdateResult result;
    result.projected_hamiltonian = std::move(projection.target);
    result.unitary = std::move(diagonalization.unitary);
    result.occupations = occupations;
    result.maximum_basis_inverse_residual =
        projection.maximum_basis_inverse_residual;
    result.maximum_basis_condition_estimate =
        projection.maximum_basis_condition_estimate;
    result.maximum_fourier_orthogonality_residual =
        projection.maximum_fourier_orthogonality_residual;
    result.maximum_source_roundtrip_relative_error =
        projection.maximum_source_roundtrip_relative_error;
    result.maximum_target_hermiticity_error =
        projection.maximum_target_hermiticity_error;
    result.maximum_target_relative_hermiticity_error =
        projection.maximum_target_relative_hermiticity_error;
    result.maximum_repaired_target_hermiticity_error =
        projection.maximum_repaired_target_hermiticity_error;
    return result;
}

} // namespace

IndependentHeadwingUpdateResult update_independent_headwing_state(
    const SpinKMatrixMap& source_hamiltonian,
    const MeanField& source_reference,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    MeanField& target_live,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const VelocityMatrix& target_reference_velocity,
    VelocityMatrix& target_live_velocity,
    const std::vector<double>& target_kpoint_weights,
    const double electron_count,
    const OperatorFourierOptions& fourier_options,
    const OccupationSettings& occupation_settings)
{
    require_distinct_headwing_state(
        target_live, target_reference, target_reference_velocity,
        target_live_velocity);
    return apply_headwing_projection(
        interpolate_fixed_basis_operator(
            source_hamiltonian, source_reference, source_kpoints,
            real_space_cells, target_reference, target_kpoints,
            fourier_options),
        target_live, target_reference, target_reference_velocity,
        target_live_velocity, target_kpoint_weights, electron_count,
        occupation_settings);
}

IndependentHeadwingUpdateResult
update_symmetry_reduced_independent_headwing_state(
    const SpinKMatrixMap& source_hamiltonian,
    const MeanField& source_reference,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<double>>& full_source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    const SymmetryContext& symmetry_context,
    const AtomicBasis& source_basis,
    MeanField& target_live,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const VelocityMatrix& target_reference_velocity,
    VelocityMatrix& target_live_velocity,
    const std::vector<double>& target_kpoint_weights,
    const double electron_count,
    const OperatorFourierOptions& fourier_options,
    const OccupationSettings& occupation_settings)
{
    require_distinct_headwing_state(
        target_live, target_reference, target_reference_velocity,
        target_live_velocity);
    return apply_headwing_projection(
        interpolate_symmetry_reduced_fixed_basis_operator(
            source_hamiltonian, source_reference, source_kpoints,
            full_source_kpoints, real_space_cells, target_reference,
            target_kpoints, symmetry_context, source_basis,
            fourier_options),
        target_live, target_reference, target_reference_velocity,
        target_live_velocity, target_kpoint_weights, electron_count,
        occupation_settings);
}

} // namespace qsgw
} // namespace librpa_int
