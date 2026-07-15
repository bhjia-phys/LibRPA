#pragma once

#include "fixed_basis.h"
#include "occupation.h"
#include "operator_fourier.h"

namespace librpa_int
{
namespace qsgw
{

struct IndependentHeadwingUpdateResult
{
    SpinKMatrixMap projected_hamiltonian;
    SpinKMatrixMap unitary;
    OccupationResult occupations;
    double maximum_basis_inverse_residual = 0.0;
    double maximum_basis_condition_estimate = 0.0;
    double maximum_fourier_orthogonality_residual = 0.0;
    double maximum_source_roundtrip_relative_error = 0.0;
    double maximum_target_hermiticity_error = 0.0;
    double maximum_target_relative_hermiticity_error = 0.0;
    double maximum_repaired_target_hermiticity_error = 0.0;
};

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
    double electron_count,
    const OperatorFourierOptions& fourier_options = {},
    const OccupationSettings& occupation_settings = {});

} // namespace qsgw
} // namespace librpa_int
