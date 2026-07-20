#pragma once

#include "matrix_map.h"

#include "../core/atomic_basis.h"
#include "../core/meanfield.h"
#include "../core/symmetry_context.h"
#include "../math/vector3_order.h"

#include <vector>

namespace librpa_int
{
namespace qsgw
{

struct OperatorFourierOptions
{
    double basis_inverse_tolerance = 1.0e-10;
    double maximum_basis_condition_estimate = 1.0e12;
    double fourier_orthogonality_tolerance = 1.0e-10;
    double source_roundtrip_tolerance = 1.0e-10;
    double hermiticity_tolerance = 1.0e-10;
    double relative_hermiticity_tolerance = 1.0e-10;
};

struct OperatorFourierResult
{
    SpinKMatrixMap target;
    double maximum_basis_inverse_residual = 0.0;
    double maximum_basis_condition_estimate = 0.0;
    double maximum_fourier_orthogonality_residual = 0.0;
    double maximum_source_roundtrip_relative_error = 0.0;
    double maximum_target_hermiticity_error = 0.0;
    double maximum_target_relative_hermiticity_error = 0.0;
    double maximum_repaired_target_hermiticity_error = 0.0;
};

// Lift an operator from the immutable source state basis to the AO basis,
// Fourier transform it through a complete BvK cell set, and project it into
// an immutable target state basis. Both reference bases must be complete.
OperatorFourierResult interpolate_fixed_basis_operator(
    const SpinKMatrixMap& source_operator,
    const MeanField& source_reference,
    const std::vector<Vector3_Order<double>>& source_kpoints,
    const std::vector<Vector3_Order<int>>& real_space_cells,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const OperatorFourierOptions& options = {});

// Expand a symmetry-reduced source operator in the AO basis before applying
// the same complete-grid Fourier interpolation as the full-BZ entry point.
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
    const OperatorFourierOptions& options = {});

} // namespace qsgw
} // namespace librpa_int
