#pragma once

#include "matrix_map.h"

#include "../core/meanfield.h"
#include "../core/symmetry_context.h"

#include <map>
#include <utility>
#include <vector>

namespace librpa_int
{
namespace qsgw
{

using DensityRMap = std::map<Vector3_Order<int>, ComplexMatrix>;
using WeightedDensityKMap = std::map<int, ComplexMatrix>;
using PeriodicOperatorRMap = std::map<
    int,
    std::map<std::pair<int, Vector3_Order<int>>, ComplexMatrix>>;

ComplexMatrix total_density_k(const MeanField& meanfield, int kpoint);

DensityRMap build_total_density_rspace(
    const MeanField& meanfield,
    const std::vector<Vector3_Order<double>>& kpoints,
    const std::vector<Vector3_Order<int>>& r_grid);

DensityRMap build_total_density_rspace_symmetry(
    const SymmetryContext& symmetry,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& meanfield,
    const std::vector<Vector3_Order<double>>& ibz_kpoints,
    const std::vector<Vector3_Order<int>>& r_grid,
    const std::map<atom_t, std::size_t>& atom_ao_sizes);

DensityRMap build_total_density_delta_rspace(
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints,
    const std::vector<Vector3_Order<int>>& r_grid);

DensityRMap build_total_density_delta_rspace_symmetry(
    const SymmetryContext& symmetry,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& ibz_kpoints,
    const std::vector<Vector3_Order<int>>& r_grid,
    const std::map<atom_t, std::size_t>& atom_ao_sizes);

WeightedDensityKMap reconstruct_weighted_full_grid_density(
    const DensityRMap& density_r,
    const std::vector<Vector3_Order<double>>& full_kpoints);

void validate_canonical_bvk_grid(
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<double>>& kpoints,
    double tolerance);

SpinKMatrixMap project_periodic_operator_to_fixed_basis(
    const PeriodicOperatorRMap& operator_r,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints);

} // namespace qsgw
} // namespace librpa_int
