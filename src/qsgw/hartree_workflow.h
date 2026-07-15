#pragma once

#include "hartree_density.h"
#include "hartree_kernel.h"

#include "../core/atomic_basis.h"
#include "../core/pbc.h"
#include "../core/ri.h"

#include <map>
#include <vector>

namespace librpa_int
{
namespace qsgw
{

struct HartreeStaticData
{
    HartreeCkMap c_k;
    HartreeVqMap v_q0;
    HartreeKNormalization normalization =
        HartreeKNormalization::weighted_occupations;
    std::map<int, int> atom_ao_sizes;
    Vector3_Order<int> period{1, 1, 1};
    std::vector<Vector3_Order<double>> full_kpoints;
    std::vector<Vector3_Order<int>> translations;
    AtomPairBvKRemap<atom_t> bvk_remap;
};

struct HartreeSymmetryData
{
    const SymmetryContext* context = nullptr;
    const std::vector<SpeciesBasisLayout>* wavefunction_layouts = nullptr;
    const std::map<atom_t, std::size_t>* atom_ao_sizes = nullptr;
};

HartreeStaticData build_hartree_static_data(
    const Cs_LRI& coefficients,
    const atpair_k_cplx_mat_t& coulomb,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis,
    const PeriodicBoundaryData& pbc,
    const AtomPairBvKRemap<atom_t>& bvk_remap,
    HartreeKNormalization normalization,
    double zero_tolerance = 1.0e-10);

Cs_LRI materialize_hartree_coefficients(
    const Cs_LRI& coefficients,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis);

HartreeCkMap build_hartree_c_k(
    const Cs_LRI& coefficients,
    const AtomicBasis& wavefunction_basis,
    const AtomicBasis& auxiliary_basis,
    const std::vector<Vector3_Order<double>>& full_kpoints);

HartreeVqMap build_hartree_v_q0(
    const atpair_k_cplx_mat_t& coulomb,
    const AtomicBasis& auxiliary_basis,
    double zero_tolerance);

HartreeDkMap split_weighted_density_by_atom(
    const WeightedDensityKMap& density_k,
    const std::map<int, int>& atom_ao_sizes);

PeriodicOperatorRMap inverse_fourier_hartree_operator(
    const HartreeDkMap& operator_k,
    const std::vector<Vector3_Order<double>>& full_kpoints,
    const std::vector<Vector3_Order<int>>& translations,
    const AtomPairBvKRemap<atom_t>* bvk_remap = nullptr);

PeriodicOperatorRMap build_hartree_delta_periodic_operator(
    const HartreeStaticData& static_data,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const HartreeSymmetryData* symmetry_data = nullptr);

SpinKMatrixMap build_hartree_delta_fixed_basis(
    const HartreeStaticData& static_data,
    const MeanField& live,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& scf_kpoints,
    const MeanField& target_reference,
    const std::vector<Vector3_Order<double>>& target_kpoints,
    const HartreeSymmetryData* symmetry_data = nullptr);

} // namespace qsgw
} // namespace librpa_int
