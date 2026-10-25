#pragma once

#include <array>
#include <map>

#include "chi0.h"
#include "dielecmodel.h"

namespace librpa_int
{

struct GwDecayQBound
{
    // Side order is positive (empty electron) then negative (occupied electron).
    std::array<double, 2> transition_max{};
    std::array<double, 2> coupling_max{};
    std::array<double, 2> coupling_trace{};
    std::array<double, 2> pole_upper{};
    double coulomb_min = 0.0;
    std::size_t coulomb_filtered = 0;
};

struct GwBodyDecayBound
{
    Chi0TransitionGrams grams;
    std::map<Vector3_Order<double>, GwDecayQBound> q_bounds;
    std::array<double, 2> electron_tail{};
    double gap = 0.0;
    double upper = 0.0;
};

struct GwOpticalDecayBound
{
    GwBodyDecayBound body;
    OpticalTransitionGrams optical;
    ArrayDesc augmented_desc;
    // Distributed 3+Naux zero embedding; its nonzero spectrum equals the
    // three Cartesian probes plus the retained regular-body subspace.
    std::array<Matz, 2> augmented_grams;
    std::array<double, 2> coupling_max{};
    std::array<double, 2> coupling_trace{};
    std::array<double, 2> pole_upper{};
    std::vector<std::complex<double>> head_direction;
    std::size_t coulomb_filtered = 0;
    double upper = 0.0;
};

// Internal finite, stable, gapped body diagnostic. Does not choose any grid.
// Coulomb is the dielectric matrix (not the external Wc or EXX matrix).
// Grams retain native distributed ordered atom-pair ownership; matrix products
// and eigenvalues use the global BLACS layout one q at a time.
GwBodyDecayBound compute_gw_body_decay_bound(
    Chi0 &chi0, const Cs_LRI &Cs, const std::vector<atpair_t> &atpairs, const AtomicBasis &abf_Cs,
    std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
    const atpair_k_cplx_mat_t &dielectric_coulomb, const BlacsCtxtHandler &blacs,
    double sqrt_coulomb_threshold, int n_bands_sigc, int n_bands_exclude_sigc,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false);

// Common full-loaded finite3D optical/body columns only. The supplied optical
// probes are exact signed endpoints; no frequency response is fitted or read.
GwOpticalDecayBound augment_gw_body_decay_bound(
    GwBodyDecayBound body, OpticalTransitionGrams optical, const AtomicBasis &basis,
    const PeriodicBoundaryData &pbc, const MpiCommHandler &comm,
    const atpair_k_cplx_mat_t &dielectric_coulomb, const BlacsCtxtHandler &blacs,
    double sqrt_coulomb_threshold, int option_dielect_func);

}  // namespace librpa_int
