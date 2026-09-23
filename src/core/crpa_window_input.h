#pragma once

#include <string>
#include <vector>

#include "atomic_basis.h"
#include "correlated_subspace.h"
#include "meanfield.h"

namespace librpa_int
{

struct CrpaWindowOptions
{
    std::string correlated_species;
    std::string ligand_species;
    //! Parent frame and output subset: d, dp, eg, or t2g, in the Cartesian frame.
    //! d|dp and eg|dp select columns after joint d+p polar normalization.
    std::string parent_orbitals = "d";
    std::string output_orbitals = "d";
    //! Inclusive absolute-Hartree bounds, stored as lower/upper pairs.
    std::vector<double> response_windows_ha;
    std::vector<double> orbital_windows_ha;
    //! Explicit zero-based KS indices, identical for all spin/k points.
    //! Each list is mutually exclusive with the corresponding energy windows.
    std::vector<int> response_bands;
    std::vector<int> orbital_bands;
    double gram_abs = 1.0e-12;
    double gram_rel = 1.0e-12;
    double gram_cond_max = 1.0e12;
    double s_abs = 1.0e-10;
    double s_rel = 1.0e-12;
    double s_cond_max = 1.0e10;
    double residual_tol = 1.0e-10;
    double t_roundtrip_tol = 1.0e-10;
};

struct CrpaWindowBuildResult
{
    std::vector<SiteOrbitalGroup> sites;
    //! Normalized output Phi[spin][k], each matrix AO x local orbitals.
    std::vector<std::vector<ComplexMatrix>> orbitals_spin_k;
    //! Original-KS state indices [spin][k][selected band], independently selected.
    std::vector<std::vector<std::vector<int>>> response_bands;
    std::vector<std::vector<std::vector<int>>> orbital_bands;
};

std::vector<std::vector<std::vector<int>>> crpa_bands_in_energy_windows(
    const MeanField& meanfield, const std::vector<double>& edges_ha);

/*! Build spin-resolved local orbitals without changing KS energies or occupations.
 * Atomic types index species_labels; unrelated species remain spectators.
 * Only the first radial shell for each selected angular momentum is used.
 * S(k) and scalar eigenvectors must use the supplied basis convention.
 * Response window A never participates in construction of output window B.
 */
CrpaWindowBuildResult build_crpa_window_input(
    const AtomicBasis& basis, const BasisConvention& convention, const std::vector<int>& atom_types,
    const std::vector<std::string>& species_labels, const std::vector<ComplexMatrix>& overlap_k,
    const std::vector<Vector3_Order<double>>& kfrac_list, const MeanField& meanfield,
    const CrpaWindowOptions& options);

}  // namespace librpa_int
