#pragma once

#include "librpa_handler.h"
#include "librpa_options.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** Numerical cRPA input. Strings and arrays remain owned by the caller.
     * Initialize with librpa_init_crpa_input before assigning fields. All input
     * values and array contents must be replicated identically on every rank of
     * the handler communicator and remain valid until the collective call returns.
     * Energy windows use inclusive absolute-Hartree lower/upper pairs.
     * A selects original KS states on both occupied and empty branches of Pd.
     * B selects the KS space used to project atomic trials; A and B are independent.
     * For each space, provide exactly one window array or explicit band list.
     * Species labels follow the zero-based type indices passed to librpa_set_atoms.
     * Scalar output uses each spin's parent frame; SOC uses one full spinor frame.
     */
    typedef struct
    {
        int n_species;
        const char* const* species_labels;
        const char* correlated_species;
        const char* ligand_species;
        const char* parent_orbitals;
        const char* output_orbitals;
        int n_response_edges;
        const double* response_windows_ha;
        int n_orbital_edges;
        const double* orbital_windows_ha;
        /** Zero-based indices, shared by all spin/k points. Each list replaces
         * its respective energy window; set the unused count to zero. */
        int n_response_bands;
        const int* response_bands;
        int n_orbital_bands;
        const int* orbital_bands;
        int n_kpoints;
        int n_aos;
        /** Spatial overlap [k][AO row][AO column][real/imag], full k grid.
         * SOC lifts this internally to I_spinor tensor S; n_aos stays spatial. */
        const double* overlap_k_ri;
        double gram_abs;
        double gram_rel;
        double gram_cond_max;
        double s_abs;
        double s_rel;
        double s_cond_max;
        double residual_tol;
        double t_roundtrip_tol;
    } LibrpaCrpaInput;

    typedef struct LibrpaCrpaResult LibrpaCrpaResult;

    /** Read-only tensor view. Arrays remain valid until the result is deleted.
     * Arrays contain M^4 complex entries, each packed as real/imag doubles.
     * Matrix row is the ordered pair (a,b), column is (c,d): a*M+b, c*M+d.
     * Energies and interactions are Hartree; frequency is positive imaginary nu.
     */
    typedef struct
    {
        int atom_index;
        int spin_left;
        int spin_right;
        int n_orbitals;
        double frequency_ha;
        const double* bare_ri;
        const double* partially_screened_ri;
        /** Null for U-only results; no fully screened interaction was computed. */
        const double* fully_screened_ri;
    } LibrpaCrpaTensor;

    void librpa_init_crpa_input(LibrpaCrpaInput* input);

    /** Read full fractional k coordinates in the current KS eigenvector order. */
    void librpa_get_crpa_kgrid(const LibrpaHandler* h, int n_kpoints, double* fractional_k);

    /** Collective calculation on the handler communicator; no input/output files.
     * All ranks must call with identical options and replicated LibrpaCrpaInput.
     * Supports scalar or two-component spinor full-grid LIBRI with replicated KS eigenvectors and native
     * positive minimax nodes. use_symmetry_rpa reduces q-space screening only,
     * restoring full kernels before contraction; original full-k states are required.
     * No static node or analytic continuation is added.
     * The result is independently owned on each rank; delete it on every rank.
     */
    LibrpaCrpaResult* librpa_compute_crpa_window(const LibrpaHandler* h,
                                                 const LibrpaOptions* options,
                                                 const LibrpaCrpaInput* input);

    /** Build the ordinary GW self-energy and cRPA U for the same entering state.
     * P0 is built once by GW and borrowed by cRPA; only Pd and constrained
     * screening are added. GW P0, optical data, W and self-energy conventions
     * remain unchanged. Requires full-grid LIBRI without symmetry or a SigC
     * file-read shortcut. Input has the same contract as the standalone call.
     * The owned result contains bare/U; fully_screened_ri is null.
     * cRPA screens and projects with full Coulomb independently of the GW
     * eps/Wc/EXX Coulomb flags, which keep their original GW meaning.
     */
    LibrpaCrpaResult* librpa_build_g0w0_sigma_crpa(LibrpaHandler* h, const LibrpaOptions* options,
                                                   const LibrpaCrpaInput* input);

    /** Response-only bare/U calculation for the handler's current E/C/f.
     * Builds P0 and Pd and screens Pr, without full-W screening or self-energy.
     * Suitable for the final updated QSGW state. Uses current velocity data
     * when optical correction is enabled. Returns an independently owned
     * result whose fully_screened_ri is null. The cRPA solve uses full Coulomb
     * regardless of the independent GW Coulomb flags.
     */
    LibrpaCrpaResult* librpa_compute_crpa_u_window(const LibrpaHandler* h,
                                                   const LibrpaOptions* options,
                                                   const LibrpaCrpaInput* input);

    void librpa_delete_crpa_result(LibrpaCrpaResult* result);
    int librpa_crpa_result_size(const LibrpaCrpaResult* result);
    void librpa_crpa_result_tensor(const LibrpaCrpaResult* result, int index,
                                   LibrpaCrpaTensor* tensor);

    /** Retrieve A (response_window != 0) or B selected original-KS band indices.
     * Band indices are zero based and the returned array is owned by result.
     */
    void librpa_crpa_result_bands(const LibrpaCrpaResult* result, int response_window, int spin,
                                  int kpoint, int* count, const int** bands);

    /** Trial label for a local output column; owned by result until deletion.
     * SOC suffix up/down identifies a trial, not a conserved spin quantum number.
     * SOC tensor spin_left/right are channel zero; spinor indices are a,b,c,d. */
    const char* librpa_crpa_result_orbital_label(const LibrpaCrpaResult* result, int atom, int orbital);

#ifdef __cplusplus
}
#endif
