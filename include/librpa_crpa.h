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
     * Output orbitals are selected from each spin's jointly normalized parent frame.
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
        /** Packed complex [k][AO row][AO column][real/imag], full scalar k grid. */
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
        const double* fully_screened_ri;
    } LibrpaCrpaTensor;

    void librpa_init_crpa_input(LibrpaCrpaInput* input);

    /** Read full fractional k coordinates in the current KS eigenvector order. */
    void librpa_get_crpa_kgrid(const LibrpaHandler* h, int n_kpoints, double* fractional_k);

    /** Collective calculation on the handler communicator; no input/output files.
     * All ranks must call with identical options and replicated LibrpaCrpaInput.
     * Supports scalar full-grid LIBRI with replicated KS eigenvectors and native
     * positive minimax nodes. No static node or analytic continuation is added.
     * The result is independently owned on each rank; delete it on every rank.
     */
    LibrpaCrpaResult* librpa_compute_crpa_window(const LibrpaHandler* h,
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

#ifdef __cplusplus
}
#endif
