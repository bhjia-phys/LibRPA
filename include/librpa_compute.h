#pragma once
/**
 * @file librpa_compute.h
 * @brief Computing data APIs for LibRPA.
 */

#include "librpa_handler.h"
#include "librpa_input.h"
#include "librpa_options.h"

// C APIs
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Construct and return frequency grids for numerical integration.
 *
 * Generates the frequency points and quadrature weights based on the grid type
 * specified in the options (e.g., Gauss-Legendre, Minimax, etc.).
 *
 * @param[in]  h        Handler.
 * @param[in]  p_opts   Runtime options.
 * @param[out] omegas  Array of frequency points (size: opts->nfreq).
 * @param[out] weights Array of quadrature weights (size: opts->nfreq).
 */
void librpa_get_imaginary_frequency_grids(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                          double *omegas, double *weights);

/**
 * @brief Compute RPA correlation energy.
 *
 * Calculates the RPA correlation energy using the input data (wavefunctions,
 * Coulomb matrices, etc.) that has been set via the input parsing functions.
 *
 * @param[in]  h                           Handler.
 * @param[in]  p_opts                      Runtime options.
 * @param[in]  n_ibz_kpoints                Number of irreducible k-points.
 * @param[out] rpa_corr_ibzk_contrib_re   Real part of correlation energy per IBZ k-point.
 * @param[out] rpa_corr_ibzk_contrib_im   Imaginary part of correlation energy per IBZ k-point.
 * @return Total RPA correlation energy (real part).
 */
double librpa_get_rpa_correlation_energy(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                         int n_ibz_kpoints, double *rpa_corr_ibzk_contrib_re,
                                         double *rpa_corr_ibzk_contrib_im);

//! Build exact-exchange matrix
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_exx(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain exact-exchange potential for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx             Exact-exchange potential for selected states.
 *                              It should be at least as long as n_spins * n_kpts_local * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                              const int n_spins, const int n_kpts_this, const int *iks_this,
                              int i_state_low, int i_state_high, double *vexx);

//! Obtain exact-exchange potential for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of band k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx_band        Exact-exchange potential for selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_band_this, const int *iks_band_this,
                               int i_state_low, int i_state_high, double *vexx_band);

//! Build self-energy matrix of G0W0, including the correlation and exchange contributions.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_g0w0_sigma(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain correlation self-energies for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc              exchange-correlation potential of the selected states.
 * @param[in]  vexx             Exact-exchange potential for the selected states.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_re          Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 * @param[out] sigc_im          Same as sigc_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_this,
                                const int *iks_this, int i_state_low, int i_state_high,
                                const double *vxc, const double *vexx, double *sigc_re, double *sigc_im);

//! Obtain correlation self-energies and QP energies for selected states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_kgrid.
 * If p_opts->use_hedin_shift is enabled, the QP solver evaluates
 * \f[
 * \Sigma_c(E - E_F - \Delta).
 * \f]
 * If p_opts->istate_ref_hedin_shift is negative, Delta is computed separately
 * for each selected state. Otherwise, that absolute zero-based reference state
 * at each k-point and spin provides Delta for all states in the same channel.
 *
 * @param[in]  h           LibRPA handler.
 * @param[in]  p_opts      Runtime options.
 * @param[in]  n_spins     Number of spin channels.
 * @param[in]  n_kpts_this Number of k-points computed on this process.
 * @param[in]  iks_this    Global indices of those k-points.
 * @param[in]  i_state_low First state index (inclusive).
 * @param[in]  i_state_high Last state index (exclusive).
 * @param[in]  vxc         XC potential for the selected states.
 * @param[in]  vexx        Exact-exchange potential for the selected states.
 * @param[out] sigc_re     Real part of the correlation self-energy.
 * @param[out] sigc_im     Imaginary part of the correlation self-energy.
 * @param[out] eqp              Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_re/sigc_im.
 */
void librpa_get_g0w0_qpe_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_this,
                               const int *iks_this, int i_state_low, int i_state_high,
                               const double *vxc, const double *vexx, double *sigc_re,
                               double *sigc_im, double *eqp);

//! Obtain spectral functions for selected k-grid states.
/**
 * Computes spectral functions under the diagonal approximation of the Green's
 * function:
 * \f$A_{n\mathbf{k}}(\omega) = -\pi^{-1}\mathrm{Im}\,G_{n\mathbf{k}}(\omega)\f$.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_this             Number of k-points to compute on this process.
 * @param[in]  iks_this                Global k-point indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc                     XC potential for selected states.
 * @param[in]  vexx                    Exact-exchange potential for selected states.
 * @param[out] spectral_function       Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][k-point][state][omega].
 * @param[out] sigc                    Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][k-point][state][omega] order as spectral_function.
 */
void librpa_get_g0w0_spectral_function_kgrid(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_this, const int *iks_this, int i_state_low, int i_state_high,
    const int n_omegas, const double *omegas, const double *vxc, const double *vexx,
    double *spectral_function, double *sigc);

//! Obtain correlation self-energies for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc_band         exchange-correlation potential of the selected states at band k-points.
 * @param[in]  vexx_band        Exact-exchange potential for the selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_band_re     Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 * @param[out] sigc_band_im     Same as sigc_band_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                 const int n_spins, const int n_kpts_band_this,
                                 const int *iks_band_this, int i_state_low, int i_state_high,
                                 const double *vxc_band, const double *vexx_band, double *sigc_band_re, double *sigc_band_im);

//! Obtain correlation self-energies and QP energies for selected band-k states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_band_k.
 * If p_opts->use_hedin_shift is enabled, the QP solver evaluates
 * \f[
 * \Sigma_c(E - E_F - \Delta).
 * \f]
 * If p_opts->istate_ref_hedin_shift is negative, Delta is computed separately
 * for each selected band state. Otherwise, that absolute zero-based reference
 * state at each band k-point and spin provides Delta for all states in the same channel.
 *
 * @param[in]  h                 LibRPA handler.
 * @param[in]  p_opts            Runtime options.
 * @param[in]  n_spins           Number of spin channels.
 * @param[in]  n_kpts_band_this  Number of band k-points computed on this process.
 * @param[in]  iks_band_this     Global indices of those band k-points.
 * @param[in]  i_state_low       First state index (inclusive).
 * @param[in]  i_state_high      Last state index (exclusive).
 * @param[in]  vxc_band          XC potential for the selected band states.
 * @param[in]  vexx_band         Exact-exchange potential for the selected band states.
 * @param[out] sigc_band_re      Real part of the correlation self-energy.
 * @param[out] sigc_band_im      Imaginary part of the correlation self-energy.
 * @param[out] eqp_band         Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_band_re/sigc_band_im.
 */
void librpa_get_g0w0_qpe_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_band_this,
                                const int *iks_band_this, int i_state_low, int i_state_high,
                                const double *vxc_band, const double *vexx_band,
                                double *sigc_band_re, double *sigc_band_im,
                                double *eqp_band);

//! Obtain spectral functions for selected band-k states.
/**
 * Same convention as librpa_get_g0w0_spectral_function_kgrid, but evaluates
 * states at band k-points.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_band_this        Number of band k-points to compute on this process.
 * @param[in]  iks_band_this           Global band-k indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc_band                XC potential for selected band states.
 * @param[in]  vexx_band               Exact-exchange potential for selected band states.
 * @param[out] spectral_function_band  Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_band_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][band k-point][state][omega].
 * @param[out] sigc_band               Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][band k-point][state][omega] order as spectral_function_band.
 */
void librpa_get_g0w0_spectral_function_band_k(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_band_this, const int *iks_band_this, int i_state_low,
    int i_state_high, const int n_omegas, const double *omegas,
    const double *vxc_band, const double *vexx_band, double *spectral_function_band,
    double *sigc_band);

/** Collective velocity computation and setter; optional analytic head/wing
 * preparation reuses the ordinary numerical API. Options use API values 3/4,
 * never driver preset 6. Optical preparation requires replace_w_head enabled.
 * Option 3 also requires use_scalapack_gw_wc to consume the analytic wing.
 * Input is borrowed only for this call; input structures are in librpa_input.h.
 */
void librpa_prepare_native_velocity(LibrpaHandler *h, const LibrpaOptions *options,
                                    const LibrpaNativeVelocityInput *root_input,
                                    int prepare_optical);

/** Independently owned, rank-local snapshot of numerical optical outputs.
 * Obtain this structure with librpa_get_optical_result and release it only with
 * librpa_delete_optical_result. Do not modify its fields, free its arrays, or
 * pass a copied structure to the deletion function. All arrays are read-only
 * and remain valid until that original result is deleted, including after the
 * handler is changed or destroyed. Empty arrays have null pointers.
 *
 * Complex arrays are row-major, interleaved [real,imag]. Velocity is
 * [n_spins][n_kpoints][3][n_states][n_states][2], in Hartree*Bohr;
 * frequencies_ha contains n_frequencies imaginary-frequency nodes in Hartree.
 * Head and wing are [n_frequencies][rows][columns][2], with their respective
 * dimensions below. Head is dimensionless; the auxiliary-basis wing uses
 * LibRPA's Coulomb-normalized response convention. No MPI gathering is done.
 */
typedef struct LibrpaOpticalResult
{
    int n_spins, n_kpoints, n_states, n_frequencies;
    int head_rows, head_columns, wing_rows, wing_columns;
    const double *velocity_ri;
    const double *frequencies_ha;
    const double *head_ri;
    const double *wing_ri;
} LibrpaOpticalResult;

/** Source-compatible borrowed view. Copying it does not copy or own the arrays. */
typedef LibrpaOpticalResult LibrpaOpticalView;

/** Noncollective snapshot; analytic head/wing preparation must be complete. */
LibrpaOpticalResult *librpa_get_optical_result(const LibrpaHandler *h);
/** Return a borrowed copy of the public fields, with the original result's lifetime. */
LibrpaOpticalView librpa_get_optical_view(const LibrpaOpticalResult *result);
/** Release a result and its arrays. Passing NULL is permitted. */
void librpa_delete_optical_result(LibrpaOpticalResult *result);

typedef enum
{
    LIBRPA_QSGW_ITERATIONS,
    LIBRPA_QSGW_EIGENVALUES,
    LIBRPA_QSGW_MATRICES,
    LIBRPA_QSGW_QUASIPARTICLES,
    LIBRPA_QSGW_ENERGY_QP,
    LIBRPA_QSGW_GW_BAND,
    LIBRPA_QSGW_KS_BAND,
    LIBRPA_QSGW_EXX_BAND,
    LIBRPA_QSGW_QSGW_BAND,
    LIBRPA_QSGW_HAMILTONIAN,
    LIBRPA_QSGW_HEAD_HAMILTONIAN
} LibrpaQsgwOutputKind;
/** Borrowed output on rank zero only, valid during the callback. spin is
 * zero-based; text_size excludes the terminating zero. The callback presents
 * output only and must not mutate or recursively use the handler/task. */
typedef struct
{
    LibrpaQsgwOutputKind kind;
    int iteration, spin, append;
    const char *text;
    size_t text_size;
    const LibrpaLocalizedOperator *hamiltonian;
} LibrpaQsgwOutput;
typedef int (*LibrpaQsgwOutputSink)(const LibrpaQsgwOutput *, void *user_data);

/** All energies are Hartree unless the field explicitly says eV. Strings and
 * numeric inputs are copied by the call receiving them. Checkpoints preserve
 * the immutable input eigenframe; original producer inputs remain required.
 * Keep the handler alive until the task is deleted. After preparation keep
 * its SCF eigenframe/grid fixed; finish band/Coulomb/velocity input before
 * run. Do not modify or reenter the handler during run or its callbacks. Call prepare and
 * set_input collectively before the single run call. */
LibrpaQsgwTask *librpa_create_qsgw_task(LibrpaHandler *, const LibrpaOptions *,
                                        const LibrpaQsgwControls *);
/** Collective validation before ordinary numerical input is installed. */
void librpa_prepare_qsgw_task(LibrpaQsgwTask *);
void librpa_set_qsgw_output_sink(LibrpaQsgwTask *, LibrpaQsgwOutputSink, void *user_data);
void librpa_run_qsgw_task(LibrpaQsgwTask *);
void librpa_delete_qsgw_task(LibrpaQsgwTask *);

/** Query the numerical dimensions needed to prepare a QSGW task's host input.
 * band_path selects band input when nonzero. This call is noncollective.
 */
void librpa_get_meanfield_dimensions(const LibrpaHandler *h, int band_path, int *nspins, int *nkpts,
                                     int *nstates, int *naos, int *nspinor);

#ifdef __cplusplus
}
#endif
