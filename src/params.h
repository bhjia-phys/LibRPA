/*!
 * @file params.h
 * @brief parameters for controlling LibRPA calculation
 */
#ifndef PARAMS_H
#define PARAMS_H

#include <string>

//! a simple struct to collect all runtime parameters
struct Params
{
    //! the task to perform in LibRPA.
    static std::string task;

    static std::string DFT_software;
    //! the path of file to store librpa mainly output
    static std::string output_file;

    //! the path of directory to store librpa output files
    static std::string output_dir;

    //! the number of frequency grid points
    static int nfreq;

    //! the type of time-frequency grids
    static std::string tfgrids_type;

    //! the number of parameters for analytic continuation
    static int n_params_anacon;

    //! type of parallel routing
    static std::string parallel_routing;

    //! threshold of R-space Green's function when construcing.
    static double gf_R_threshold;

    //! threshold of RI coefficient when parsing. The atomic block with maximal element smaller than
    //! it will be filtered.
    static double cs_threshold;

    //! threshold of Coulomb matrix when parsing. The atom-pair block of Coulomb matrix with maximal
    //! element smaller than it will be filtered
    static double vq_threshold;

    //! threshold to filter when computing the square root of Coulomb matrix
    static double sqrt_coulomb_threshold;

    //! Cs threshold parsed to RPA object of LibRI.
    static double libri_chi0_threshold_C;

    //! Green's function threshold parsed to RPA object of LibRI.
    static double libri_chi0_threshold_G;

    //! switch of using ScaLAPACK for EcRPA calculation
    static bool use_scalapack_ecrpa;

    //! Cs threshold parsed to EXX object of LibRI.
    static double libri_exx_threshold_C;

    //! Density matrix threshold parsed to EXX object of LibRI.
    static double libri_exx_threshold_D;

    //! Coulomb matrix threshold parsed to EXX object of LibRI.
    static double libri_exx_threshold_V;

    //! Cs threshold parsed to G0W0 object of LibRI.
    static double libri_g0w0_threshold_C;

    //! Green's function threshold parsed to G0W0 object of LibRI.
    static double libri_g0w0_threshold_G;

    //! Screened Coulomb matrix threshold parsed to EXX object of LibRI.
    static double libri_g0w0_threshold_Wc;

    //! set gap by hand to generate frequency points in minimax time-frequency grids, for gapless
    //! systems
    static double minimax_min_gap;
    //! maximum transition frequency in minimax time-frequency grids
    //! for test supercell to fix the minimax frequency points
    static double minimax_max_transition;
    //! switch of using full Coulomb interaction in EXX
    //! test for abacus 2d system
    static bool use_fullcoul_exx;
    //! switch of reducing EXX real-space contractions with ABACUS irreducible sectors
    //! the ABACUS IBZ density-matrix restoration remains active independently
    static bool use_abacus_exx_symmetry;
    //! switch of restoring GW Green's functions from ABACUS IBZ k-stars
    static bool use_abacus_gw_symmetry;
    //! switch of dumping selected GW Green's-function slices for symmetry validation
    static bool output_abacus_gw_gf;
    //! switch of using full Coulomb interaction in Wc=eps-1v
    //! test for abacus 2d system
    static bool use_fullcoul_wc;
    //! switch of using ScaLAPACK for computing Wc from chi0
    static bool use_scalapack_gw_wc;

    //! switch of run-time debug mode
    static bool debug;

    //! switch of replacing head of screened interaction by macroscopic dielectric function
    static bool replace_w_head;
    //! switch of shrinking number of auxiliary basis by reading shrink_sinvS_0.txt
    static bool use_shrink_abfs;
    //! switch of shrinking chi0 matrix
    //! if false, chi0 will be calculated in small abfs
    //! and only unfold Wc
    //! faster 10 times and accurate in most cases
    static bool use_shrink_chi;
    //! switch of using spin-orbit coupling correction
    static bool use_soc;
    //! switch of using 2D dielectric function
    static bool use_2d_dielectric;
    //! switch of using pyatb_meanfield for head/wing calculation
    static bool use_pyatb;

    //! in task "g0w0_band", continue from previous self-energy matrix in NAO (R, iw)
    static bool band_continue;
    
    //! option of computing dielectric function on imaginary axis
    /*!
     * Available values:
     * - 0: direct read from input
     * - 1: dielectric model fitting
     * - 2: cubic-spline interpolation
     */
    static int option_dielect_func;

    /* ==========================================================
     * output options
     */
    /*
    switch of outputting Wc matrix in Abs (real space, imaginary frequency domain)
    Available values:
    - 0: do not output
    - 1: output lowerest frequency
    - 2: output all frequencies
    */
    static int output_Wc_Rf_mat;

    //! output energy_qp file for BSE calculation outside
    static bool output_energy_qp;

    //! output correlation self-energy matrix (reciprocal space, imaginary frequency domain)
    static bool output_gw_sigc_mat;

    //! output correlation self-energy matrix in NAO (real space, imaginary time domain)
    static bool output_gw_sigc_mat_rt;

    //! output correlation self-energy matrix in NAO (real space, imaginary frequency domain)
    static bool output_gw_sigc_mat_rf;

    //! output gw energy for HamGNN mechine learning
    static bool output_hamgnn;

    //! sum of nbands in Green's function. nbands < 0 meanns sum over all states.
    static int nbands_G;

    //! topology mesh dimension along the first reciprocal direction
    static int topology_nk1;

    //! topology mesh dimension along the second reciprocal direction
    static int topology_nk2;

    //! number of occupied bands per spin channel for topology. Negative means infer from KS bands.
    static int topology_nocc;

    //! shift the topological Hamiltonian by the chemical potential
    static bool topology_shift_mu;

    //! dump Sigma_c(i0,k) matrices in MatrixMarket format
    static bool topology_dump_sigma0;

    //! dump H_top(k) matrices in MatrixMarket format
    static bool topology_dump_hmat;

    //! dump occupied H_top eigenvectors in KS basis in MatrixMarket format
    static bool topology_dump_occ_evec;

    //! allow a diagonal Vxc fallback from band_vxc_k_*.txt when full band_vxc_mat_spin_*_k_*.csc
    //! matrices are unavailable. This is an approximation and should be enabled explicitly.
    static bool topology_allow_diag_vxc_fallback;

    //! enable restart from a saved QSGW checkpoint
    static bool qsgw_restart;

    //! checkpoint root used for QSGW restart. Empty means use current output_dir/qsgw_checkpoints/
    static std::string qsgw_restart_dir;

    //! restart from the requested iteration. Non-positive means use latest checkpoint
    static int qsgw_restart_iteration;

    //! write a QSGW checkpoint every N iterations. Non-positive disables periodic checkpoints
    static int qsgw_checkpoint_every;

    //! refresh the pyatb-style head/wing bundle from the current QSGW meanfield every iteration
    static bool qsgw_iterative_headwing;

    //! output directory for the refreshed pyatb-style head/wing bundle. Empty means
    //! output_dir/pyatb_librpa_df_iterative/
    static std::string qsgw_headwing_bundle_dir;

    static void check_consistency();
    static void print();
};

#endif
