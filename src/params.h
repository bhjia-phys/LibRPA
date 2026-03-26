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

    //! switch of using ScaLAPACK for computing Wc from chi0
    static bool use_scalapack_gw_wc;

    //! switch of run-time debug mode
    static bool debug;

    //! switch of replacing head of screened interaction by macroscopic dielectric function
    static bool replace_w_head;
    //! switch of shrinking number of auxiliary basis by reading shrink_sinvS_0.txt
    static bool use_shrink_abfs;
    //! switch of using spin-orbit coupling correction
    static bool use_soc;

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
    //! output correlation self-energy matrix (reciprocal space, imaginary frequency domain)
    static bool output_gw_sigc_mat;

    //! output correlation self-energy matrix in NAO (real space, imaginary time domain)
    static bool output_gw_sigc_mat_rt;

    //! output correlation self-energy matrix in NAO (real space, imaginary frequency domain)
    static bool output_gw_sigc_mat_rf;
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

    static void check_consistency();
    static void print();
};

#endif
