#include "params.h"

#include <string>
#include <utility>
#include <vector>

#include "utils_io.h"

// default setting

std::string Params::task = "rpa";
std::string Params::output_file = "stdout";
std::string Params::output_dir = "librpa.d/";
std::string Params::tfgrids_type = "minimax";
std::string Params::DFT_software = "auto";
std::string Params::parallel_routing = "auto";

int Params::nfreq = 0;
int Params::n_params_anacon = -1;
int Params::option_dielect_func = 2;

double Params::gf_R_threshold = 1e-4;
double Params::cs_threshold = 1e-4;
double Params::vq_threshold = 0;
double Params::sqrt_coulomb_threshold = 1e-8;
double Params::libri_chi0_threshold_C = 0.0;
double Params::libri_chi0_threshold_G = 0.0;
double Params::libri_exx_threshold_C = 0.0;
double Params::libri_exx_threshold_D = 0.0;
double Params::libri_exx_threshold_V = 0.0;
double Params::libri_g0w0_threshold_C = 0.0;
double Params::libri_g0w0_threshold_G = 0.0;
double Params::libri_g0w0_threshold_Wc = 0.0;

bool Params::use_scalapack_ecrpa = true;
bool Params::use_scalapack_gw_wc = false;
bool Params::debug = false;
bool Params::replace_w_head = true;
bool Params::use_shrink_abfs = false;
bool Params::use_soc = false;

/* ==========================================================
 * output options begin
 * ========================================================== */
bool Params::output_gw_sigc_mat = false;
bool Params::output_gw_sigc_mat_rt = false;
bool Params::output_gw_sigc_mat_rf = false;
int Params::nbands_G = -1;
int Params::topology_nk1 = 0;
int Params::topology_nk2 = 0;
int Params::topology_nocc = -1;
bool Params::topology_shift_mu = true;
bool Params::topology_dump_sigma0 = false;
bool Params::topology_dump_hmat = false;
bool Params::topology_dump_occ_evec = false;
bool Params::topology_allow_diag_vxc_fallback = false;
bool Params::qsgw_restart = false;
std::string Params::qsgw_restart_dir = "";
int Params::qsgw_restart_iteration = -1;
int Params::qsgw_checkpoint_every = 1;
/* ==========================================================
 * output options end
 * ========================================================== */

void Params::check_consistency()
{
    if (n_params_anacon < 0)
    {
        n_params_anacon = nfreq;
    }
}

void Params::print()
{
    const std::vector<std::pair<std::string, double>> double_params{
        {"gf_R_threshold", gf_R_threshold},
        {"cs_R_threshold", cs_threshold},
        {"vq_threshold", vq_threshold},
        {"sqrt_coulomb_threshold", sqrt_coulomb_threshold},
        {"libri_chi0_threshold_C", libri_chi0_threshold_C},
        {"libri_chi0_threshold_G", libri_chi0_threshold_G},
        {"libri_exx_threshold_C", libri_exx_threshold_C},
        {"libri_exx_threshold_D", libri_exx_threshold_D},
        {"libri_exx_threshold_V", libri_exx_threshold_V},
        {"libri_g0w0_threshold_C", libri_g0w0_threshold_C},
        {"libri_g0w0_threshold_G", libri_g0w0_threshold_G},
        {"libri_g0w0_threshold_Wc", libri_g0w0_threshold_Wc},
    };

    const std::vector<std::pair<std::string, int>> int_params{
        {"nfreq", nfreq},
        {"n_params_anacon", n_params_anacon},
        {"option_dielect_func", option_dielect_func},
        {"nbands_G", nbands_G},
        {"topology_nk1", topology_nk1},
        {"topology_nk2", topology_nk2},
        {"topology_nocc", topology_nocc},
        {"qsgw_restart_iteration", qsgw_restart_iteration},
        {"qsgw_checkpoint_every", qsgw_checkpoint_every},
    };

    const std::vector<std::pair<std::string, std::string>> str_params{
        {"task", task},
        {"output_dir", output_dir},
        {"output_file", output_file},
        {"tfgrids_type", tfgrids_type},
        {"parallel_routing", parallel_routing},
        {"qsgw_restart_dir", qsgw_restart_dir.empty() ? "(current-output-dir)" : qsgw_restart_dir},
    };

    const std::vector<std::pair<std::string, bool>> bool_params{
        {"debug", debug},
        {"use_scalapack_ecrpa", use_scalapack_ecrpa},
        {"use_scalapack_gw_wc", use_scalapack_gw_wc},
        {"output_gw_sigc_mat", output_gw_sigc_mat},
        {"replace_w_head", replace_w_head},
        {"use_shrink_abfs", use_shrink_abfs},
        {"use_soc", use_soc},
        {"topology_shift_mu", topology_shift_mu},
        {"topology_dump_sigma0", topology_dump_sigma0},
        {"topology_dump_hmat", topology_dump_hmat},
        {"topology_dump_occ_evec", topology_dump_occ_evec},
        {"topology_allow_diag_vxc_fallback", topology_allow_diag_vxc_fallback},
        {"qsgw_restart", qsgw_restart},
    };

    for (const auto &param : str_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second.c_str());

    for (const auto &param : int_params)
        LIBRPA::utils::lib_printf("%s = %d\n", param.first.c_str(), param.second);

    for (const auto &param : double_params)
        LIBRPA::utils::lib_printf("%s = %f\n", param.first.c_str(), param.second);

    for (const auto &param : bool_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second ? "T" : "F");
}

Params params;
