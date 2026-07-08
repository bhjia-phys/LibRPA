#include "params.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../io/global_io.h"

namespace librpa_int {

// default setting

std::string Params::output_file = "stdout";
std::string Params::output_dir = "librpa.d/";
std::string Params::tfgrids_type = "minimax";
std::string Params::DFT_software = "auto";
std::string Params::parallel_routing = "auto";

int Params::nfreq = 0;
int Params::n_params_anacon = -1;
int Params::option_dielect_func = 2;

// Opt-in regularized-rational (ridge / ridge_guard) analytic continuation.
// Defaults reproduce the legacy bare-Thiele behaviour exactly.
std::string Params::anacon_method = "thiele";
double Params::pade_ridge_lambda = 1e-10;
double Params::pade_ridge_den_weight = 1.0;
double Params::pade_denominator_floor = 1e-12;
double Params::pade_thiele_den_cut = 1e-3;

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
double Params::minimax_min_gap = -1.0;
double Params::minimax_max_transition = -1.0;
bool Params::use_fullcoul_exx = false;
bool Params::use_fullcoul_wc = false;

bool Params::use_scalapack_ecrpa = true;
bool Params::use_scalapack_gw_wc = false;
bool Params::debug = false;
bool Params::replace_w_head = true;
bool Params::use_shrink_chi = true;
bool Params::use_shrink_abfs = false;
bool Params::use_soc = false;
bool Params::use_2d_dielectric = false;
bool Params::use_pyatb = true;

bool Params::band_continue = false;

/* ==========================================================
 * output options begin
 * ========================================================== */
int Params::output_Wc_Rf_mat = 0;
bool Params::output_energy_qp = false;
bool Params::output_gw_sigc_mat = false;
bool Params::output_gw_sigc_mat_rt = false;
bool Params::output_gw_sigc_mat_rf = false;
bool Params::output_hamgnn = false;
int Params::nbands_G = -1;
/* ==========================================================
 * output options end
 * ========================================================== */

std::string Params::qsgw_mixer = "linear";
double Params::qsgw_mixing_beta = 0.25;
int Params::qsgw_mixing_history = 12;
int Params::qsgw_linear_mixing_steps = 3;
int Params::qsgw_min_iter = 1;
int Params::qsgw_max_iter = 1;
bool Params::qsgw_dump_iter1 = false;
std::string Params::qsgw_dump_dir = "";

void Params::check_consistency()
{
    if (n_params_anacon < 0)
    {
        n_params_anacon = nfreq;
    }

    // Normalize and validate the opt-in analytic-continuation method.
    std::transform(anacon_method.begin(), anacon_method.end(), anacon_method.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (anacon_method == "pade")
    {
        anacon_method = "thiele";
    }
    if (anacon_method == "ridge-guard")
    {
        anacon_method = "ridge_guard";
    }
    if (anacon_method != "thiele" && anacon_method != "ridge" && anacon_method != "ridge_guard")
    {
        throw std::logic_error("Unknown anacon_method (" + anacon_method
                               + "). Available values: thiele, ridge, ridge_guard.");
    }
    if (pade_ridge_lambda < 0.0) pade_ridge_lambda = 0.0;
    if (pade_ridge_den_weight < 0.0) pade_ridge_den_weight = 0.0;
    if (pade_denominator_floor < 0.0) pade_denominator_floor = 0.0;
    if (pade_thiele_den_cut < 0.0) pade_thiele_den_cut = 0.0;
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
        {"minimax_min_gap", minimax_min_gap},
        {"minimax_max_transition", minimax_max_transition},
        {"pade_ridge_lambda", pade_ridge_lambda},
        {"pade_ridge_den_weight", pade_ridge_den_weight},
        {"pade_denominator_floor", pade_denominator_floor},
        {"pade_thiele_den_cut", pade_thiele_den_cut},
        {"qsgw_mixing_beta", qsgw_mixing_beta},
    };

    const std::vector<std::pair<std::string, int>> int_params{
        {"nfreq", nfreq},
        {"n_params_anacon", n_params_anacon},
        {"option_dielect_func", option_dielect_func},
        {"output_Wc_Rf_mat", output_Wc_Rf_mat},
        {"nbands_G", nbands_G},
        {"qsgw_mixing_history", qsgw_mixing_history},
        {"qsgw_linear_mixing_steps", qsgw_linear_mixing_steps},
        {"qsgw_min_iter", qsgw_min_iter},
        {"qsgw_max_iter", qsgw_max_iter},
    };

    const std::vector<std::pair<std::string, std::string>> str_params
        {
            {"output_dir", output_dir},
            {"output_file", output_file},
            {"tfgrids_type", tfgrids_type},
            {"parallel_routing", parallel_routing},
            {"anacon_method", anacon_method},
            {"qsgw_mixer", qsgw_mixer},
            {"qsgw_dump_dir", qsgw_dump_dir},
        };

    const std::vector<std::pair<std::string, bool>> bool_params{
        {"debug", debug},
        {"band_continue", band_continue},
        {"use_scalapack_ecrpa", use_scalapack_ecrpa},
        {"use_scalapack_gw_wc", use_scalapack_gw_wc},
        {"output_energy_qp", output_energy_qp},
        {"output_gw_sigc_mat", output_gw_sigc_mat},
        {"output_gw_sigc_mat_rt", output_gw_sigc_mat_rt},
        {"output_gw_sigc_mat_rf", output_gw_sigc_mat_rf},
        {"replace_w_head", replace_w_head},
        {"use_shrink_abfs", use_shrink_abfs},
        {"use_shrink_chi", use_shrink_chi},
        {"use_soc", use_soc},
        {"use_fullcoul_exx", use_fullcoul_exx},
        {"use_fullcoul_wc", use_fullcoul_wc},
        {"output_hamgnn", output_hamgnn},
        {"use_2d_dielectric", use_2d_dielectric},
        {"use_pyatb", use_pyatb},
        {"qsgw_dump_iter1", qsgw_dump_iter1},
    };

    for (const auto &param: str_params)
        librpa_int::global::lib_printf("%s = %s\n", param.first.c_str(), param.second.c_str());

    for (const auto &param: int_params)
        librpa_int::global::lib_printf("%s = %d\n", param.first.c_str(), param.second);

    for (const auto &param: double_params)
        librpa_int::global::lib_printf("%s = %f\n", param.first.c_str(), param.second);

    for (const auto &param: bool_params)
        librpa_int::global::lib_printf("%s = %s\n", param.first.c_str(), param.second? "T": "F");
}

Params params;

}
