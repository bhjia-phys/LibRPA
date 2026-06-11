#include "params.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils_io.h"

// default setting

std::string Params::task = "rpa";
std::string Params::output_file = "stdout";
std::string Params::output_dir = "librpa.d/";
std::string Params::tfgrids_type = "minimax";
std::string Params::DFT_software =  "auto";
std::string Params::parallel_routing = "auto";

int Params::nfreq = 0;
int Params::n_params_anacon = -1;
int Params::option_dielect_func = 2;
std::string Params::anacon_method = "thiele";

double Params::gf_R_threshold = 1e-4;
double Params::cs_threshold = 1e-4;
double Params::vq_threshold = 0;
double Params::sqrt_coulomb_threshold = 1e-8;
double Params::pade_ridge_lambda = 1e-10;
double Params::pade_ridge_den_weight = 1.0;
double Params::pade_denominator_floor = 1e-12;
double Params::pade_thiele_den_cut = 1e-3;
double Params::libri_chi0_threshold_C = 0.0;
double Params::libri_chi0_threshold_G = 0.0;
double Params::libri_exx_threshold_C = 0.0;
double Params::libri_exx_threshold_D = 0.0;
double Params::libri_exx_threshold_V = 0.0;
double Params::libri_g0w0_threshold_C  = 0.0;
double Params::libri_g0w0_threshold_G  = 0.0;
double Params::libri_g0w0_threshold_Wc = 0.0;

bool Params::use_scalapack_ecrpa = true;
bool Params::use_scalapack_gw_wc = false;
bool Params::debug = false;
bool Params::replace_w_head = true;

/* ==========================================================
 * output options begin
 * ========================================================== */
bool Params::output_gw_sigc_mat = false;
bool Params::output_gw_sigc_mat_rt = false;
bool Params::output_gw_sigc_mat_rf = false;
/* ==========================================================
 * output options end
 * ========================================================== */

void Params::check_consistency()
{
    if (n_params_anacon < 0)
    {
        n_params_anacon = nfreq;
    }
    if (n_params_anacon < 1)
    {
        n_params_anacon = 1;
    }

    std::transform(anacon_method.begin(), anacon_method.end(), anacon_method.begin(),
            [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (anacon_method == "pade")
    {
        anacon_method = "thiele";
    }
    if (anacon_method == "ridge-guard")
    {
        anacon_method = "ridge_guard";
    }
    if (anacon_method != "thiele" && anacon_method != "ridge"
            && anacon_method != "ridge_guard")
    {
        throw std::logic_error("Unknown anacon_method (" + anacon_method
                + "). Available values: thiele, ridge, ridge_guard.");
    }
    if (pade_ridge_lambda < 0.0)
    {
        pade_ridge_lambda = 0.0;
    }
    if (pade_ridge_den_weight < 0.0)
    {
        pade_ridge_den_weight = 0.0;
    }
    if (pade_denominator_floor < 0.0)
    {
        pade_denominator_floor = 0.0;
    }
    if (pade_thiele_den_cut < 0.0)
    {
        pade_thiele_den_cut = 0.0;
    }
}

void Params::print()
{
    const std::vector<std::pair<std::string, double>> double_params
        {
            {"gf_R_threshold", gf_R_threshold},
            {"cs_R_threshold", cs_threshold},
            {"vq_threshold", vq_threshold},
            {"sqrt_coulomb_threshold", sqrt_coulomb_threshold},
            {"pade_ridge_lambda", pade_ridge_lambda},
            {"pade_ridge_den_weight", pade_ridge_den_weight},
            {"pade_denominator_floor", pade_denominator_floor},
            {"pade_thiele_den_cut", pade_thiele_den_cut},
            {"libri_chi0_threshold_C", libri_chi0_threshold_C},
            {"libri_chi0_threshold_G", libri_chi0_threshold_G},
            {"libri_exx_threshold_C", libri_exx_threshold_C},
            {"libri_exx_threshold_D", libri_exx_threshold_D},
            {"libri_exx_threshold_V", libri_exx_threshold_V},
            {"libri_g0w0_threshold_C", libri_g0w0_threshold_C},
            {"libri_g0w0_threshold_G", libri_g0w0_threshold_G},
            {"libri_g0w0_threshold_Wc", libri_g0w0_threshold_Wc},
        };

    const std::vector<std::pair<std::string, int>> int_params
        {
            {"nfreq", nfreq},
            {"n_params_anacon", n_params_anacon},
            {"option_dielect_func", option_dielect_func},
        };

    const std::vector<std::pair<std::string, std::string>> str_params
        {
            {"task", task},
            {"output_dir", output_dir},
            {"output_file", output_file},
            {"tfgrids_type", tfgrids_type},
            {"parallel_routing", parallel_routing},
            {"anacon_method", anacon_method},
        };

    const std::vector<std::pair<std::string, bool>> bool_params
        {
            {"debug", debug},
            {"use_scalapack_ecrpa", use_scalapack_ecrpa},
            {"use_scalapack_gw_wc", use_scalapack_gw_wc},
            {"output_gw_sigc_mat", output_gw_sigc_mat},
            {"replace_w_head", replace_w_head},
        };

    for (const auto &param: str_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second.c_str());

    for (const auto &param: int_params)
        LIBRPA::utils::lib_printf("%s = %d\n", param.first.c_str(), param.second);

    for (const auto &param: double_params)
        LIBRPA::utils::lib_printf("%s = %f\n", param.first.c_str(), param.second);

    for (const auto &param: bool_params)
        LIBRPA::utils::lib_printf("%s = %s\n", param.first.c_str(), param.second? "T": "F");
}

Params params;
