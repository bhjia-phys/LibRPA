#include "driver_utils.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <fstream>
#include <stdexcept>

#include "dielecmodel.h"
#include "driver_params.h"
#include "envs_mpi.h"
#include "fitting.h"
#include "interpolate.h"
#include "meanfield.h"
#include "params.h"
#include "read_data.h"
#include "ri.h"

namespace
{

std::string ensure_trailing_slash(const std::string &dir_path)
{
    if (dir_path.empty() || dir_path.back() == '/')
    {
        return dir_path;
    }
    return dir_path + "/";
}

void load_pyatb_headwing_bundle(const std::string &bundle_dir,
                                const std::vector<double> &frequencies_target)
{
    const auto normalized_dir = ensure_trailing_slash(bundle_dir);
    const auto velocity_file = normalized_dir + "velocity_matrix";

    std::ifstream infile_pyatb(velocity_file);
    if (!infile_pyatb.is_open())
    {
        throw std::runtime_error("use_pyatb is enabled but cannot find " + velocity_file);
    }

    read_scf_occ_eigenvalues(normalized_dir + "band_out", pyatb_meanfield);
    read_eigenvector(normalized_dir, pyatb_meanfield);
    read_velocity(velocity_file, pyatb_meanfield);

    int n_basis = 0;
    int n_states = 0;
    int n_spin = 0;
    int flag = 0;
    std::vector<Vector3_Order<double>> kfrac_band =
        read_band_kpath_info(normalized_dir + "k_path_info", n_basis, n_states, n_spin, flag);
    if (flag != 0)
    {
        throw std::runtime_error("Failed to read k_path_info from " + normalized_dir);
    }
    if (Params::use_soc)
    {
        assert(n_basis % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_basis = n_basis / 2;
    }
    df_headwing.set(pyatb_meanfield, kfrac_band, frequencies_target, n_basis, n_states, n_spin);
}

void load_direct_headwing_inputs(const std::vector<double> &frequencies_target)
{
    int n_basis = 0;
    int n_states = 0;
    int n_spin = 0;

    const std::string file_abacus = driver_params.input_dir + "velocity_matrix";
    const std::string file_aims = driver_params.input_dir + "moment_KS_spin_01_kpt_000001.dat";
    std::ifstream infile_abacus(file_abacus);
    std::ifstream infile_aims(file_aims);
    if (infile_abacus.is_open())
    {
        read_velocity(file_abacus, meanfield);
        n_basis = meanfield.get_n_aos();
        n_states = meanfield.get_n_bands();
        n_spin = meanfield.get_n_spins();
        df_headwing.set(meanfield, kfrac_list, frequencies_target, n_basis, n_states, n_spin);
    }
    else if (infile_aims.is_open())
    {
        read_velocity_aims(meanfield, driver_params.input_dir);
        n_basis = meanfield.get_n_aos();
        n_states = meanfield.get_n_bands();
        n_spin = meanfield.get_n_spins();
        df_headwing.set(meanfield, kfrac_list, frequencies_target, n_basis, n_states, n_spin);
    }
    else
    {
        throw std::runtime_error("Cannot find moment files for head/wing!");
    }
}

void prepare_headwing_inputs(const std::vector<double> &frequencies_target)
{
    if (Params::use_pyatb)
    {
        load_pyatb_headwing_bundle(resolve_active_pyatb_headwing_dir(), frequencies_target);
    }
    else
    {
        load_direct_headwing_inputs(frequencies_target);
    }
}

} // namespace

bool use_iterative_pyatb_headwing_bundle()
{
    std::string task_lower = Params::task;
    std::transform(task_lower.begin(), task_lower.end(), task_lower.begin(), ::tolower);
    const bool task_supported = (task_lower == "qsgw" || task_lower == "qsgw_band");
    return Params::qsgw_iterative_headwing && Params::replace_w_head && Params::use_pyatb
           && (Params::option_dielect_func == 3 || Params::option_dielect_func == 4)
           && task_supported;
}

std::string get_iterative_pyatb_headwing_bundle_dir()
{
    if (!Params::qsgw_headwing_bundle_dir.empty())
    {
        return ensure_trailing_slash(Params::qsgw_headwing_bundle_dir);
    }
    return Params::output_dir + "pyatb_librpa_df_iterative/";
}

std::string resolve_active_pyatb_headwing_dir()
{
    if (use_iterative_pyatb_headwing_bundle())
    {
        return get_iterative_pyatb_headwing_bundle_dir();
    }
    return resolve_input_dir_with_pyatb_fallback(driver_params.input_dir, "velocity_matrix");
}

void initialize_headwing_velocity_from_input(MeanField &mf)
{
    if (Params::use_pyatb)
    {
        const auto velocity_file =
            resolve_input_file_with_pyatb_fallback(driver_params.input_dir, "velocity_matrix");
        std::ifstream infile(velocity_file);
        if (!infile.is_open())
        {
            throw std::runtime_error("Failed to seed iterative head/wing velocity from "
                                     + velocity_file);
        }
        read_velocity(velocity_file, mf);
        return;
    }

    const std::string file_abacus = driver_params.input_dir + "velocity_matrix";
    const std::string file_aims = driver_params.input_dir + "moment_KS_spin_01_kpt_000001.dat";
    std::ifstream infile_abacus(file_abacus);
    std::ifstream infile_aims(file_aims);
    if (infile_abacus.is_open())
    {
        read_velocity(file_abacus, mf);
    }
    else if (infile_aims.is_open())
    {
        read_velocity_aims(mf, driver_params.input_dir);
    }
    else
    {
        throw std::runtime_error("Cannot find a velocity seed for iterative head/wing refresh");
    }
}

void refresh_pyatb_headwing_bundle(const MeanField &mf,
                                   const std::vector<Vector3_Order<double>> &kfrac,
                                   const std::string &bundle_dir)
{
    if (!Params::use_pyatb)
    {
        return;
    }
    const auto target_dir =
        bundle_dir.empty() ? get_iterative_pyatb_headwing_bundle_dir() : ensure_trailing_slash(bundle_dir);
    write_pyatb_bundle(target_dir, mf, kfrac);
}

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target)
{
    std::vector<double> df_target;

    switch (option)
    {
        case 0: /* No extrapolation, copy the input data to target */
        {
            assert(frequencies_in.size() == frequencies_target.size());
            df_target = df_in;
            break;
        }
        case 1: /* Use spline interpolation */
        {
            df_target =
                LIBRPA::utils::interp_cubic_spline(frequencies_in, df_in, frequencies_target);
            break;
        }
        case 2: /* Use dielectric model for fitting */
        {
            LIBRPA::utils::LevMarqFitting levmarq;
            std::vector<double> pars(DoubleHavriliakNegami::d_npar, 1);
            pars[0] = pars[4] = df_in[0];
            df_target =
                levmarq.fit_eval(pars, frequencies_in, df_in, DoubleHavriliakNegami::func_imfreq,
                                 DoubleHavriliakNegami::grad_imfreq, frequencies_target);
            break;
        }
        case 3: /* Read velocity matrix and calculate head and wing */
        {
            prepare_headwing_inputs(frequencies_target);
            df_headwing.cal_head();
            df_target = df_headwing.get_head_vec();
            df_headwing.test_head();
            df_headwing.cal_wing();
            if (Params::debug)
            {
                df_headwing.test_wing();
            }
            break;
        }
        case 4: /* Read velocity matrix and calculate head only */
        {
            prepare_headwing_inputs(frequencies_target);
            df_headwing.cal_head();
            df_target = df_headwing.get_head_vec();
            df_headwing.test_head();
            break;
        }
        default:
            throw std::logic_error("Unsupported value for option");
    }

    return df_target;
}
