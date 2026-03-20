#include "task_gw_band.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "abacus_symmetry.h"
#include "analycont.h"
#include "chi0.h"
#include "constants.h"
#include "coulmat.h"
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"
#include "exx.h"
#include "gw.h"
#include "meanfield.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "qpe_solver.h"
#include "read_data.h"
#include "ri.h"
#include "utils_timefreq.h"
#include "write_aims.h"

namespace
{

using ExxDiagMap = std::map<int, std::map<int, std::map<int, double>>>;

struct GwBandStateDebugSpec
{
    bool enabled = false;
    int spin = -1;
    int kpoint = -1;
    int state = -1;
};

struct QpeIterationTracePoint
{
    int iter = 0;
    double e_qp = 0.0;
    double diff = 0.0;
    cplxdb sigc{0.0, 0.0};
};

bool need_full_cut_coulomb_for_abacus_symmetry()
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return Params::use_abacus_gw_symmetry
           && ctx.available
           && ctx.has_abf_shell_layout()
           && !ctx.kstars.empty()
           && ctx.kstars.size() == kfrac_list.size()
           && static_cast<int>(klist.size()) < get_full_bz_kpoint_count();
}

bool can_expand_gw_output_to_full_bz()
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return ctx.available && !ctx.kstars.empty()
           && ctx.kstars.size() == static_cast<std::size_t>(meanfield.get_n_kpoints())
           && static_cast<int>(ctx.count_kstar_members()) > meanfield.get_n_kpoints();
}

bool enable_exx_kgrid_sigc_probe()
{
    const char* env_value = std::getenv("LIBRPA_DEBUG_EXX_KGRID_SIGC");
    if (env_value == nullptr)
    {
        return false;
    }
    const std::string value(env_value);
    return !(value.empty() || value == "0" || value == "f" || value == "F" || value == "false"
             || value == "FALSE");
}

void write_exx_kgrid_debug_files(const ExxDiagMap& exx_diag, const std::string& file_tag)
{
    using LIBRPA::envs::mpi_comm_global_h;
    if (!enable_exx_kgrid_sigc_probe() || !mpi_comm_global_h.is_root())
    {
        return;
    }

    const auto full_k_members =
        can_expand_gw_output_to_full_bz()
            ? LIBRPA::build_abacus_full_kpoint_member_list(LIBRPA::abacus_symmetry_ctx,
                                                           kfrac_list)
            : std::vector<LIBRPA::AbacusFullKpointMemberEntry>{};

    for (int i_spin = 0; i_spin < meanfield.get_n_spins(); ++i_spin)
    {
        std::ofstream ofs(file_tag + std::to_string(i_spin + 1) + ".dat");
        ofs << std::fixed;
        if (!full_k_members.empty())
        {
            for (int ifull = 0; ifull != static_cast<int>(full_k_members.size()); ++ifull)
            {
                const auto& member = full_k_members[static_cast<std::size_t>(ifull)];
                const int i_kpoint = member.ik_ibz;
                const auto& k = member.k_bz;
                for (int i_state = 0; i_state < meanfield.get_n_bands(); ++i_state)
                {
                    ofs << std::setw(5) << ifull + 1 << std::setw(15) << std::setprecision(7)
                        << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                        << std::setprecision(7) << k.z << std::setw(8) << i_state + 1
                        << std::setw(20) << std::setprecision(10)
                        << exx_diag.at(i_spin).at(i_kpoint).at(i_state) * HA2EV << '\n';
                }
            }
            continue;
        }

        for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); ++i_kpoint)
        {
            const auto& k = kfrac_list[i_kpoint];
            for (int i_state = 0; i_state < meanfield.get_n_bands(); ++i_state)
            {
                ofs << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                    << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                    << std::setprecision(7) << k.z << std::setw(8) << i_state + 1
                    << std::setw(20) << std::setprecision(10)
                    << exx_diag.at(i_spin).at(i_kpoint).at(i_state) * HA2EV << '\n';
            }
        }
    }
}

void report_exx_diag_debug_difference(const ExxDiagMap& exx_before_sigc,
                                      const ExxDiagMap& exx_after_sigc)
{
    using LIBRPA::envs::mpi_comm_global_h;
    if (!enable_exx_kgrid_sigc_probe() || !mpi_comm_global_h.is_root())
    {
        return;
    }

    double max_abs_diff = 0.0;
    int max_spin = -1;
    int max_kpoint = -1;
    int max_state = -1;
    for (const auto& spin_entry : exx_before_sigc)
    {
        const int i_spin = spin_entry.first;
        const auto after_spin_iter = exx_after_sigc.find(i_spin);
        if (after_spin_iter == exx_after_sigc.end())
        {
            continue;
        }
        for (const auto& k_entry : spin_entry.second)
        {
            const int i_kpoint = k_entry.first;
            const auto after_k_iter = after_spin_iter->second.find(i_kpoint);
            if (after_k_iter == after_spin_iter->second.end())
            {
                continue;
            }
            for (const auto& state_entry : k_entry.second)
            {
                const int i_state = state_entry.first;
                const auto after_state_iter = after_k_iter->second.find(i_state);
                if (after_state_iter == after_k_iter->second.end())
                {
                    continue;
                }
                const double abs_diff = std::abs(state_entry.second - after_state_iter->second);
                if (abs_diff > max_abs_diff)
                {
                    max_abs_diff = abs_diff;
                    max_spin = i_spin;
                    max_kpoint = i_kpoint;
                    max_state = i_state;
                }
            }
        }
    }

    LIBRPA::utils::lib_printf(
        "Debug EXX k-grid check across Sigc KS build: max |delta v_exx| = %.12e Ha"
        " at spin %d, ik %d, state %d\n",
        max_abs_diff, max_spin + 1, max_kpoint + 1, max_state + 1);
}

const GwBandStateDebugSpec& get_gw_band_state_debug_spec()
{
    static const GwBandStateDebugSpec spec = []() {
        GwBandStateDebugSpec parsed;
        const char* env_value = std::getenv("LIBRPA_DEBUG_GW_BAND_STATE");
        if (env_value == nullptr)
        {
            return parsed;
        }

        std::string value(env_value);
        for (char& ch : value)
        {
            if (ch == ',' || ch == ':' || ch == ';')
            {
                ch = ' ';
            }
        }

        std::istringstream iss(value);
        int spin = 0;
        int kpoint = 0;
        int state = 0;
        if (!(iss >> spin >> kpoint >> state))
        {
            return parsed;
        }
        if (spin <= 0 || kpoint <= 0 || state <= 0)
        {
            return parsed;
        }

        parsed.enabled = true;
        parsed.spin = spin - 1;
        parsed.kpoint = kpoint - 1;
        parsed.state = state - 1;
        return parsed;
    }();
    return spec;
}

bool need_gw_band_state_debug_dump(const int spin, const int kpoint, const int state)
{
    const auto& spec = get_gw_band_state_debug_spec();
    return spec.enabled && spec.spin == spin && spec.kpoint == kpoint && spec.state == state;
}

std::vector<QpeIterationTracePoint> trace_qpe_solver_iterations(const LIBRPA::AnalyContPade& pade,
                                                                const double e_mf,
                                                                const double e_fermi,
                                                                const double vxc,
                                                                const double sigma_x,
                                                                const double thres = 1.0e-5)
{
    constexpr double escale = 0.1;
    constexpr int n_iter_max = 10000;

    std::vector<QpeIterationTracePoint> trace;
    trace.reserve(256);

    int n_iter = 0;
    double e_qp = e_mf;
    double diff = 1.0;
    cplxdb sigc{0.0, 0.0};

    while (n_iter++ < n_iter_max)
    {
        if (std::abs(diff) > 10.0 * thres)
        {
            e_qp += escale * diff;
            sigc = pade.get(static_cast<cplxdb>(e_qp - e_fermi));
            diff = e_mf - vxc + sigma_x + sigc.real() - e_qp;
            trace.push_back({n_iter, e_qp, diff, sigc});
            if (n_iter == n_iter_max - 1)
            {
                break;
            }
        }
        else
        {
            e_qp += escale * diff * 0.1;
            sigc = pade.get(static_cast<cplxdb>(e_qp - e_fermi));
            diff = e_mf - vxc + sigma_x + sigc.real() - e_qp;
            trace.push_back({n_iter, e_qp, diff, sigc});
            if (std::abs(diff) < thres || n_iter == n_iter_max - 1)
            {
                break;
            }
        }
    }

    return trace;
}

void dump_gw_band_state_debug(const int spin,
                              const int kpoint,
                              const int state,
                              const Vector3_Order<double>& kfrac,
                              const std::vector<cplxdb>& imagfreqs,
                              const std::vector<cplxdb>& sigc_state,
                              const LIBRPA::AnalyContPade& pade,
                              const double e_mf,
                              const double e_fermi,
                              const double vxc,
                              const double sigma_x,
                              const double e_qp,
                              const cplxdb& sigc_qp,
                              const int qpe_flag)
{
    using LIBRPA::envs::mpi_comm_global_h;
    if (!need_gw_band_state_debug_dump(spin, kpoint, state) || !mpi_comm_global_h.is_root())
    {
        return;
    }

    std::ostringstream prefix;
    prefix << "GW_band_state_debug_spin_" << spin + 1 << "_k_" << kpoint + 1 << "_state_"
           << state + 1;

    {
        std::ofstream ofs(prefix.str() + "_iw.dat");
        ofs << std::scientific << std::setprecision(15);
        ofs << "# spin=" << spin + 1 << " kpoint=" << kpoint + 1 << " state=" << state + 1
            << '\n';
        ofs << "# kfrac " << kfrac.x << ' ' << kfrac.y << ' ' << kfrac.z << '\n';
        ofs << "# e_mf_Ha " << e_mf << '\n';
        ofs << "# e_fermi_Ha " << e_fermi << '\n';
        ofs << "# vxc_Ha " << vxc << '\n';
        ofs << "# sigma_x_Ha " << sigma_x << '\n';
        ofs << "# e_qp_Ha " << e_qp << '\n';
        ofs << "# e_qp_eV " << e_qp * HA2EV << '\n';
        ofs << "# sigc_qp_real_Ha " << sigc_qp.real() << '\n';
        ofs << "# sigc_qp_imag_Ha " << sigc_qp.imag() << '\n';
        ofs << "# qpe_flag " << qpe_flag << '\n';
        ofs << "# omega_imag_Ha ReSigc_Ha ImSigc_Ha\n";
        for (std::size_t i = 0; i != imagfreqs.size() && i != sigc_state.size(); ++i)
        {
            ofs << imagfreqs[i].imag() << ' ' << sigc_state[i].real() << ' ' << sigc_state[i].imag()
                << '\n';
        }
    }

    const auto trace = trace_qpe_solver_iterations(pade, e_mf, e_fermi, vxc, sigma_x);
    {
        std::ofstream ofs(prefix.str() + "_iter.dat");
        ofs << std::scientific << std::setprecision(15);
        ofs << "# iter e_qp_Ha e_qp_eV ReSigc_Ha ImSigc_Ha residual_Ha\n";
        for (const auto& point : trace)
        {
            ofs << point.iter << ' ' << point.e_qp << ' ' << point.e_qp * HA2EV << ' '
                << point.sigc.real() << ' ' << point.sigc.imag() << ' ' << point.diff << '\n';
        }
    }

    const double half_width =
        std::max(15.0, 0.5 * std::abs(e_qp - e_mf) + 5.0);
    const double energy_min = std::min(e_mf, e_qp) - half_width;
    const double energy_max = std::max(e_mf, e_qp) + half_width;
    constexpr int n_samples = 801;
    {
        std::ofstream ofs(prefix.str() + "_scan.dat");
        ofs << std::scientific << std::setprecision(15);
        ofs << "# energy_Ha energy_eV omega_minus_ef_Ha ReSigc_Ha ImSigc_Ha residual_Ha\n";
        for (int isample = 0; isample != n_samples; ++isample)
        {
            const double alpha = (n_samples == 1) ? 0.0 : static_cast<double>(isample) / (n_samples - 1);
            const double energy = energy_min + (energy_max - energy_min) * alpha;
            const double omega = energy - e_fermi;
            const cplxdb sigc = pade.get(cplxdb{omega, 0.0});
            const double residual = e_mf - vxc + sigma_x + sigc.real() - energy;
            ofs << energy << ' ' << energy * HA2EV << ' ' << omega << ' ' << sigc.real() << ' '
                << sigc.imag() << ' ' << residual << '\n';
        }
    }
}

} // namespace

void task_g0w0_band(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;

    Profiler::start("g0w0_band", "G0W0 quasi-particle band structure calculation");

    if (mpi_comm_global_h.is_root())
    {
        const auto& debug_spec = get_gw_band_state_debug_spec();
        if (debug_spec.enabled)
        {
            lib_printf("Debug GW band-state dump enabled for spin %d, k-point %d, state %d\n",
                       debug_spec.spin + 1, debug_spec.kpoint + 1, debug_spec.state + 1);
        }
    }

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    // Preserve the loaded IBZ q-index order instead of iterating the sorted `irk_weight` map.
    vector<Vector3_Order<double>> qlist = klist;

    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU
        || need_full_cut_coulomb_for_abacus_symmetry())
    {
        if (need_full_cut_coulomb_for_abacus_symmetry() && mpi_comm_global_h.is_root())
        {
            lib_printf("ABACUS GW/EXX symmetry builds `V(R)` directly from the full IBZ operator;"
                       " switching to `read_Vq_full`\n");
        }
        // The ABACUS symmetry-aware `FT_Vq()` path needs the complete IBZ q-space operator on the
        // local rank before it can apply the q-star rotations and accumulate irreducible-sector
        // `V(R)`. The row-distributed cut-Coulomb reader does not provide that view.
        read_Vq_full(driver_params.input_dir, "coulomb_cut_", true);
    }
    else
    {
        // NOTE: local_atpair already set in the main.cpp.
        //       It can consists of distributed atom pairs of only upper half.
        //       Setup of local_atpair may be better to extracted as some util function,
        //       instead of in the main driver.
        read_Vq_row(driver_params.input_dir, "coulomb_cut_", Params::vq_threshold, local_atpair,
                    true);
    }
    Profiler::cease("read_vq_cut");

    Profiler::start("read_vxc", "Load DFT xc potential");
    std::vector<matrix> vxc;
    int flag_read_vxc = read_vxc(driver_params.input_dir + "vxc_out", vxc);
    Profiler::stop("read_vxc");

    if (flag_read_vxc != 0)
    {
        if (mpi_comm_global_h.myid == 0)
        {
            lib_printf("Error in reading Vxc on kgrid, task failed!\n");
        }
        Profiler::stop("g0w0_band");
        return;
    }
    // ================== EXX part: Compute Hexx(R) ==================
    Profiler::start("g0w0_exx", "Build exchange self-energy");
    auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
    {
        atpair_R_mat_t VR;
        if (Params::use_fullcoul_exx)
        {
            Profiler::start("ft_vq_full", "Fourier transform full Coulomb");
            VR = FT_Vq(Vq, get_full_bz_kpoint_count(), Rlist, true);
            Profiler::stop("ft_vq_full");
        }
        else
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            VR = FT_Vq(Vq_cut, get_full_bz_kpoint_count(), Rlist, true);
            Profiler::stop("ft_vq_cut");
        }

        Profiler::start("g0w0_exx_real_work");
        const auto& exx_cs = Params::use_shrink_abfs ? Cs_shrinked_data : Cs_data;
        if (Params::use_soc)
            exx.build<std::complex<double>>(exx_cs, Rlist, VR);
        else
            exx.build<double>(exx_cs, Rlist, VR);
        Profiler::stop("g0w0_exx_real_work");
    }
    Profiler::stop("g0w0_exx");
    std::flush(ofs_myid);

    // ================== GW part: Compute Sigma_c(R, iw)  ==================
    // Prepare time-frequency grids
    auto tfg =
        LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);
    
    LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, tfg, period);
    
    if (Params::band_continue)
    {
        s_g0w0.read_sigc(driver_params.input_dir + "librpa.d/", Rlist);
    }
    else
    {
        Chi0 chi0(meanfield, klist, tfg);
        chi0.gf_R_threshold = Params::gf_R_threshold;

        std::vector<double> epsmac_LF_imagfreq_re;
        if (Params::replace_w_head)
        {
            std::vector<double> omegas_dielect;
            std::vector<double> dielect_func;
            if (Params::option_dielect_func != 3 && Params::option_dielect_func != 4)
                read_dielec_func(driver_params.input_dir + "dielecfunc_out", omegas_dielect,
                                dielect_func);

            epsmac_LF_imagfreq_re = interpolate_dielec_func(Params::option_dielect_func, omegas_dielect,
                                                            dielect_func, chi0.tfg.get_freq_nodes());
        }

        chi0.set_input_dir(driver_params.input_dir);
        Profiler::start("chi0_build", "Build response function chi0");
        if (Params::use_shrink_chi)
        {
            chi0.build(Cs_data, Rlist, period, local_atpair, qlist, sinvS);
        }
        else
        {
            chi0.build(Cs_shrinked_data, Rlist, period, local_atpair, qlist, sinvS);
        }

        Profiler::stop("chi0_build");

        std::flush(ofs_myid);
        mpi_comm_global_h.barrier();

        if (Params::debug)
        {  // debug, check chi0
            char fn[80];
            for (const auto &chi0q : chi0.get_chi0_q())
            {
                const int ifreq = chi0.tfg.get_freq_index(chi0q.first);
                for (const auto &q_IJchi0 : chi0q.second)
                {
                    const int iq = std::distance(klist.begin(),
                                                std::find(klist.begin(), klist.end(), q_IJchi0.first));
                    for (const auto &I_Jchi0 : q_IJchi0.second)
                    {
                        const auto &I = I_Jchi0.first;
                        for (const auto &J_chi0 : I_Jchi0.second)
                        {
                            const auto &J = J_chi0.first;
                            sprintf(fn, "chi0fq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I, J,
                                    mpi_comm_global_h.myid);
                            print_complex_matrix_mm(J_chi0.second, Params::output_dir + "/" + fn,
                                                    1e-15);
                        }
                    }
                }
            }
        }

        Profiler::start("g0w0_wc", "Build screened interaction");
        vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(),
                                                        epsmac_LF_imagfreq_re.cend());
        map<double,
            atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
            Wc_freq_q;
        if (Params::use_scalapack_gw_wc)
        {
            if (Params::use_fullcoul_wc)
            {
                Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq, epsmac_LF_imagfreq);
            }
            else
            {
                Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
            }
        }
        else
        {
            if (Params::use_fullcoul_wc)
            {
                Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq, epsmac_LF_imagfreq);
            }
            else
            {
                Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
            }
        }
        Profiler::stop("g0w0_wc");
        if (Params::debug)
        {  // debug, check Wc(q, iw)
            char fn[80];
            for (const auto &Wc : Wc_freq_q)
            {
                const int ifreq = chi0.tfg.get_freq_index(Wc.first);
                for (const auto &I_JqWc : Wc.second)
                {
                    const auto &I = I_JqWc.first;
                    for (const auto &J_qWc : I_JqWc.second)
                    {
                        const auto &J = J_qWc.first;
                        for (const auto &q_Wc : J_qWc.second)
                        {
                            const int iq = std::distance(
                                klist.begin(), std::find(klist.begin(), klist.end(), q_Wc.first));
                            sprintf(fn, "Wcfq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I, J,
                                    mpi_comm_global_h.myid);
                            print_matrix_mm_file(q_Wc.second, Params::output_dir + "/" + fn, 1e-15);
                        }
                    }
                }
            }
        }

        if (Params::use_shrink_abfs)
        {
            Profiler::start("read_shrink_sinvS_fold", "Load shrink transformation");
            // change atom_mu: number of {Mu,mu} in the later calculations
            read_shrink_sinvS(driver_params.input_dir, "shrink_sinvS_", sinvS);
            Profiler::stop("read_shrink_sinvS_fold");
        }

        Profiler::start("g0w0_sigc_IJ", "Build real-space correlation self-energy");

        if (Params::use_soc)
            s_g0w0.build_spacetime<std::complex<double>>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        else
            s_g0w0.build_spacetime<double>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);

        Profiler::stop("g0w0_sigc_IJ");
        std::flush(ofs_myid);
    }

    // ================== Compute the QP energies on k-grid ==================
    Profiler::start("g0w0_exx_ks_kgrid");
    exx.build_KS_kgrid();
    Profiler::stop("g0w0_exx_ks_kgrid");
    ExxDiagMap exx_diag_before_sigc;
    if (enable_exx_kgrid_sigc_probe() && mpi_comm_global_h.is_root())
    {
        // Preserve the freshly projected EXX diagonal before the later Sigc KS build so that
        // any accidental in-memory corruption can be detected in a single debug rerun.
        exx_diag_before_sigc = exx.Eexx;
        write_exx_kgrid_debug_files(exx_diag_before_sigc, "EXX_kgrid_pre_sigc_spin_");
    }
    Profiler::start("g0w0_sigc_ks_kgrid");
    s_g0w0.build_sigc_matrix_KS_kgrid();
    Profiler::stop("g0w0_sigc_ks_kgrid");
    if (enable_exx_kgrid_sigc_probe() && mpi_comm_global_h.is_root())
    {
        write_exx_kgrid_debug_files(exx.Eexx, "EXX_kgrid_post_sigc_spin_");
        report_exx_diag_debug_difference(exx_diag_before_sigc, exx.Eexx);
    }

    // imaginary freqencies for analytic continuation
    std::vector<cplxdb> imagfreqs;
    for (const auto &freq : tfg.get_freq_nodes())
    {
        imagfreqs.push_back(cplxdb{0.0, freq});
    }

    Profiler::start("g0w0_solve_qpe_kgrid", "Solve quasi-particle equation");
    if (mpi_comm_global_h.is_root())
    {
        std::cout << "Solving quasi-particle equation for states at k-points of regular grid\n";
    }

    // TODO: parallelize analytic continuation and QPE solver among tasks
    if (mpi_comm_global_h.is_root())
    {
        map<int, map<int, map<int, double>>> e_qp_all;
        map<int, map<int, map<int, cplxdb>>> sigc_all;
        const auto full_k_members =
            can_expand_gw_output_to_full_bz()
                ? LIBRPA::build_abacus_full_kpoint_member_list(LIBRPA::abacus_symmetry_ctx,
                                                               kfrac_list)
                : std::vector<LIBRPA::AbacusFullKpointMemberEntry>{};
        const int occupation_scale =
            full_k_members.empty() ? meanfield.get_n_kpoints() : get_full_bz_kpoint_count();
        const auto efermi = meanfield.get_efermi();
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state);
                    const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state];
                    const auto &vxc_state = vxc[i_spin](i_kpoint, i_state);
                    std::vector<cplxdb> sigc_state;
                    for (const auto &freq : tfg.get_freq_nodes())
                    {
                        sigc_state.push_back(sigc_sk.at(freq)(i_state, i_state));
                    }
                    LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_state);
                    double e_qp;
                    cplxdb sigc;
                    int flag_qpe_solver = LIBRPA::qpe_solver_pade_self_consistent(
                        pade, eks_state, efermi, vxc_state, exx_state, e_qp, sigc);
                    dump_gw_band_state_debug(i_spin, i_kpoint, i_state, kfrac_list[i_kpoint],
                                             imagfreqs, sigc_state, pade, eks_state, efermi,
                                             vxc_state, exx_state, e_qp, sigc, flag_qpe_solver);
                    if (flag_qpe_solver == 0)
                    {
                        e_qp_all[i_spin][i_kpoint][i_state] = e_qp;
                        sigc_all[i_spin][i_kpoint][i_state] = sigc;
                    }
                    else
                    {
                        printf("Warning! QPE solver failed for spin %d, kpoint %d, state %d\n",
                               i_spin + 1, i_kpoint + 1, i_state + 1);
                        e_qp_all[i_spin][i_kpoint][i_state] =
                            std::numeric_limits<double>::quiet_NaN();
                        sigc_all[i_spin][i_kpoint][i_state] =
                            std::numeric_limits<cplxdb>::quiet_NaN();
                    }
                }
            }
        }

        // display results
        const std::string banner(124, '-');
        printf("Printing quasi-particle energy [unit: eV]\n\n");
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            if (!full_k_members.empty())
            {
                for (int ifull = 0; ifull != static_cast<int>(full_k_members.size()); ++ifull)
                {
                    const auto& member = full_k_members[static_cast<std::size_t>(ifull)];
                    const int i_kpoint = member.ik_ibz;
                    const auto& k = member.k_bz;
                    printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1, ifull + 1,
                           k.x, k.y, k.z);
                    printf("%124s\n", banner.c_str());
                    printf("%5s %16s %16s %16s %16s %16s %16s %16s\n", "State", "occ", "e_mf",
                           "v_xc", "v_exx", "ReSigc", "ImSigc", "e_qp");
                    printf("%124s\n", banner.c_str());
                    for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                    {
                        const auto& occ_state =
                            meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                        const auto& eks_state =
                            meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                        const auto& exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                        const auto& vxc_state = vxc[i_spin](i_kpoint, i_state) * HA2EV;
                        const auto& resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                        const auto& imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                        const auto& eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                        printf("%5d %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f\n",
                               i_state + 1, occ_state, eks_state, vxc_state, exx_state, resigc,
                               imsigc, eqp);
                    }
                    printf("\n");
                }
                continue;
            }

            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_list[i_kpoint];
                printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1, i_kpoint + 1,
                       k.x, k.y, k.z);
                printf("%124s\n", banner.c_str());
                printf("%5s %16s %16s %16s %16s %16s %16s %16s\n", "State", "occ", "e_mf", "v_xc",
                       "v_exx", "ReSigc", "ImSigc", "e_qp");
                printf("%124s\n", banner.c_str());
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &occ_state =
                        meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                    const auto &eks_state =
                        meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                    const auto &vxc_state = vxc[i_spin](i_kpoint, i_state) * HA2EV;
                    const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                    const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                    const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                    printf("%5d %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f\n", i_state + 1,
                           occ_state, eks_state, vxc_state, exx_state, resigc, imsigc, eqp);
                }
                printf("\n");
            }
        }
        if (Params::output_energy_qp) {
            std::ofstream ofs("energy_qp");
            ofs << "  state     occ_num        e_gs(Ha)        e_qp(Ha)"<<std::endl;
            ofs << banner << std::endl;
            for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
            {
                if (!full_k_members.empty())
                {
                    for (int ifull = 0; ifull != static_cast<int>(full_k_members.size()); ++ifull)
                    {
                        const auto& member = full_k_members[static_cast<std::size_t>(ifull)];
                        const int i_kpoint = member.ik_ibz;
                        const auto& k = member.k_bz;
                        ofs << " K_point " << ifull + 1 << " :" << std::setw(10)
                            << std::setprecision(7) << k.x << std::setw(10) << k.y
                            << std::setw(10) << k.z << std::setw(10) << " Spin " << i_spin + 1
                            << std::endl;
                        ofs << banner << std::endl;
                        for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                        {
                            const auto& occ_state =
                                meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                            const auto& eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state);
                            const auto& eqp = e_qp_all[i_spin][i_kpoint][i_state];
                            ofs << std::setw(7) << i_state + 1 << std::setw(10)
                                << std::setprecision(5) << occ_state << std::setw(18)
                                << std::setprecision(10) << eks_state << std::setw(18)
                                << std::setprecision(10) << eqp << std::endl;
                        }
                        ofs << banner << std::endl;
                        ofs << std::endl;
                    }
                    continue;
                }

                for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
                {
                    const auto &k = kfrac_list[i_kpoint];
                    ofs << " K_point " << i_kpoint + 1 << " :" << std::setw(10)
                        << std::setprecision(7) << k.x << std::setw(10) << k.y << std::setw(10)
                        << k.z << std::setw(10) <<" Spin " << i_spin + 1 << std::endl;
                    ofs << banner << std::endl;
                    for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                    {
                        const auto &occ_state =
                            meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                        const auto &eks_state =
                            meanfield.get_eigenvals()[i_spin](i_kpoint, i_state);
                        const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state];
                        ofs << std::setw(7) << i_state + 1 << std::setw(10) << std::setprecision(5)
                            << occ_state << std::setw(18) << std::setprecision(10) << eks_state
                            << std::setw(18) << std::setprecision(10) << eqp << std::endl;
                    }
                    ofs << banner << std::endl;
                    ofs << std::endl;
                }
            }
            ofs.close();
        }
        // output for HamGNN
        if (Params::output_hamgnn)
        {
            std::ofstream ofs_hamgnn;
            std::stringstream fn;
            fn << "GW_gridk_hamgnn" << ".dat";
            ofs_hamgnn.open(fn.str());
            ofs_hamgnn << std::fixed;

            for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
            {
                if (!full_k_members.empty())
                {
                    for (int ifull = 0; ifull != static_cast<int>(full_k_members.size()); ++ifull)
                    {
                        const auto& member = full_k_members[static_cast<std::size_t>(ifull)];
                        const int i_kpoint = member.ik_ibz;
                        const auto& k = member.k_bz;
                        ofs_hamgnn << "spin " << i_spin << ", k-point " << ifull + 1 << ": ("
                                   << std::setw(10) << std::setprecision(7) << k.x << ", "
                                   << std::setw(10) << std::setprecision(7) << k.y << ", "
                                   << std::setw(10) << std::setprecision(7) << k.z << ")" << std::endl;
                        ofs_hamgnn << std::setw(5) << "State"
                                   << " " << std::setw(16) << "occ"
                                   << " " << std::setw(25) << "e_qp(eV)" << std::endl;
                        for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                        {
                            const auto& occ_state =
                                meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                            const auto& eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                            ofs_hamgnn << std::setw(5) << i_state + 1 << " " << std::setw(16)
                                       << std::setprecision(5) << occ_state << " " << std::setw(25)
                                       << std::setprecision(8) << eqp << std::endl;
                        }
                        printf("\n");
                    }
                    continue;
                }

                for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
                {
                    const auto &k = kfrac_list[i_kpoint];
                    ofs_hamgnn << "spin " << i_spin << ", k-point " << i_kpoint << ": ("
                               << std::setw(10) << std::setprecision(7) << k.x << ", "
                               << std::setw(10) << std::setprecision(7) << k.y << ", "
                               << std::setw(10) << std::setprecision(7) << k.z << ")" << std::endl;
                    ofs_hamgnn << std::setw(5) << "State"
                               << " " << std::setw(16) << "occ"
                               << " " << std::setw(25) << "e_qp(eV)" << std::endl;
                    for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                    {
                        const auto &occ_state =
                            meanfield.get_weight()[i_spin](i_kpoint, i_state) * occupation_scale;
                        const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                        ofs_hamgnn << std::setw(5) << i_state + 1 << " " << std::setw(16)
                                   << std::setprecision(5) << occ_state << " " << std::setw(25)
                                   << std::setprecision(8) << eqp << std::endl;
                    }
                    printf("\n");
                }
            }
        }
    }
    Profiler::stop("g0w0_solve_qpe_kgrid");

    // ================== Compute the QP energies on k-path ==================
    // Reset k-space EXX and Sigmac matrices to avoid warning from internal reset
    exx.reset_kspace();
    s_g0w0.reset_kspace();

    /* Below we handle the band k-points data
     * First load the information of k-points along the k-path */
    Profiler::start("g0w0_band_load_kpath");
    int n_basis_band, n_states_band, n_spin_band;
    int flag;
    std::vector<Vector3_Order<double>> kfrac_band =
        read_band_kpath_info(driver_params.input_dir + "band_kpath_info", n_basis_band,
                             n_states_band, n_spin_band, flag);
    Profiler::stop("g0w0_band_load_kpath");

    if (flag == 0)
    {
        // Success
        if (mpi_comm_global_h.is_root())
        {
            std::cout << "Band k-points to compute:" << std::endl;
            for (int ik = 0; ik < kfrac_band.size(); ik++)
            {
                const auto &k = kfrac_band[ik];
                lib_printf("%5d %12.7f %12.7f %12.7f\n", ik + 1, k.x, k.y, k.z);
            }
        }
        mpi_comm_global_h.barrier();
    }
    else
    {
        if (mpi_comm_global_h.is_root())
        {
            const auto fn = driver_params.input_dir + "band_kpath_info";
            std::cout << "Warning! Failed to read " << fn << " , skip band structure" << std::endl
                      << std::endl;
        }
        mpi_comm_global_h.barrier();
        Profiler::stop("g0w0_band");
        return;
    }

    Profiler::start("g0w0_band_load_band_mf", "Read eigen solutions at band kpoints");
    auto meanfield_band = read_meanfield_band(driver_params.input_dir, n_basis_band, n_states_band,
                                              n_spin_band, kfrac_band.size());

    /* Set the same Fermi energy as in SCF */
    meanfield_band.get_efermi() = meanfield.get_efermi();
    Profiler::stop("g0w0_band_load_band_mf");

    Profiler::start("g0w0_sigx_rotate_KS");
    exx.build_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);
    Profiler::stop("g0w0_sigx_rotate_KS");
    std::flush(ofs_myid);

    Profiler::start("g0w0_sigc_rotate_KS");
    s_g0w0.build_sigc_matrix_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);
    Profiler::stop("g0w0_sigc_rotate_KS");
    std::flush(ofs_myid);

    Profiler::start("read_vxc", "Load DFT xc potential");
    auto vxc_band =
        read_vxc_band(driver_params.input_dir, n_states_band, n_spin_band, kfrac_band.size());
    Profiler::stop("read_vxc");
    std::flush(ofs_myid);

    Profiler::start("g0w0_solve_band_qpe", "Solve quasi-particle equation");
    if (mpi_comm_global_h.is_root())
    {
        std::cout << "Solving quasi-particle equation\n";
    }

    // TODO: parallelize analytic continuation and QPE solver among tasks
    if (mpi_comm_global_h.is_root())
    {
        const auto &mf = meanfield_band;
        map<int, map<int, map<int, double>>> e_qp_all;
        map<int, map<int, map<int, cplxdb>>> sigc_all;
        const auto efermi = mf.get_efermi();
        for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < mf.get_n_kpoints(); i_kpoint++)
            {
                const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
                for (int i_state = 0; i_state < mf.get_n_bands(); i_state++)
                {
                    const auto &eks_state = mf.get_eigenvals()[i_spin](i_kpoint, i_state);
                    const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state];
                    const auto &vxc_state = vxc_band[i_spin](i_kpoint, i_state);
                    std::vector<cplxdb> sigc_state;
                    for (const auto &freq : tfg.get_freq_nodes())
                    {
                        sigc_state.push_back(sigc_sk.at(freq)(i_state, i_state));
                    }
                    LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_state);
                    double e_qp;
                    cplxdb sigc;
                    int flag_qpe_solver = LIBRPA::qpe_solver_pade_self_consistent(
                        pade, eks_state, efermi, vxc_state, exx_state, e_qp, sigc);
                    dump_gw_band_state_debug(i_spin, i_kpoint, i_state, kfrac_band[i_kpoint],
                                             imagfreqs, sigc_state, pade, eks_state, efermi,
                                             vxc_state, exx_state, e_qp, sigc, flag_qpe_solver);
                    if (flag_qpe_solver == 0)
                    {
                        e_qp_all[i_spin][i_kpoint][i_state] = e_qp;
                        sigc_all[i_spin][i_kpoint][i_state] = sigc;
                    }
                    else
                    {
                        printf("Warning! QPE solver failed for spin %d, kpoint %d, state %d\n",
                               i_spin + 1, i_kpoint + 1, i_state + 1);
                        e_qp_all[i_spin][i_kpoint][i_state] =
                            std::numeric_limits<double>::quiet_NaN();
                        sigc_all[i_spin][i_kpoint][i_state] =
                            std::numeric_limits<cplxdb>::quiet_NaN();
                    }
                }
            }
        }
        // output bandgap
        double gw_bandgap = 0.0;
        double gw_valence = -1.e10;
        double gw_conduct = 1.e10;
        double exx_bandgap = 0.0;
        double exx_valence = -1.e10;
        double exx_conduct = 1.e10;
        double dft_bandgap = 0.0;
        double dft_valence = -1.e10;
        double dft_conduct = 1.e10;
        int ik_val_gw = 0;
        int ik_cond_gw = 0;
        int ik_val_exx = 0;
        int ik_cond_exx = 0;
        int ik_val_dft = 0;
        int ik_cond_dft = 0;
        int nocc = 0;
        auto &wg = meanfield.get_weight()[0];
        for (int i = 0; i != wg.size; i++)
        {
            if (wg.c[i] == 0.)
            {
                nocc = i;
                break;
            }
        }
        lib_printf("Bands of occupation: %4d \n", nocc);

        // display results
        for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
        {
            std::ofstream ofs_ks;
            std::ofstream ofs_hf;
            std::ofstream ofs_gw;
            std::stringstream fn;

            fn << "GW_band_spin_" << i_spin + 1 << ".dat";
            ofs_gw.open(fn.str());

            fn.str("");
            fn.clear();
            fn << "EXX_band_spin_" << i_spin + 1 << ".dat";
            ofs_hf.open(fn.str());

            fn.str("");
            fn.clear();
            fn << "KS_band_spin_" << i_spin + 1 << ".dat";
            ofs_ks.open(fn.str());

            ofs_gw << std::fixed;
            ofs_hf << std::fixed;
            ofs_ks << std::fixed;

            for (int i_kpoint = 0; i_kpoint < mf.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_band[i_kpoint];
                ofs_ks << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                       << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                       << std::setprecision(7) << k.z;
                ofs_gw << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                       << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                       << std::setprecision(7) << k.z;
                ofs_hf << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                       << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                       << std::setprecision(7) << k.z;
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &occ_state = mf.get_weight()[i_spin](i_kpoint, i_state) * mf.get_n_kpoints();
                    const auto &eks_state = mf.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                    const auto &vxc_state = vxc_band[i_spin](i_kpoint, i_state) * HA2EV;
                    // const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                    // const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                    const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                    ofs_ks << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15)
                           << std::setprecision(5) << eks_state;
                    ofs_gw << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15)
                           << std::setprecision(5) << eqp;
                    ofs_hf << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15)
                           << std::setprecision(5) << eks_state - vxc_state + exx_state;

                    // output GW bandgap
                    if (i_state == nocc - 1)  // HOMO
                    {
                        if (eqp > gw_valence)
                        {
                            gw_valence = eqp;
                            ik_val_gw = i_kpoint;
                        }
                    }
                    else if (i_state == nocc)  // LUMO
                    {
                        if (eqp < gw_conduct)
                        {
                            gw_conduct = eqp;
                            ik_cond_gw = i_kpoint;
                        }
                    }
                    // output EXX bandgap
                    if (i_state == nocc - 1)  // HOMO
                    {
                        if (eks_state - vxc_state + exx_state > exx_valence)
                        {
                            exx_valence = eks_state - vxc_state + exx_state;
                            ik_val_exx = i_kpoint;
                        }
                    }
                    else if (i_state == nocc)  // LUMO
                    {
                        if (eks_state - vxc_state + exx_state < exx_conduct)
                        {
                            exx_conduct = eks_state - vxc_state + exx_state;
                            ik_cond_exx = i_kpoint;
                        }
                    }
                    // output DFT bandgap
                    if (i_state == nocc - 1)  // HOMO
                    {
                        if (eks_state > dft_valence)
                        {
                            dft_valence = eks_state;
                            ik_val_dft = i_kpoint;
                        }
                    }
                    else if (i_state == nocc)  // LUMO
                    {
                        if (eks_state < dft_conduct)
                        {
                            dft_conduct = eks_state;
                            ik_cond_dft = i_kpoint;
                        }
                    }
                }
                ofs_gw << "\n";
                ofs_hf << "\n";
                ofs_ks << "\n";
            }
        }
        gw_bandgap = gw_conduct - gw_valence;
        exx_bandgap = exx_conduct - exx_valence;
        dft_bandgap = dft_conduct - dft_valence;
        const auto &k_val_gw = kfrac_band[ik_val_gw];
        const auto &k_cond_gw = kfrac_band[ik_cond_gw];
        printf("GW VBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_val_gw + 1, k_val_gw.x, k_val_gw.y,
               k_val_gw.z);
        printf("GW CBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_cond_gw + 1, k_cond_gw.x,
               k_cond_gw.y, k_cond_gw.z);
        lib_printf("GW bandgap(eV): %12.7f \n", gw_bandgap);
        const auto &k_val_exx = kfrac_band[ik_val_exx];
        const auto &k_cond_exx = kfrac_band[ik_cond_exx];
        printf("EXX VBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_val_exx + 1, k_val_exx.x,
               k_val_exx.y, k_val_exx.z);
        printf("EXX CBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_cond_exx + 1, k_cond_exx.x,
               k_cond_exx.y, k_cond_exx.z);
        lib_printf("EXX bandgap(eV): %12.7f \n", exx_bandgap);
        const auto &k_val_dft = kfrac_band[ik_val_dft];
        const auto &k_cond_dft = kfrac_band[ik_cond_dft];
        printf("DFT VBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_val_dft + 1, k_val_dft.x,
               k_val_dft.y, k_val_dft.z);
        printf("DFT CBM: k-point %4d: (%.5f, %.5f, %.5f) \n", ik_cond_dft + 1, k_cond_dft.x,
               k_cond_dft.y, k_cond_dft.z);
        lib_printf("DFT bandgap(eV): %12.7f \n", dft_bandgap);
    }
    Profiler::stop("g0w0_solve_band_qpe");

    Profiler::stop("g0w0_band");
}
