#include "task_topo_gw_band.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "analycont.h"
#include "chi0.h"
#include "constants.h"
#include "convert_csc.h"
#include "coulmat.h"
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"
#include "exx.h"
#include "gw.h"
#include "matrix_m.h"
#include "meanfield.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "read_data.h"
#include "ri.h"
#include "utils_timefreq.h"

namespace
{
using cplxdb = std::complex<double>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSigcTruncInScale = 1.0e8;
constexpr double kAcTruncOutScale = 1.0e6;
constexpr double kPadeThreshold = 1.0e-6;
constexpr double kTinyLinkThreshold = 1.0e-14;

struct SpinTopologySummary
{
    int nocc = 0;
    double chern_raw = 0.0;
    long long chern_rounded = 0;
    double min_link_abs = std::numeric_limits<double>::infinity();
    double max_sigma_antiherm = 0.0;
    double min_gap_ha = std::numeric_limits<double>::infinity();
};

struct TopologyVxcBundle
{
    std::map<int, std::map<int, Matz>> matrices;
    std::string mode = "full_matrix";
};

int band_mesh_index(const int ik1, const int ik2, const int nk1)
{
    return ik2 * nk1 + ik1;
}

void ensure_dir(const std::string &dir)
{
    std::system(("mkdir -p " + dir).c_str());
}

cplxdb round_complex(const cplxdb &z, const double scale)
{
    return cplxdb{
        std::round(z.real() * scale) / scale,
        std::round(z.imag() * scale) / scale,
    };
}

cplxdb sanitize_complex(const cplxdb &z)
{
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        return cplxdb{0.0, 0.0};
    return z;
}

Matz extract_occupied_columns(const Matz &eigvec_ks, const int nocc)
{
    Matz occ(eigvec_ks.nr(), nocc, MAJOR::COL);
    for (int ib = 0; ib < eigvec_ks.nr(); ++ib)
        for (int iocc = 0; iocc < nocc; ++iocc)
            occ(ib, iocc) = eigvec_ks(ib, iocc);
    return occ;
}

double max_antihermitian_residual(const Matz &mat)
{
    double max_residual = 0.0;
    for (int i = 0; i < mat.nr(); ++i)
    {
        for (int j = 0; j < mat.nc(); ++j)
        {
            const double residual = std::abs(mat(i, j) - std::conj(mat(j, i)));
            max_residual = std::max(max_residual, residual);
        }
    }
    return max_residual;
}

std::map<int, std::map<int, Matz>> build_band_ks_hamiltonian(const MeanField &meanfield_band)
{
    std::map<int, std::map<int, Matz>> h_ks_band;
    const int n_spins = meanfield_band.get_n_spins();
    const int n_kpoints = meanfield_band.get_n_kpoints();
    const int n_bands = meanfield_band.get_n_bands();
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz h_ks(n_bands, n_bands, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
                h_ks(ib, ib) = meanfield_band.get_eigenvals()[ispin](ikpt, ib);
            h_ks_band[ispin][ikpt] = h_ks;
        }
    }
    return h_ks_band;
}

std::map<int, std::map<int, Matz>> read_vxc_band_matrices(
    const std::string &input_dir, const int n_spins, const int n_kpoints, const int n_bands)
{
    std::map<int, std::map<int, Matz>> vxc_band;
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            std::map<std::string, Matz> arrays_band;
            std::string key_vxc_band;
            std::ostringstream oss_vxc_band;
            oss_vxc_band << input_dir << "band_vxc_mat_spin_" << (ispin + 1) << "_k_"
                         << std::setw(5) << std::setfill('0') << (ikpt + 1) << ".csc";

            if (!convert_csc(oss_vxc_band.str(), arrays_band, key_vxc_band))
            {
                throw std::runtime_error("Failed to parse " + oss_vxc_band.str());
            }
            if (arrays_band.count(key_vxc_band) == 0)
            {
                throw std::runtime_error("Missing parsed matrix in " + oss_vxc_band.str());
            }

            const auto &mat = arrays_band.at(key_vxc_band);
            if (mat.nr() != n_bands || mat.nc() != n_bands)
            {
                std::ostringstream err;
                err << "band_vxc matrix dimension mismatch in " << oss_vxc_band.str()
                    << ": expected " << n_bands << "x" << n_bands
                    << ", got " << mat.nr() << "x" << mat.nc();
                throw std::runtime_error(err.str());
            }
            vxc_band[ispin][ikpt] = mat;
        }
    }
    return vxc_band;
}

std::string make_band_vxc_matrix_filename(
    const std::string &input_dir, const int ispin, const int ikpt)
{
    std::ostringstream oss_vxc_band;
    oss_vxc_band << input_dir << "band_vxc_mat_spin_" << (ispin + 1) << "_k_"
                 << std::setw(5) << std::setfill('0') << (ikpt + 1) << ".csc";
    return oss_vxc_band.str();
}

bool band_vxc_matrix_exists(const std::string &input_dir, const int ispin, const int ikpt)
{
    std::ifstream ifs(make_band_vxc_matrix_filename(input_dir, ispin, ikpt));
    return ifs.good();
}

TopologyVxcBundle read_topology_vxc(
    const std::string &input_dir, const int n_spins, const int n_kpoints, const int n_bands)
{
    TopologyVxcBundle bundle;
    const auto matrix_input_dir =
        resolve_input_dir_with_pyatb_fallback(input_dir, "band_vxc_mat_spin_1_k_00001.csc");
    if (band_vxc_matrix_exists(matrix_input_dir, 0, 0))
    {
        bundle.matrices = read_vxc_band_matrices(matrix_input_dir, n_spins, n_kpoints, n_bands);
        bundle.mode = "full_matrix";
        return bundle;
    }

    if (!Params::topology_allow_diag_vxc_fallback)
    {
        std::ostringstream err;
        err << "Missing " << make_band_vxc_matrix_filename(matrix_input_dir, 0, 0)
            << ". topo_gw_band needs full band_vxc_mat matrices for the exact topological "
               "Hamiltonian. If you want to use the ABACUS-style diagonal approximation from "
               "band_vxc_k_*.txt, set topology_allow_diag_vxc_fallback = true explicitly.";
        throw std::runtime_error(err.str());
    }

    const auto diag_input_dir =
        resolve_input_dir_with_pyatb_fallback(input_dir, "band_vxc_k_00001.txt");
    const auto vxc_diag = read_vxc_band(diag_input_dir, n_bands, n_spins, n_kpoints);
    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz vxc_here(n_bands, n_bands, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
                vxc_here(ib, ib) = vxc_diag[ispin](ikpt, ib);
            bundle.matrices[ispin][ikpt] = vxc_here;
        }
    }
    bundle.mode = "diag_expectation_fallback";
    return bundle;
}

Matz build_sigma_c_i0_matrix(
    const std::map<double, Matz> &sigc_sk,
    const std::vector<double> &freq_nodes,
    const std::vector<cplxdb> &imagfreqs,
    const int n_bands,
    double &antiherm_max)
{
    Matz sigma0_raw(n_bands, n_bands, MAJOR::COL);

    for (int irow = 0; irow < n_bands; ++irow)
    {
        for (int icol = 0; icol < n_bands; ++icol)
        {
            std::vector<cplxdb> sigc_mn;
            double max_magnitude = 0.0;
            for (const auto &freq : freq_nodes)
            {
                auto value = sigc_sk.at(freq)(irow, icol);
                value = round_complex(value, kSigcTruncInScale);
                max_magnitude = std::max(max_magnitude, std::abs(value));
                sigc_mn.push_back(value);
            }

            cplxdb sigma0 = cplxdb{0.0, 0.0};
            if (max_magnitude > kPadeThreshold)
            {
                try
                {
                    LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);
                    sigma0 = pade.get(0.0);
                }
                catch (...)
                {
                    if (!sigc_mn.empty())
                        sigma0 = sigc_mn.front();
                }
            }
            else if (!sigc_mn.empty())
            {
                sigma0 = sigc_mn.front();
            }

            sigma0 = sanitize_complex(round_complex(sigma0, kAcTruncOutScale));
            sigma0_raw(irow, icol) = sigma0;
        }
    }

    antiherm_max = max_antihermitian_residual(sigma0_raw);
    return 0.5 * (sigma0_raw + transpose(sigma0_raw, true));
}

Matz construct_topological_hamiltonian(
    const Matz &h_ks,
    const Matz &vxc_band,
    const Matz &hexx_band,
    const Matz &sigma0,
    const bool shift_mu,
    const double mu)
{
    Matz h_top = h_ks - vxc_band + hexx_band + sigma0;
    if (shift_mu)
    {
        for (int ib = 0; ib < h_top.nr(); ++ib)
            h_top(ib, ib) -= mu;
    }
    return 0.5 * (h_top + transpose(h_top, true));
}

int infer_nocc(const MeanField &meanfield_band, const int ispin)
{
    if (Params::topology_nocc > 0)
        return Params::topology_nocc;

    const int n_bands = meanfield_band.get_n_bands();
    const int n_kpoints = meanfield_band.get_n_kpoints();
    const double mu = meanfield_band.get_efermi();
    const double tol = 1.0e-8;

    auto count_occ = [&](const int ikpt) {
        int nocc = 0;
        for (int ib = 0; ib < n_bands; ++ib)
        {
            if (meanfield_band.get_eigenvals()[ispin](ikpt, ib) < mu + tol)
                ++nocc;
        }
        return nocc;
    };

    const int nocc_ref = count_occ(0);
    for (int ikpt = 1; ikpt < n_kpoints; ++ikpt)
    {
        const int nocc_here = count_occ(ikpt);
        if (nocc_here != nocc_ref)
        {
            std::ostringstream err;
            err << "topo_gw_band requires an insulating band manifold with fixed occupied count;"
                << " spin " << (ispin + 1)
                << " has nocc=" << nocc_ref << " at k=1 but nocc=" << nocc_here
                << " at k=" << (ikpt + 1);
            throw std::runtime_error(err.str());
        }
    }

    return nocc_ref;
}

cplxdb normalized_link(const Matz &overlap, double &det_abs)
{
    const cplxdb det = get_determinant(overlap);
    det_abs = std::abs(det);
    if (det_abs < kTinyLinkThreshold)
        return cplxdb{1.0, 0.0};
    return det / det_abs;
}

double compute_fhs_chern(
    const std::vector<Matz> &eigvecs_ks,
    const int nk1,
    const int nk2,
    const int nocc,
    double &min_link_abs)
{
    std::vector<Matz> occ_states(eigvecs_ks.size());
    for (size_t ik = 0; ik < eigvecs_ks.size(); ++ik)
        occ_states[ik] = extract_occupied_columns(eigvecs_ks[ik], nocc);

    min_link_abs = std::numeric_limits<double>::infinity();
    double total_phase = 0.0;
    for (int ik2 = 0; ik2 < nk2; ++ik2)
    {
        for (int ik1 = 0; ik1 < nk1; ++ik1)
        {
            const int ik = band_mesh_index(ik1, ik2, nk1);
            const int ik_x = band_mesh_index((ik1 + 1) % nk1, ik2, nk1);
            const int ik_y = band_mesh_index(ik1, (ik2 + 1) % nk2, nk1);
            const int ik_xy = band_mesh_index((ik1 + 1) % nk1, (ik2 + 1) % nk2, nk1);

            double abs_x = 0.0;
            double abs_y = 0.0;
            double abs_x_y = 0.0;
            double abs_y_x = 0.0;

            const cplxdb u_x = normalized_link(transpose(occ_states[ik], true) * occ_states[ik_x], abs_x);
            const cplxdb u_y = normalized_link(transpose(occ_states[ik], true) * occ_states[ik_y], abs_y);
            const cplxdb u_x_y = normalized_link(transpose(occ_states[ik_y], true) * occ_states[ik_xy], abs_x_y);
            const cplxdb u_y_x = normalized_link(transpose(occ_states[ik_x], true) * occ_states[ik_xy], abs_y_x);

            min_link_abs = std::min(min_link_abs, abs_x);
            min_link_abs = std::min(min_link_abs, abs_y);
            min_link_abs = std::min(min_link_abs, abs_x_y);
            min_link_abs = std::min(min_link_abs, abs_y_x);

            const cplxdb plaquette = u_x * u_y_x * std::conj(u_x_y) * std::conj(u_y);
            total_phase += std::arg(plaquette);
        }
    }

    return total_phase / (2.0 * kPi);
}

void write_band_spectrum(
    const std::string &filename,
    const std::vector<Vector3_Order<double>> &kfrac_band,
    const std::vector<std::vector<double>> &eigvals_ha)
{
    std::ofstream ofs(filename);
    ofs << std::fixed;
    for (size_t ik = 0; ik < kfrac_band.size(); ++ik)
    {
        ofs << std::setw(6) << (ik + 1)
            << std::setw(16) << std::setprecision(8) << kfrac_band[ik].x
            << std::setw(16) << std::setprecision(8) << kfrac_band[ik].y
            << std::setw(16) << std::setprecision(8) << kfrac_band[ik].z;
        for (const auto &eval : eigvals_ha[ik])
            ofs << std::setw(18) << std::setprecision(8) << (eval * HA2EV);
        ofs << "\n";
    }
}

void write_topology_summary(
    const std::string &filename,
    const int nk1,
    const int nk2,
    const int n_kpoints,
    const std::string &vxc_mode,
    const bool shift_mu,
    const double mu,
    const std::vector<SpinTopologySummary> &spin_summaries)
{
    double total_chern_raw = 0.0;
    long long total_chern_rounded = 0;
    for (const auto &summary : spin_summaries)
    {
        total_chern_raw += summary.chern_raw;
        total_chern_rounded += summary.chern_rounded;
    }

    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(16);
    ofs << "{\n";
    ofs << "  \"task\": \"topo_gw_band\",\n";
    ofs << "  \"mesh_order\": \"k1_fast_row_major\",\n";
    ofs << "  \"nk1\": " << nk1 << ",\n";
    ofs << "  \"nk2\": " << nk2 << ",\n";
    ofs << "  \"n_kpoints\": " << n_kpoints << ",\n";
    ofs << "  \"vxc_treatment\": \"" << vxc_mode << "\",\n";
    ofs << "  \"shift_mu\": " << (shift_mu ? "true" : "false") << ",\n";
    ofs << "  \"mu_ha\": " << mu << ",\n";
    ofs << "  \"mu_ev\": " << mu * HA2EV << ",\n";
    ofs << "  \"total_chern_raw\": " << total_chern_raw << ",\n";
    ofs << "  \"total_chern_rounded\": " << total_chern_rounded << ",\n";
    ofs << "  \"spins\": [\n";
    for (size_t ispin = 0; ispin < spin_summaries.size(); ++ispin)
    {
        const auto &summary = spin_summaries[ispin];
        ofs << "    {\n";
        ofs << "      \"spin\": " << (ispin + 1) << ",\n";
        ofs << "      \"nocc\": " << summary.nocc << ",\n";
        ofs << "      \"chern_raw\": " << summary.chern_raw << ",\n";
        ofs << "      \"chern_rounded\": " << summary.chern_rounded << ",\n";
        ofs << "      \"min_link_abs\": " << summary.min_link_abs << ",\n";
        ofs << "      \"max_sigma_antiherm\": " << summary.max_sigma_antiherm << ",\n";
        ofs << "      \"min_gap_ha\": " << summary.min_gap_ha << ",\n";
        ofs << "      \"min_gap_ev\": " << summary.min_gap_ha * HA2EV << "\n";
        ofs << "    }" << (ispin + 1 == spin_summaries.size() ? "\n" : ",\n");
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

}  // namespace

void task_topo_gw_band(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;

    if (Params::topology_nk1 <= 1 || Params::topology_nk2 <= 1)
    {
        throw std::logic_error("topo_gw_band requires topology_nk1 > 1 and topology_nk2 > 1");
    }

    Profiler::start("topo_gw_band", "GW topological Hamiltonian on a 2D band mesh");

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    std::vector<Vector3_Order<double>> qlist;
    for (const auto &q_weight : irk_weight)
        qlist.push_back(q_weight.first);

    auto tfg =
        LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);

    Chi0 chi0(meanfield, klist, tfg);
    chi0.gf_R_threshold = Params::gf_R_threshold;
    chi0.set_input_dir(driver_params.input_dir);

    Profiler::start("chi0_build", "Build response function chi0");
    chi0.build(Cs_data, Rlist, period, local_atpair, qlist, sinvS);
    Profiler::stop("chi0_build");
    std::flush(ofs_myid);
    mpi_comm_global_h.barrier();

    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
    {
        read_Vq_full(driver_params.input_dir, "coulomb_cut_", true);
    }
    else
    {
        read_Vq_row(driver_params.input_dir, "coulomb_cut_", Params::vq_threshold, local_atpair,
                    true);
    }
    Profiler::stop("read_vq_cut");

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

    Profiler::start("g0w0_exx", "Build exchange self-energy");
    auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
    {
        Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
        const auto VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
        Profiler::stop("ft_vq_cut");

        Profiler::start("g0w0_exx_real_work");
        if (Params::use_shrink_abfs)
        {
            if (Params::use_soc)
                exx.build<std::complex<double>>(Cs_shrinked_data, Rlist, VR);
            else
                exx.build<double>(Cs_shrinked_data, Rlist, VR);
        }
        else
        {
            if (Params::use_soc)
                exx.build<std::complex<double>>(Cs_data, Rlist, VR);
            else
                exx.build<double>(Cs_data, Rlist, VR);
        }
        Profiler::stop("g0w0_exx_real_work");
    }
    Profiler::stop("g0w0_exx");
    std::flush(ofs_myid);

    Profiler::start("g0w0_wc", "Build screened interaction");
    std::vector<cplxdb> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(),
                                           epsmac_LF_imagfreq_re.cend());
    std::map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old>
        Wc_freq_q;
    if (Params::use_scalapack_gw_wc)
    {
        Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
    }
    else
    {
        Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
    }
    Profiler::stop("g0w0_wc");

    if (Params::use_shrink_abfs)
    {
        Profiler::start("read_shrink_sinvS_fold", "Load shrink transformation");
        read_shrink_sinvS(driver_params.input_dir, "shrink_sinvS_", sinvS);
        Profiler::stop("read_shrink_sinvS_fold");
    }

    LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg, period);
    Profiler::start("g0w0_sigc_IJ", "Build real-space correlation self-energy");
    if (Params::use_soc)
        s_g0w0.build_spacetime<std::complex<double>>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
    else
        s_g0w0.build_spacetime<double>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
    Profiler::stop("g0w0_sigc_IJ");
    std::flush(ofs_myid);

    Profiler::start("topo_band_load_kmesh", "Load topology band mesh");
    int n_basis_band = 0;
    int n_states_band = 0;
    int n_spin_band = 0;
    int flag = 0;
    const auto band_bundle_dir =
        resolve_input_dir_with_pyatb_fallback(driver_params.input_dir, "band_kpath_info");
    const auto kfrac_band = read_band_kpath_info(band_bundle_dir + "band_kpath_info",
                                                 n_basis_band, n_states_band, n_spin_band, flag);
    Profiler::stop("topo_band_load_kmesh");

    if (flag != 0)
    {
        if (mpi_comm_global_h.is_root())
        {
            lib_printf("Warning! Failed to read %sband_kpath_info , skip topo_gw_band\n",
                       driver_params.input_dir.c_str());
        }
        mpi_comm_global_h.barrier();
        Profiler::stop("topo_gw_band");
        return;
    }

    if (static_cast<int>(kfrac_band.size()) != Params::topology_nk1 * Params::topology_nk2)
    {
        std::ostringstream err;
        err << "topology_nk1 * topology_nk2 = "
            << (Params::topology_nk1 * Params::topology_nk2)
            << " does not match the number of band k-points " << kfrac_band.size();
        throw std::runtime_error(err.str());
    }

    Profiler::start("topo_load_band_mf", "Read band eigen solutions");
    auto meanfield_band = read_meanfield_band(band_bundle_dir, n_basis_band, n_states_band,
                                              n_spin_band, static_cast<int>(kfrac_band.size()));
    meanfield_band.get_efermi() = meanfield.get_efermi();
    Profiler::stop("topo_load_band_mf");

    exx.reset_kspace();
    s_g0w0.reset_kspace();

    Profiler::start("topo_sigx_rotate_KS");
    exx.build_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);
    Profiler::stop("topo_sigx_rotate_KS");

    Profiler::start("topo_sigc_rotate_KS");
    s_g0w0.build_sigc_matrix_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);
    Profiler::stop("topo_sigc_rotate_KS");
    std::flush(ofs_myid);
    mpi_comm_global_h.barrier();

    if (mpi_comm_global_h.is_root())
    {
        const int n_spins_band = meanfield_band.get_n_spins();
        const int n_kpoints_band = meanfield_band.get_n_kpoints();
        const int n_bands_band = meanfield_band.get_n_bands();
        const double mu = meanfield_band.get_efermi();

        const auto h_ks_band = build_band_ks_hamiltonian(meanfield_band);
        const auto vxc_bundle =
            read_topology_vxc(driver_params.input_dir, n_spins_band, n_kpoints_band,
                              n_bands_band);
        const auto &vxc_band = vxc_bundle.matrices;
        if (vxc_bundle.mode != "full_matrix")
        {
            lib_printf("Warning: topo_gw_band is using %s for Vxc because full "
                       "band_vxc_mat_spin_*_k_*.csc files are absent.\n",
                       vxc_bundle.mode.c_str());
        }

        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
            imagfreqs.push_back(cplxdb{0.0, freq});

        if (Params::topology_dump_sigma0)
            ensure_dir(Params::output_dir + "topology_sigma0");
        if (Params::topology_dump_hmat)
            ensure_dir(Params::output_dir + "topology_hmat");
        if (Params::topology_dump_occ_evec)
            ensure_dir(Params::output_dir + "topology_occ_evec");

        std::vector<SpinTopologySummary> spin_summaries(n_spins_band);

        for (int ispin = 0; ispin < n_spins_band; ++ispin)
        {
            const int nocc = infer_nocc(meanfield_band, ispin);
            if (nocc <= 0 || nocc >= n_bands_band)
            {
                std::ostringstream err;
                err << "Invalid occupied-band count for topology on spin " << (ispin + 1)
                    << ": nocc=" << nocc << " with n_bands=" << n_bands_band;
                throw std::runtime_error(err.str());
            }

            spin_summaries[ispin].nocc = nocc;
            std::vector<Matz> eigvecs_ks(n_kpoints_band);
            std::vector<std::vector<double>> eigvals_ha(
                n_kpoints_band, std::vector<double>(n_bands_band, 0.0));

            for (int ikpt = 0; ikpt < n_kpoints_band; ++ikpt)
            {
                double antiherm_max = 0.0;
                const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS.at(ispin).at(ikpt);
                const auto sigma0 =
                    build_sigma_c_i0_matrix(sigc_sk, chi0.tfg.get_freq_nodes(), imagfreqs,
                                            n_bands_band, antiherm_max);
                const auto h_top = construct_topological_hamiltonian(
                    h_ks_band.at(ispin).at(ikpt), vxc_band.at(ispin).at(ikpt),
                    exx.exx_is_ik_KS.at(ispin).at(ikpt), sigma0, Params::topology_shift_mu, mu);

                std::vector<double> eigvals_here;
                Matz eigvec_here;
                eigsh(h_top, eigvals_here, eigvec_here);
                eigvecs_ks[ikpt] = eigvec_here;
                eigvals_ha[ikpt] = eigvals_here;

                const double gap_here = eigvals_here[nocc] - eigvals_here[nocc - 1];
                spin_summaries[ispin].min_gap_ha =
                    std::min(spin_summaries[ispin].min_gap_ha, gap_here);
                spin_summaries[ispin].max_sigma_antiherm =
                    std::max(spin_summaries[ispin].max_sigma_antiherm, antiherm_max);

                if (Params::topology_dump_sigma0)
                {
                    std::ostringstream fn;
                    fn << Params::output_dir << "topology_sigma0/sigma0_spin_" << (ispin + 1)
                       << "_k_" << std::setw(5) << std::setfill('0') << (ikpt + 1) << ".mtx";
                    print_matrix_mm_file(sigma0, fn.str());
                }
                if (Params::topology_dump_hmat)
                {
                    std::ostringstream fn;
                    fn << Params::output_dir << "topology_hmat/Htop_spin_" << (ispin + 1)
                       << "_k_" << std::setw(5) << std::setfill('0') << (ikpt + 1) << ".mtx";
                    print_matrix_mm_file(h_top, fn.str());
                }
                if (Params::topology_dump_occ_evec)
                {
                    std::ostringstream fn;
                    fn << Params::output_dir << "topology_occ_evec/Uocc_spin_" << (ispin + 1)
                       << "_k_" << std::setw(5) << std::setfill('0') << (ikpt + 1) << ".mtx";
                    print_matrix_mm_file(extract_occupied_columns(eigvec_here, nocc), fn.str());
                }
            }

            spin_summaries[ispin].chern_raw = compute_fhs_chern(
                eigvecs_ks, Params::topology_nk1, Params::topology_nk2, nocc,
                spin_summaries[ispin].min_link_abs);
            spin_summaries[ispin].chern_rounded =
                static_cast<long long>(std::llround(spin_summaries[ispin].chern_raw));

            std::ostringstream fn_band;
            fn_band << Params::output_dir << "topology_band_spin_" << (ispin + 1) << ".dat";
            write_band_spectrum(fn_band.str(), kfrac_band, eigvals_ha);

            lib_printf("topo_gw_band spin %d: nocc=%d raw Chern=% .12f rounded=%lld "
                       "min_link_abs=%.6e min_gap=%.6e Ha max_sigma_antiH=%.6e\n",
                       ispin + 1, nocc, spin_summaries[ispin].chern_raw,
                       spin_summaries[ispin].chern_rounded,
                       spin_summaries[ispin].min_link_abs,
                       spin_summaries[ispin].min_gap_ha,
                       spin_summaries[ispin].max_sigma_antiherm);
        }

        write_topology_summary(Params::output_dir + "topology_summary.json", Params::topology_nk1,
                               Params::topology_nk2, n_kpoints_band, vxc_bundle.mode,
                               Params::topology_shift_mu, mu, spin_summaries);
    }

    mpi_comm_global_h.barrier();
    Profiler::stop("topo_gw_band");
}
