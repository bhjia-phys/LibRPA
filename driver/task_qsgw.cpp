// 标准库头文件
#include <iostream>         // 用于输入输出操作
#include <filesystem>       // 用于文件系统操作
#include <map>              // 用于std::map容器
#include <string>           // 用于std::string类
#include <regex>            // 用于正则表达式操作

// 自定义头文件
#include "task_qsgw.h"              // 任务调度相关
#include "envs_mpi.h"               // MPI环境相关
#include "meanfield.h"              // MeanField类相关
#include "params.h"                 // 参数设置相关
#include "pbc.h"                    // 周期性边界条件相关
#include "chi0.h"                   // 响应函数相关
#include "gw.h"                     // GW计算相关
#include "analycont.h"              // 分析延拓相关
#include "qpe_solver.h"             // 准粒子方程求解器
#include "epsilon.h"                // 介电函数相关
#include "exx.h"                    // Exact exchange相关
#include "constants.h"              // 常量定义
#include "coulmat.h"                // 库仑矩阵相关
#include "profiler.h"               // 性能分析工具
#include "ri.h"                     // RI相关
#include "matrix.h"                     // RI相关
#include "read_data.h"              // 数据读取相关
#include "fermi_energy_occupation.h"// 费米能和占据数计算相关
#include "convert_csc.h"
#include "Hamiltonian.h"            // 哈密顿量相关
#include "driver_utils.h"                  // 包含 process_csc 和其他实用工具函数
#include "read_data.h"

void task_qsgw()
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::utils::lib_printf;

    Profiler::start("qsgw", "QSGW quasi-particle calculation");

    Vector3_Order<int> period {kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    vector<Vector3_Order<double>> qlist;
    for (auto q_weight: irk_weight)
    {
        qlist.push_back(q_weight.first);
    }

    Chi0 chi0(meanfield, klist, Params::nfreq);
    chi0.gf_R_threshold = Params::gf_R_threshold;

    Profiler::start("chi0_build", "Build response function chi0");
    chi0.build(Cs_data, Rlist, period, local_atpair, qlist,
               TFGrids::get_grid_type(Params::tfgrids_type), true);
    Profiler::stop("chi0_build");


    // 初始化并读取当前文件夹中的所有 .csc 文件
    Profiler::start("read_vxc_vexx_HKS");

    std::map<int, std::map<int, Matz>> vxc;  // 当前vxc
    std::map<int, std::map<int, Matz>> vxc0; // 存储初始vxc
    std::map<int, std::map<int, Matz>> H_KS; // H_KS矩阵

    bool all_files_processed_successfully = true;

    // 自旋和k点的循环，读取初始数据
    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            // 初始化格式化字符串和文件路径
            std::map<std::string, std::string> format_fn = {
                {"xc", "xc_matr_spin_" + to_string(ispin + 1) + "_kpt_" + to_string(ikpt + 1) + ".csc"},
                // FIXME: you are not supposed to read H and C matrices, only xc.
                // C has been read and stored in the MeanField object
                // H should be constructed using eigenvalues and eigenvectors of Meanfield.
                // {"H", "H_spin_" + to_string(ispin + 1) + "_kpt_" + to_string(ikpt + 1) + ".csc"},
                // {"C", "C_spin_" + to_string(ispin + 1) + "_kpt_" + to_string(ikpt + 1) + ".csc"}
            };

            std::map<std::string, Matz> arrays;

            // 遍历所有格式化的文件路径，并读取对应的矩阵
            for (const auto& pair : format_fn) {
                if (!convert_csc(pair.second, arrays)) {
                    all_files_processed_successfully = false;
                    cerr << "Failed to process file: " << pair.second << endl;
                }
            }

            if (!all_files_processed_successfully) {
                // 如果任何文件处理失败，跳过这个k点的处理
                continue;
            }

            // 获取状态限制
            // int lb, ub;
            // tie(lb, ub) = read_aims_state_limits();

            // 将NHO哈密顿量转换为Kohn-Sham空间
            // arrays["H_KS"] = arrays["C"].adjoint() * arrays["H"] * arrays["C"];
            // arrays["H_KS"] = arrays["H_KS"].block(lb, lb, ub - lb, ub - lb);

            // 存储初始的vxc0，构建哈密顿量时有用
            vxc0[ispin][ikpt] = arrays["xc"];

            // 初始化vxc为vxc0
            vxc[ispin][ikpt] = vxc0[ispin][ikpt].copy();

            // 存储 H_KS 矩阵
            // TODO: construct H_KS using eigenvluaes and eigenvectors stored in MeanField
            // H_KS[ispin][ikpt] = arrays["H_KS"];
        }
    }

    // 判断是否所有文件都成功处理
    if (mpi_comm_global_h.myid == 0) {
        if (all_files_processed_successfully) {
            cout << "* Success: Read DFT xc potential, will solve quasi-particle equation\n";
        } else {
            cout << "* Error: Some files failed to process, switch off solving quasi-particle equation\n";
        }
    }

    Profiler::stop("read_vxc_vexx_HKS");

    // 初始化费米能，单位为hatree
    const auto efermi = meanfield.get_efermi() * 0.5;
    //计算体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    // 读取库伦相互作用
    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    read_Vq_full("./", "coulomb_cut_", true);
    const auto VR = FT_Vq(Vq_cut, Rlist, true);  // 对库伦势进行傅里叶变换
    Profiler::stop("read_vq_cut");

    // 读取和处理介电函数
    std::vector<double> epsmac_LF_imagfreq_re;
    if (Params::replace_w_head)
    {
        std::vector<double> omegas_dielect;
        std::vector<double> dielect_func;
        read_dielec_func("dielecfunc_out", omegas_dielect, dielect_func);

        epsmac_LF_imagfreq_re = interpolate_dielec_func(
                Params::option_dielect_func, omegas_dielect, dielect_func,
                chi0.tfg.get_freq_nodes());


        if (Params::debug) {
            if (mpi_comm_global_h.is_root()) {
                lib_printf("Dielectric function parsed:\n");
                for (int i = 0; i < chi0.tfg.get_freq_nodes().size(); i++)
                    lib_printf("%d %f %f\n", i+1, chi0.tfg.get_freq_nodes()[i], epsmac_LF_imagfreq_re[i]);
            }
            mpi_comm_global_h.barrier();
        }
    }

    // 设置收敛条件
    double eigenvalue_tolerance = 1e-6; // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations = 100;           // 最大迭代次数
    int iteration = 0;
    bool converged = false;

    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues;

    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations) {
        iteration++;

        // 更新前一轮的本征值
        previous_eigenvalues = meanfield.get_eigenvals();

        // 构建V^{exx}矩阵,得到Hexx_nband_nband
        Profiler::start("qsgw_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list);
        exx.build(Cs_data, Rlist, period, VR);
        exx.build_KS_kgrid();
        Profiler::stop("qsgw_exx");

        // Build screened interaction
        Profiler::start("qsgw_wc", "Build screened interaction");
        vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(), epsmac_LF_imagfreq_re.cend());
        map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old> Wc_freq_q;
        if (Params::use_scalapack_gw_wc) {
            Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        } else {
            Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        Profiler::stop("qsgw_wc");

        // if (Params::debug)
        // { 
        //     char fn[80];
        //     for (const auto &Wc: Wc_freq_q)
        //     {
        //         const int ifreq = chi0.tfg.get_freq_index(Wc.first);
        //         for (const auto &I_JqWc: Wc.second)
        //         {
        //             const auto &I = I_JqWc.first;
        //             for (const auto &J_qWc: I_JqWc.second)
        //             {
        //                 const auto &J = J_qWc.first;
        //                 for (const auto &q_Wc: J_qWc.second)
        //                 {
        //                     const int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q_Wc.first));
        //                     sprintf(fn, "Wcfq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I, J, mpi_comm_global_h.myid);
        //                     print_matrix_mm_file(q_Wc.second, Params::output_dir + "/" + fn, 1e-15);
        //                 }
        //             }
        //         }
        //     }
        // }

        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg);
        Profiler::start("qsgw_sigc_IJ", "Build correlation self-energy");
        s_g0w0.build_spacetime(Cs_data, Wc_freq_q, Rlist, period);
        Profiler::stop("qsgw_sigc_IJ");

        Profiler::start("qsgw_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid();
        Profiler::stop("qsgw_sigc_rotate_KS");

        // 检测；Solve quasi-particle equation，并对于全自能矩阵进行解析延拓方便下一步构建哈密顿量

        if (all_files_processed_successfully)
        {
            Profiler::start("qsgw_solve_qpe", "Solve quasi-particle equation");

            if (mpi_comm_global_h.is_root()) {
                std::cout << "Solving quasi-particle equation\n";
            }

            // 构建虚频点列表
            std::vector<cplxdb> imagfreqs;
            for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                imagfreqs.push_back(cplxdb{0.0, freq});
            }

            if (mpi_comm_global_h.is_root()) {
                // 初始化存储自能的容器
                map<int, map<int, map<int, map<int, cplxdb>>>> sigc_all;
                map<int, map<int, map<int, map<double, cplxdb>>>> sigc_real_all;

                // 定义实频点数组
                // MYZ: Why do you need this?
                std::vector<double> real_freqs = {/* 实频点数组，例如：-10.0, -5.0, 0.0, 5.0, 10.0 */};

                // 遍历自旋、k点和能带状态
                for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++) {
                    for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++) {
                        const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];

                        for (int i_state_row = 0; i_state_row < meanfield.get_n_bands(); i_state_row++) {
                            for (int i_state_col = 0; i_state_col < meanfield.get_n_bands(); i_state_col++) {

                                // FIXME: eks_state_mn is wrong: it is always the diagonal element. Should extend to any element
                                const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                                const auto &exx_state = exx.exx_is_ik_KS[i_spin][i_kpoint](i_state_row, i_state_col);
                                const auto &vxc_mn = vxc[i_spin][i_kpoint](i_state_row, i_state_col);

                                // 获取自能的虚频值
                                std::vector<cplxdb> sigc_state;
                                for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                    sigc_state.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                                }

                                // 构建 Pade 近似对象
                                LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_state);

                                // 解析延拓到每个实频点
                                for (const auto& real_freq : real_freqs) {
                                    cplxdb sigc_real = pade.get(static_cast<cplxdb>(real_freq - efermi));
                                    sigc_real_all[i_spin][i_kpoint][i_state_row][i_state_col][real_freq] = sigc_real;
                                }
                            }
                            // QPE求解
                            double e_qp;
                            cplxdb sigc_qp;
                            int flag_qpe_solver = LIBRPA::qpe_solver_pade_self_consistent(
                                pade, eks_state_row, efermi, vxc_state_row, exx_state_row, e_qp, sigc_qp
                            );

                            if (flag_qpe_solver == 0) {
                                e_qp_all[i_spin][i_kpoint][i_state_row] = e_qp;
                                sigc_all[i_spin][i_kpoint][i_state_row] = sigc_qp;
                            } else {
                                std::cout << "Warning! QPE solver failed for spin " << i_spin + 1
                                        << ", kpoint " << i_kpoint + 1
                                        << ", state (" << i_state_row + 1 << ", " << i_state_col + 1 << ")\n";
                                e_qp_all[i_spin][i_kpoint][i_state_row] = std::numeric_limits<double>::quiet_NaN();
                                sigc_all[i_spin][i_kpoint][i_state_row] = std::numeric_limits<cplxdb>::quiet_NaN();
                            }
                        }
                    }
                }
            }

            Profiler::stop("qsgw_solve_qpe");
        }

        //构建哈密顿量矩阵并对角化，旋转基底，并存储本征值，本征矢量
        // 第一步：构建关联势矩阵
        auto Vc_all = build_correlation_potential(sigc_real_all, e_qp_all, n_spins, n_kpoints, n_bands);

        // 第二步：使用构建好的关联势矩阵构建 GW 哈密顿量
        std::vector<Matrix> H_KS_vec, vxc0_vec, Hexx_vec;

        // 将 H_KS 和 vxc0 转换为二维矩阵向量
        for (int ispin = 0; ispin < n_spins; ++ispin) {
            for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
                H_KS_vec.push_back(H_KS[ispin][ikpt].block(0, 0, n_bands, n_bands));
                vxc0_vec.push_back(vxc0[ispin][ikpt].block(0, 0, n_bands, n_bands));
                Hexx_vec.push_back(exx.Hexx_KS[ispin][ikpt].block(0, 0, n_bands, n_bands));
            }
        }

        auto H0_GW_all = construct_H0_GW(H_KS_vec, vxc0_vec, Hexx_vec, Vc_all, n_spins, n_kpoints, n_bands);

        // 第三步：对 Hamiltonian 进行对角化并存储本征值
        diagonalize_and_store(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);

        // 第四步：存储新矩阵到 MeanField 对象的 wfc 矩阵中
        std::vector<std::vector<std::vector<std::vector<double>>>> newMatrix(n_spins, std::vector<std::vector<std::vector<double>>>(n_kpoints, std::vector<std::vector<double>>(n_bands, std::vector<double>(n_aos, 0.0))));
        // MYZ: Now you ARE saving zeros to meanfield.
        store_newMatrix_to_wfc(meanfield, newMatrix, n_spins, n_kpoints, n_bands, n_aos);

        // 计算全局费米能和占据数
    
        // 计算费米能级并更新占据数
        double efermi = calculate_fermi_energy_and_occupations(meanfield, temperature, total_electrons);

        // 将费米能级更新到 MeanField 对象中
        meanfield.get_efermi() = efermi;

        // 更新vxc数据
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                // 获取 Hexx_vec 和 Vc_all 对应的矩阵
                const auto& Hexx_matrix = Hexx_vec[ispin][ikpt];
                const auto& Vc_matrix = Vc_all[ispin][ikpt];

                // 获取矩阵的维度（假设 Hexx_matrix 和 Vc_matrix 具有相同的维度）
                int n_rows = Hexx_matrix.size();
                int n_cols = Hexx_matrix[0].size();

                // 遍历矩阵中的每个元素，并更新 vxc[ispin][ikpt] 矩阵
                for (int i = 0; i < n_rows; ++i) {
                    for (int j = 0; j < n_cols; ++j) {
                        vxc[ispin][ikpt](i, j) = Hexx_matrix[i][j] + Vc_matrix[i][j];
                    }
                }
            }
        }

        // 比较本轮和前一轮的本征值，判断是否收敛
        converged = true;
        for (int ispin = 0; ispin < n_spins; ++ispin) {
            for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
                const auto &current_eigenvals = meanfield.get_eigenvals()[ispin].row(ikpt);
                if (!previous_eigenvalues.empty()) {
                    const auto &previous_eigenvals = previous_eigenvalues[ispin * n_kpoints + ikpt];
                    for (int ib = 0; ib < n_bands; ++ib) {
                        if (std::abs(current_eigenvals(ib) - previous_eigenvals(ib)) > eigenvalue_tolerance) {
                            converged = false;
                            break;
                        }
                    }
                }
                if (!converged) break;
            }
            if (!converged) break;
        }

        // 如果已经收敛或达到最大迭代次数，退出循环
        if (converged) {
            std::cout << "Converged after " << iteration << " iterations.\n";
            break;
        }

        if (iteration == max_iterations) {
            std::cout << "Reached maximum number of iterations.\n";
        }
    }

    // 输出最终的QSGW迭代结果
    const std::string final_banner(90, '-');
    printf("Final Quasi-Particle Energy after QSGW Iterations [unit: eV]\n\n");
    for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
    {
        for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        {
            const auto &k = kfrac_list[i_kpoint];
            printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                   i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
            printf("%77s\n", final_banner.c_str());
            printf("%5s %16s %16s %16s %16s %16s %16s\n", "State", "e_mf", "v_xc", "v_exx", "ReSigc", "ImSigc", "e_qp");
            printf("%77s\n", final_banner.c_str());
            for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
            {
                const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * RY2EV;
                const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                printf("%5d %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f\n",
                       i_state + 1, eks_state, vxc_state, exx_state, resigc, imsigc, eqp);
            }
            printf("\n");
        }
    }

    Profiler::stop("qsgw");
}
