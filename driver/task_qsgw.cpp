#include "task_qsgw.h"

// 标准库头文件
#include <iostream>         // 用于输入输出操作
#include <map>              // 用于std::map容器
#include <string>           // 用于std::string类
#include <fstream> // 用于文件存在检查
#include <sstream>
#include <iomanip> // 用于格式化
#include <vector>
#include <cmath>
// 自定义头文件
#include "envs_mpi.h"               // MPI环境相关
#include "utils_io.h"
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
#include "matrix.h"
#include "read_data.h"              // 数据读取相关
#include "fermi_energy_occupation.h"// 费米能和占据数计算相关
#include "convert_csc.h"
#include "Hamiltonian.h"            // 哈密顿量相关
#include "driver_utils.h"
#include "read_data.h"

std::vector<double> efermi_values;  
std::vector<double> homo_values;
std::vector<double> lumo_values;
std::vector<int> iteration_numbers;

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

    const auto n_spins = meanfield.get_n_spins();
    const auto n_bands = meanfield.get_n_bands();
    const auto n_kpoints = meanfield.get_n_kpoints();
    const auto n_aos = meanfield.get_n_aos();
    // 初始化
    Profiler::start("read_vxc_HKS");
    std::map<int, std::map<int, Matz>> hf_nao;  
    std::map<int, std::map<int, Matz>> vxc;  
    std::map<int, std::map<int, Matz>> hf;
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int,std::map<int, std::map<int, Matz>>> Hexx_matrix_temp;
    std::map<int, std::map<int, Matz>> H_KS; // H_KS矩阵
    std::map<int, std::map<int, Matz>> H_KS0;
    std::map<int, std::map<int, Matz>> H_KS1;//用于混合迭代
    bool all_files_processed_successfully = true;

        // 自旋和 k 点的循环，读取初始数据
    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            std::map<std::string, Matz> arrays;
            std::string key_hf, key_vxc;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_hf, oss_vxc;
            oss_hf << "hf_exchange_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_vxc << "xc_matr_spin_" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";

            std::string hfFilePath = oss_hf.str();
            std::string vxcFilePath = oss_vxc.str();

            Matz wfc1(n_bands, n_aos, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib) {
                for (int iao = 0; iao < n_aos; iao++) {
                    wfc1(ib, iao) = meanfield.get_eigenvectors()[ispin][ikpt](ib, iao);
                    meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao) = wfc1(ib, iao);
                }
            }
            hf_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  
            vxc[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            // 初始化 hf 和 vxc 矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i) {
                for (int j = 0; j < n_aos; ++j) {
                    hf_nao[ispin][ikpt](i, j) = 0.0;
                    vxc[ispin][ikpt](i, j) = 0.0;
                }
            }
          

            bool hf_file_found = false;
            bool vxc_file_found = false;

            // 读取 hf 文件
            std::ifstream hf_file(hfFilePath.c_str());
            if (hf_file.good()) {
                if (!convert_csc(hfFilePath, arrays, key_hf)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << hfFilePath << std::endl;
                } else {
                    hf_nao[ispin][ikpt] = arrays[key_hf];
                    hf_file_found = true;
                }
            } else {
                std::cerr << "HF file not found: " << hfFilePath << std::endl;
            }

            // 读取 vxc 文件
            std::ifstream vxc_file(vxcFilePath.c_str());
            if (vxc_file.good()) {
                if (!convert_csc(vxcFilePath, arrays, key_vxc)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath << std::endl;
                } else {
                    vxc[ispin][ikpt] = arrays[key_vxc]; 
                    vxc_file_found = true;
                }
            } else {
                std::cerr << "VXC file not found: " << vxcFilePath << std::endl;
            }

            // 如果两个文件都不存在，报错并跳过该 k 点
            if (!hf_file_found && !vxc_file_found) {
                all_files_processed_successfully = false;
                std::cerr << "Both HF and VXC files not found for spin " << ispin + 1 << ", k-point " << ikpt + 1 << std::endl;
                continue;
            }

            // 生成 H_KS 和 H_KS0 矩阵
            hf[ispin][ikpt] = wfc1 * hf_nao[ispin][ikpt] * transpose(wfc1);

            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵
            vxc[ispin][ikpt] = vxc[ispin][ikpt] + hf[ispin][ikpt];
            vxc0[ispin][ikpt] = vxc[ispin][ikpt];

            // 构建 H_KS 矩阵，使用哈密顿量中的本征值
            H_KS[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            H_KS0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band) {
                H_KS[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
                H_KS0[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
            }
        }
    }


    // // 判断是否所有文件都成功处理
    // if (mpi_comm_global_h.myid == 0) {
    //     if (all_files_processed_successfully) {
    //         std::cout << "* Success: Read DFT xc potential, will solve quasi-particle equation\n";
    //     } else {
    //         std::cout << "* Error: Some files failed to process, switch off solving quasi-particle equation\n";
    //     }
    // }

    Profiler::stop("read_vxc_HKS");

    
    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    double homo = -1e6;
    double lumo = 1e6;

    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            int homo_level = -1;
            for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) {
                double weight = meanfield.get_weight()[ispin](ikpt, ib);
                double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) {
                    homo_level = ib;
                }
            }

            if (homo_level != -1) {
                homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
            }
        }
    }

    // 保存初始状态数据
    homo_values.push_back(homo * HA2EV);  // 初始 HOMO 值
    lumo_values.push_back(lumo * HA2EV);  // 初始 LUMO 值
    efermi_values.push_back(efermi * HA2EV);  // 初始费米能级
    iteration_numbers.push_back(0);  // 初始迭代次数为 0

    std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
              << "LUMO = " << lumo * HA2EV << " eV, "
              << "Fermi Energy = " << efermi * HA2EV << " eV\n";
    plot_homo_lumo_vs_iterations();
    
    //计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n","Total_electrons");
    printf("%5f\n",total_electrons);
    // 读取库伦相互作用
    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    read_Vq_full("./", "coulomb_cut_", true);
    const auto VR = FT_Vq(Vq_cut, Rlist, true);  // 对库伦势进行傅里叶变换
    Profiler::stop("read_vq_cut");


    // 设置收敛条件
    double eigenvalue_tolerance = -1e-4; // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations = 50;           // 最大迭代次数
    int iteration = 0;
    const double temperature = 0.001;
    bool converged = false;
    int frequency = n_bands + 1; 
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    
    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations) {
        iteration++;

        // 更新前一轮的本征值
        for (int i_spin = 0; i_spin < n_spins; i_spin++)
        {
            previous_eigenvalues[i_spin] = meanfield.get_eigenvals()[i_spin];
        }
        
        
        
        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS
        Profiler::start("qsgw_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list);
        exx.build(Cs_data, Rlist, period, VR);
        exx.build_KS_kgrid0();
        Profiler::stop("qsgw_exx");
        Hexx_matrix_temp[iteration] = exx.exx_is_ik_KS ;
        Hexx_matrix_temp[0] = exx.exx_is_ik_KS ;

        Chi0 chi0(meanfield, klist, Params::nfreq);
        chi0.gf_R_threshold = Params::gf_R_threshold;

        Profiler::start("chi0_build", "Build response function chi0");
        chi0.build(Cs_data, Rlist, period, local_atpair, qlist,
                   TFGrids::get_grid_type(Params::tfgrids_type), true);
        Profiler::stop("chi0_build");
        
        const std::string final_banner(90, '-');
        // //check
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %16.5f %16.5f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix-vxc Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%16.6f ", exx_value.real()-vxc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //     }
        // }
        
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
        }
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


        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg);
        Profiler::start("g0w0_sigc_IJ", "Build correlation self-energy");
        s_g0w0.build_spacetime(Cs_data, Wc_freq_q, Rlist, period);
        Profiler::stop("g0w0_sigc_IJ");

        Profiler::start("g0w0_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid0();
        Profiler::stop("g0w0_sigc_rotate_KS");

        // 构建哈密顿量矩阵并对角化，旋转基底，并存储本征值，本征矢量
        // 第一步：构建关联势矩阵
        std::map<int, std::map<int, Matz>> Vc_all;

        // 构建虚频点列表
        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes()) {
            imagfreqs.push_back(cplxdb{0.0, freq});
        }

        std::map<int, std::map<int, std::map<int, double>>> e_qp_all;
        std::map<int, std::map<int, std::map<int, cplxdb>>> sigc_all;
        
        if (all_files_processed_successfully)
        {
            Profiler::start("qsgw_solve_qpe", "Solve quasi-particle equation");

            if (mpi_comm_global_h.is_root()) {
                std::cout << "Solving quasi-particle equation\n";
            }

            if (mpi_comm_global_h.is_root()) {
                // 遍历自旋、k点和能带状态
                for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                    for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                        std::vector<std::vector<std::vector<cplxdb>>> sigcmat(
                            n_bands, std::vector<std::vector<cplxdb>>(n_bands, std::vector<cplxdb>(n_bands + 1))
                        );
                  
                        const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];

                        for (int i_state_row = 0; i_state_row < n_bands; i_state_row++) {
                            // 检测；Solve quasi-particle equation，并对于全自能矩阵进行解析延拓方便下一步构建哈密顿量
                            const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                            const auto &exx_state = exx.exx_is_ik_KS[i_spin][i_kpoint](i_state_row, i_state_row);
                            const auto &vxc_state = vxc[i_spin][i_kpoint](i_state_row, i_state_row);
                       
                            // 获取自能
                            std::vector<cplxdb> sigc_state;

                            for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                sigc_state.push_back(sigc_sk.at(freq)(i_state_row, i_state_row));
                            }
                            // 定义阈值
                            double threshold = 1e-5;
                            // 指定特定的自旋和 k 点
                            int target_spin = 0;
                            int target_kpoint = 0;

                            // 第一次迭代中的查找
                            if (iteration == 1) {  
                                std::vector<std::tuple<int, int, double>> non_diagonal_elements;
                                for (int i_state_row = 0; i_state_row < n_bands; i_state_row++) {
                                    for (int i_state_col = 0; i_state_col < n_bands; i_state_col++) {
                                        if (i_state_row != i_state_col) {  // 排除对角元
                                            for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                                // 只关注特定自旋和 k 点的情况
                                                cplxdb sigc_value = s_g0w0.sigc_is_ik_f_KS[target_spin][target_kpoint].at(freq)(i_state_row, i_state_col);                                               
                                                non_diagonal_elements.emplace_back(i_state_row, i_state_col, sigc_value.real());
                                            }
                                        }
                                    }
                                }
                                // 按实部大小排序，找到最大的三个非对角元
                                std::sort(non_diagonal_elements.begin(), non_diagonal_elements.end(),
                                        [](const auto &a, const auto &b) {
                                            return std::get<2>(a) > std::get<2>(b);  // 按实部降序排序
                                        });

                                // 只保留最大的三个元素（可能少于三个）
                                for (int i = 0; i < std::min(3, static_cast<int>(non_diagonal_elements.size())); ++i) {
                                    significant_positions.emplace_back(std::get<0>(non_diagonal_elements[i]), std::get<1>(non_diagonal_elements[i]));
                                // 输出位置和实部值
                                std::cout << "Significant Position: (" 
                                        << std::get<0>(non_diagonal_elements[i]) << ", " 
                                        << std::get<1>(non_diagonal_elements[i]) << ") "
                                        << "with Real Part = " 
                                        << std::get<2>(non_diagonal_elements[i]) << std::endl;
                                }
                            }
                            // 构建 Pade 近似对象
                            LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_state);
                            // QPE求解
                            double e_qp;
                            cplxdb sigc_qp;
                            int flag_qpe_solver = LIBRPA::qpe_solver_pade_self_consistent(
                                pade, eks_state, efermi, vxc_state.real(), exx_state.real(), e_qp, sigc_qp
                            );

                            if (flag_qpe_solver != 0) {
                                std::cout << "Warning! QPE solver failed for spin " << i_spin + 1
                                        << ", kpoint " << i_kpoint + 1
                                        << ", state " << i_state_row + 1 << "\n";
                                e_qp = std::numeric_limits<double>::quiet_NaN();
                                sigc_qp = std::numeric_limits<cplxdb>::quiet_NaN();
                            }
                            else
                            {
                                sigcmat[i_state_row][i_state_row][i_state_row] = sigc_qp;
                                e_qp_all[i_spin][i_kpoint][i_state_row] = e_qp;
                                sigc_all[i_spin][i_kpoint][i_state_row] = sigc_qp;
                                // // 输出 e_qp 的值
                                // std::cout << "e_qp for spin " << i_spin + 1 
                                //         << ", kpoint " << i_kpoint + 1 
                                //         << ", state " << i_state_row + 1 
                                //         << " = " << e_qp << std::endl;
                            }
                            for (int i_state_col = 0; i_state_col < meanfield.get_n_bands(); i_state_col++) {
                                if (i_state_col == i_state_row) {
                                    continue;  // 跳过 i_state_col 等于 i_state_row 的情况
                                }
                                std::vector<cplxdb> sigc_mn;
                                for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                    sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                                    // if (std::abs(sigc_mn.real()) < threshold) {
                                    //     sigc_mn.real(0.0);
                                    // }
                                    // if (std::abs(sigc_mn.imag()) < threshold) {
                                    //     sigc_mn.imag(0.0);
                                    // }
                                }    
                                LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);
                                
                                // 计算得到的值
                                auto result = pade.get(e_qp - efermi);
                                auto result1 = pade.get(0.0);
                                // 存储值到 sigcmat
                                sigcmat[i_state_row][i_state_col][i_state_row] = result;
                                sigcmat[i_state_row][i_state_col][n_bands] = result1;
                                
                                // // 输出当前计算结果
                                // std::cout << "sigcmat[" << i_state_row << "][" << i_state_col << "][" << i_state_row 
                                //         << "] = " << result << std::endl;
                            }
                            

                        }

                        Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);
                    }
                }
            }
            Profiler::stop("qsgw_solve_qpe");
        }

        //检查输入
        
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_list[i_kpoint];
                printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                        i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                printf("%77s\n", final_banner.c_str());
                printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
                printf("%77s\n", final_banner.c_str());
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
                    const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
                    printf("%5d %20.15f %20.15f\n",
                        i_state + 1, eks_state, vxc_state.real());
                }
                printf("%77s\n", final_banner.c_str());
                printf("H Matrix_real:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &H_value = H_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", H_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                printf("%77s\n", final_banner.c_str());
                printf("H Matrix_image:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &H_value = H_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", H_value.imag()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                // 输出 exx_is_ik_KS 矩阵
                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix1:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", exx_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                printf("%77s\n", final_banner.c_str());
                printf("vxc Matrix1:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", vxc_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix-vxc Matrix1:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
                        const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
                        printf("%16.6f ", exx_value.real()-vxc_value.real()); 
                    }
                    printf("\n"); 
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
            }
        }
        auto H0_GW_all = construct_H0_GW(H_KS, vxc, exx.exx_is_ik_KS, Vc_all, n_spins, n_kpoints, n_bands);
        //混合
        // if(iteration > 1){
        //     for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        //         for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
        //             H0_GW_all[ispin][ikpt] = 0.2 * H0_GW_all[ispin][ikpt] + 0.8 * H_KS[ispin][ikpt];
        //         }
        //     }
        // }
        //检查输入
        
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %20.15f %20.15f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H0_GW_all[i_spin][i_kpoint](i, j) ;
        //                 printf("%20.15f ", H_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_image:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H0_GW_all[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", H_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix2:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("eigenvectors:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &eigenvectors = meanfield.get_eigenvectors()[i_spin][i_kpoint](i, j) ;
        //                 printf("%20.15f ", eigenvectors.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix2:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix-vxc Matrix2:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%16.6f ", exx_value.real()-vxc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //     }
        // }

        // 第三步：对 Hamiltonian 进行对角化并存储本征值
        diagonalize_and_store(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);
        
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %20.15f %20.15f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix3:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("eigenvectors2:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &eigenvectors = meanfield.get_eigenvectors()[i_spin][i_kpoint](i, j) ;
        //                 printf("%20.15f ", eigenvectors.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix3:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("Vc+exx Matrix-vxc Matrix:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &Vc_value = Vc_all[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%20.15f ", Vc_value.real() + exx_value.real() - vxc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //     }
        // }

        // 计算全局费米能和占据数
        const auto &Efermi0 = meanfield.get_efermi() ;
        printf("%5s\n","efermi0");
        printf("%5f\n",Efermi0);
        // 计算费米能级
        
        double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
        printf("%5s\n","efermi0");
        printf("%5f\n",efermi);


         //将占据数和费米能级更新到 MeanField 对象中
        update_fermi_energy_and_occupations(meanfield, temperature, efermi);
        efermi_values.push_back(efermi * HA2EV);  
        // 比较本轮和前一轮的本征值判断是否收敛
        converged = true;
        for (int ispin = 0; ispin < n_spins; ++ispin) {
            const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
            const auto max_diff = (current_eigenvals - previous_eigenvalues[ispin]).absmax();
            if (max_diff > eigenvalue_tolerance) {
                converged = false;
                break;
            }
        }
        std::cout << "Converged after " << iteration << " iterations.\n";
        // const std::string final_banner(90, '-');
        lib_printf("Final Quasi-Particle Energy after QSGW Iterations [unit: eV]\n\n");
        const auto &Efermi = meanfield.get_efermi() ;
        printf("%5s\n","efermi");
        printf("%5f\n",Efermi);
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s %16s %16s %16s %16s\n", "State", "e_mf", "v_xc", "v_exx", "ReSigc", "ImSigc", "e_qp");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
        //             const auto &exx_state = exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state) * HA2EV;
        //             // exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
        //             const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
        //             const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
        //             const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
        //             printf("%5d %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f\n",
        //                    i_state + 1, eks_state, vxc_state.real(), exx_state.real(), resigc, imsigc, eqp);
        //         }
        //         printf("\n");
        //     }
        // }

        
        // 更新vxc数据
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                
                const auto& Vc_matrix = Vc_all[ispin][ikpt];
                const auto& Hexx_matrix = exx.exx_is_ik_KS[ispin][ikpt];
                vxc[ispin][ikpt] = Hexx_matrix + Vc_matrix;
                // vxc[ispin][ikpt] = Hexx_matrix ;
               
            }
        }


        // 更新 H_KS
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                for (int i_row = 0; i_row < n_bands; i_row++)
                {
                    // H_KS[ispin][ikpt](i_row, i_row) = meanfield.get_eigenvals()[ispin](ikpt, i_row);
                    for(int i_col = 0; i_col < n_bands;i_col++){
                        const auto& H0_GW_all_const = H0_GW_all[ispin][ikpt](i_row, i_col);
                        H_KS[ispin][ikpt](i_row, i_col) = H0_GW_all_const;
                    }
                }
            }
        }
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %16.5f %16.5f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix4:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix4:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix-vxc Matrix4:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%16.6f ", exx_value.real()-vxc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //     }
        // }
        
        std::ofstream exx_output_file("exx_output_all_iterations.dat", std::ios::app);
        if (!exx_output_file.is_open()) {
            std::cerr << "Error: Unable to open file for writing." << std::endl;
            return;
        }
        // 写入当前迭代次数标识
        exx_output_file << "Iteration " << iteration << "\n";
        for (int ispin = 0; ispin < n_spins; ++ispin) { 
            for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
                exx_output_file << "Spin " << ispin << ", K-point " << ikpt << ":\n";

                // 获取 exx 矩阵并写入文件
                for (int i = 0; i < meanfield.get_n_bands(); ++i) {
                    for (int j = 0; j < meanfield.get_n_bands(); ++j) {
                        const auto& exx_value = exx.exx_is_ik_KS[ispin][ikpt](i, j);
                        exx_output_file << exx_value.real() << " "; // 假设只写入实部
                    }
                    exx_output_file << "\n";
                }
                exx_output_file << "\n"; // 分隔不同自旋或 k 点

                // // 输出 H_KS
                // print_matrix_mm_file(H_KS[ispin][ikpt], "H_KS_output_ispin" + std::to_string(ispin) + "_ikpt" + std::to_string(ikpt) + ".dat");
                // //exx
                // print_matrix_mm_file(exx.exx_is_ik_KS[ispin][ikpt], "exx_output_ispin" + std::to_string(ispin) + "_ikpt" + std::to_string(ikpt) + ".dat");
                
                // // 输出 vxc
                // print_matrix_mm_file(vxc[ispin][ikpt], "vxc_output_ispin" + std::to_string(ispin) + "_ikpt" + std::to_string(ikpt) + ".dat");
            }
        }

        // 计算 HOMO 和 LUMO
        homo = -1e6;  // 
        lumo = 1e6;   // 
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                int homo_level = -1;
                for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) {
                    double weight = meanfield.get_weight()[ispin](ikpt, ib);
                    double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                    // 
                    if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) {
                        homo_level = ib;
                    }
                }

                // 
                if (homo_level != -1) {
                    // 
                    homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                    // 
                    lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
                }
            }
        }

        // 
        homo_values.push_back(homo * HA2EV);  // 
        lumo_values.push_back(lumo * HA2EV);  // 
        iteration_numbers.push_back(iteration);

        // 输出当前 HOMO 和 LUMO 值
        std::cout << "Iteration " << iteration
          << ": HOMO = " << homo * HA2EV << " eV, "
          << "LUMO = " << lumo * HA2EV << " eV, "
          << "Efermi = " << efermi * HA2EV << " eV\n";
        
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_list[i_kpoint];
                printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                        i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                printf("%77s\n", final_banner.c_str());
                printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
                printf("%77s\n", final_banner.c_str());
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
                    const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
                    printf("%5d %16.5f %16.5f\n",
                        i_state + 1, eks_state, vxc_state.real());
                }
                // 输出 exx_is_ik_KS 矩阵
                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix5:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", exx_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                printf("%77s\n", final_banner.c_str());
                printf("vxc Matrix5:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", vxc_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix-vxc Matrix5:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
                        const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
                        printf("%16.6f ", exx_value.real()-vxc_value.real()); 
                    }
                    printf("\n"); 
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
            }
        }

        // 如果已经收敛或达到最大迭代次数，输出最终的QSGW迭代结果，退出循环
        if (converged) {
            break;
        }

        if (iteration == max_iterations) {
            std::cout << "Reached maximum number of iterations.\n";
        }

    }

    plot_homo_lumo_vs_iterations();

    


    Profiler::stop("qsgw");
}
void plot_homo_lumo_vs_iterations() {
    // 将 HOMO、LUMO 和费米能级数据保存到文件
    std::ofstream file("homo_lumo_vs_iterations.dat");
    for (size_t i = 0; i < iteration_numbers.size(); ++i) {
        file << iteration_numbers[i] << " "
             << homo_values[i] << " "
             << lumo_values[i] << " "
             << efermi_values[i] << std::endl;
    }
    file.close();

}
