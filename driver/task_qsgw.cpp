#include "task_qsgw.h"
// 标准库头文件
#include <cmath>
#include <fstream>   // 用于文件存在检查
#include <iomanip>   // 用于格式化
#include <iostream>  // 用于输入输出操作
#include <map>       // 用于std::map容器
#include <sstream>
#include <string>  // 用于std::string类
#include <vector>
// 自定义头文件

#include "Hamiltonian.h"  // 哈密顿量相关
#include "analycont.h"    // 分析延拓相关
#include "chi0.h"         // 响应函数相关
#include "constants.h"    // 常量定义
#include "convert_csc.h"
#include "coulmat.h"  // 库仑矩阵相关
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"    
#include "hartree.h"
#include "exx.h"                      // Exact exchange相关
#include "fermi_energy_occupation.h"  // 费米能和占据数计算相关
#include "gw.h"                       // GW计算相关
#include "matrix.h"
#include "meanfield.h"  // MeanField类相关
#include "mpi.h"
#include "params.h"      // 参数设置相关
#include "pbc.h"         // 周期性边界条件相关
#include "profiler.h"    // 性能分析工具
#include "qpe_solver.h"  // 准粒子方程求解器
#include "read_data.h"
#include "ri.h"
#include "utils_io.h"
#include "utils_timefreq.h"
#include "write_aims.h"
#include "pulay_mixing.h" 
std::vector<double> efermi_values;
std::vector<double> homo_values;
std::vector<double> lumo_values;
std::vector<int> iteration_numbers;

void task_qsgw(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;

    Profiler::start("qsgw", "QSGW quasi-particle calculation");

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    vector<Vector3_Order<double>> qlist;
    for (auto q_weight : irk_weight)
    {
        qlist.push_back(q_weight.first);
    }

    // 读取 meanfield 数据
    const auto n_spins = meanfield.get_n_spins();
    const auto n_bands = meanfield.get_n_bands();
    const auto n_kpoints = meanfield.get_n_kpoints();
    const auto n_aos = meanfield.get_n_aos();
    const auto n_soc = meanfield.get_n_soc();

    // 初始化
    Profiler::start("read_vxc_HKS");
    std::map<int, std::map<int, Matz>> hf_nao;
    std::map<int, std::map<int, Matz>> vxc;
    std::map<int, std::map<int, Matz>> hf;
    std::map<int, std::map<int, Matz>> s_nao;
    std::map<int, std::map<int, Matz>> s_band;
    std::map<int, std::map<int, Matz>> s_inverse;
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int, std::map<int, Matz>> vxc1;
    std::map<int, std::map<int, Matz>> vxc_band;
    std::map<int, std::map<int, Matz>> exx0;
    std::map<int, std::map<int, std::map<int, Matz>>> Hexx_matrix_temp;
    std::map<int, std::map<int, Matz>> H_KS;  // H_KS矩阵
    std::map<int, std::map<int, Matz>> H_KS0;
    std::map<int, std::map<int, Matz>> H_KS0_band;
    std::map<int, std::map<int, Matz>> Vc_nao;
    std::map<int, std::map<int, Matz>> H_nao;
    std::map<int, std::map<int, Matz>> H_KS1;  
    std::map<int, std::map<int, Matz>> Hartree_0;
    std::map<int, std::map<int, Matz>> Hartree_i;
    std::map<int, std::map<int, Matz>> Hartree_i_delta;
    std::map<int, std::map<int, Matz>> H_DFT_nao;
    std::map<int, std::map<int, Matz>> H_DFT_band_nao;
    map<int, map<int, map<int, map<Vector3_Order<int>, Matz>>>> H_nao_R;
    map<int, map<int, map<int, map<Vector3_Order<int>, Matz>>>> Vc_nao_R;
    bool all_files_processed_successfully = true;
    const std::string final_banner(90, '-');

    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt)
        {
            std::map<std::string, Matz> arrays;
            std::string key_hf, key_vxc,key_s;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_hf, oss_vxc, oss_s,oss_s2;  // 新增S矩阵文件名生成器
            oss_hf << "hf_exchange_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_vxc << "xc_matr_spin_" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_s << "sks" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";  // 新增S矩阵文件名
            oss_s2 << "S_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";


            std::string hfFilePath = oss_hf.str();
            std::string vxcFilePath = oss_vxc.str();
            std::string sFilePath = oss_s.str();  // 获取S矩阵文件路径
            std::string sFilePath_2 = oss_s2.str();
            
            Matz wfc1(n_bands, n_aos * n_soc, MAJOR::COL);
            for (int ib1 = 0; ib1 < n_bands; ++ib1)
            {
                for (int isoc = 0; isoc < n_soc; isoc++)
                {
                    for (int iao = 0; iao < n_aos; iao++)
                    {
                        int ib2 = iao * n_soc + isoc;
                        wfc1(ib1, ib2) = meanfield.get_eigenvectors()[ispin][isoc][ikpt](ib1, iao);
                        meanfield.get_eigenvectors0()[ispin][isoc][ikpt](ib1, iao) = wfc1(ib1, ib2);
                    }
                }
            }

            // 初始化 hf 和 vxc 矩阵
            hf_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  
            vxc0[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            s_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  // 新增S矩阵初始化
            s_inverse[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  // 新增S矩阵初始化
            
            // 初始化矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i) {
                for (int j = 0; j < n_aos; ++j) {
                    hf_nao[ispin][ikpt](i, j) = 0.0;
                    vxc0[ispin][ikpt](i, j) = 0.0;
                    s_nao[ispin][ikpt](i, j) = 0.0;  // 初始化S矩阵元素
                    s_inverse[ispin][ikpt](i, j) = 0.0;  // 初始化S矩阵元素
                }
            }

            bool hf_file_found = false;
            bool vxc_file_found = false;
            bool s_file_found = false;  // 新增S文件存在标志

            
            // 首先尝试读取原有格式的 S 矩阵文件
            std::ifstream s_file_2(sFilePath_2.c_str());
            if (s_file_2.good()) {
                if (!convert_csc(sFilePath_2, arrays, key_s)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << sFilePath_2 << std::endl;
                }
                else {
                    s_nao[ispin][ikpt] = arrays[key_s];
                    s_file_found = true;
            }
            }
                
            

            // 如果原有格式未找到或读取失败，尝试新格式
            if (!s_file_found) {
                // 构建新格式文件名（与原有代码一致）
                std::ostringstream oss_s_new;
                oss_s_new << "s" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                std::string sNewFilePath = oss_s_new.str();

                std::ifstream s_new_format_file(sNewFilePath.c_str());
                if (s_new_format_file.good()) {
                    try {
                        // 读取矩阵维数（第一个数）
                        int matrix_size = 0;
                        s_new_format_file >> matrix_size;
                        if (matrix_size <= 0) {
                            throw std::runtime_error("Invalid matrix size in file: " + sNewFilePath);
                        }

                        // 初始化矩阵
                        Matz s_matrix(matrix_size, matrix_size, MAJOR::COL);

                        // 读取矩阵数据
                        for (int row = 0; row < matrix_size; ++row) {
                            for (int col = row; col < matrix_size; ++col) {
                                char ch; // 用于读取括号和逗号
                                double real_part, imag_part;

                                // 读取格式为 (real,imag)
                                s_new_format_file >> ch >> real_part >> ch >> imag_part >> ch;
                                if (s_new_format_file.fail()) {
                                    throw std::runtime_error("Error reading matrix data in file: " + sNewFilePath);
                                }

                                // 存储到上三角矩阵
                                s_matrix(row, col) = std::complex<double>(real_part, imag_part);
                            }
                        }

                        // 将上三角矩阵扩展为完整矩阵
                        for (int i = 0; i < matrix_size; ++i) {
                            for (int j = 0; j < i; ++j) {
                                s_matrix(i, j) = std::conj(s_matrix(j, i));
                            }
                        }

                        // 存储到 s_nao
                        s_nao[ispin][ikpt] = s_matrix;
                        s_file_found = true;
                    } catch (const std::exception& e) {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process new format file: " << sNewFilePath
                                << ". Error: " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "S matrix file not found: " << sFilePath << " or " << sNewFilePath << std::endl;
                }
            }
            
                
            // // 修改文件存在性检查，包含S矩阵
            // if (!hf_file_found && !vxc_file_found && !s_file_found) {
            //     all_files_processed_successfully = false;
            //     std::cerr << "HF, VXC and S files not found for spin " << ispin + 1 << ", k-point " << ikpt + 1 << std::endl;
            //     continue;
            // }

            // 读取 hf 文件
            std::ifstream hf_file(hfFilePath.c_str());
            if (hf_file.good()) {
                if (!convert_csc(hfFilePath, arrays, key_hf)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << hfFilePath << std::endl;
                } 
                else 
                {
                    hf_nao[ispin][ikpt] = arrays[key_hf];
                    hf_file_found = true;
                }
            } 
            else 
            {
                std::cerr << "HF file not found: " << hfFilePath << std::endl;
            }

            // 读取 vxc 文件
            std::ifstream vxc_file(vxcFilePath.c_str());
            if (vxc_file.good()) {
                // 尝试读取原有格式的 vxc 矩阵
                if (!convert_csc(vxcFilePath, arrays, key_vxc)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath << std::endl;
                } else {
                    vxc0[ispin][ikpt] = arrays[key_vxc];
                    vxc_file_found = true;
                }
            } else {
                // 如果原有格式的文件未找到，尝试新的格式读取
                // 在使用 vxcNewFilePath 之前定义并赋值
                std::ostringstream oss_vxc_new;
                oss_vxc_new << "vxcs" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                std::string vxcNewFilePath = oss_vxc_new.str();

                // 打开文件
                std::ifstream vxc_new_format_file(vxcNewFilePath.c_str());
                if (vxc_new_format_file.good()) {
                    try {
                        // 读取矩阵维数（第一个数）
                        int matrix_size = 0;
                        vxc_new_format_file >> matrix_size;
                        if (matrix_size <= 0) {
                            throw std::runtime_error("Invalid matrix size in file: " + vxcNewFilePath);
                        }

                        // 初始化矩阵
                        Matz vxc_matrix(matrix_size, matrix_size, MAJOR::COL);

                        // 读取矩阵数据
                        for (int row = 0; row < matrix_size; ++row) {
                            for (int col = row; col < matrix_size; ++col) {
                                char ch; // 用于读取括号和逗号
                                double real_part, imag_part;

                                // 读取格式为 (real,imag)
                                vxc_new_format_file >> ch >> real_part >> ch >> imag_part >> ch;
                                if (vxc_new_format_file.fail()) {
                                    throw std::runtime_error("Error reading matrix data in file: " + vxcNewFilePath);
                                }

                                // 存储到上三角矩阵
                                vxc_matrix(row, col) = std::complex<double>(real_part, imag_part);
                            }
                        }

                        // 将上三角矩阵扩展为完整矩阵
                        for (int i = 0; i < matrix_size; ++i) {
                            for (int j = 0; j < i; ++j) {
                                vxc_matrix(i, j) = std::conj(vxc_matrix(j, i));
                            }
                        }

                        // 存储到 vxc0
                        vxc0[ispin][ikpt] = 0.5 * vxc_matrix;
                        vxc_file_found = true;
                    } catch (const std::exception& e) {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process new format file: " << vxcNewFilePath
                                << ". Error: " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "VXC file not found: " << vxcFilePath << " or " << vxcNewFilePath << std::endl;
                }
            }

            // 如果两个文件都不存在，报错并跳过该 k 点
            if (!hf_file_found && !vxc_file_found) 
            {
                all_files_processed_successfully = false;
                std::cerr << "Both HF and VXC files not found for spin " << ispin + 1 << ", k-point " << ikpt + 1 << std::endl;
                continue;
            }

            // 生成 H_KS 和 H_KS0 矩阵
            hf[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);
            hf[ispin][ikpt] = conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);  // row hf,KS
                                                                                   // basis

            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵

            vxc[ispin][ikpt] = vxc0[ispin][ikpt] + hf[ispin][ikpt];
            vxc0[ispin][ikpt] = vxc[ispin][ikpt];
            
            // 构建 H_KS 矩阵，使用哈密顿量中的本征值
            H_KS[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            H_KS0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band)
            {
                H_KS[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
                H_KS0[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
            }
            // auto H_KS_check = conj(wfc1) * s_nao[ispin][ikpt] * transpose(wfc1)  * conj(wfc1) * transpose(s_nao[ispin][ikpt],true) * transpose(wfc1);
            // for (int i_band = 0; i_band < n_bands; ++i_band)
            // {
            //     auto H_check = H_KS_check(i_band,i_band);
            //     printf("iband = %2d, H_KS_check_value = (%.5f, %.5f) \n", i_band, H_check.real(), H_check.imag());
            // }

            H_DFT_nao[ispin][ikpt] = s_nao[ispin][ikpt] * transpose(wfc1) * ( H_KS0[ispin][ikpt] - vxc0[ispin][ikpt] ) * conj(wfc1) * transpose(s_nao[ispin][ikpt],true);
        }
        
    }

    Profiler::stop("read_vxc_HKS");
    mpi_comm_global_h.barrier();
    std::flush(ofs_myid);

    // 读取库伦相互作用
    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU)
    {
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
    Profiler::stop("read_vq_cut");
    
    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    double homo = -1e6;
    double lumo = 1e6;

    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt)
        {
            int homo_level = -1;
            for (int ib = 0; ib < meanfield.get_n_bands(); ++ib)
            {
                double weight = meanfield.get_weight()[ispin](ikpt, ib);
                double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints()))
                {
                    homo_level = ib;
                }
            }

            if (homo_level != -1)
            {
                homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
            }
        }
    }

    // 保存初始状态数据
    homo_values.push_back(homo * HA2EV);      // 初始 HOMO 值
    lumo_values.push_back(lumo * HA2EV);      // 初始 LUMO 值
    efermi_values.push_back(efermi * HA2EV);  // 初始费米能级
    iteration_numbers.push_back(0);           // 初始迭代次数为 0

    std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
              << "LUMO = " << lumo * HA2EV << " eV, "
              << "Fermi Energy = " << efermi * HA2EV << " eV\n";
    plot_homo_lumo_vs_iterations();

    // 计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n", "Total_electrons");
    printf("%5f\n", total_electrons);

    // 设置收敛条件
    double eigenvalue_tolerance = 1e-4;  // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations = 500;              // 最大迭代次数i
    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1;
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    
    // ==========================================
    // Initialize Pulay Mixer
    // ==========================================
    // History size: 7, Mixing beta: 0.5
    // 建议：对于难收敛体系，可将 beta 调小至 0.2-0.3，history 增加至 10-12
    PulayMixer mixer(12, 0.2); 
    bool mixer_initialized = false;

    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::trunc);
        file.close();
    }
    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations)
    {
        iteration++;

        // 更新前一轮的本征值
        if (mpi_comm_global_h.is_root())
        {
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                previous_eigenvalues[i_spin] = meanfield.get_eigenvals()[i_spin];
            }
        }
        mpi_comm_global_h.barrier();

        //构建V^{Hartree}矩阵
        Profiler::start("qsgw_hartree", "Build Hartree potential");
        auto Hartree = LIBRPA::Hartree(meanfield, kfrac_list, period);
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            const auto VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
            Profiler::stop("ft_vq_cut"); 
            Profiler::start("qsgw_hartree_real_work");
            Hartree.build(Cs_data, Rlist, VR); 
            // // 新增调试输出
            // for (int isp = 0; isp < meanfield.get_n_spins(); ++isp) {
            //     for (int is1 = 0; is1 < meanfield.get_n_soc(); ++is1) {
            //         for (int is2 = 0; is2 < meanfield.get_n_soc(); ++is2) {
            //             // 遍历R空间
            //             for (const auto& R_entry : Hartree.hartree[isp][is1][is2]) {
            //                 Vector3_Order<int> R = R_entry.first;
            //                 // 只检查R=0的情况
            //                 // if (R.x == 0 && R.y == 0 && R.z == 0) {
            //                     for (const auto& P_entry : R_entry.second) {
            //                         atom_t P = P_entry.first;
            //                         for (const auto& Q_entry : P_entry.second) {
            //                             atom_t Q = Q_entry.first;
            //                             const Matd& hartree_mat = Q_entry.second;
                                        
            //                             // 输出矩阵基本信息
            //                             std::cout << "Hartree[" << isp << "][" << is1 << "][" << is2 << "]" 
            //                                     << "[R=(" << R.x << "," << R.y << "," << R.z << ")]"
            //                                     << "[P=" << P << "][Q=" << Q << "] Matrix:" << std::endl;
                                        
            //                             // 输出矩阵前3x3部分
            //                             for (int i = 0; i < 20 && i < hartree_mat.nr(); ++i) {
            //                                 for (int j = 0; j < 20 && j < hartree_mat.nc(); ++j) {
            //                                     std::cout << std::setw(12) << hartree_mat(i,j) << " ";
            //                                 }
            //                                 std::cout << std::endl;
            //                             }
            //                         }
            //                     }
            //                 // }
            //             }
            //         }
            //     }
            // }
            Hartree.build_KS_kgrid0();//rotate  
            Profiler::stop("qsgw_hartree_real_work");
        
        }

        Profiler::stop("qsgw_hartree");
        std::flush(ofs_myid);
        mpi_comm_global_h.barrier();
 
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {    
                Hartree_i[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                if(iteration==1){
                    Hartree_0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                }
                Hartree_i_delta[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                for (int i = 0; i < n_bands; ++i) {
                    
                    const auto &hartree0_k_ks_value = Hartree.EHartree[ispin][ikpt][i];
                    printf("%16.6f ", hartree0_k_ks_value ); 
                    for (int j = 0; j < n_bands;++j) {

                        const auto &hartree_k_ks_value = Hartree.Hartree_is_ik_KS[ispin][ikpt](i, j);
                        
                        Hartree_i[ispin][ikpt](i, j) = hartree_k_ks_value;
                        
                        
                        
                        if(iteration==1){
                            Hartree_0[ispin][ikpt](i, j) = Hartree.Hartree_is_ik_KS[ispin][ikpt](i,j);
                        }
                        else{
                            Hartree_i_delta[ispin][ikpt](i, j) = Hartree_i[ispin][ikpt](i, j) - Hartree_0[ispin][ikpt](i,j);
                            
                        }
                    }
                    printf("\n");
                }
                printf("\n");
            }
        
        }
        // Prepare time-frequency grids
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

        if (Params::debug)
        {  // debug, check chi0
            char fn[80];
            for (const auto &chi0q : chi0.get_chi0_q())
            {
                const int ifreq = chi0.tfg.get_freq_index(chi0q.first);
                for (const auto &q_IJchi0 : chi0q.second)
                {
                    const int iq = std::distance(
                        klist.begin(), std::find(klist.begin(), klist.end(), q_IJchi0.first));
                    for (const auto &I_Jchi0 : q_IJchi0.second)
                    {
                        const auto &I = I_Jchi0.first;
                        for (const auto &J_chi0 : I_Jchi0.second)
                        {
                            const auto &J = J_chi0.first;
                            sprintf(fn, "chi0fq_ifreq_%d_iq_%d_I_%d_J_%d_id_%d.mtx", ifreq, iq, I,
                                    J, mpi_comm_global_h.myid);
                            print_complex_matrix_mm(J_chi0.second, Params::output_dir + "/" + fn,
                                                    1e-15);
                        }
                    }
                }
            }
        }

        // 读取和处理介电函数
        std::vector<double> epsmac_LF_imagfreq_re;
        if (Params::replace_w_head)
        {
            std::vector<double> omegas_dielect;
            std::vector<double> dielect_func;
            read_dielec_func(driver_params.input_dir + "dielecfunc_out", omegas_dielect,
                             dielect_func);

            epsmac_LF_imagfreq_re =
                interpolate_dielec_func(Params::option_dielect_func, omegas_dielect, dielect_func,
                                        chi0.tfg.get_freq_nodes());
            if (Params::debug)
            {
                if (mpi_comm_global_h.is_root())
                {
                    lib_printf("Dielectric function parsed:\n");
                    for (int i = 0; i < chi0.tfg.get_freq_nodes().size(); i++)
                        lib_printf("%d %f %f\n", i + 1, chi0.tfg.get_freq_nodes()[i],
                                   epsmac_LF_imagfreq_re[i]);
                }
                mpi_comm_global_h.barrier();
            }
        }


        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS

        Profiler::start("qsgw_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            const auto VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
            Profiler::stop("ft_vq_cut");

            Profiler::start("g0w0_exx_real_work");
            if (Params::use_soc)
                exx.build<std::complex<double>>(Cs_data, Rlist, VR);
            else
                exx.build<double>(Cs_data, Rlist, VR);
            exx.build_KS_kgrid0();  // rotate
            Profiler::stop("g0w0_exx_real_work");
            for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
                for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                    exx0[ispin][ikpt] = Matz(n_bands, n_aos, MAJOR::COL);
                    exx0[ispin][ikpt] = exx.exx_is_ik_KS[ispin][ikpt];
                    Matz wfc2(n_bands, n_aos, MAJOR::COL);
                    // for (int ib = 0; ib < n_bands; ++ib) {
                    //     const auto &exx0_k_ks_value = exx.exx_is_ik_KS[ispin][ikpt](ib, ib);
                    //     printf("%16.6f ", exx0_k_ks_value.real()* HA2EV);
                    //     printf("\n");
                    //     // for (int iao = 0; iao < n_aos; iao++) {
                    //     //     wfc2(ib, iao) = meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao); 
                    //     // }
                    // }
                    // printf("\n"); 
                    // exx0[ispin][ikpt] = transpose(wfc2) * exx0[ispin][ikpt] * conj(wfc2);//for check
                }
            }
            
        }
        Profiler::stop("qsgw_exx");
        std::flush(ofs_myid);

        mpi_comm_global_h.barrier();

        

        // Build screened interaction
        Profiler::start("qsgw_wc", "Build screened interaction");
        vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(),
                                                        epsmac_LF_imagfreq_re.cend());
        map<double,
            atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
            Wc_freq_q;
        if (Params::use_scalapack_gw_wc)
        {
            Wc_freq_q = compute_Wc_freq_q_blacs(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        else
        {
            Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        Profiler::stop("qsgw_wc");

        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg, period);
        Profiler::start("g0w0_sigc_IJ", "Build correlation self-energy");
        if (Params::use_soc)
            s_g0w0.build_spacetime<std::complex<double>>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        else
            s_g0w0.build_spacetime<double>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        Profiler::stop("g0w0_sigc_IJ");
        std::flush(ofs_myid);
        Profiler::start("g0w0_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid0();  // rotate
        Profiler::stop("g0w0_sigc_rotate_KS");

        // 构建哈密顿量矩阵并对角化，旋转基底，并存储本征值，本征矢量
        // 第一步：构建关联势矩阵
        std::map<int, std::map<int, Matz>> Vc_all;

        // 构建虚频点列表
        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            imagfreqs.push_back(cplxdb{0.0, freq});
        }

        std::map<int, std::map<int, std::map<int, double>>> e_qp_all;
        std::map<int, std::map<int, std::map<int, cplxdb>>> sigc_all;

        if (all_files_processed_successfully)
        {
            Profiler::start("qsgw_solve_qpe", "Solve quasi-particle equation");

            if (mpi_comm_global_h.is_root())
            {
                std::cout << "Solving quasi-particle equation\n";
            }

            if (mpi_comm_global_h.is_root())
            {
                // 遍历自旋、k点和能带状态
                for (int i_spin = 0; i_spin < n_spins; i_spin++)
                {
                    for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++)
                    {
                        std::vector<std::vector<std::vector<cplxdb>>> sigcmat(
                            n_bands, std::vector<std::vector<cplxdb>>(
                                         n_bands, std::vector<cplxdb>(n_bands + 1)));
                        const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
                        // const auto& freq = chi0.tfg.get_freq_nodes();
                        // std::vector<cplxdb> Omega_values(n_bands);
                        // // printf("%zu\n ",freq.size());
                        // const auto& f_weight= chi0.tfg.get_freq_weights();
                        // auto G0_matrix= build_G0(meanfield,freq,i_spin,i_kpoint,n_bands);
                        for (int i_state_row = 0; i_state_row < n_bands; i_state_row++)
                        {
                            for (int i_state_col = 0; i_state_col < meanfield.get_n_bands();
                                 i_state_col++)
                            {
                                std::vector<cplxdb> sigc_mn;
                                // if(i_state_row==i_state_col){
                                //     for (size_t w = 0; w < f_weight.size(); ++w) {
                                //         cplxdb sigc_nn_iw = sigc_sk.at(freq[w])(i_state_row,
                                //         i_state_row);
                                //         Sigma_iskwnn[iteration-1][i_spin][i_kpoint][w][i_state_row]
                                //         = sigc_nn_iw; Omega_values[i_state_row] += f_weight[w] *
                                //         sigc_nn_iw * G0_matrix[i_state_row][w] *
                                //         G0_matrix[i_state_row][w];
                                //     }
                                // }
                                for (const auto &freq : chi0.tfg.get_freq_nodes())
                                {
                                    sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                                    // std::cout << "sigc_sk[" << i_state_row << "][" << i_state_col
                                    //           << "][" << freq << "] = " << sigc_sk.at(freq)(i_state_row,
                                    //                                                       i_state_col)
                                    //           << std::endl;
                                }
                                
                                LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs,
                                                           sigc_mn);

                                auto energy0 =
                                    meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                                efermi = meanfield.get_efermi();
                                // 计算得到的值
                                auto result = pade.get(energy0 - efermi);
                                auto result1 = pade.get(0.0);
                                // 存储值到 sigcmat
                                sigcmat[i_state_row][i_state_col][i_state_row] = result;
                                sigcmat[i_state_row][i_state_col][n_bands] = result1;

                                // // 输出当前计算结果
                                // std::cout << "sigcmat[" << i_state_row << "][" << i_state_col <<
                                // "][" << i_state_row
                                //         << "] = " << result << std::endl;
                            }
                        }
                        // Omega_total[iteration-1][i_spin][i_kpoint] = Omega_values;

                        Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);

                        // Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k_modeA(sigcmat,n_bands);
                        if(iteration>1){
                            Matz delta_Hartree_is_ik(n_bands, n_bands, MAJOR::COL);
                            delta_Hartree_is_ik = Hartree_i_delta[i_spin][i_kpoint];
                            Vc_all[i_spin][i_kpoint] = Vc_all[i_spin][i_kpoint] + delta_Hartree_is_ik;
                        }

                        

                        
                    }

                    
                }
                Profiler::stop("qsgw_solve_qpe");

                auto H0_GW_all = construct_H0_GW(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                                                 n_spins, n_kpoints, n_bands);

                // auto H0_GW_all = construct_H0_GW_new_basis(meanfield, H_KS0, H_DFT_nao, exx.exx_is_ik_KS, Vc_all,
                //                                  n_spins, n_kpoints, n_bands);

              
                // ==========================================
                // Pulay Mixing Execution
                // ==========================================
                {
                    std::cout << "Performing Pulay Mixing..." << std::endl;
                    
                    // 1. Prepare data dimensions
                    int total_matrices = n_spins * n_kpoints;
                    int rows_per_matrix = n_bands;
                    int cols_per_matrix = n_bands;
                    
                    // We pack all k-points and spins into one large matrix.
                    // To handle complex numbers, we double the width: [Real, Imag]
                    matrix mixed_input(total_matrices * rows_per_matrix, 2 * cols_per_matrix);
                    
                    // 2. Pack H0_GW_all into mixed_input
                    int row_offset = 0;
                    for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                        for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                            const Matz& mat = H0_GW_all[i_spin][i_kpoint];
                            for (int i = 0; i < rows_per_matrix; i++) {
                                for (int j = 0; j < cols_per_matrix; j++) {
                                    std::complex<double> val = mat(i, j);
                                    mixed_input(row_offset + i, j) = val.real();
                                    mixed_input(row_offset + i, j + cols_per_matrix) = val.imag();
                                }
                            }
                            row_offset += rows_per_matrix;
                        }
                    }

                    // 3. Execute Mixing
                    if (!mixer_initialized) {
                        mixer.initialize(mixed_input);
                        mixer_initialized = true;
                        std::cout << "Pulay Mixer Initialized with dimension " << mixed_input.nr << "x" << mixed_input.nc << std::endl;
                    } else {
                        try {
                            matrix mixed_output = mixer.mix(mixed_input);
                            
                            // 4. Unpack result back to H0_GW_all
                            row_offset = 0;
                            for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                                for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                                    Matz& mat = H0_GW_all[i_spin][i_kpoint];
                                    for (int i = 0; i < rows_per_matrix; i++) {
                                        for (int j = 0; j < cols_per_matrix; j++) {
                                            double re = mixed_output(row_offset + i, j);
                                            double im = mixed_output(row_offset + i, j + cols_per_matrix);
                                            mat(i, j) = std::complex<double>(re, im);
                                        }
                                    }
                                    row_offset += rows_per_matrix;
                                }
                            }
                            std::cout << "Pulay Mixing applied successfully." << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Pulay Mixing failed: " << e.what() << ". Continuing with unmixed Hamiltonian." << std::endl;
                            // If mixing fails, we just use the current H0_GW_all (unmixed)
                        }
                    }
                }
                // ==========================================
                // End Pulay Mixing
                // ==========================================

                //  第三步：对 Hamiltonian 进行对角化并存储本征值
                // diagonalize_and_store(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);
                diagonalize_and_store_fixed_basis(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);
                // 计算全局费米能和占据数
                const auto &Efermi0 = meanfield.get_efermi();
                printf("%5s\n", "efermi0");
                printf("%5f\n", Efermi0);
                // 计算费米能级

                double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
                printf("%5s\n", "efermi0");
                printf("%5f\n", efermi);

                update_fermi_energy_and_occupations(meanfield, temperature, efermi);
                efermi_values.push_back(efermi * HA2EV);
                // 比较本轮和前一轮的本征值判断是否收敛
                converged = true;
                for (int ispin = 0; ispin < n_spins; ++ispin)
                {
                    const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
                    const auto max_diff =
                        (current_eigenvals - previous_eigenvalues[ispin]).absmax();
                    if (max_diff > eigenvalue_tolerance)
                    {
                        converged = false;
                        break;
                    }
                }
                std::cout << "Converged after " << iteration << " iterations.\n";
                // const std::string final_banner(90, '-');
                lib_printf("Final Quasi-Particle Energy after QSGW Iterations [unit: eV]\n\n");
                const auto &Efermi = meanfield.get_efermi();
                printf("%5s\n", "efermi");
                printf("%5f\n", Efermi);
                for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
                {
                    for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
                    {
                        const auto &k = kfrac_list[i_kpoint];
                        printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                                i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                        printf("%77s\n", final_banner.c_str());
                        printf("%5s %16s %16s %16s \n", "State", "e_mf", "v_xc", "v_exx");
                        printf("%77s\n", final_banner.c_str());
                        for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                        {
                            const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                            const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                            // const auto &hartree_state = Hartree.Hartree_is_ik_KS[i_spin][i_kpoint](i_state, i_state)* HA2EV;
                            const auto &vxc_state = vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                            // const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                            // const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                            // const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                            // printf("%5d %20.15f %16.5f %16.5f  \n",
                            //     i_state + 1, eks_state, vxc_state.real(), hartree_state, exx_state);
                            printf("%5d %20.15f %16.5f %16.5f  \n",
                                i_state + 1, eks_state, vxc_state.real(), exx_state);
                        }
                        printf("\n");
                    }
                }

                // 计算 HOMO 和 LUMO
                homo = -1e6;  //
                lumo = 1e6;   //
                for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin)
                {
                    for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt)
                    {
                        int homo_level = -1;
                        for (int ib = 0; ib < meanfield.get_n_bands(); ++ib)
                        {
                            double weight = meanfield.get_weight()[ispin](ikpt, ib);
                            double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                            //
                            if (weight >=
                                1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints()))
                            {
                                homo_level = ib;
                            }
                        }

                        //
                        if (homo_level != -1)
                        {
                            //
                            homo =
                                std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                            //
                            lumo = std::min(lumo,
                                            meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
                        }
                    }
                }

                //
                homo_values.push_back(homo * HA2EV);  //
                lumo_values.push_back(lumo * HA2EV);  //
                iteration_numbers.push_back(iteration);

                // 输出当前 HOMO 和 LUMO 值
                std::cout << "Iteration " << iteration << ": HOMO = " << homo * HA2EV << " eV, "
                          << "LUMO = " << lumo * HA2EV << " eV, "
                          << "Efermi = " << efermi * HA2EV << " eV\n";

                // 保存 HOMO、LUMO 和费米能级数据
                {
                    std::ofstream file("homo_lumo_vs_iterations.dat",
                                       std::ios::app);  // 使用 std::ios::app 以追加模式打开文件
                    file << iteration << " " << homo_values[iteration] << " "
                         << lumo_values[iteration] << " " << efermi_values[iteration] << std::endl;
                }
                // 比较本轮和前一轮的本征值判断是否收敛
                converged = true;
                for (int ispin = 0; ispin < n_spins; ++ispin)
                {
                    const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
                    const auto max_diff =
                        (current_eigenvals - previous_eigenvalues[ispin]).absmax();
                    if (max_diff > eigenvalue_tolerance)
                    {
                        converged = false;
                        break;
                    }
                }

                std::cout << "Converged after " << iteration << " iterations.\n";
            }
        }
        mpi_comm_global_h.barrier();

        mpi_comm_global_h.broadcast(converged, 0);
        mpi_comm_global_h.barrier();
        meanfield.broadcast(mpi_comm_global_h, 0);
        mpi_comm_global_h.barrier();

        if (converged || iteration == max_iterations)
        {
            if (mpi_comm_global_h.is_root())
            {
                std::cout << " iterations: " << iteration;
            }
            break;
        }

        mpi_comm_global_h.barrier();
    }

    Profiler::stop("qsgw");
}

void plot_homo_lumo_vs_iterations()
{
    // 将 HOMO、LUMO 和费米能级数据保存到文件
    std::ofstream file("homo_lumo_vs_iterations.dat");
    for (size_t i = 0; i < iteration_numbers.size(); ++i)
    {
        file << iteration_numbers[i] << " " << homo_values[i] << " " << lumo_values[i] << " "
             << efermi_values[i] << std::endl;
    }
    file.close();
}
