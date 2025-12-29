#include "task_qsgw_band_0.h"
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
#include "dielecmodel.h"
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"       
#include "hartree.h"         
#include "exx.h"                     
#include "fermi_energy_occupation.h"  
#include "gw.h"                      
#include "matrix.h"
#include "meanfield.h" 
#include "mpi.h"
#include "params.h"     
#include "pbc.h"         
#include "profiler.h"   
#include "qpe_solver.h"  
#include "read_data.h"
#include "ri.h"
#include "utils_io.h"
#include "utils_timefreq.h"
#include "write_aims.h"

void task_qsgw_band_0(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;

    Profiler::start("qsgw_band", "QSGW quasi-particle calculation");

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    // 读取CSR文件并提取Rlist信息
    std::vector<Vector3_Order<int>> Rlist_abacus;
    std::string csr_filename = "hrs1_nao.csr"; 
    std::ifstream csr_file(csr_filename.c_str());
    
    if (csr_file.good()) {
        std::cout << "开始读取CSR文件: " << csr_filename << std::endl;
        
        std::string line;
        int line_number = 0;
        int matrix_dimension = 0;
        bool header_processed = false;
        
        while (std::getline(csr_file, line)) {
            line_number++;
            std::istringstream iss(line);
            std::string token;
            
            if (!header_processed) {
                // 处理文件头
                if (line.find("STEP:") != std::string::npos) {
                    // STEP行，跳过
                    continue;
                } else if (line.find("Matrix Dimension of H(R):") != std::string::npos) {
                    // 矩阵维数行
                    size_t pos = line.find(":");
                    if (pos != std::string::npos) {
                        matrix_dimension = std::stoi(line.substr(pos + 1));
                    }
                    header_processed = true;
                }
                continue;
            }
            
            // 处理R向量行 (格式: dRx dRy dRz nonZero)
            int dRx, dRy, dRz, nonZero;
            if (iss >> dRx >> dRy >> dRz >> nonZero) {
                // 只需要有非零元素的R向量
                if (nonZero > 0) {
                    Rlist_abacus.push_back(Vector3_Order<int>{dRx, dRy, dRz});
                }
                
                // 跳过接下来的3行（矩阵值、列指标、行非零元素累计和）
                for (int i = 0; i < 3; ++i) {
                    if (!std::getline(csr_file, line)) {
                        break;
                    }
                    line_number++;
                }
            }
        }
        
        csr_file.close();
        std::cout << "CSR文件读取完成，共提取 " << Rlist_abacus.size() << " 个R向量" << std::endl;
        
        // 输出前几个R向量作为验证
        int max_display = std::min(5, (int)Rlist_abacus.size());
        for (int i = 0; i < max_display; ++i) {
            std::cout << "R[" << i << "] = (" << Rlist_abacus[i].x << ", " 
                      << Rlist_abacus[i].y << ", " << Rlist_abacus[i].z << ")" << std::endl;
        }
        
    } else {
        std::cerr << "无法打开CSR文件: " << csr_filename << std::endl;
    }
    // vector<Vector3_Order<int>> Rlist_hartree;
    // // const auto Cs_0R = Cs_data.data_libri.at(0).at({0, {}});
    // for (const auto&I_JR : Cs_data.data_libri) {
    //     const auto& I = I_JR.first;
    //     for (const auto& JR_Cs : I_JR.second) {
    //         const auto& J = JR_Cs.first.first;
    //         const auto& Rc = JR_Cs.first.second;
    //         if( I==1 && J==1 ){
    //             Rlist_hartree.push_back({Rc[0], Rc[1], Rc[2]});
    //             // printf("Checking  R111: (%d,%d,%d)\n", 
    //             //             Rc[0], Rc[1], Rc[2]);
    //         }
            
    //         // printf("Checking  R: (%d,%d,%d)\n", 
    //         //                 Rc[0], Rc[2], Rc[2]);
            
    //     }
    // }

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
    std::map<int, std::map<int, Matz>> hf_nao_band;
    std::map<int, std::map<int, Matz>> vxc;
    std::map<int, std::map<int, Matz>> hf;
    std::map<int, std::map<int, Matz>> hf_band;
    std::map<int, std::map<int, Matz>> s_nao;
    std::map<int, std::map<int, Matz>> s_band;
    std::map<int, std::map<int, Matz>> s_inverse;
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int, std::map<int, Matz>> vxc1;
    std::map<int, std::map<int, Matz>> vxc_band;
    std::map<int, std::map<int, Matz>> vxc0_band;
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
            std::string sFilePath = oss_s.str();  
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
                // std::ostringstream oss_s_new;
                // oss_s_new << "s" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                std::string sNewFilePath = oss_s.str();

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
                    std::cerr << "S matrix file not found: " << sFilePath_2 << " or " << sNewFilePath << std::endl;
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
            hf[ispin][ikpt] = conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);

            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵
            // vxc[ispin][ikpt] = vxc0[ispin][ikpt];
            // vxc[ispin][ikpt] = hf[ispin][ikpt];
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

    // initialize the QSGW_band object
    /* Below we handle the band k-points data
     * First load the information of k-points along the k-path */
    int n_basis_band, n_states_band, n_spin_band;
    int flag;
    std::vector<Vector3_Order<double>> kfrac_band = read_band_kpath_info(
        driver_params.input_dir + "band_kpath_info", n_basis_band, n_states_band, n_spin_band, flag);
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
        Profiler::stop("qsgw_band");
        return;
    }
    mpi_comm_global_h.barrier();

    Profiler::start("qsgw_band_load_band_mf", "Read eigen solutions at band kpoints");
    auto meanfield_band = read_meanfield_band(driver_params.input_dir, n_basis_band, n_states_band,
                                              n_spin_band, kfrac_band.size());
    
    if (mpi_comm_global_h.is_root())
    {                                         
        const auto &mf = meanfield_band;
        for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < mf.get_n_kpoints(); i_kpoint++)
            {                                       
                const auto &k = kfrac_band[i_kpoint];
                printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1, i_kpoint + 1,
                       k.x, k.y, k.z);
                for (int ib1 = 0; ib1 < n_states_band; ++ib1)
                {
                    // meanfield_band.get_weight0()[i_spin](i_kpoint, ib1) = meanfield_band.get_weight()[i_spin](i_kpoint, ib1);
                    for (int isoc = 0; isoc < n_soc; isoc++)
                    {
                        for (int iao = 0; iao < n_basis_band; iao++)
                        {
                            // int ib2 = iao * n_soc + isoc;
                            auto wfc0_value = meanfield_band.get_eigenvectors()[i_spin][isoc][i_kpoint](ib1, iao);
                            // if(ib1 == 0 && iao == 0){
                            //     printf("ispin = %2d, ikpoint = %4d, wfc0_value = (%.5f, %.5f) \n", i_spin, i_kpoint, wfc0_value.real(), wfc0_value.imag() );
                            // }
                            

                            
                        }
                    }
                } 
            }
        }
    }
    Profiler::stop("qsgw_band_load_band_mf");

    Profiler::start("read_vxc_band", "Load DFT xc potential");

    // 读取 vxc_band 文件,H_KS0_band,S_band
    for (int i_spin = 0; i_spin < meanfield_band.get_n_spins(); i_spin++)
    {
        for (int i_kpoint = 0; i_kpoint < meanfield_band.get_n_kpoints(); i_kpoint++)
        {
            std::map<std::string, Matz> arrays_band;
            std::map<std::string, Matz> arrays;
            std::string key_vxc_band;
            std::string key_s;
            std::string key_hf;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_vxc_band;
            std::ostringstream oss_hf_band;
            oss_vxc_band << "band_vxc_mat_spin_" << (i_spin + 1) << "_k_" << std::setw(5)
                        << std::setfill('0') << (i_kpoint + 1) << ".csc";
            oss_hf_band << "band_hf_exchange_spin_0" << (i_spin + 1) << "_k_" << std::setw(5) << std::setfill('0') << (i_kpoint + 1) << ".csc";
            
            std::string vxcFilePath_band = oss_vxc_band.str();
            std::string hfFilePath_band = oss_hf_band.str();

            vxc_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);
            vxc0_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);
            hf_nao_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);
            bool vxc_band_file_found = false;
            bool hf_band_file_found = false;

            // 初始化矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i) {
                for (int j = 0; j < n_aos; ++j) {
                    hf_nao_band[i_spin][i_kpoint](i, j) = 0.0;
                }
            }
            // 读取 vxc_band 文件
            std::ifstream vxc_band_file(vxcFilePath_band.c_str());
            if (vxc_band_file.good()) {
                // 尝试读取原有格式的 vxc_band 矩阵
                if (!convert_csc(vxcFilePath_band, arrays_band, key_vxc_band)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath_band << std::endl;
                } else {
                    vxc0_band[i_spin][i_kpoint] = arrays_band[key_vxc_band];
                    vxc_band_file_found = true;
                }
            } else {
                // 如果原有格式的文件未找到，尝试新的格式读取
                std::ostringstream oss_vxc_band_new;
                oss_vxc_band_new << "band_vxcs" << (i_spin + 1) << "k" << (i_kpoint + 1) << "_nao.txt";
                std::string vxcBandNewFilePath = oss_vxc_band_new.str();

                // 打开文件
                std::ifstream vxc_band_new_format_file(vxcBandNewFilePath.c_str());
                if (vxc_band_new_format_file.good()) {
                    try {
                        // 读取矩阵维数（第一个数）
                        int matrix_size = 0;
                        vxc_band_new_format_file >> matrix_size;
                        if (matrix_size <= 0) {
                            throw std::runtime_error("Invalid matrix size in file: " + vxcBandNewFilePath);
                        }

                        // 初始化矩阵
                        Matz vxc_band_matrix(matrix_size, matrix_size, MAJOR::COL);

                        // 读取矩阵数据
                        for (int row = 0; row < matrix_size; ++row) {
                            for (int col = row; col < matrix_size; ++col) {
                                char ch; // 用于读取括号和逗号
                                double real_part, imag_part;

                                // 读取格式为 (real,imag)
                                vxc_band_new_format_file >> ch >> real_part >> ch >> imag_part >> ch;
                                if (vxc_band_new_format_file.fail()) {
                                    throw std::runtime_error("Error reading matrix data in file: " + vxcBandNewFilePath);
                                }

                                // 存储到上三角矩阵
                                vxc_band_matrix(row, col) = std::complex<double>(real_part, imag_part);
                            }
                        }

                        // 将上三角矩阵扩展为完整矩阵
                        for (int i = 0; i < matrix_size; ++i) {
                            for (int j = 0; j < i; ++j) {
                                vxc_band_matrix(i, j) = std::conj(vxc_band_matrix(j, i));
                            }
                        }

                        // 存储到 vxc_band
                        vxc0_band[i_spin][i_kpoint] = 0.5 * vxc_band_matrix;//单位转换
                        vxc_band_file_found = true;
                    } catch (const std::exception& e) {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process new format file: " << vxcBandNewFilePath
                                << ". Error: " << e.what() << std::endl;
                    }
                } else {
                    std::cerr << "VXC_band file not found: " << vxcFilePath_band << " or " << vxcBandNewFilePath << std::endl;
                }
            }    
            
            // 读取 hf_band 文件
            std::ifstream hf_file_band(hfFilePath_band.c_str());
            if (hf_file_band.good()) {
                if (!convert_csc(hfFilePath_band, arrays, key_hf)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << hfFilePath_band << std::endl;
                } 
                else 
                {
                    hf_nao_band[i_spin][i_kpoint] = arrays[key_hf];
                    hf_band_file_found = true;
                }
            } 
            else 
            {
                std::cerr << "HF_band file not found: " << hfFilePath_band << std::endl;
            }

            // 生成旧格式文件名
            std::ostringstream oss_band_ovlp;
            oss_band_ovlp << "band_ovlp_k_" << std::setw(5) << std::setfill('0') << (i_kpoint + 1) << ".csc";
            std::string bandOvlpFilePath = oss_band_ovlp.str();

            s_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);
            bool s_band_file_found = false;

            // 首先尝试读取旧格式文件 (band_ovlp_k_xxxxx.csc)
            std::ifstream band_ovlp_file(bandOvlpFilePath.c_str());
            if (band_ovlp_file.good()) {
                // 使用 convert_csc 函数读取 CSC 格式文件
                if (!convert_csc(bandOvlpFilePath, arrays, key_s)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << bandOvlpFilePath << std::endl;
                } else {
                    Matz s_band_matrix = arrays[key_s];
                    s_band[i_spin][i_kpoint] = s_band_matrix;
                    s_band_file_found = true;
                }  
            } else {
                // 构建新格式文件名
                std::ostringstream oss_s_band;
                oss_s_band << "band_sks" << (i_spin + 1) << "k" << (i_kpoint + 1) << "_nao.txt";
                std::string sBandFilePath = oss_s_band.str();

                std::ifstream s_band_file(sBandFilePath.c_str());
                if (s_band_file.good()) {
                    try {
                        // 读取矩阵维数（第一个数）
                        int matrix_size = 0;
                        s_band_file >> matrix_size;
                        if (matrix_size <= 0) {
                            throw std::runtime_error("Invalid matrix size in file: " + sBandFilePath);
                        }
                        
                        // 检查矩阵维度是否匹配预期
                        if (matrix_size != n_aos) {
                            throw std::runtime_error("Matrix size mismatch: expected " + 
                                                std::to_string(n_aos) + ", got " + 
                                                std::to_string(matrix_size));
                        }

                        // 初始化矩阵
                        Matz s_band_matrix(matrix_size, matrix_size, MAJOR::COL);

                        // 读取矩阵数据（上三角部分）
                        for (int row = 0; row < matrix_size; ++row) {
                            for (int col = row; col < matrix_size; ++col) {
                                char bracket, comma, closing_bracket;
                                double real_part, imag_part;

                                // 读取格式为 (real,imag)
                                s_band_file >> bracket >> real_part >> comma >> imag_part >> closing_bracket;
                                
                                if (s_band_file.fail() || bracket != '(' || comma != ',' || closing_bracket != ')') {
                                    throw std::runtime_error("Error reading matrix data at position (" + 
                                                        std::to_string(row) + "," + 
                                                        std::to_string(col) + ") in file: " + sBandFilePath);
                                }

                                // 存储到上三角矩阵
                                s_band_matrix(row, col) = std::complex<double>(real_part, imag_part);
                            }
                        }

                        // 将上三角矩阵扩展为完整矩阵（共轭对称）
                        for (int i = 0; i < matrix_size; ++i) {
                            for (int j = 0; j < i; ++j) {
                                s_band_matrix(i, j) = std::conj(s_band_matrix(j, i));
                            }
                        }

                        // 存储到 s_band
                        s_band[i_spin][i_kpoint] = s_band_matrix;
                        s_band_file_found = true;
                        
                    } catch (const std::exception& e) {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process s_band file: " << sBandFilePath 
                                << ". Error: " << e.what() << std::endl;
                    }
                } else {
                    all_files_processed_successfully = false;
                    std::cerr << "S_band file not found: " << bandOvlpFilePath << " or " << sBandFilePath << std::endl;
                }
            }

            
            
            Matz wfc5(n_bands, n_aos * n_soc, MAJOR::COL);
            for (int ib1 = 0; ib1 < n_bands; ++ib1)
            {
                meanfield_band.get_weight0()[i_spin](i_kpoint, ib1) =
                    meanfield_band.get_weight()[i_spin](i_kpoint, ib1);
                for (int isoc = 0; isoc < n_soc; isoc++)
                {
                    for (int iao = 0; iao < n_aos; iao++)
                    {
                        int ib2 = iao * n_soc + isoc;
                        wfc5(ib1, ib2) = meanfield_band.get_eigenvectors()[i_spin][isoc][i_kpoint](ib1, iao);
                        meanfield_band.get_eigenvectors0()[i_spin][isoc][i_kpoint](ib1, iao) = wfc5(ib1, ib2);
                    }
                }
            }

            hf_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);
            hf_band[i_spin][i_kpoint] = conj(wfc5) * hf_nao_band[i_spin][i_kpoint] * transpose(wfc5);  // row hf,KS
            // hf_band[i_spin][i_kpoint] = hf_nao_band[i_spin][i_kpoint] ; 
                                                                                 

            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵
            vxc_band[i_spin][i_kpoint] = vxc0_band[i_spin][i_kpoint];
            // vxc_band[i_spin][i_kpoint] = vxc0_band[i_spin][i_kpoint] + hf_band[i_spin][i_kpoint];
            vxc0_band[i_spin][i_kpoint] = vxc_band[i_spin][i_kpoint];

            H_KS0_band[i_spin][i_kpoint] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band)
            {
                H_KS0_band[i_spin][i_kpoint](i_band, i_band) = meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_band);
            }

            H_DFT_band_nao[i_spin][i_kpoint] = s_band[i_spin][i_kpoint] * transpose(wfc5) * (H_KS0_band[i_spin][i_kpoint] - vxc_band[i_spin][i_kpoint]) * conj(wfc5) * transpose(s_band[i_spin][i_kpoint],true);
            
        }
    }

    Profiler::stop("read_vxc_band");
    // std::flush(ofs_myid);


    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    printf("%5s\n", "efermi_band1");
    printf("%5f\n", efermi);
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
    int max_iterations = 1;             // 最大迭代次数
    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1;
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::trunc);
        file.close();
    }
    meanfield_band.get_efermi() = meanfield.get_efermi();
    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations)
    {
        iteration++;

        if (mpi_comm_global_h.is_root())
        {
            double efermi_band2 = meanfield_band.get_efermi();
            printf("%5s\n", "efermi_band2");
            printf("%5f\n", efermi_band2);
            // 更新前一轮的本征值
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                previous_eigenvalues[i_spin] = meanfield.get_eigenvals()[i_spin];
            }
        }
        mpi_comm_global_h.barrier();
        // Prepare time-frequency grids
        auto tfg =
        LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);
 
        Chi0 chi0(meanfield, klist, tfg);
        chi0.gf_R_threshold = Params::gf_R_threshold;

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
        Profiler::cease("read_vq_cut");

        std::vector<double> epsmac_LF_imagfreq_re;
        if (Params::replace_w_head )
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

        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS

        Profiler::start("qsgw_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
        {
            atpair_R_mat_t VR;
            if (Params::use_fullcoul_exx)
            {
                Profiler::start("ft_vq_full", "Fourier transform full Coulomb");
                VR = FT_Vq(Vq, meanfield.get_n_kpoints(), Rlist, true);
                Profiler::stop("ft_vq_full");
            }
            else
            {
                Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
                VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
                Profiler::stop("ft_vq_cut");
            }

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
            exx.build_KS_kgrid0();  // rotate
            Profiler::stop("g0w0_exx_real_work");
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

        if (Params::debug)
        {  // debug, check Wc
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

        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg, period);
        Profiler::start("g0w0_sigc_IJ", "Build real-space correlation self-energy");

        if (Params::use_soc)
            s_g0w0.build_spacetime<std::complex<double>>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);
        else
            s_g0w0.build_spacetime<double>(Cs_data, Wc_freq_q, Rlist, qlist, sinvS);

        Profiler::stop("g0w0_sigc_IJ");

        std::flush(ofs_myid);
        Profiler::start("g0w0_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid0();  // rotate
        Profiler::stop("g0w0_sigc_rotate_KS");

        std::map<int, std::map<int, Matz>> Vc_all;

        // 构建虚频点列表
        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            imagfreqs.push_back(cplxdb{0.0, freq});
        }

        std::map<int, std::map<int, std::map<int, double>>> e_qp_all;
        std::map<int, std::map<int, std::map<int, cplxdb>>> sigc_all;

        
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
                    for (int i_state_row = 0; i_state_row < n_bands; i_state_row++)
                    {
                        for (int i_state_col = 0; i_state_col < meanfield.get_n_bands();
                                i_state_col++)
                        {
                            std::vector<cplxdb> sigc_mn;
                            for (const auto &freq : chi0.tfg.get_freq_nodes())
                            {
                                sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                            }
                            LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs,
                                                        sigc_mn);
                            // LIBRPA::AnalyContNevanlinna nevanlinna(Params::n_params_anacon, imagfreqs,
                            //                            sigc_mn);
                            auto energy0 =
                                meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                            efermi = meanfield.get_efermi();
                            // 计算得到的值
                            auto result = pade.get(energy0 - efermi);
                            auto result1 = pade.get(0.0);
                            // auto result = nevanlinna.get(energy0 - efermi);
                            // auto result1 = nevanlinna.get(0.0);
                            // 存储值到 sigcmat
                            sigcmat[i_state_row][i_state_col][i_state_row] = result;
                            sigcmat[i_state_row][i_state_col][n_bands] = result1;
                        }
                    }

                    Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);
                    // Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k_modeA(sigcmat, n_bands);
                }
            }
            Profiler::stop("qsgw_solve_qpe");
            
            auto H0_GW_all = construct_H0_GW_cut(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                                                n_spins, n_kpoints, n_bands);
            
            
            //check
            //output H_nao_R
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                for(int i_soc1 = 0 ; i_soc1 < n_soc; i_soc1++)
                {
                    for(int i_soc2 =0 ; i_soc2 < n_soc; i_soc2++)
                    {
                        for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++)
                        {

                            Matz wfc2(n_bands, n_aos * n_soc, MAJOR::COL);

                            for (int ib1 = 0; ib1 < n_bands; ++ib1)
                            {
                                for (int isoc = 0; isoc < n_soc; isoc++)
                                {
                                    for (int iao = 0; iao < n_aos; iao++)
                                    {
                                        int ib2 = iao * n_soc + isoc;
                                        wfc2(ib1, ib2) = meanfield.get_eigenvectors0()[i_spin][isoc][i_kpoint](ib1, iao);       
                                        
                                    }
                                }
                            }
                            // conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);
                            
                            H_nao[i_spin][i_kpoint] = s_nao[i_spin][i_kpoint] * transpose(wfc2) * H0_GW_all[i_spin][i_kpoint] * conj(wfc2) * transpose(s_nao[i_spin][i_kpoint],true);
                            // H_nao[i_spin][i_kpoint] = s_nao[i_spin][i_kpoint] * transpose(wfc2) * H_KS0[i_spin][i_kpoint] * conj(wfc2) * transpose(s_nao[i_spin][i_kpoint],true);
                            
                        }
                                
                        for (const auto &R : Rlist_abacus){
                            Matz mat_R_cplx(meanfield.get_n_aos(), meanfield.get_n_aos());
                            for(int i=0 ; i < n_bands; i++)
                            {
                                for(int j=0 ; j < n_bands; j++)
                                {
                                    for (int ik = 0; ik != meanfield.get_n_kpoints(); ik++)
                                    {
                                        
                                        auto ang = -(kfrac_list[ik] * R) * TWO_PI;
                                        complex<double> kphase = complex<double>(cos(ang), sin(ang));
                                        mat_R_cplx(i,j) += kphase * H_nao[i_spin][ik](i,j) ;
                                    }
                                    // printf("R=(%d,%d,%d) H_nao_R(%d,%d)=(%f,%f)\n", R.x, R.y, R.z, i, j, mat_R_cplx(i,j).real(), mat_R_cplx(i,j).imag());
                                    mat_R_cplx(i,j) = mat_R_cplx(i,j) / static_cast<double>(meanfield.get_n_kpoints());

                                }
                                
                            
                            }
                            H_nao_R[i_spin][i_soc1][i_soc2][R] = mat_R_cplx ;
                            
                        }
                        
                    }
                }
            }
            std::cout << "开始写入 hrs_nao 文件" << std::endl;

            
            std::ostringstream filename;
            filename << "hrs1_nao_" << iteration << ".csr";
            std::ofstream ofs(filename.str());
            // if (!ofs) {
            //     std::cerr << "无法创建或打开 hrs1_nao_1.csr 文件，错误码: " << errno << std::endl;
                
            // }

            // 写入文件头信息
            int nlocal = meanfield.get_n_aos();
            ofs << "STEP: 0" << std::endl;
            ofs << "Matrix Dimension of H(R): " << nlocal << std::endl;
            
            // 统计有效的R向量数量
            int output_R_number = 0;
            for (int i_spin = 0; i_spin < n_spins; ++i_spin) {
                for (int i_soc1 = 0; i_soc1 < n_soc; ++i_soc1) {
                    for (int i_soc2 = 0; i_soc2 < n_soc; ++i_soc2) {
                        for (const auto& R_entry : H_nao_R[i_spin][i_soc1][i_soc2]) {
                            bool has_nonzero = false;
                            const Matz& mat = R_entry.second;
                            int rows = mat.nr();
                            int cols = mat.nc();
                            for (int row = 0; row < rows; ++row) {
                                for (int col = 0; col < cols; ++col) {
                                    std::complex<double> value = mat(row, col);
                                    if (std::abs(value.real()*2.0) > 1e-10) {
                                        has_nonzero = true;
                                        break;//imaginary part
                                    }
                                }
                                if (has_nonzero) break;
                            }
                            if (has_nonzero) {
                                ++output_R_number;
                            }
                        }
                    }
                }
            }
            ofs << "Matrix number of H(R): " << output_R_number << std::endl;

            // 遍历 H_nao_R，按照新格式写入数据
            for (int i_spin = 0; i_spin < n_spins; ++i_spin) {
                for (int i_soc1 = 0; i_soc1 < n_soc; ++i_soc1) {
                    for (int i_soc2 = 0; i_soc2 < n_soc; ++i_soc2) {
                        for (const auto& R_entry : H_nao_R[i_spin][i_soc1][i_soc2]) {
                            const auto& R = R_entry.first;
                            int dRx = R.x, dRy = R.y, dRz = R.z;
                            const Matz& mat = R_entry.second;
                            int rows = mat.nr();
                            int cols = mat.nc();

                            // 收集非零元素信息
                            std::vector<double> matrix_values;
                            std::vector<int> col_indices;
                            std::vector<int> row_nonzero_counts;
                            int nonZero = 0;
                            
                            // 按行收集数据
                            for (int row = 0; row < rows; ++row) {
                                int row_nonzero = 0;
                                for (int col = 0; col < cols; ++col) {
                                    std::complex<double> value = mat(row, col);
                                    if (std::abs(value.real() * 2.0) > 1e-10) {
                                        matrix_values.push_back(value.real());
                                        // matrix_values.push_back(value.imag());
                                        col_indices.push_back(col);
                                        ++nonZero;
                                        ++row_nonzero;
                                    }
                                }
                                row_nonzero_counts.push_back(row_nonzero);
                            }

                            // 只写入有非零元素的R向量
                            if (nonZero > 0) {
                                // 第4行：Rlist的点和对应R点的非0矩阵元的个数
                                ofs << dRx << " " << dRy << " " << dRz << " " << nonZero << std::endl;

                                // 第5行：该R点的非0矩阵元按行输出到一整行
                                for (size_t i = 0; i < matrix_values.size(); ++i) {
                                    ofs << " " << std::scientific << std::setprecision(16) << matrix_values[i] * 2.0;//Ryd
                                }
                                ofs << std::endl;

                                // 第6行：记录的是第五行非0矩阵元的列指标
                                for (size_t i = 0; i < col_indices.size(); ++i) {
                                    ofs << " " << col_indices[i];
                                }
                                ofs << std::endl;

                                // 第7行：先输出一个0，然后依次输出矩阵的各行存在的非0矩阵元的个数
                                ofs << " 0";
                                int cumulative_sum = 0;
                                for (size_t i = 0; i < row_nonzero_counts.size(); ++i) {
                                    cumulative_sum += row_nonzero_counts[i];
                                    ofs << " " << cumulative_sum;
                                }
                                ofs << std::endl;
                            }
                        }
                    }
                }
            }

            ofs.close();
            std::cout << "hrs1_nao_iteration.csr 文件 (文本格式) 写入完成" << std::endl;
            // 混合
            //  if(iteration > 1){
            //      for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            //          for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            //              H0_GW_all[ispin][ikpt] = 0.2 * H0_GW_all[ispin][ikpt] + 0.8 *
            //              H_KS[ispin][ikpt];
            //          }
            //      }
            //  }

            // 第三步：对 Hamiltonian 进行对角化并存储本征值
            diagonalize_and_store_fixed_basis(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);

            // 计算全局费米能和占据数
            const auto &Efermi0 = meanfield.get_efermi();
            printf("%5s\n", "efermi0");
            printf("%5f\n", Efermi0);
            // 计算费米能级

            double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
            printf("%5s\n", "efermi0");
            printf("%5f\n", efermi);

            // 将占据数和费米能级更新到 MeanField 对象中
            update_fermi_energy_and_occupations(meanfield, temperature, efermi);
            efermi_values.push_back(efermi * HA2EV);

            // const std::string final_banner(90, '-');
            lib_printf("Quasi-Particle Energy after QSGW Iterations [unit: eV]\n\n");
            const auto &Efermi = meanfield.get_efermi();
            printf("%5s\n", "efermi");
            printf("%5f\n", Efermi);
            for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
                {
                    const auto &k = kfrac_list[i_kpoint];
                    printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1,
                            i_kpoint + 1, k.x, k.y, k.z);
                    printf("%77s\n", final_banner.c_str());
                    printf("%5s %16s %16s %16s \n", "State", "e_mf", "v_xc",
                            "v_exx");
                    printf("%77s\n", final_banner.c_str());
                    for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                    {
                        const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                        // const auto &eks_state = H0_GW_all[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &exx_state1 = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                        const auto &exx_state2 =
                            exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &vxc_state =
                            vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                        const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                        printf("%5d %20.15f %16.5f %16.5f \n", i_state + 1,eks_state, vxc_state.real(), exx_state1);
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
        
        mpi_comm_global_h.barrier();

        mpi_comm_global_h.broadcast(converged, 0);
        mpi_comm_global_h.barrier();
        meanfield.broadcast(mpi_comm_global_h, 0);
        mpi_comm_global_h.barrier();

        // QSGW_band iteration
        /*
         * Compute the QP energies on band k-paths
         */
        // Reset k-space EXX and Sigmac matrices to avoid warning from internal reset
        exx.reset_kspace();
        s_g0w0.reset_kspace();
        /* reconstruct  exx, sigma_c matrix on k_band_path*/
        Profiler::start("g0w0_sigx_rotate_KS");
        exx.build_KS_band(meanfield_band.get_eigenvectors0(), kfrac_band);
        // exx.build_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);

        Profiler::stop("g0w0_sigx_rotate_KS");
        std::flush(ofs_myid);

        Profiler::start("g0w0_sigc_rotate_KS");
        s_g0w0.build_sigc_matrix_KS_band(meanfield_band.get_eigenvectors0(), kfrac_band);
        // s_g0w0.build_sigc_matrix_KS_band(meanfield_band.get_eigenvectors(), kfrac_band);
        Profiler::stop("g0w0_sigc_rotate_KS");
        std::flush(ofs_myid);
        mpi_comm_global_h.barrier();
        if (mpi_comm_global_h.is_root())
        {
            /*qpe solver*/
            Profiler::start("g0w0_solve_qpe", "Solve quasi-particle equation");

            std::cout << "Solving quasi-particle equation\n";

            // TODO: parallelize analytic continuation and QPE solver among tasks

            map<int, map<int, map<int, double>>> e_qp_all;
            map<int, map<int, map<int, cplxdb>>> sigc_all;

            for (int i_spin = 0; i_spin < meanfield_band.get_n_spins(); i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < meanfield_band.get_n_kpoints(); i_kpoint++)
                {
                    std::vector<std::vector<std::vector<cplxdb>>> sigcmat(
                        n_bands, std::vector<std::vector<cplxdb>>(
                                     n_bands, std::vector<cplxdb>(n_bands + 1)));
                    const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
                    const auto &k = kfrac_band[i_kpoint];
                    // printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                    //     i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                    for (int i_state_row = 0; i_state_row < meanfield_band.get_n_bands();
                         i_state_row++)
                    {
                        for (int i_state_col = 0; i_state_col < meanfield_band.get_n_bands();
                             i_state_col++)
                        {
                            std::vector<cplxdb> sigc_mn;
                            for (const auto &freq : chi0.tfg.get_freq_nodes())
                            {
                                sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                            }

                            LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);
                            // LIBRPA::AnalyContNevanlinna nevanlinna(Params::n_params_anacon, imagfreqs,
                            //                                sigc_mn);
                            auto energy0 = meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                            efermi = meanfield_band.get_efermi();
                            // 计算得到的值
                            auto result = pade.get(energy0 - efermi);
                            auto result1 = pade.get(0.0);
                            // auto result = nevanlinna.get(energy0 - efermi);
                            // auto result1 = nevanlinna.get(0.0);
                            // 存储值到 sigcmat
                            sigcmat[i_state_row][i_state_col][i_state_row] = result;
                            sigcmat[i_state_row][i_state_col][n_bands] = result1;
                        }
                    }
                    Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);
                    // Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k_modeA(sigcmat, n_bands);

            }
            }

            // reconstruct H0_GW_all
            auto H0_GW_all_band = construct_H0_GW_cut(
                meanfield_band, H_KS0_band, vxc_band, exx.exx_is_ik_KS, Vc_all,
                meanfield_band.get_n_spins(), meanfield_band.get_n_kpoints(), n_bands);

            diagonalize_and_store_fixed_basis(meanfield_band, H0_GW_all_band, meanfield_band.get_n_spins(),
                                  meanfield_band.get_n_kpoints(), n_bands);

            double total_electrons_band = total_electrons;
            printf("%5s\n", "Total_electrons_band");
            printf("%5f\n", total_electrons_band);
            double efermi_band0 =
                calculate_fermi_energy(meanfield_band, temperature, total_electrons_band);
            printf("%5s\n", "efermi_band0");
            printf("%5f\n", efermi_band0);
            update_fermi_energy_and_occupations(meanfield_band, temperature, efermi_band0);
            double efermi_band1 = meanfield_band.get_efermi();
            printf("%5s\n", "efermi_band1");
            printf("%5f\n", efermi_band1);
            meanfield_band.get_efermi() = meanfield.get_efermi();


            for (int i_spin = 0; i_spin < meanfield_band.get_n_spins(); i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < meanfield_band.get_n_kpoints(); i_kpoint++)
                {
                    const auto &k = kfrac_band[i_kpoint];
                    printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1,
                            i_kpoint + 1, k.x, k.y, k.z);
                    printf("%77s\n", final_banner.c_str());
                    printf("%5s %16s %16s %16s \n", "State", "e_mf", "v_xc",
                            "v_exx");
                    printf("%77s\n", final_banner.c_str());
                    for (int i_state = 0; i_state < meanfield_band.get_n_bands(); i_state++)
                    {
                        const auto &eks_state = meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                        // const auto &eks_state = H0_GW_all[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &exx_state1 = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                        const auto &exx_state2 =
                            exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &vxc_state =
                            vxc0_band[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
                        printf("%5d %20.15f %16.5f %16.5f \n", i_state + 1,eks_state, vxc_state.real(), exx_state1);
                    }
                    printf("\n");
                }
            }
                
            // display results
            for (int i_spin = 0; i_spin < meanfield_band.get_n_spins(); i_spin++)
            {
                std::ofstream ofs_ks;
                std::ofstream ofs_hf;
                std::ofstream ofs_qsgw;
                std::stringstream fn;

                fn << "EXX_band_spin_" << i_spin + 1 << "_" << iteration << ".dat";
                ofs_hf.open(fn.str());

                fn.str("");
                fn.clear();
                fn << "KS_band_spin_" << i_spin + 1 << "_" << iteration << ".dat";
                ofs_ks.open(fn.str());

                fn.str("");
                fn.clear();
                fn << "QSGW_band_spin_" << i_spin + 1 << "_" << iteration << "_cut10.dat";
                ofs_qsgw.open(fn.str());

                ofs_hf << std::fixed;
                ofs_ks << std::fixed;
                ofs_qsgw << std::fixed;

                for (int i_kpoint = 0; i_kpoint < meanfield_band.get_n_kpoints(); i_kpoint++)
                {
                    const auto &k = kfrac_band[i_kpoint];
                    ofs_ks << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                           << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                           << std::setprecision(7) << k.z;
                    ofs_hf << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7)
                           << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15)
                           << std::setprecision(7) << k.z;
                    ofs_qsgw << std::setw(5) << i_kpoint + 1 << std::setw(15)
                             << std::setprecision(7) << k.x << std::setw(15) << std::setprecision(7)
                             << k.y << std::setw(15) << std::setprecision(7) << k.z;

                    for (int i_state = 0; i_state < meanfield_band.get_n_bands(); i_state++)
                    {
                        const auto &occ_state0 =
                            meanfield_band.get_weight0()[i_spin](i_kpoint, i_state) *
                            (meanfield_band.get_n_kpoints() * meanfield_band.get_n_spins());
                        const auto &occ_state =
                            meanfield_band.get_weight()[i_spin](i_kpoint, i_state) *
                            (meanfield_band.get_n_kpoints() * meanfield_band.get_n_spins());
                        const auto &eks_state =
                            H_KS0_band[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &exx_state =
                            exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &vxc_state =
                            vxc_band[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        const auto &H_state = meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                         

                        ofs_ks << std::setw(15) << std::setprecision(5) << occ_state0
                               << std::setw(15) << std::setprecision(5)
                               << std::real(eks_state.real());
                        ofs_hf << std::setw(15) << std::setprecision(5) << occ_state0
                               << std::setw(15) << std::setprecision(5)
                               << std::real(eks_state.real() - vxc_state + exx_state);
                        ofs_qsgw << std::setw(15) << std::setprecision(5) << occ_state
                                 << std::setw(15) << std::setprecision(5) << H_state;
                    }
                    ofs_hf << "\n";
                    ofs_ks << "\n";
                    ofs_qsgw << "\n";
                }
            }
        }
        mpi_comm_global_h.barrier();
        meanfield_band.broadcast(mpi_comm_global_h, 0);
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
    Profiler::stop("qsgw_band");
}
