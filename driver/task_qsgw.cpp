#include "task_qsgw.h"
// 标准库头文件
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>   // 用于文件存在检查
#include <iomanip>   // 用于格式化
#include <iostream>  // 用于输入输出操作
#include <limits>   // std::numeric_limits
#include <map>       // 用于std::map容器
#include <sstream>
#include <string>  // 用于std::string类
#include <vector>
// 自定义头文件

#include "Hamiltonian.h"  // 哈密顿量相关
#include "abacus_symmetry.h"
#include "analycont.h"    // 分析延拓相关
#include "chi0.h"         // 响应函数相关
#include "constants.h"    // 常量定义
#include "convert_csc.h"
#include "coulmat.h"  // 库仑矩阵相关
#include "driver_params.h"
#include "driver_utils.h"
#include "inputfile.h"
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

namespace
{

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
void reset_iteration_history()
{
    efermi_values.clear();
    homo_values.clear();
    lumo_values.clear();
    iteration_numbers.clear();
}

void append_iteration_history(const int iteration, const double homo_ev, const double lumo_ev,
                              const double efermi_ev)
{
    iteration_numbers.push_back(iteration);
    homo_values.push_back(homo_ev);
    lumo_values.push_back(lumo_ev);
    efermi_values.push_back(efermi_ev);
}

void write_iteration_history_file(const std::string &path)
{
    std::ofstream file(path);
    if (!file.good())
    {
        throw std::runtime_error("Failed to open history file for write: " + path);
    }

    for (size_t i = 0; i < iteration_numbers.size(); ++i)
    {
        file << iteration_numbers[i] << " " << homo_values[i] << " " << lumo_values[i] << " "
             << efermi_values[i] << std::endl;
    }
}

bool load_iteration_history_file(const std::string &path, const int max_iteration)
{
    std::ifstream file(path);
    if (!file.good())
    {
        return false;
    }

    reset_iteration_history();

    int iteration = 0;
    double homo_ev = 0.0;
    double lumo_ev = 0.0;
    double efermi_ev = 0.0;
    while (file >> iteration >> homo_ev >> lumo_ev >> efermi_ev)
    {
        if (max_iteration >= 0 && iteration > max_iteration)
        {
            break;
        }
        append_iteration_history(iteration, homo_ev, lumo_ev, efermi_ev);
    }

    return !iteration_numbers.empty();
}

void compute_homo_lumo_ha(const MeanField &mf, double &homo_ha, double &lumo_ha)
{
    homo_ha = -1e6;
    lumo_ha = 1e6;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt)
        {
            int homo_level = -1;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                const double weight = mf.get_weight()[ispin](ikpt, ib);
                if (weight >= 1.0 / (mf.get_n_spins() * mf.get_n_kpoints()))
                {
                    homo_level = ib;
                }
            }

            if (homo_level >= 0)
            {
                homo_ha = std::max(homo_ha, mf.get_eigenvals()[ispin](ikpt, homo_level));
                if (homo_level + 1 < mf.get_n_bands())
                {
                    lumo_ha = std::min(lumo_ha,
                                       mf.get_eigenvals()[ispin](ikpt, homo_level + 1));
                }
            }
        }
    }
}

void ensure_dir(const std::string &dir)
{
    std::system(("mkdir -p " + dir).c_str());
}

std::string qsgw_checkpoint_save_root()
{
    return Params::output_dir + "qsgw_checkpoints/";
}

std::string qsgw_checkpoint_load_root()
{
    if (!Params::qsgw_restart_dir.empty())
    {
        return Params::qsgw_restart_dir;
    }
    return qsgw_checkpoint_save_root();
}

std::string checkpoint_iteration_dir(const std::string &checkpoint_root, const int iteration)
{
    std::ostringstream oss;
    oss << checkpoint_root << "iter_" << std::setw(5) << std::setfill('0') << iteration << "/";
    return oss.str();
}

std::string checkpoint_matrix_file(const std::string &checkpoint_dir, const int ispin,
                                   const int ikpt)
{
    std::ostringstream oss;
    oss << checkpoint_dir << "H0_GW_spin_" << std::setw(2) << std::setfill('0') << (ispin + 1)
        << "_k_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".bin";
    return oss.str();
}

std::string checkpoint_hartree0_file(const std::string &checkpoint_dir, const int ispin,
                                     const int ikpt)
{
    std::ostringstream oss;
    oss << checkpoint_dir << "Hartree0_spin_" << std::setw(2) << std::setfill('0') << (ispin + 1)
        << "_k_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".bin";
    return oss.str();
}

void write_matz_binary(const Matz &mat, const std::string &path)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.good())
    {
        throw std::runtime_error("Failed to open checkpoint matrix for write: " + path);
    }

    const std::int32_t nr = mat.nr();
    const std::int32_t nc = mat.nc();
    ofs.write(reinterpret_cast<const char *>(&nr), sizeof(nr));
    ofs.write(reinterpret_cast<const char *>(&nc), sizeof(nc));
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nc; ++j)
        {
            const auto value = mat(i, j);
            const double re = value.real();
            const double im = value.imag();
            ofs.write(reinterpret_cast<const char *>(&re), sizeof(re));
            ofs.write(reinterpret_cast<const char *>(&im), sizeof(im));
        }
    }
}

Matz read_matz_binary(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good())
    {
        throw std::runtime_error("Failed to open checkpoint matrix for read: " + path);
    }

    std::int32_t nr = 0;
    std::int32_t nc = 0;
    ifs.read(reinterpret_cast<char *>(&nr), sizeof(nr));
    ifs.read(reinterpret_cast<char *>(&nc), sizeof(nc));
    if (!ifs.good() || nr <= 0 || nc <= 0)
    {
        throw std::runtime_error("Invalid checkpoint matrix header: " + path);
    }

    Matz mat(nr, nc, MAJOR::COL);
    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nc; ++j)
        {
            double re = 0.0;
            double im = 0.0;
            ifs.read(reinterpret_cast<char *>(&re), sizeof(re));
            ifs.read(reinterpret_cast<char *>(&im), sizeof(im));
            if (!ifs.good())
            {
                throw std::runtime_error("Failed to read checkpoint matrix body: " + path);
            }
            mat(i, j) = std::complex<double>(re, im);
        }
    }

    return mat;
}

void write_qsgw_checkpoint(const std::string &checkpoint_root, const int iteration,
                           const std::map<int, std::map<int, Matz>> &H0_GW_all,
                           const double efermi_ha,
                           const std::map<int, std::map<int, Matz>> *hartree0 = nullptr)
{
    ensure_dir(checkpoint_root);
    const auto checkpoint_dir = checkpoint_iteration_dir(checkpoint_root, iteration);
    ensure_dir(checkpoint_dir);
    const bool has_hartree0 = (hartree0 != nullptr && !hartree0->empty());

    {
        std::ofstream meta(checkpoint_dir + "checkpoint.meta");
        if (!meta.good())
        {
            throw std::runtime_error("Failed to open checkpoint meta for write: " + checkpoint_dir);
        }
        meta << "iteration " << iteration << "\n";
        meta << "efermi_ha " << std::setprecision(17) << efermi_ha << "\n";
        meta << "has_hartree0 " << (has_hartree0 ? 1 : 0) << "\n";
    }

    for (const auto &spin_entry : H0_GW_all)
    {
        for (const auto &k_entry : spin_entry.second)
        {
            write_matz_binary(k_entry.second,
                              checkpoint_matrix_file(checkpoint_dir, spin_entry.first, k_entry.first));
        }
    }

    if (has_hartree0)
    {
        for (const auto &spin_entry : *hartree0)
        {
            for (const auto &k_entry : spin_entry.second)
            {
                write_matz_binary(
                    k_entry.second,
                    checkpoint_hartree0_file(checkpoint_dir, spin_entry.first, k_entry.first));
            }
        }
    }

    {
        std::ofstream latest(checkpoint_root + "latest_iteration.txt");
        if (!latest.good())
        {
            throw std::runtime_error("Failed to open latest checkpoint marker for write: " +
                                     checkpoint_root);
        }
        latest << iteration << "\n";
    }
}

struct QsgwCheckpointState
{
    int iteration = -1;
    double efermi_ha = 0.0;
    std::map<int, std::map<int, Matz>> H0_GW_all;
    std::map<int, std::map<int, Matz>> Hartree_0;
    bool has_hartree0 = false;
};

QsgwCheckpointState load_qsgw_checkpoint(const std::string &checkpoint_root,
                                         const int requested_iteration, const int n_spins,
                                         const int n_kpoints)
{
    QsgwCheckpointState state;
    if (requested_iteration > 0)
    {
        state.iteration = requested_iteration;
    }
    else
    {
        std::ifstream latest(checkpoint_root + "latest_iteration.txt");
        if (!latest.good())
        {
            throw std::runtime_error("Cannot find latest_iteration.txt in " + checkpoint_root);
        }
        latest >> state.iteration;
    }

    if (state.iteration <= 0)
    {
        throw std::runtime_error("Invalid QSGW restart iteration in " + checkpoint_root);
    }

    const auto checkpoint_dir = checkpoint_iteration_dir(checkpoint_root, state.iteration);
    {
        std::ifstream meta(checkpoint_dir + "checkpoint.meta");
        if (!meta.good())
        {
            throw std::runtime_error("Cannot open checkpoint meta: " + checkpoint_dir);
        }

        std::string key;
        while (meta >> key)
        {
            if (key == "iteration")
            {
                meta >> state.iteration;
            }
            else if (key == "efermi_ha")
            {
                meta >> state.efermi_ha;
            }
            else if (key == "has_hartree0")
            {
                int has_hartree0 = 0;
                meta >> has_hartree0;
                state.has_hartree0 = (has_hartree0 != 0);
            }
        }
    }

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            state.H0_GW_all[ispin][ikpt] =
                read_matz_binary(checkpoint_matrix_file(checkpoint_dir, ispin, ikpt));
            if (state.has_hartree0)
            {
                state.Hartree_0[ispin][ikpt] =
                    read_matz_binary(checkpoint_hartree0_file(checkpoint_dir, ispin, ikpt));
            }
        }
    }

    return state;
}
}  // namespace

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


            const std::string input_dir = driver_params.input_dir;
            std::string hfFilePath = input_dir + "/" + oss_hf.str();
            std::string vxcFilePath = input_dir + "/" + oss_vxc.str();
            std::string sFilePath = input_dir + "/" + oss_s.str();  // 获取S矩阵文件路径
            std::string sFilePath_2 = input_dir + "/" + oss_s2.str();
            
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
                    s_nao[ispin][ikpt](i, j) = (i == j ? 1.0 : 0.0);  // default: identity if no S file
                    s_inverse[ispin][ikpt](i, j) = (i == j ? 1.0 : 0.0);  // default: identity if no S file
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
                        std::cerr << "Failed to process new format file: " + sNewFilePath
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
                        std::cerr << "Failed to process new format file: " + vxcNewFilePath
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
    if (LIBRPA::parallel_routing == LIBRPA::ParallelRouting::R_TAU
        || need_full_cut_coulomb_for_abacus_symmetry())
    {
        if (need_full_cut_coulomb_for_abacus_symmetry() && mpi_comm_global_h.is_root())
        {
            lib_printf("ABACUS GW/EXX symmetry builds `V(R)` directly from the full IBZ operator;"
                       " switching to `read_Vq_full`\n");
        }
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
    compute_homo_lumo_ha(meanfield, homo, lumo);

    // 计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n", "Total_electrons");
    printf("%5f\n", total_electrons);

    // ==============================
    // QSGW convergence + mixing controls
    // ==============================
    // NOTE: driver/inputfile.cpp does NOT parse max_iter/mixing/mixing_history into Params.
    // We re-parse librpa.in locally here so users can control these via input file.
    double eigenvalue_diff_tolerance = 1e-3;  // eV (strict)
    int max_iterations = 500;

    // Convergence focus window around Fermi (0 => full-band). Default: focus 10 bands near Fermi.
    // For an insulator (efermi in the gap), this corresponds roughly to {HOMO-4..HOMO} + {LUMO..LUMO+4}.
    int eigdiff_focus_nbands = 10;

    // Aggressive: focus-window weighted Hamiltonian mixing (0 => disable; default ties to eigdiff_focus_nbands)
    int mix_focus_nbands = eigdiff_focus_nbands;
    // Damping factor for updates outside focus window (0 => freeze outside window)
    double mix_focus_outside_damp = 0.1;

    // Repro scheme (from LibRPA-develop-fix/task_qsgwA.cpp): fixed Pulay mixer knobs.
    // Optional diagnostic override via environment variables:
    //   LIBRPA_QSGW_MIXING_HISTORY (int >= 1)
    //   LIBRPA_QSGW_MIXING_BETA    (double > 0)
    double mixing_beta = 0.2;
    int mixing_history = 12;
    const int linear_mixing_steps = 3; // first N steps rely on mixer’s linear fallback

    {
        const char* env_hist = std::getenv("LIBRPA_QSGW_MIXING_HISTORY");
        if (env_hist != nullptr) {
            try {
                mixing_history = std::stoi(std::string(env_hist));
            } catch (...) {
                if (mpi_comm_global_h.is_root()) {
                    std::cerr << "[QSGW] Warning: invalid LIBRPA_QSGW_MIXING_HISTORY='" << env_hist
                              << "', keep default " << mixing_history << std::endl;
                }
            }
        }
        if (mixing_history < 1) {
            if (mpi_comm_global_h.is_root()) {
                std::cerr << "[QSGW] Warning: mixing_history=" << mixing_history
                          << " < 1, reset to 1." << std::endl;
            }
            mixing_history = 1;
        }

        const char* env_beta = std::getenv("LIBRPA_QSGW_MIXING_BETA");
        if (env_beta != nullptr) {
            try {
                mixing_beta = std::stod(std::string(env_beta));
            } catch (...) {
                if (mpi_comm_global_h.is_root()) {
                    std::cerr << "[QSGW] Warning: invalid LIBRPA_QSGW_MIXING_BETA='" << env_beta
                              << "', keep default " << mixing_beta << std::endl;
                }
            }
        }
        if (!(mixing_beta > 0.0)) {
            if (mpi_comm_global_h.is_root()) {
                std::cerr << "[QSGW] Warning: mixing_beta=" << mixing_beta
                          << " <= 0, reset to 0.2." << std::endl;
            }
            mixing_beta = 0.2;
        }

        if (mpi_comm_global_h.is_root()) {
            std::cout << "[QSGW] Mixer knobs: history=" << mixing_history
                      << ", beta=" << mixing_beta << std::endl;
        }
    }

    // Optional Hamiltonian band window (diagnostic acceleration)
    int hamiltonian_cut_above_fermi = -1;
    double hamiltonian_cut_diag_shift_ev = 20.0;

    {
        int flag = 0;
        InputFile inputf;
        auto parser = inputf.load(input_filename, false);
        parser.parse_int("max_iter", max_iterations, max_iterations, flag);
        // NOTE: ignore librpa.in mixing knobs for reproducibility with historical scheme
        parser.parse_int("hamiltonian_cut_above_fermi", hamiltonian_cut_above_fermi, hamiltonian_cut_above_fermi, flag);
        parser.parse_double("hamiltonian_cut_diag_shift_ev", hamiltonian_cut_diag_shift_ev, hamiltonian_cut_diag_shift_ev, flag);
        // optional override
        parser.parse_double("eigenvalue_diff_tolerance", eigenvalue_diff_tolerance, eigenvalue_diff_tolerance, flag);
        parser.parse_int("eigdiff_focus_nbands", eigdiff_focus_nbands, eigdiff_focus_nbands, flag);
        parser.parse_int("mix_focus_nbands", mix_focus_nbands, mix_focus_nbands, flag);
        parser.parse_double("mix_focus_outside_damp", mix_focus_outside_damp, mix_focus_outside_damp, flag);
    }

    // No beta clamping / adaptive caps here: follow historical fixed settings.

    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1;
    std::vector<std::pair<int, int>> significant_positions;

    // For convergence check: store previous iteration eigenvalues
    std::vector<matrix> previous_eigenvalues(n_spins);

    // ==========================================
    // Initialize Pulay Mixer (mixing Hamiltonian H0_GW_all)
    // ==========================================
    PulayMixer mixer(mixing_history, mixing_beta);
    bool mixer_initialized = false;
    const auto checkpoint_save_root = qsgw_checkpoint_save_root();
    const auto checkpoint_load_root = qsgw_checkpoint_load_root();

    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        reset_iteration_history();
        ensure_dir(checkpoint_save_root);
        if (use_iterative_pyatb_headwing_bundle())
        {
            initialize_headwing_velocity_from_input(meanfield);
        }

        if (Params::qsgw_restart)
        {
            const auto checkpoint =
                load_qsgw_checkpoint(checkpoint_load_root, Params::qsgw_restart_iteration,
                                     n_spins, n_kpoints);
            diagonalize_and_store_fixed_basis(meanfield, checkpoint.H0_GW_all, n_spins, n_kpoints,
                                              n_bands);
            update_fermi_energy_and_occupations(meanfield, temperature, checkpoint.efermi_ha);
            compute_homo_lumo_ha(meanfield, homo, lumo);
            iteration = checkpoint.iteration;
            Hartree_0 = checkpoint.Hartree_0;

            if (!checkpoint.has_hartree0)
            {
                throw std::runtime_error(
                    "QSGW restart checkpoint is missing Hartree_0. "
                    "Regenerate checkpoints with the updated code before resuming.");
            }

            const auto history_loaded =
                load_iteration_history_file(checkpoint_load_root + "homo_lumo_vs_iterations.dat",
                                            iteration);
            if (!history_loaded)
            {
                append_iteration_history(iteration, homo * HA2EV, lumo * HA2EV,
                                         checkpoint.efermi_ha * HA2EV);
            }

            std::cout << "[QSGW] Restarting from checkpoint iteration " << iteration << " at "
                      << checkpoint_load_root << std::endl;
            std::cout << "[QSGW] Restored HOMO = " << homo * HA2EV << " eV, "
                      << "LUMO = " << lumo * HA2EV << " eV, "
                      << "Fermi Energy = " << checkpoint.efermi_ha * HA2EV << " eV\n";
        }
        else
        {
            append_iteration_history(0, homo * HA2EV, lumo * HA2EV, efermi * HA2EV);
            std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
                      << "LUMO = " << lumo * HA2EV << " eV, "
                      << "Fermi Energy = " << efermi * HA2EV << " eV\n";
        }

        if (use_iterative_pyatb_headwing_bundle())
        {
            refresh_pyatb_headwing_bundle(meanfield, kfrac_list);
        }

        plot_homo_lumo_vs_iterations();
        write_iteration_history_file(checkpoint_save_root + "homo_lumo_vs_iterations.dat");
    }
    mpi_comm_global_h.broadcast(iteration, 0);
    meanfield.broadcast(mpi_comm_global_h, 0);
    mpi_comm_global_h.barrier();
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
            if (Params::use_shrink_abfs)
            {
                Hartree.build(Cs_shrinked_data, Rlist, VR);
            }
            else
            {
                Hartree.build(Cs_data, Rlist, VR);
            }
            
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
                }
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
            if (Params::option_dielect_func != 3 && Params::option_dielect_func != 4)
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
            const auto VR = FT_Vq(Vq_cut, get_full_bz_kpoint_count(), Rlist, true);
            Profiler::stop("ft_vq_cut");

            Profiler::start("g0w0_exx_real_work");
            const auto& exx_cs = Params::use_shrink_abfs ? Cs_shrinked_data : Cs_data;
            if (Params::use_soc)
                exx.build<std::complex<double>>(exx_cs, Rlist, VR);
            else
                exx.build<double>(exx_cs, Rlist, VR);
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
        // 新增：用于存储纯 GW 部分的关联势（不含 Hartree 修正），在内循环中保持固定
        std::map<int, std::map<int, Matz>> Vc_GW_pure;

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

                        // Optional: evaluate retarded Sigma at w + i*eta (stabilize AC)
                        // AC scheme (from LibRPA-develop-fix/task_qsgwA.cpp):
                        // - pre-AC truncate sigc(iω) to 1e-8
                        // - if max(|sigc(iω)|) <= 1e-6: skip Pade and use sigc(iω0) as constant
                        // - post-AC truncate result to 1e-6 (and sanitize NaN/Inf -> 0)
                        const double sigc_trunc_in_scale = 1.0e8;
                        const double ac_trunc_out_scale = 1.0e6;
                        const double pade_threshold = 1.0e-6;


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
                                double max_magnitude = 0.0;
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
                                    auto val = sigc_sk.at(freq)(i_state_row, i_state_col);

                                    // pre-AC rounding (match historical qsgwA scheme)
                                    {
                                        const double re = std::round(val.real() * sigc_trunc_in_scale) / sigc_trunc_in_scale;
                                        const double im = std::round(val.imag() * sigc_trunc_in_scale) / sigc_trunc_in_scale;
                                        val = cplxdb{re, im};
                                    }

                                    const double mag = std::abs(val);
                                    if (mag > max_magnitude) max_magnitude = mag;
                                    sigc_mn.push_back(val);
                                }

                                auto energy0 =
                                    meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                                efermi = meanfield.get_efermi();
                                if (!std::isfinite(energy0)) energy0 = efermi;

                                auto result = cplxdb{0.0, 0.0};
                                auto result1 = cplxdb{0.0, 0.0};

                                if (max_magnitude > pade_threshold)
                                {
                                    try
                                    {
                                        LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);
                                        result = pade.get(energy0 - efermi);
                                        result1 = pade.get(0.0);
                                    }
                                    catch (...)
                                    {
                                        if (!sigc_mn.empty())
                                        {
                                            result = sigc_mn[0];
                                            result1 = sigc_mn[0];
                                        }
                                    }
                                }
                                else
                                {
                                    if (!sigc_mn.empty())
                                    {
                                        result = sigc_mn[0];
                                        result1 = sigc_mn[0];
                                    }
                                }

                                if (!std::isfinite(result.real()) || !std::isfinite(result.imag())) result = cplxdb{0.0, 0.0};
                                if (!std::isfinite(result1.real()) || !std::isfinite(result1.imag())) result1 = cplxdb{0.0, 0.0};

                                {
                                    // Try nearbyint (ties-to-even) for bitwise reproducibility vs historical reference.
                                    // const double re = std::round(result.real() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    // const double im = std::round(result.imag() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    const double re = std::round(result.real() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    const double im = std::round(result.imag() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    result = cplxdb{re, im};
                                    // const double re1 = std::round(result1.real() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    // const double im1 = std::round(result1.imag() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    const double re1 = std::round(result1.real() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    const double im1 = std::round(result1.imag() * ac_trunc_out_scale) / ac_trunc_out_scale;
                                    result1 = cplxdb{re1, im1};
                                }

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

                        // 修改：先保存纯 GW 势，再计算 Vc_all
                        Vc_GW_pure[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);
                        Vc_all[i_spin][i_kpoint] = Vc_GW_pure[i_spin][i_kpoint];

                        // Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k_modeA(sigcmat, n_bands);
                        if(iteration>1){
                            Matz delta_Hartree_is_ik(n_bands, n_bands, MAJOR::COL);
                            delta_Hartree_is_ik = Hartree_i_delta[i_spin][i_kpoint];
                            Vc_all[i_spin][i_kpoint] = Vc_all[i_spin][i_kpoint] + delta_Hartree_is_ik;
                        }

                        

                        
                    }

                    
                }
                Profiler::stop("qsgw_solve_qpe");

                // // ======================================================================================
                // // Inner Hartree Loop: Fix Vxc (GW), Update Hartree Self-Consistently
                // // ======================================================================================
                
                // auto Vc_all_outer = Vc_all; 

                // int max_inner_iterations = 100; 
                // double inner_tolerance = 1e-6;
                // bool inner_converged = false;
                // int inner_iter = 0;
                
                // // 策略调整：对于小分子/简单体系，使用极小的 alpha 防止过冲
                // double mixing_alpha = 0.05; 
                // PulayMixer inner_mixer(10, mixing_alpha); 
                // bool inner_mixer_initialized = false;
                // double prev_max_inner_diff = 1e10;
                
                // // 新增：用于手动线性混合的历史记录
                // matrix H_prev_step;

                // if (mpi_comm_global_h.is_root()) std::cout << ">>> Starting Inner Hartree Self-Consistency Loop <<<\n";

                // Profiler::start("inner_hartree_prep");
                // const auto VR_inner = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
                // Profiler::stop("inner_hartree_prep");

                // while (!inner_converged && inner_iter < max_inner_iterations) {
                //     inner_iter++;

                //     // 1. 更新 Hartree 势
                //     if (inner_iter > 1) {
                //         Profiler::start("inner_hartree_update");
                //         Hartree.build(Cs_data, Rlist, VR_inner); 
                //         Hartree.build_KS_kgrid0(); // rotate

                //         for (int ispin = 0; ispin < n_spins; ++ispin) {
                //             for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
                //                 Matz current_Hartree_val(n_bands, n_bands, MAJOR::COL);
                //                 for(int i=0; i<n_bands; ++i) {
                //                     for(int j=0; j<n_bands; ++j) {
                //                         current_Hartree_val(i,j) = Hartree.Hartree_is_ik_KS[ispin][ikpt](i,j);
                //                     }
                //                 }
                //                 Matz delta = current_Hartree_val - Hartree_0[ispin][ikpt];
                //                 Vc_all[ispin][ikpt] = Vc_all_outer[ispin][ikpt] + delta;
                //             }
                //         }
                //         Profiler::stop("inner_hartree_update");
                //     }

                //     // 2. 构建哈密顿量 (H_out)
                //     auto H0_GW_all = construct_H0_GW(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                //                                 n_spins, n_kpoints, n_bands);

                //     // ==========================================
                //     // Robust Mixing Strategy: Linear -> Pulay
                //     // ==========================================
                //     if (mpi_comm_global_h.is_root()) {
                //         int total_matrices = n_spins * n_kpoints;
                //         int rows_per_matrix = n_bands;
                //         int cols_per_matrix = n_bands;
                        
                //         matrix mixed_input(total_matrices * rows_per_matrix, 2 * cols_per_matrix);
                        
                //         int row_offset = 0;
                //         for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                //             for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                //                 const Matz& mat = H0_GW_all[i_spin][i_kpoint];
                //                 for (int i = 0; i < rows_per_matrix; i++) {
                //                     for (int j = 0; j < cols_per_matrix; j++) {
                //                         std::complex<double> val = mat(i, j);
                //                         mixed_input(row_offset + i, j) = val.real();
                //                         mixed_input(row_offset + i, j + cols_per_matrix) = val.imag();
                //                     }
                //                 }
                //                 row_offset += rows_per_matrix;
                //             }
                //         }

                //         // 策略：前5步强制使用线性混合 (Simple Mixing) 稳定方向
                //         // 之后如果误差小于 0.1 才启用 Pulay
                //         bool use_pulay = (inner_iter > 5 && prev_max_inner_diff < 0.1);
                //         matrix mixed_output;

                //         if (use_pulay) {
                //             if (!inner_mixer_initialized) {
                //                 // 切换到 Pulay 时，使用上一步的结果作为初始历史
                //                 inner_mixer.initialize(H_prev_step);
                //                 inner_mixer_initialized = true;
                //             } 
                            
                //             try {
                //                 mixed_output = inner_mixer.mix(mixed_input);
                //             } catch (...) {
                //                 // Pulay 失败回退到线性
                //                 std::cout << "    Pulay failed. Fallback to Linear Mixing." << std::endl;
                //                 inner_mixer.reset();
                //                 inner_mixer_initialized = false;
                //                 mixed_output = (1.0 - mixing_alpha) * H_prev_step + mixing_alpha * mixed_input;
                //             }
                //         } else {
                //             // 强制线性混合
                //             if (inner_iter == 1) {
                //                 mixed_output = mixed_input;
                //             } else {
                //                 mixed_output = (1.0 - mixing_alpha) * H_prev_step + mixing_alpha * mixed_input;
                //             }
                            
                //             // 保持 Pulay 状态重置
                //             if (inner_mixer_initialized) {
                //                 inner_mixer.reset();
                //                 inner_mixer_initialized = false;
                //             }
                //         }
                        
                //         // 更新历史
                //         H_prev_step = mixed_output;
                            
                //         // Unpack
                //         row_offset = 0;
                //         for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                //             for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                //                 Matz& mat = H0_GW_all[i_spin][i_kpoint];
                //                 for (int i = 0; i < rows_per_matrix; i++) {
                //                     for (int j = 0; j < cols_per_matrix; j++) {
                //                         double re = mixed_output(row_offset + i, j);
                //                         double im = mixed_output(row_offset + i, j + cols_per_matrix);
                //                         mat(i, j) = std::complex<double>(re, im);
                //                     }
                //                 }
                //                 row_offset += rows_per_matrix;
                //             }
                //         }
                //     }
                    
                //     // Broadcast mixed Hamiltonian (Fixed for std::vector)
                //     for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                //         for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                //              Matz& mat = H0_GW_all[i_spin][i_kpoint];
                //              int size = mat.nr() * mat.nc();
                //              if (size > 0) {
                //                  std::vector<double> buffer(size * 2);
                //                  if (mpi_comm_global_h.is_root()) {
                //                      double* ptr = reinterpret_cast<double*>(&mat(0, 0));
                //                      std::copy(ptr, ptr + size * 2, buffer.begin());
                //                  }
                //                  mpi_comm_global_h.broadcast(buffer, 0);
                //                  if (!mpi_comm_global_h.is_root()) {
                //                      double* ptr = reinterpret_cast<double*>(&mat(0, 0));
                //                      std::copy(buffer.begin(), buffer.end(), ptr);
                //                  }
                //              }
                //         }
                //     }
                //     // ==========================================

                //     std::vector<matrix> prev_inner_eigs(n_spins);
                //     if (mpi_comm_global_h.is_root()) {
                //         for(int s=0; s<n_spins; ++s) prev_inner_eigs[s] = meanfield.get_eigenvals()[s];
                //     }

                //     // 3. 对角化
                //     diagonalize_and_store_fixed_basis(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);

                //     // 4. 更新费米能级
                //     double efermi_inner = calculate_fermi_energy(meanfield, temperature, total_electrons);
                //     update_fermi_energy_and_occupations(meanfield, temperature, efermi_inner);

                //     // 5. 检查收敛性 & 自适应重置
                //     inner_converged = true;
                //     double max_inner_diff = 0.0;
                //     if (mpi_comm_global_h.is_root()) {
                //         for (int ispin = 0; ispin < n_spins; ++ispin) {
                //             const auto &curr = meanfield.get_eigenvals()[ispin];
                //             double diff = (curr - prev_inner_eigs[ispin]).absmax();
                //             if (diff > max_inner_diff) max_inner_diff = diff;
                //         }
                        
                //         printf("    Inner Iter %2d: Max Eigenval Diff = %12.6e (Fermi = %12.6f eV)\n", 
                //                inner_iter, max_inner_diff, efermi_inner * HA2EV);

                //         if (max_inner_diff > inner_tolerance) {
                //             inner_converged = false;
                //             // 激进的重置策略：只要误差增加，立即重置混合器
                //             if (inner_iter > 1 && max_inner_diff > prev_max_inner_diff) {
                //                 std::cout << "    Warning: Error increased. Resetting Pulay Mixer." << std::endl;
                //                 inner_mixer_initialized = false; 
                //             }
                //         }
                //         prev_max_inner_diff = max_inner_diff;
                //     }
                    
                //     mpi_comm_global_h.broadcast(inner_converged, 0);
                //     mpi_comm_global_h.broadcast(inner_mixer_initialized, 0); 
                //     mpi_comm_global_h.barrier();
                //     meanfield.broadcast(mpi_comm_global_h, 0); 
                // }
                
                // if (mpi_comm_global_h.is_root()) std::cout << ">>> Inner Loop Finished <<<\n";
                // // ======================================================================================

                // 重新构建 H0_GW_all 以供外循环使用
                // auto H0_GW_all = construct_H0_GW_new_basis(meanfield, H_KS0, H_DFT_nao, exx.exx_is_ik_KS, Vc_all,
                //                                  n_spins, n_kpoints, n_bands);
                std::map<int, std::map<int, Matz>> H0_GW_all;
                if (hamiltonian_cut_above_fermi >= 0) {
                    H0_GW_all = construct_H0_GW_fermi_window(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                                                           n_spins, n_kpoints, n_bands,
                                                           hamiltonian_cut_above_fermi,
                                                           hamiltonian_cut_diag_shift_ev);
                } else {
                    H0_GW_all = construct_H0_GW(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                                               n_spins, n_kpoints, n_bands);
                }
              
                // ==========================================
                // Pulay Mixing Execution (Outer Loop)
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
                        if (mpi_comm_global_h.is_root()) {
                            std::cout << "Pulay Mixer Initialized (History=" << mixing_history << ", Beta=" << mixer.get_mixing_beta() << ")" << std::endl;
                        }
                    } else {
                        try {
                            matrix mixed_output;

                            // 引入 Linear / Pulay 切换策略
                            if (iteration <= linear_mixing_steps) {
                                // Linear Mixing: H_new = (1-beta)*H_old + beta*H_calc
                                // PulayMixer 类如果没有显式提供 linear_mix 方法，我们可以手动实现，
                                // 或者通常 PulayMixer 在 history 填满前行为类似 linear。
                                // 这里假设我们暂时依赖 mixer 的默认行为，但通过降低 beta 来稳健化。

                                // 如果您的 PulayMixer 实现支持在初期退化为 Linear，那最好。
                                // 否则，最简单的改进是：直接使用较小的 beta (0.2)。
                                mixed_output = mixer.mix(mixed_input);
                                if (mpi_comm_global_h.is_root()) std::cout << " mixing (iter " << iteration << ")..." << std::endl;
                            } else {
                                mixed_output = mixer.mix(mixed_input);
                            }



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
                diagonalize_and_store_fixed_basis(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);

                // ==========================================
                // 矩阵调试输出（已注释以减少内存占用）
                // 如果需要调试，可以取消注释
                // ==========================================
                // 打印 H0_GW_all 矩阵内容以供检查
                
                if (Params::debug)
                {
                std::cout << "\n" << std::string(60, '=') << "\n";
                std::cout << " DEBUG: H0_GW_all Matrix Dump (Iteration " << iteration << ")\n";
                std::cout << std::string(60, '=') << "\n";
                for (int ispin = 0; ispin < n_spins; ++ispin) {
                    for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
                        std::cout << "Spin " << ispin + 1 << ", K-point " << ikpt + 1 << ":\n";
                        if (H0_GW_all.count(ispin) && H0_GW_all[ispin].count(ikpt)) {
                            const auto& mat = H0_GW_all[ispin][ikpt];
                            for (int i = 0; i < mat.nr(); ++i) {
                                for (int j = 0; j < mat.nc(); ++j) {
                                    // 格式化输出复数
                                    std::cout << mat(i, j) << " ";
                                }
                                std::cout << "\n";
                            }
                        } else {
                            std::cout << " (Matrix not found)\n";
                        }
                        std::cout << "\n";
                    }
                }
                std::cout << std::string(60, '=') << "\n";
                
                }

                // ==========================================
                // 计算全局费米能和占据数
                const auto &Efermi0 = meanfield.get_efermi();
                printf("%5s\n", "efermi_input");
                printf("%5f\n", Efermi0);
                // 计算费米能级

                double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
                printf("%5s\n", "efermi_qsgw");
                printf("%5f\n", efermi);

                update_fermi_energy_and_occupations(meanfield, temperature, efermi);
                compute_homo_lumo_ha(meanfield, homo, lumo);

                if (use_iterative_pyatb_headwing_bundle())
                {
                    refresh_pyatb_headwing_bundle(meanfield, kfrac_list);
                }

                // 输出当前 HOMO 和 LUMO 值
                std::cout << "Iteration " << iteration << ": HOMO = " << homo * HA2EV << " eV, "
                          << "LUMO = " << lumo * HA2EV << " eV, "
                          << "Efermi = " << efermi * HA2EV << " eV\n";


                // Convergence check:
                //  - Full-band sorted diff is permutation-invariant diagnostics.
                //  - Focus-window sorted diff near Fermi (eigdiff_focus_nbands) is used for convergence.
                double max_eigenvalue_diff_ev = 0.0;
                double max_eigenvalue_diff_ev_sorted = 0.0;
                double max_eigenvalue_diff_ev_sorted_focus = 0.0;
                int max_spin = -1, max_kpt = -1, max_band = -1;

                if (mpi_comm_global_h.is_root())
                {
                    for (int i_spin = 0; i_spin < n_spins; i_spin++)
                    {
                        const auto &curr_mat = meanfield.get_eigenvals()[i_spin];
                        const auto &prev_mat = previous_eigenvalues[i_spin];

                        for (int ikpt = 0; ikpt < n_kpoints; ikpt++)
                        {
                            std::vector<double> curr_eigs(n_bands);
                            std::vector<double> prev_eigs(n_bands);
                            for (int ib = 0; ib < n_bands; ib++)
                            {
                                double c = curr_mat(ikpt, ib);
                                double p_ = prev_mat(ikpt, ib);
                                curr_eigs[ib] = c;
                                prev_eigs[ib] = p_;

                                double d = std::abs(c - p_) * HA2EV;
                                if (d > max_eigenvalue_diff_ev)
                                {
                                    max_eigenvalue_diff_ev = d;
                                    max_spin = i_spin;
                                    max_kpt = ikpt;
                                    max_band = ib;
                                }
                            }

                            std::sort(curr_eigs.begin(), curr_eigs.end());
                            std::sort(prev_eigs.begin(), prev_eigs.end());

                            for (int ib = 0; ib < n_bands; ib++)
                            {
                                double d = std::abs(curr_eigs[ib] - prev_eigs[ib]) * HA2EV;
                                if (d > max_eigenvalue_diff_ev_sorted)
                                {
                                    max_eigenvalue_diff_ev_sorted = d;
                                }
                            }

                            if (eigdiff_focus_nbands > 0)
                            {
                                // Determine occupied count by energy relative to current efermi.
                                // For an insulator, efermi is in the gap so Nocc is stable.
                                const int nbelow = eigdiff_focus_nbands / 2;
                                const int nabove = eigdiff_focus_nbands - nbelow;
                                int Nocc = 0;
                                for (int ib = 0; ib < n_bands; ib++)
                                {
                                    if (curr_mat(ikpt, ib) < efermi)
                                        Nocc++;
                                }
                                int start_ib = std::max(0, Nocc - nbelow);
                                int end_ib = std::min(n_bands, Nocc + nabove);
                                if (end_ib > start_ib)
                                {
                                    std::vector<double> curr_focus;
                                    std::vector<double> prev_focus;
                                    curr_focus.reserve(end_ib - start_ib);
                                    prev_focus.reserve(end_ib - start_ib);
                                    for (int ib = start_ib; ib < end_ib; ib++)
                                    {
                                        curr_focus.push_back(curr_mat(ikpt, ib));
                                        prev_focus.push_back(prev_mat(ikpt, ib));
                                    }

                                    // Sort within the window (permutation-invariant within focus subspace).
                                    std::sort(curr_focus.begin(), curr_focus.end());
                                    std::sort(prev_focus.begin(), prev_focus.end());
                                    for (size_t ii = 0; ii < curr_focus.size(); ii++)
                                    {
                                        double d = std::abs(curr_focus[ii] - prev_focus[ii]) * HA2EV;
                                        if (d > max_eigenvalue_diff_ev_sorted_focus)
                                        {
                                            max_eigenvalue_diff_ev_sorted_focus = d;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    std::cout << "    max eigenvalue diff (raw)    = " << max_eigenvalue_diff_ev;
                    if (max_spin >= 0)
                    {
                        std::cout << " eV @ (spin=" << max_spin << ", k=" << max_kpt
                                  << ", band=" << max_band << ")";
                    }
                    std::cout << std::endl;

                    std::cout << "    max eigenvalue diff (sorted) = " << max_eigenvalue_diff_ev_sorted
                              << " eV (tol=" << eigenvalue_diff_tolerance << " eV)" << std::endl;

                    if (eigdiff_focus_nbands > 0)
                    {
                        std::cout << "    max eigenvalue diff (sorted, focus=" << eigdiff_focus_nbands
                                  << ") = " << max_eigenvalue_diff_ev_sorted_focus
                                  << " eV (tol=" << eigenvalue_diff_tolerance << " eV)" << std::endl;
                    }

                    const double eigdiff_for_conv = (eigdiff_focus_nbands > 0)
                        ? max_eigenvalue_diff_ev_sorted_focus
                        : max_eigenvalue_diff_ev_sorted;

                    converged = (eigdiff_for_conv < eigenvalue_diff_tolerance);
                    if (converged)
                    {
                        std::cout << "Converged after " << iteration << " iterations: "
                                  << "max eigenvalue diff(";
                        if (eigdiff_focus_nbands > 0)
                            std::cout << "sorted, focus=" << eigdiff_focus_nbands;
                        else
                            std::cout << "sorted";
                        std::cout << ") = " << eigdiff_for_conv << " eV" << std::endl;
                    }

                    append_iteration_history(iteration, homo * HA2EV, lumo * HA2EV,
                                             efermi * HA2EV);
                    plot_homo_lumo_vs_iterations();
                    write_iteration_history_file(checkpoint_save_root +
                                                 "homo_lumo_vs_iterations.dat");

                    const bool should_write_checkpoint =
                        (Params::qsgw_checkpoint_every > 0 &&
                         (iteration % Params::qsgw_checkpoint_every == 0)) ||
                        converged || iteration == max_iterations;
                    if (should_write_checkpoint)
                    {
                        write_qsgw_checkpoint(checkpoint_save_root, iteration, H0_GW_all, efermi,
                                              &Hartree_0);
                    }
                }
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
    write_iteration_history_file("homo_lumo_vs_iterations.dat");
}
