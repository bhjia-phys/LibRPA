#include "task_qsgw_band_0.h"

#include "task_qsgw.h"
// 标准库头文件
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>   // 用于文件存在检查
#include <iomanip>   // 用于格式化
#include <iostream>  // 用于输入输出操作
#include <map>       // 用于std::map容器
#include <regex>
#include <sstream>
#include <stdexcept>
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
#include "envs_io.h"
#include "envs_mpi.h"
#include "epsilon.h"                  // 介电函数相关
#include "exx.h"                      // Exact exchange相关
#include "fermi_energy_occupation.h"  // 费米能和占据数计算相关
#include "gw.h"                       // GW计算相关
#include "hartree.h"
#include "inputfile.h"
#include "matrix.h"
#include "meanfield.h"   // MeanField类相关
#include "params.h"      // 参数设置相关
#include "pbc.h"         // 周期性边界条件相关
#include "profiler.h"    // 性能分析工具
#include "qpe_solver.h"  // 准粒子方程求解器
#include "read_data.h"
#include "ri.h"
#include "utils_timefreq.h"
#include "write_aims.h"

namespace
{

void reset_iteration_history_band()
{
    iteration_numbers.clear();
    homo_values.clear();
    lumo_values.clear();
    efermi_values.clear();
}

void append_iteration_history_band(const int iteration, const double homo_ev, const double lumo_ev,
                                   const double efermi_ev)
{
    iteration_numbers.push_back(iteration);
    homo_values.push_back(homo_ev);
    lumo_values.push_back(lumo_ev);
    efermi_values.push_back(efermi_ev);
}

void write_iteration_history_file_band(const std::string &path)
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

bool load_iteration_history_file_band(const std::string &path, const int max_iteration)
{
    std::ifstream file(path);
    if (!file.good())
    {
        return false;
    }

    reset_iteration_history_band();
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
        append_iteration_history_band(iteration, homo_ev, lumo_ev, efermi_ev);
    }
    return !iteration_numbers.empty();
}

void restore_band_meanfield_from_qsgw_band_files(MeanField &mf_band,
                                                 const std::vector<Vector3_Order<double>> &kfrac_band,
                                                 const std::string &input_dir,
                                                 const int iteration)
{
    const int n_spins = mf_band.get_n_spins();
    const int n_kpoints = mf_band.get_n_kpoints();
    const int n_bands = mf_band.get_n_bands();
    const double occ_scale = static_cast<double>(n_kpoints * n_spins);
    constexpr double k_tol = 5.0e-7;

    if (iteration <= 0)
    {
        return;
    }

    for (int i_spin = 0; i_spin < n_spins; ++i_spin)
    {
        std::ostringstream filename;
        filename << input_dir << "QSGW_band_spin_" << i_spin + 1 << "_" << iteration << ".dat";
        std::ifstream file(filename.str());
        if (!file.good())
        {
            throw std::runtime_error("QSGW band0 restart requires previous band file: "
                                     + filename.str());
        }

        for (int i_kpoint = 0; i_kpoint < n_kpoints; ++i_kpoint)
        {
            int row_index = 0;
            double kx = 0.0;
            double ky = 0.0;
            double kz = 0.0;
            if (!(file >> row_index >> kx >> ky >> kz))
            {
                throw std::runtime_error("Malformed previous QSGW band file: " + filename.str());
            }
            if (row_index != i_kpoint + 1)
            {
                throw std::runtime_error("Unexpected k-point row index in previous QSGW band file: "
                                         + filename.str());
            }
            const auto &k_ref = kfrac_band[i_kpoint];
            if (std::abs(kx - k_ref.x) > k_tol || std::abs(ky - k_ref.y) > k_tol
                || std::abs(kz - k_ref.z) > k_tol)
            {
                throw std::runtime_error("K-point mismatch in previous QSGW band file: "
                                         + filename.str());
            }

            for (int i_band = 0; i_band < n_bands; ++i_band)
            {
                double occ_printed = 0.0;
                double energy_ev = 0.0;
                if (!(file >> occ_printed >> energy_ev))
                {
                    throw std::runtime_error("Missing band entries in previous QSGW band file: "
                                             + filename.str());
                }
                mf_band.get_weight()[i_spin](i_kpoint, i_band) = occ_printed / occ_scale;
                mf_band.get_eigenvals()[i_spin](i_kpoint, i_band) = energy_ev / HA2EV;
            }
        }
    }

    std::cout << "QSGW band0: restored band-path eigenvalues from iteration " << iteration
              << " QSGW_band files." << std::endl;
}

void compute_homo_lumo_ha_band(const MeanField &mf, double &homo_ha, double &lumo_ha)
{
    homo_ha = -1e6;
    lumo_ha = 1e6;
    constexpr double occupation_tol = 1.0e-10;
    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt)
        {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                const double weight = mf.get_weight()[ispin](ikpt, ib);
                const double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                if (weight > occupation_tol)
                {
                    homo_ha = std::max(homo_ha, energy);
                }
                else
                {
                    lumo_ha = std::min(lumo_ha, energy);
                }
            }
        }
    }
}

void ensure_dir_band(const std::string &dir)
{
    std::system(("mkdir -p " + dir).c_str());
}

std::string qsgw_checkpoint_save_root_band()
{
    return Params::output_dir + "qsgw_checkpoints/";
}

std::string qsgw_checkpoint_load_root_band()
{
    if (!Params::qsgw_restart_dir.empty())
    {
        return Params::qsgw_restart_dir;
    }
    return qsgw_checkpoint_save_root_band();
}

std::string checkpoint_iteration_dir_band(const std::string &checkpoint_root, const int iteration)
{
    std::ostringstream oss;
    oss << checkpoint_root << "iter_" << std::setw(5) << std::setfill('0') << iteration << "/";
    return oss.str();
}

std::string checkpoint_matrix_file_band(const std::string &checkpoint_dir, const int ispin,
                                        const int ikpt)
{
    std::ostringstream oss;
    oss << checkpoint_dir << "H0_GW_spin_" << std::setw(2) << std::setfill('0') << (ispin + 1)
        << "_k_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".bin";
    return oss.str();
}

std::string checkpoint_hartree0_matrix_file_band(const std::string &checkpoint_dir,
                                                 const int ispin, const int ikpt)
{
    std::ostringstream oss;
    oss << checkpoint_dir << "Hartree0_spin_" << std::setw(2) << std::setfill('0')
        << (ispin + 1) << "_k_" << std::setw(6) << std::setfill('0') << (ikpt + 1)
        << ".bin";
    return oss.str();
}

void write_matz_binary_band(const Matz &mat, const std::string &path)
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

Matz read_matz_binary_band(const std::string &path)
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

void write_qsgw_checkpoint_band(const std::string &checkpoint_root, const int iteration,
                                const std::map<int, std::map<int, Matz>> &H0_GW_all,
                                const double efermi_ha,
                                const std::map<int, std::map<int, Matz>> *Hartree_0)
{
    ensure_dir_band(checkpoint_root);
    const auto checkpoint_dir = checkpoint_iteration_dir_band(checkpoint_root, iteration);
    ensure_dir_band(checkpoint_dir);

    std::ofstream meta(checkpoint_dir + "checkpoint.meta");
    if (!meta.good())
    {
        throw std::runtime_error("Failed to open checkpoint meta for write: " + checkpoint_dir);
    }
    meta << "iteration " << iteration << "\n";
    meta << "efermi_ha " << std::setprecision(17) << efermi_ha << "\n";
    meta << "has_hartree0 " << (Hartree_0 != nullptr ? 1 : 0) << "\n";

    for (const auto &spin_entry : H0_GW_all)
    {
        for (const auto &k_entry : spin_entry.second)
        {
            write_matz_binary_band(
                k_entry.second,
                checkpoint_matrix_file_band(checkpoint_dir, spin_entry.first, k_entry.first));
        }
    }

    if (Hartree_0 != nullptr)
    {
        for (const auto &spin_entry : *Hartree_0)
        {
            for (const auto &k_entry : spin_entry.second)
            {
                write_matz_binary_band(
                    k_entry.second,
                    checkpoint_hartree0_matrix_file_band(
                        checkpoint_dir, spin_entry.first, k_entry.first));
            }
        }
    }

    std::ofstream latest(checkpoint_root + "latest_iteration.txt");
    if (!latest.good())
    {
        throw std::runtime_error("Failed to open latest checkpoint marker for write: " +
                                 checkpoint_root);
    }
    latest << iteration << "\n";
}

struct QsgwCheckpointStateBand
{
    int iteration = -1;
    double efermi_ha = 0.0;
    bool has_hartree0 = false;
    std::map<int, std::map<int, Matz>> H0_GW_all;
    std::map<int, std::map<int, Matz>> Hartree_0;
};

QsgwCheckpointStateBand load_qsgw_checkpoint_band(const std::string &checkpoint_root,
                                                  const int requested_iteration, const int n_spins,
                                                  const int n_kpoints)
{
    QsgwCheckpointStateBand state;
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

    const auto checkpoint_dir = checkpoint_iteration_dir_band(checkpoint_root, state.iteration);
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

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            state.H0_GW_all[ispin][ikpt] =
                read_matz_binary_band(checkpoint_matrix_file_band(checkpoint_dir, ispin, ikpt));
            if (state.has_hartree0)
            {
                state.Hartree_0[ispin][ikpt] = read_matz_binary_band(
                    checkpoint_hartree0_matrix_file_band(checkpoint_dir, ispin, ikpt));
            }
        }
    }
    return state;
}

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

bool read_abacus_upper_triangle_matrix(const std::string& file_path, Matz& mat,
                                       const double scale = 1.0)
{
    std::ifstream input(file_path.c_str());
    if (!input.good())
    {
        return false;
    }

    std::ostringstream content_stream;
    content_stream << input.rdbuf();
    const std::string content = content_stream.str();

    int matrix_size = 0;
    {
        std::istringstream lines(content);
        std::string line;
        std::regex rows_regex("^\\s*#\\s*rows\\s+([0-9]+)\\s*$",
                              std::regex_constants::icase);
        std::regex cols_regex("^\\s*#\\s*columns\\s+([0-9]+)\\s*$",
                              std::regex_constants::icase);
        int rows = 0;
        int cols = 0;
        while (std::getline(lines, line))
        {
            std::smatch match;
            if (std::regex_match(line, match, rows_regex))
            {
                rows = std::stoi(match[1].str());
            }
            else if (std::regex_match(line, match, cols_regex))
            {
                cols = std::stoi(match[1].str());
            }
        }
        if (rows > 0 || cols > 0)
        {
            if (rows <= 0 || cols <= 0 || rows != cols)
            {
                throw std::runtime_error("Invalid ABACUS text matrix header in file: " +
                                         file_path);
            }
            matrix_size = rows;
        }
    }

    if (matrix_size <= 0)
    {
        std::istringstream first_token_stream(content);
        first_token_stream >> matrix_size;
        if (matrix_size <= 0)
        {
            throw std::runtime_error("Invalid matrix size in file: " + file_path);
        }
    }

    std::vector<std::complex<double>> values;
    const std::regex complex_regex(
        "\\(\\s*([^,\\s\\)]+)\\s*,\\s*([^\\s\\)]+)\\s*\\)");
    for (auto it = std::sregex_iterator(content.begin(), content.end(), complex_regex);
         it != std::sregex_iterator(); ++it)
    {
        const std::smatch match = *it;
        values.emplace_back(std::stod(match[1].str()), std::stod(match[2].str()));
    }

    Matz parsed(matrix_size, matrix_size, MAJOR::COL);
    const std::size_t full_count =
        static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size);
    const std::size_t upper_count =
        static_cast<std::size_t>(matrix_size) * static_cast<std::size_t>(matrix_size + 1) / 2;

    if (values.size() == full_count)
    {
        std::size_t index = 0;
        for (int row = 0; row < matrix_size; ++row)
        {
            for (int col = 0; col < matrix_size; ++col)
            {
                parsed(row, col) = scale * values[index++];
            }
        }
    }
    else if (values.size() == upper_count)
    {
        std::size_t index = 0;
        for (int row = 0; row < matrix_size; ++row)
        {
            for (int col = row; col < matrix_size; ++col)
            {
                parsed(row, col) = scale * values[index++];
            }
        }
        for (int row = 0; row < matrix_size; ++row)
        {
            for (int col = 0; col < row; ++col)
            {
                parsed(row, col) = std::conj(parsed(col, row));
            }
        }
    }
    else
    {
        throw std::runtime_error("Unexpected matrix entry count in file: " + file_path);
    }

    mat = parsed;
    return true;
}

ComplexMatrix matz_to_complex_matrix(const Matz& mat)
{
    ComplexMatrix converted(mat.nr(), mat.nc());
    for (int row = 0; row < mat.nr(); ++row)
    {
        for (int col = 0; col < mat.nc(); ++col)
        {
            converted(row, col) = mat(row, col);
        }
    }
    return converted;
}

void accumulate_complex_matrix_to_matz(Matz& accumulator,
                                       const ComplexMatrix& value,
                                       const std::complex<double>& factor)
{
    if (accumulator.nr() != value.nr || accumulator.nc() != value.nc)
    {
        throw std::runtime_error("QSGW H(R) export accumulator has inconsistent dimensions");
    }
    for (int row = 0; row < accumulator.nr(); ++row)
    {
        for (int col = 0; col < accumulator.nc(); ++col)
        {
            accumulator(row, col) += factor * value(row, col);
        }
    }
}

void ensure_atom_nw_for_qsgw_hr_export()
{
    if (!atom_nw.empty())
    {
        return;
    }
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    if (!ctx.has_ao_shell_layout() || ctx.atom_to_type.empty())
    {
        return;
    }
    for (const auto& atom_type : ctx.atom_to_type)
    {
        const auto type_index = atom_type.second;
        if (type_index < 0 || type_index >= static_cast<int>(ctx.ao_type_layouts.size()))
        {
            atom_nw.clear();
            return;
        }
        atom_nw[atom_type.first] = ctx.ao_type_layouts[static_cast<std::size_t>(type_index)].nao;
    }
}

bool can_restore_qsgw_hr_export_with_abacus_symmetry(const int n_kpoints)
{
    const auto& ctx = LIBRPA::abacus_symmetry_ctx;
    return Params::use_abacus_gw_symmetry
           && ctx.available
           && ctx.has_ao_shell_layout()
           && !ctx.kstars.empty()
           && ctx.kstars.size() == kfrac_list.size()
           && n_kpoints == static_cast<int>(ctx.kstars.size())
           && static_cast<int>(klist.size()) < get_full_bz_kpoint_count()
           && static_cast<int>(ctx.count_kstar_members()) == get_full_bz_kpoint_count()
           && ctx.atom_to_type.size() == atom_nw.size()
           && ctx.input_coord_frac.size() == atom_nw.size();
}

std::vector<Vector3_Order<int>> read_csr_rlist(const std::string& file_path)
{
    std::ifstream input(file_path.c_str());
    if (!input.good())
    {
        throw std::runtime_error("Failed to open CSR file: " + file_path);
    }

    std::vector<Vector3_Order<int>> rlist;
    std::string line;
    bool header_processed = false;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        if (!header_processed)
        {
            if (line.find("Matrix Dimension of H(R):") != std::string::npos)
            {
                header_processed = true;
            }
            continue;
        }

        std::istringstream iss(line);
        int rx = 0;
        int ry = 0;
        int rz = 0;
        int nnz = 0;
        if (!(iss >> rx >> ry >> rz >> nnz))
        {
            continue;
        }
        if (nnz > 0)
        {
            rlist.push_back(Vector3_Order<int>{rx, ry, rz});
        }
        for (int i = 0; i < 3 && std::getline(input, line); ++i)
        {
        }
    }

    if (rlist.empty())
    {
        throw std::runtime_error("No non-empty R blocks found in CSR file: " + file_path);
    }
    return rlist;
}

void write_real_csr_from_dense_blocks(const std::string& file_path,
                                      const std::map<Vector3_Order<int>, Matz>& blocks,
                                      const int nlocal)
{
    constexpr double threshold = 1.0e-10;
    std::ofstream output(file_path.c_str());
    if (!output.good())
    {
        throw std::runtime_error("Failed to open CSR output file: " + file_path);
    }

    int n_blocks = 0;
    for (const auto& r_block : blocks)
    {
        const Matz& mat = r_block.second;
        bool has_nonzero = false;
        for (int row = 0; row < mat.nr() && !has_nonzero; ++row)
        {
            for (int col = 0; col < mat.nc(); ++col)
            {
                if (std::abs(mat(row, col).real() * 2.0) > threshold)
                {
                    has_nonzero = true;
                    break;
                }
            }
        }
        if (has_nonzero)
        {
            ++n_blocks;
        }
    }

    output << "STEP: 0\n";
    output << "Matrix Dimension of H(R): " << nlocal << '\n';
    output << "Matrix number of H(R): " << n_blocks << '\n';

    for (const auto& r_block : blocks)
    {
        const auto& R = r_block.first;
        const Matz& mat = r_block.second;
        std::vector<double> values;
        std::vector<int> cols;
        std::vector<int> row_ptr;
        row_ptr.reserve(static_cast<std::size_t>(nlocal) + 1);
        row_ptr.push_back(0);

        for (int row = 0; row < nlocal; ++row)
        {
            for (int col = 0; col < nlocal; ++col)
            {
                const double value_ry = mat(row, col).real() * 2.0;
                if (std::abs(value_ry) > threshold)
                {
                    values.push_back(value_ry);
                    cols.push_back(col);
                }
            }
            row_ptr.push_back(static_cast<int>(values.size()));
        }

        if (values.empty())
        {
            continue;
        }

        output << R.x << ' ' << R.y << ' ' << R.z << ' ' << values.size() << '\n';
        for (const double value : values)
        {
            output << ' ' << std::scientific << std::setprecision(16) << value;
        }
        output << '\n';
        for (const int col : cols)
        {
            output << ' ' << col;
        }
        output << '\n';
        for (const int ptr : row_ptr)
        {
            output << ' ' << ptr;
        }
        output << '\n';
    }
}


void export_band_basis_hamiltonian_to_abacus_csr(
    const std::vector<std::string>& output_files,
    const std::map<int, std::map<int, Matz>>& h_band,
    const std::map<int, std::map<int, Matz>>& s_nao,
    const MeanField& meanfield_ref,
    const int n_spins,
    const int n_kpoints,
    const int n_bands,
    const int n_aos,
    const int n_soc,
    const std::vector<Vector3_Order<int>>& rlist_abacus,
    const std::string& log_label)
{
    if (static_cast<int>(output_files.size()) != n_spins)
    {
        throw std::runtime_error("QSGW H(R) export received inconsistent spin file count");
    }
    if (rlist_abacus.empty())
    {
        throw std::runtime_error("QSGW H(R) export requires a non-empty ABACUS R list");
    }
    if (n_soc != 1)
    {
        std::cerr << "QSGW H(R) export is currently implemented for non-SOC ABACUS cases only; "
                  << "skip export for n_soc=" << n_soc << std::endl;
        return;
    }

    for (int i_spin = 0; i_spin < n_spins; ++i_spin)
    {
        const auto& symmetry_ctx = LIBRPA::abacus_symmetry_ctx;
        const bool use_export_symmetry = can_restore_qsgw_hr_export_with_abacus_symmetry(n_kpoints);
        const int nsym_space = static_cast<int>(symmetry_ctx.rspace_operations.size());
        const int nk_export = use_export_symmetry
                                  ? static_cast<int>(symmetry_ctx.count_kstar_members())
                                  : n_kpoints;
        const double inv_nk_export = 1.0 / static_cast<double>(nk_export);
        if (use_export_symmetry)
        {
            LIBRPA::utils::lib_printf(
                "%s: H(R) export restores full %d-point BZ from %d ABACUS IBZ k-stars\n",
                log_label.c_str(), nk_export, n_kpoints);
        }
        else if (Params::use_abacus_gw_symmetry)
        {
            LIBRPA::utils::lib_printf(
                "%s: H(R) export could not use ABACUS k-star restoration; falling back to %d loaded k-points\n",
                log_label.c_str(), n_kpoints);
        }

        std::vector<std::pair<Vector3_Order<double>, ComplexMatrix>> h_nao_full_k;
        h_nao_full_k.reserve(static_cast<std::size_t>(nk_export));
        for (int i_kpoint = 0; i_kpoint < n_kpoints; ++i_kpoint)
        {
            Matz wfc2(n_bands, n_aos, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib)
            {
                for (int iao = 0; iao < n_aos; ++iao)
                {
                    wfc2(ib, iao) =
                        meanfield_ref.get_eigenvectors0()[i_spin][0][i_kpoint](ib, iao);
                }
            }

            const Matz h_nao_k = s_nao.at(i_spin).at(i_kpoint) * transpose(wfc2)
                                 * h_band.at(i_spin).at(i_kpoint) * conj(wfc2)
                                 * transpose(s_nao.at(i_spin).at(i_kpoint), true);
            const ComplexMatrix h_nao_k_ibz = matz_to_complex_matrix(h_nao_k);

            if (use_export_symmetry)
            {
                const auto& k_ibz = kfrac_list[i_kpoint];
                const auto& star = LIBRPA::find_abacus_kstar_for_ibz_kpoint(symmetry_ctx, k_ibz);
                for (const auto& member : star.members)
                {
                    const bool use_time_reversal = member.isym >= nsym_space;
                    const ComplexMatrix h_nao_k_bz = LIBRPA::rotate_abacus_kspace_matrix(
                        symmetry_ctx, member, h_nao_k_ibz, atom_nw, k_ibz,
                        symmetry_ctx.input_coord_frac, use_time_reversal, nullptr);
                    h_nao_full_k.push_back({member.k_bz, h_nao_k_bz});
                }
            }
            else
            {
                h_nao_full_k.push_back({kfrac_list[i_kpoint], h_nao_k_ibz});
            }
        }

        if (static_cast<int>(h_nao_full_k.size()) != nk_export)
        {
            throw std::runtime_error("QSGW H(R) export produced an inconsistent full-k list");
        }

        std::map<Vector3_Order<int>, Matz> h_nao_R;
        for (const auto& R : rlist_abacus)
        {
            Matz mat_R(n_aos, n_aos, MAJOR::COL);
            for (int row = 0; row < n_aos; ++row)
            {
                for (int col = 0; col < n_aos; ++col)
                {
                    mat_R(row, col) = 0.0;
                }
            }

            for (const auto& k_h_pair : h_nao_full_k)
            {
                const auto ang = -(k_h_pair.first * R) * TWO_PI;
                const std::complex<double> kphase(std::cos(ang), std::sin(ang));
                accumulate_complex_matrix_to_matz(mat_R, k_h_pair.second, kphase * inv_nk_export);
            }
            h_nao_R[R] = mat_R;
        }

        write_real_csr_from_dense_blocks(output_files[i_spin], h_nao_R, n_aos);
        std::cout << log_label << ": exported " << output_files[i_spin]
                  << " for PyATB" << std::endl;
    }
}

} // namespace

void task_qsgw_band_0(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::envs::ofs_myid;
    using LIBRPA::utils::lib_printf;

    Profiler::start("qsgw_band0", "QSGW band0 fixed-basis quasi-particle calculation");

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    // Preserve the loaded IBZ q-index order.  The ABACUS symmetry-aware
    // Coulomb and chi0 paths rely on the same ordering as g0w0_band.
    vector<Vector3_Order<double>> qlist = klist;

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
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int, std::map<int, Matz>> vxc1;
    std::map<int, std::map<int, Matz>> vxc_band;
    std::map<int, std::map<int, Matz>> s_nao;
    std::map<int, std::map<int, Matz>> exx0;
    std::map<int, std::map<int, std::map<int, Matz>>> Hexx_matrix_temp;
    std::map<int, std::map<int, Matz>> H_KS;  // H_KS矩阵
    std::map<int, std::map<int, Matz>> H_KS0;
    std::map<int, std::map<int, Matz>> H_KS0_band;
    std::map<int, std::map<int, Matz>> H_KS1;  // 用于混合迭代
    std::map<int, std::map<int, Matz>> Hartree_0;
    std::map<int, std::map<int, Matz>> Hartree_i_delta;
    std::map<int, std::map<int, Matz>> Hartree_0_band;
    std::map<int, std::map<int, Matz>> Hartree_i_delta_band;
    bool hartree_reference_ready = false;
    bool hartree_band_reference_ready = false;
    bool all_files_processed_successfully = true;
    const std::string final_banner(90, '-');
    bool export_hamiltonian_for_pyatb = false;
    bool debug_export_ks_hamiltonian_for_pyatb = false;
    bool debug_export_ks_hamiltonian_only = false;
    bool debug_export_h0_cut_variants_for_pyatb = false;
    bool qsgw_hr_export_full_mp_rgrid = false;
    int qsgw_band0_unoccupied_keep = 10;
    int qsgw_band0_cut_mode = 2;
    double qsgw_band0_cut_shift_ha = 20.0;
    bool qsgw_band0_update_hartree = false;
    {
        int flag = 0;
        if (mpi_comm_global_h.is_root())
        {
            InputFile inputf;
            auto parser = inputf.load("librpa.in", false);
            parser.parse_bool("qsgw_export_hamiltonian_for_pyatb",
                              export_hamiltonian_for_pyatb, false, flag);
            parser.parse_bool("qsgw_debug_export_ks_hamiltonian_for_pyatb",
                              debug_export_ks_hamiltonian_for_pyatb, false, flag);
            parser.parse_bool("qsgw_debug_export_ks_hamiltonian_only",
                              debug_export_ks_hamiltonian_only, false, flag);
            parser.parse_bool("qsgw_debug_export_h0_cut_variants_for_pyatb",
                              debug_export_h0_cut_variants_for_pyatb, false, flag);
            parser.parse_bool("qsgw_hr_export_full_mp_rgrid",
                              qsgw_hr_export_full_mp_rgrid, false, flag);
            parser.parse_int("qsgw_band0_unoccupied_keep",
                             qsgw_band0_unoccupied_keep, qsgw_band0_unoccupied_keep, flag);
            parser.parse_int("qsgw_band0_cut_mode",
                             qsgw_band0_cut_mode, qsgw_band0_cut_mode, flag);
            parser.parse_double("qsgw_band0_cut_shift_ha",
                                qsgw_band0_cut_shift_ha, qsgw_band0_cut_shift_ha, flag);
            parser.parse_bool("qsgw_band0_update_hartree",
                              qsgw_band0_update_hartree, false, flag);
            if (qsgw_band0_cut_mode < 0 || qsgw_band0_cut_mode > 2)
            {
                lib_printf("QSGW band0: unsupported qsgw_band0_cut_mode=%d; using shifted cut mode\n",
                           qsgw_band0_cut_mode);
                qsgw_band0_cut_mode = 2;
            }
            if (qsgw_band0_unoccupied_keep < 0)
            {
                lib_printf("QSGW band0: qsgw_band0_unoccupied_keep=%d is invalid; using 10\n",
                           qsgw_band0_unoccupied_keep);
                qsgw_band0_unoccupied_keep = 10;
            }
            lib_printf("QSGW band0: H0 cut mode %d, keep %d unoccupied bands, shift %.6f Ha\n",
                       qsgw_band0_cut_mode, qsgw_band0_unoccupied_keep, qsgw_band0_cut_shift_ha);
            if (qsgw_band0_update_hartree)
            {
                lib_printf("QSGW band0: Hartree update is ENABLED; Delta V_H will be added after the reference iteration\n");
            }
            else
            {
                lib_printf("QSGW band0: Hartree update is disabled; legacy head-wing behavior is preserved\n");
            }
            if (debug_export_ks_hamiltonian_for_pyatb)
            {
                export_hamiltonian_for_pyatb = true;
                lib_printf("QSGW band0: will export input KS H(R) for PyATB self-check\n");
            }
            if (debug_export_h0_cut_variants_for_pyatb)
            {
                export_hamiltonian_for_pyatb = true;
                lib_printf("QSGW band0: will export H0 cut-variant H(R) files for PyATB diagnostics\n");
            }
            if (export_hamiltonian_for_pyatb)
            {
                lib_printf("QSGW band0: will export H(R) for PyATB after each iteration\n");
                if (qsgw_hr_export_full_mp_rgrid)
                {
                    lib_printf("QSGW band0: H(R) export will use the full Monkhorst-Pack R grid\n");
                }
            }
        }
        mpi_comm_global_h.broadcast(export_hamiltonian_for_pyatb, 0);
        mpi_comm_global_h.broadcast(debug_export_ks_hamiltonian_for_pyatb, 0);
        mpi_comm_global_h.broadcast(debug_export_ks_hamiltonian_only, 0);
        mpi_comm_global_h.broadcast(debug_export_h0_cut_variants_for_pyatb, 0);
        mpi_comm_global_h.broadcast(qsgw_hr_export_full_mp_rgrid, 0);
        mpi_comm_global_h.broadcast(qsgw_band0_unoccupied_keep, 0);
        mpi_comm_global_h.broadcast(qsgw_band0_cut_mode, 0);
        mpi_comm_global_h.broadcast(qsgw_band0_cut_shift_ha, 0);
        mpi_comm_global_h.broadcast(qsgw_band0_update_hartree, 0);
    }

    // 自旋和 k 点的循环，读取初始数据
    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt)
        {
            std::map<std::string, Matz> arrays;
            std::string key_hf, key_vxc;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_hf, oss_vxc;
            oss_hf << "hf_exchange_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6)
                   << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_vxc << "xc_matr_spin_" << (ispin + 1) << "_kpt_" << std::setw(6)
                    << std::setfill('0') << (ikpt + 1) << ".csc";

            std::string hfFilePath = oss_hf.str();
            std::string vxcFilePath = oss_vxc.str();

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

            hf_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);
            vxc0[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);
            s_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);
            // 初始化 hf 和 vxc 矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i)
            {
                for (int j = 0; j < n_aos; ++j)
                {
                    hf_nao[ispin][ikpt](i, j) = 0.0;
                    vxc0[ispin][ikpt](i, j) = 0.0;
                    s_nao[ispin][ikpt](i, j) = (i == j ? 1.0 : 0.0);
                }
            }

            bool hf_file_found = false;
            bool vxc_file_found = false;
            bool s_file_found = false;

            // 读取 hf 文件
            std::ifstream hf_file(hfFilePath.c_str());
            if (hf_file.good())
            {
                if (!convert_csc(hfFilePath, arrays, key_hf))
                {
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
            if (vxc_file.good())
            {
                if (!convert_csc(vxcFilePath, arrays, key_vxc))
                {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath << std::endl;
                }
                else
                {
                    vxc0[ispin][ikpt] = arrays[key_vxc];
                    vxc_file_found = true;
                }
            }
            else
            {
                std::ostringstream oss_vxc_text;
                oss_vxc_text << "vxcs" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                std::ostringstream oss_vxc_text_k;
                oss_vxc_text_k << "vxck" << (ikpt + 1) << "_nao.txt";
                const std::string vxc_text_path = oss_vxc_text.str();
                const std::string vxc_text_k_path = oss_vxc_text_k.str();
                try
                {
                    if (read_abacus_upper_triangle_matrix(vxc_text_path, vxc0[ispin][ikpt], 0.5)
                        || read_abacus_upper_triangle_matrix(vxc_text_k_path, vxc0[ispin][ikpt],
                                                             0.5))
                    {
                        vxc_file_found = true;
                    }
                    else
                    {
                        std::cerr << "VXC file not found: " << vxcFilePath << " or "
                                  << vxc_text_path << " or " << vxc_text_k_path << std::endl;
                    }
                }
                catch (const std::exception& e)
                {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process ABACUS VXC text file for spin " << ispin + 1
                              << ", k-point " << ikpt + 1 << ": " << e.what() << std::endl;
                }
            }

            {
                std::map<std::string, Matz> arrays_s;
                std::string key_s;
                std::ostringstream oss_s_csc;
                oss_s_csc << "S_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6)
                          << std::setfill('0') << (ikpt + 1) << ".csc";
                const std::string s_csc_path = oss_s_csc.str();
                std::ifstream s_csc_file(s_csc_path.c_str());
                if (s_csc_file.good())
                {
                    if (!convert_csc(s_csc_path, arrays_s, key_s))
                    {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process file: " << s_csc_path << std::endl;
                    }
                    else
                    {
                        s_nao[ispin][ikpt] = arrays_s[key_s];
                        s_file_found = true;
                    }
                }
                else
                {
                    std::ostringstream oss_s_text;
                    oss_s_text << "sks" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                    std::ostringstream oss_s_text_short;
                    oss_s_text_short << "s" << (ispin + 1) << "k" << (ikpt + 1) << "_nao.txt";
                    const std::string s_text_path = oss_s_text.str();
                    const std::string s_text_short_path = oss_s_text_short.str();
                    try
                    {
                        if (read_abacus_upper_triangle_matrix(s_text_path, s_nao[ispin][ikpt])
                            || read_abacus_upper_triangle_matrix(s_text_short_path,
                                                                 s_nao[ispin][ikpt]))
                        {
                            s_file_found = true;
                        }
                    }
                    catch (const std::exception& e)
                    {
                        all_files_processed_successfully = false;
                        std::cerr << "Failed to process overlap matrix for spin " << ispin + 1
                                  << ", k-point " << ikpt + 1 << ": " << e.what() << std::endl;
                    }
                }
                if (export_hamiltonian_for_pyatb && !s_file_found)
                {
                    all_files_processed_successfully = false;
                    std::cerr << "QSGW H(R) export needs overlap matrix for spin " << ispin + 1
                              << ", k-point " << ikpt + 1 << std::endl;
                }
            }

            // 如果两个文件都不存在，报错并跳过该 k 点
            if (!hf_file_found && !vxc_file_found)
            {
                all_files_processed_successfully = false;
                std::cerr << "Both HF and VXC files not found for spin " << ispin + 1
                          << ", k-point " << ikpt + 1 << std::endl;
                continue;
            }

            // ABACUS out_mat_xc writes the Vxc matrix in KS-orbital representation;
            // the filename still contains "nao" for historical reasons.
            hf[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            hf[ispin][ikpt] = conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);

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
    if (mpi_comm_global_h.is_root())
    {
        std::cout << "Band k-points to compute:\n";
        for (int ik = 0; ik < kfrac_band.size(); ik++)
        {
            const auto &k = kfrac_band[ik];
            lib_printf("%5d %12.7f %12.7f %12.7f\n", ik + 1, k.x, k.y, k.z);
        }
    }
    mpi_comm_global_h.barrier();

    Profiler::start("g0w0_band_load_band_mf", "Read eigen solutions at band kpoints");
    auto meanfield_band = read_meanfield_band(driver_params.input_dir, n_basis_band, n_states_band,
                                              n_spin_band, kfrac_band.size());

    Profiler::stop("g0w0_band_load_band_mf");

    Profiler::start("read_vxc_band", "Load DFT xc potential");

    // 读取 vxc_band 文件,H_KS0_band
    for (int i_spin = 0; i_spin < meanfield_band.get_n_spins(); i_spin++)
    {
        for (int i_kpoint = 0; i_kpoint < meanfield_band.get_n_kpoints(); i_kpoint++)
        {
            std::map<std::string, Matz> arrays_band;
            std::string key_vxc_band;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_vxc_band;

            oss_vxc_band << "band_vxc_mat_spin_" << (i_spin + 1) << "_k_" << std::setw(5)
                         << std::setfill('0') << (i_kpoint + 1) << ".csc";
            std::string vxcFilePath_band = oss_vxc_band.str();

            vxc_band[i_spin][i_kpoint] = Matz(n_aos, n_aos, MAJOR::COL);

            // 初始化 vxc_band 矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i)
            {
                for (int j = 0; j < n_aos; ++j)
                {
                    vxc_band[i_spin][i_kpoint](i, j) = 0.0;
                }
            }
            bool vxc_band_file_found = false;

            // 读取 vxc_band 文件
            std::ifstream vxc_band_file(vxcFilePath_band.c_str());
            if (vxc_band_file.good())
            {
                if (!convert_csc(vxcFilePath_band, arrays_band, key_vxc_band))
                {
                    std::cerr << "Failed to process file: " << vxcFilePath_band << std::endl;
                }
                else
                {
                    vxc_band[i_spin][i_kpoint] = arrays_band[key_vxc_band];
                    vxc_band_file_found = true;
                }
            }
            else
            {
                std::ostringstream oss_vxc_band_text;
                oss_vxc_band_text << "band_vxcs" << (i_spin + 1) << "k" << (i_kpoint + 1)
                                  << "_nao.txt";
                std::ostringstream oss_vxc_band_text_k;
                oss_vxc_band_text_k << "band_vxck" << (i_kpoint + 1) << "_nao.txt";
                const std::string vxc_band_text_path = oss_vxc_band_text.str();
                const std::string vxc_band_text_k_path = oss_vxc_band_text_k.str();
                try
                {
                    if (read_abacus_upper_triangle_matrix(vxc_band_text_path,
                                                          vxc_band[i_spin][i_kpoint], 0.5)
                        || read_abacus_upper_triangle_matrix(vxc_band_text_k_path,
                                                             vxc_band[i_spin][i_kpoint], 0.5))
                    {
                        vxc_band_file_found = true;
                    }
                    else
                    {
                        std::cerr << "VXC_band file not found: " << vxcFilePath_band << " or "
                                  << vxc_band_text_path << " or " << vxc_band_text_k_path
                                  << std::endl;
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Failed to process ABACUS band VXC text file for spin "
                              << i_spin + 1 << ", k-point " << i_kpoint + 1 << ": " << e.what()
                              << std::endl;
                }
            }

            H_KS0_band[i_spin][i_kpoint] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band)
            {
                H_KS0_band[i_spin][i_kpoint](i_band, i_band) =
                    meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_band);
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
                        wfc5(ib1, ib2) =
                            meanfield_band.get_eigenvectors()[i_spin][isoc][i_kpoint](ib1, iao);
                        meanfield_band.get_eigenvectors0()[i_spin][isoc][i_kpoint](ib1, iao) =
                            wfc5(ib1, ib2);
                    }
                }
            }

            // band_vxck*_nao.txt is also in KS-orbital representation despite the suffix.
        }
    }

    Profiler::stop("read_vxc_band");
    std::flush(ofs_myid);
    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    double homo = -1e6;
    double lumo = 1e6;
    printf("%5s\n", "efermi_band1");
    printf("%5f\n", efermi);

    // 计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n", "Total_electrons");
    printf("%5f\n", total_electrons);

    // 设置收敛条件
    double eigenvalue_tolerance = 1e-5;  // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations = 10;             // 最大迭代次数
    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1;
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    {
        int flag = 0;
        if (mpi_comm_global_h.is_root())
        {
            InputFile inputf;
            auto parser = inputf.load("librpa.in", false);
            parser.parse_int("max_iter", max_iterations, max_iterations, flag);
            lib_printf("QSGW band0: max_iterations = %d\n", max_iterations);
            if (Params::qsgw_restart)
            {
                lib_printf("QSGW band0: restart enabled from %s, requested iteration = %d\n",
                           qsgw_checkpoint_load_root_band().c_str(),
                           Params::qsgw_restart_iteration);
            }
            lib_printf("QSGW band0: checkpoint interval = %d\n", Params::qsgw_checkpoint_every);
        }
        mpi_comm_global_h.broadcast(max_iterations, 0);
    }

    const auto checkpoint_save_root = qsgw_checkpoint_save_root_band();
    const auto checkpoint_load_root = qsgw_checkpoint_load_root_band();
    std::map<int, std::map<int, Matz>> restart_H0_GW_all;
    bool have_restart_H0_GW_all = false;
    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        reset_iteration_history_band();
        ensure_dir_band(checkpoint_save_root);

        if (Params::qsgw_restart)
        {
            const auto checkpoint = load_qsgw_checkpoint_band(
                checkpoint_load_root, Params::qsgw_restart_iteration, n_spins, n_kpoints);
            restart_H0_GW_all = checkpoint.H0_GW_all;
            have_restart_H0_GW_all = true;
            if (checkpoint.has_hartree0)
            {
                Hartree_0 = checkpoint.Hartree_0;
                hartree_reference_ready = true;
                if (qsgw_band0_update_hartree)
                {
                    lib_printf("QSGW band0: restored Hartree reference from checkpoint\n");
                }
            }
            else if (qsgw_band0_update_hartree)
            {
                lib_printf("QSGW band0: restart checkpoint has no Hartree reference; the first resumed Hartree update step will define a new reference and use zero Delta V_H\n");
            }
            diagonalize_and_store_fixed_basis(meanfield, checkpoint.H0_GW_all, n_spins, n_kpoints, n_bands);
            update_fermi_energy_and_occupations(meanfield, temperature, checkpoint.efermi_ha);
            compute_homo_lumo_ha_band(meanfield, homo, lumo);
            efermi = checkpoint.efermi_ha;
            iteration = checkpoint.iteration;

            const bool history_loaded = load_iteration_history_file_band(
                checkpoint_load_root + "homo_lumo_vs_iterations.dat", iteration);
            if (!history_loaded)
            {
                append_iteration_history_band(iteration, homo * HA2EV, lumo * HA2EV,
                                              efermi * HA2EV);
            }

            std::cout << "[QSGW_BAND] Restarting from checkpoint iteration " << iteration
                      << " at " << checkpoint_load_root << std::endl;
            std::cout << "[QSGW_BAND] Restored HOMO = " << homo * HA2EV << " eV, "
                      << "LUMO = " << lumo * HA2EV << " eV, "
                      << "Fermi Energy = " << efermi * HA2EV << " eV\n";
        }
        else
        {
            compute_homo_lumo_ha_band(meanfield, homo, lumo);
            append_iteration_history_band(0, homo * HA2EV, lumo * HA2EV, efermi * HA2EV);
            std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
                      << "LUMO = " << lumo * HA2EV << " eV, "
                      << "Fermi Energy = " << efermi * HA2EV << " eV\n";
        }

        plot_homo_lumo_vs_iterations();
        write_iteration_history_file_band("homo_lumo_vs_iterations.dat");
        write_iteration_history_file_band(checkpoint_save_root + "homo_lumo_vs_iterations.dat");
    }
    mpi_comm_global_h.broadcast(iteration, 0);
    mpi_comm_global_h.broadcast(have_restart_H0_GW_all, 0);
    meanfield.broadcast(mpi_comm_global_h, 0);
    mpi_comm_global_h.barrier();
    meanfield_band.get_efermi() = meanfield.get_efermi();
    if (Params::qsgw_restart && iteration < max_iterations)
    {
        if (mpi_comm_global_h.is_root())
        {
            restore_band_meanfield_from_qsgw_band_files(
                meanfield_band, kfrac_band, driver_params.input_dir, iteration);
            meanfield_band.get_efermi() = meanfield.get_efermi();
        }
        meanfield_band.broadcast(mpi_comm_global_h, 0);
        mpi_comm_global_h.barrier();
    }

    // 初始化完毕，开始循环
    if (export_hamiltonian_for_pyatb && mpi_comm_global_h.is_root())
    {
        ensure_atom_nw_for_qsgw_hr_export();
    }
    std::vector<Vector3_Order<int>> Rlist_abacus;
    if (export_hamiltonian_for_pyatb && mpi_comm_global_h.is_root())
    {
        if (qsgw_hr_export_full_mp_rgrid)
        {
            Rlist_abacus = construct_R_grid(period);
            lib_printf("QSGW band0: H(R) export will use the full %zu-point MP R grid\n",
                       Rlist_abacus.size());
        }
        else
        {
            Rlist_abacus = read_csr_rlist(driver_params.input_dir + "hrs1_nao.csr");
            lib_printf("QSGW band0: H(R) export will use %zu R blocks from hrs1_nao.csr\n",
                       Rlist_abacus.size());
        }
    }
    if (debug_export_ks_hamiltonian_for_pyatb)
    {
        if (mpi_comm_global_h.is_root())
        {
            std::vector<std::string> ks_export_files;
            ks_export_files.reserve(static_cast<std::size_t>(n_spins));
            for (int i_spin = 0; i_spin < n_spins; ++i_spin)
            {
                std::ostringstream filename;
                filename << "hrs" << (i_spin + 1) << "_nao_ks_librpa.csr";
                ks_export_files.push_back(filename.str());
            }
            export_band_basis_hamiltonian_to_abacus_csr(
                ks_export_files, H_KS0, s_nao, meanfield, n_spins, n_kpoints,
                n_bands, n_aos, n_soc, Rlist_abacus, "QSGW band0 KS-debug");
        }
        mpi_comm_global_h.barrier();
        if (debug_export_ks_hamiltonian_only)
        {
            if (mpi_comm_global_h.is_root())
            {
                std::cout << "QSGW band0: debug-only KS H(R) export complete" << std::endl;
            }
            Profiler::stop("qsgw_band0");
            return;
        }
    }
    if (Params::qsgw_restart && export_hamiltonian_for_pyatb
        && iteration >= max_iterations && mpi_comm_global_h.is_root())
    {
        if (!have_restart_H0_GW_all)
        {
            throw std::runtime_error("QSGW band0 restart H(R) export requested but no checkpoint H0 was loaded");
        }
        std::vector<std::string> restart_export_files;
        restart_export_files.reserve(static_cast<std::size_t>(n_spins));
        for (int i_spin = 0; i_spin < n_spins; ++i_spin)
        {
            std::ostringstream filename;
            filename << "hrs" << (i_spin + 1) << "_nao_qsgw_restart_iter_"
                     << std::setw(4) << std::setfill('0') << iteration << ".csr";
            restart_export_files.push_back(filename.str());
        }
        export_band_basis_hamiltonian_to_abacus_csr(
            restart_export_files, restart_H0_GW_all, s_nao, meanfield, n_spins,
            n_kpoints, n_bands, n_aos, n_soc, Rlist_abacus, "QSGW band0 restart");
    }
    mpi_comm_global_h.barrier();
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

        chi0.set_input_dir(driver_params.input_dir);
        Profiler::start("chi0_build", "Build response function chi0");
        const auto& chi0_cs =
            (Params::use_shrink_abfs && !Params::use_shrink_chi) ? Cs_shrinked_data : Cs_data;
        chi0.build(chi0_cs, Rlist, period, local_atpair, qlist, sinvS);
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
                            sprintf(fn, "chi0fq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I,
                                    J, mpi_comm_global_h.myid);
                            print_complex_matrix_mm(J_chi0.second, Params::output_dir + "/" + fn,
                                                    1e-15);
                        }
                    }
                }
            }
        }
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
        Profiler::cease("read_vq_cut");

        if (qsgw_band0_update_hartree)
        {
            Profiler::start("qsgw_hartree", "Build Hartree potential");
            auto Hartree = LIBRPA::Hartree(meanfield, kfrac_list, period);
            {
                Profiler::start("ft_vq_cut_hartree", "Fourier transform truncated Coulomb for Hartree");
                const auto VR_hartree = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
                Profiler::stop("ft_vq_cut_hartree");

                Profiler::start("qsgw_hartree_real_work");
                Hartree.build(Cs_data, Rlist, VR_hartree);
                Hartree.build_KS_kgrid0();
                Profiler::stop("qsgw_hartree_real_work");
            }

            if (mpi_comm_global_h.is_root())
            {
                Hartree_i_delta.clear();
                double max_abs_hartree_delta = 0.0;
                for (int ispin = 0; ispin < n_spins; ++ispin)
                {
                    for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
                    {
                        const auto &hartree_now = Hartree.Hartree_is_ik_KS[ispin][ikpt];
                        if (!hartree_reference_ready)
                        {
                            Hartree_0[ispin][ikpt] = hartree_now.copy();
                        }

                        Hartree_i_delta[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                        Hartree_i_delta[ispin][ikpt].zero_out();
                        if (hartree_reference_ready)
                        {
                            for (int i = 0; i < n_bands; ++i)
                            {
                                for (int j = 0; j < n_bands; ++j)
                                {
                                    const auto delta = hartree_now(i, j) - Hartree_0[ispin][ikpt](i, j);
                                    Hartree_i_delta[ispin][ikpt](i, j) = delta;
                                    max_abs_hartree_delta =
                                        std::max(max_abs_hartree_delta, std::abs(delta));
                                }
                            }
                        }
                    }
                }

                if (!hartree_reference_ready)
                {
                    hartree_reference_ready = true;
                    lib_printf("QSGW band0: captured regular-grid Hartree reference; Delta V_H is zero for iteration %d\n",
                               iteration);
                }
                else
                {
                    lib_printf("QSGW band0: regular-grid max |Delta V_H| = %.8e Ha at iteration %d\n",
                               max_abs_hartree_delta, iteration);
                }
            }

            Hartree.reset_kspace();
            Hartree.build_KS_band(meanfield_band.get_eigenvectors0(), kfrac_band);
            if (mpi_comm_global_h.is_root())
            {
                Hartree_i_delta_band.clear();
                double max_abs_hartree_delta_band = 0.0;
                for (int ispin = 0; ispin < meanfield_band.get_n_spins(); ++ispin)
                {
                    for (int ikpt = 0; ikpt < meanfield_band.get_n_kpoints(); ++ikpt)
                    {
                        const auto &hartree_now = Hartree.Hartree_is_ik_KS[ispin][ikpt];
                        if (!hartree_band_reference_ready)
                        {
                            Hartree_0_band[ispin][ikpt] = hartree_now.copy();
                        }

                        Hartree_i_delta_band[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
                        Hartree_i_delta_band[ispin][ikpt].zero_out();
                        if (hartree_band_reference_ready)
                        {
                            for (int i = 0; i < n_bands; ++i)
                            {
                                for (int j = 0; j < n_bands; ++j)
                                {
                                    const auto delta =
                                        hartree_now(i, j) - Hartree_0_band[ispin][ikpt](i, j);
                                    Hartree_i_delta_band[ispin][ikpt](i, j) = delta;
                                    max_abs_hartree_delta_band =
                                        std::max(max_abs_hartree_delta_band, std::abs(delta));
                                }
                            }
                        }
                    }
                }

                if (!hartree_band_reference_ready)
                {
                    hartree_band_reference_ready = true;
                    lib_printf("QSGW band0: captured band-path Hartree reference; Delta V_H is zero for iteration %d\n",
                               iteration);
                }
                else
                {
                    lib_printf("QSGW band0: band-path max |Delta V_H| = %.8e Ha at iteration %d\n",
                               max_abs_hartree_delta_band, iteration);
                }
            }
            Profiler::stop("qsgw_hartree");
            std::flush(ofs_myid);
            mpi_comm_global_h.barrier();
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
        }

        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS

        Profiler::start("qsgw_exx", "Build exchange self-energy");
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
                            sprintf(fn, "Wcfq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I,
                                    J, mpi_comm_global_h.myid);
                            print_matrix_mm_file(q_Wc.second, Params::output_dir + "/" + fn, 1e-15);
                        }
                    }
                }
            }
        }

        if (Params::use_shrink_abfs)
        {
            Profiler::start("read_shrink_sinvS_fold", "Load shrink transformation");
            read_shrink_sinvS(driver_params.input_dir, "shrink_sinvS_", sinvS);
            Profiler::stop("read_shrink_sinvS_fold");
        }

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
                                auto energy0 =
                                    meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                                efermi = meanfield.get_efermi();
                                // 计算得到的值
                                auto result = pade.get(energy0 - efermi);
                                auto result1 = pade.get(0.0);
                                // 存储值到 sigcmat
                                sigcmat[i_state_row][i_state_col][i_state_row] = result;
                                sigcmat[i_state_row][i_state_col][n_bands] = result1;
                            }
                        }

                        Vc_all[i_spin][i_kpoint] =
                            build_correlation_potential_spin_k(sigcmat, n_bands);
                        if (qsgw_band0_update_hartree && iteration > 1)
                        {
                            Vc_all[i_spin][i_kpoint] =
                                Vc_all[i_spin][i_kpoint] + Hartree_i_delta[i_spin][i_kpoint];
                        }
                    }
                }
                Profiler::stop("qsgw_solve_qpe");

                auto H0_GW_all = construct_H0_GW_cut(
                    meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                    n_spins, n_kpoints, n_bands, qsgw_band0_unoccupied_keep,
                    qsgw_band0_cut_mode, qsgw_band0_cut_shift_ha);

                if (export_hamiltonian_for_pyatb)
                {
                    std::vector<std::string> qsgw_export_files;
                    qsgw_export_files.reserve(static_cast<std::size_t>(n_spins));
                    for (int i_spin = 0; i_spin < n_spins; ++i_spin)
                    {
                        std::ostringstream filename;
                        filename << "hrs" << (i_spin + 1) << "_nao_qsgw_iter_"
                                 << std::setw(4) << std::setfill('0') << iteration << ".csr";
                        qsgw_export_files.push_back(filename.str());
                    }
                    export_band_basis_hamiltonian_to_abacus_csr(
                        qsgw_export_files, H0_GW_all, s_nao, meanfield, n_spins,
                        n_kpoints, n_bands, n_aos, n_soc, Rlist_abacus, "QSGW band0");

                    if (debug_export_h0_cut_variants_for_pyatb)
                    {
                        for (int variant_mode = 0; variant_mode <= 2; ++variant_mode)
                        {
                            auto H0_GW_variant = construct_H0_GW_cut(
                                meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                                n_spins, n_kpoints, n_bands, qsgw_band0_unoccupied_keep,
                                variant_mode, qsgw_band0_cut_shift_ha);
                            std::vector<std::string> variant_files;
                            variant_files.reserve(static_cast<std::size_t>(n_spins));
                            for (int i_spin = 0; i_spin < n_spins; ++i_spin)
                            {
                                std::ostringstream filename;
                                filename << "hrs" << (i_spin + 1) << "_nao_qsgw_iter_"
                                         << std::setw(4) << std::setfill('0') << iteration
                                         << "_cutmode" << variant_mode << ".csr";
                                variant_files.push_back(filename.str());
                            }
                            std::ostringstream label;
                            label << "QSGW band0 cut-mode " << variant_mode;
                            export_band_basis_hamiltonian_to_abacus_csr(
                                variant_files, H0_GW_variant, s_nao, meanfield, n_spins,
                                n_kpoints, n_bands, n_aos, n_soc, Rlist_abacus, label.str());
                        }
                    }
                }

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
                        printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n", i_spin + 1,
                               i_kpoint + 1, k.x, k.y, k.z);
                        printf("%77s\n", final_banner.c_str());
                        printf("%5s %16s %16s %16s %16s %16s %16s %16s\n", "State", "e_mf", "v_xc",
                               "v_exx1", "v_exx2", "ReSigc", "ImSigc", "e_qp");
                        printf("%77s\n", final_banner.c_str());
                        for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                        {
                            const auto &eks_state =
                                meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                            const auto &exx_state1 = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                            const auto &exx_state2 =
                                exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                            const auto &vxc_state =
                                vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                            const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                            const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                            printf("%5d %20.15f %16.5f %16.5f %16.5f %16.5f %16.5f \n", i_state + 1,
                                   eks_state, vxc_state.real(), exx_state1, exx_state2.real(),
                                   resigc, imsigc);
                        }
                        printf("\n");
                    }
                }

                // 计算 HOMO 和 LUMO
                compute_homo_lumo_ha_band(meanfield, homo, lumo);

                // 输出当前 HOMO 和 LUMO 值
                std::cout << "Iteration " << iteration << ": HOMO = " << homo * HA2EV << " eV, "
                          << "LUMO = " << lumo * HA2EV << " eV, "
                          << "Efermi = " << efermi * HA2EV << " eV\n";
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

                append_iteration_history_band(iteration, homo * HA2EV, lumo * HA2EV,
                                              efermi * HA2EV);
                plot_homo_lumo_vs_iterations();
                write_iteration_history_file_band("homo_lumo_vs_iterations.dat");
                write_iteration_history_file_band(checkpoint_save_root +
                                                  "homo_lumo_vs_iterations.dat");

                const bool should_write_checkpoint =
                    (Params::qsgw_checkpoint_every > 0 &&
                     (iteration % Params::qsgw_checkpoint_every == 0)) ||
                    converged || iteration == max_iterations;
                if (should_write_checkpoint)
                {
                    write_qsgw_checkpoint_band(
                        checkpoint_save_root, iteration, H0_GW_all, efermi,
                        qsgw_band0_update_hartree ? &Hartree_0 : nullptr);
                }

                if (converged)
                {
                    std::cout << "Converged after " << iteration << " iterations.\n";
                }
            }
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

        Profiler::stop("g0w0_sigx_rotate_KS");
        std::flush(ofs_myid);

        Profiler::start("g0w0_sigc_rotate_KS");
        s_g0w0.build_sigc_matrix_KS_band(meanfield_band.get_eigenvectors0(), kfrac_band);
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
                            auto energy0 =
                                meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_state_row);
                            efermi = meanfield_band.get_efermi();
                            // 计算得到的值
                            auto result = pade.get(energy0 - efermi);
                            auto result1 = pade.get(0.0);
                            // 存储值到 sigcmat
                            sigcmat[i_state_row][i_state_col][i_state_row] = result;
                            sigcmat[i_state_row][i_state_col][n_bands] = result1;
                        }
                    }
                    Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k(sigcmat, n_bands);
                    if (qsgw_band0_update_hartree && iteration > 1)
                    {
                        Vc_all[i_spin][i_kpoint] =
                            Vc_all[i_spin][i_kpoint] + Hartree_i_delta_band[i_spin][i_kpoint];
                    }
                }
            }

            // reconstruct H0_GW_all
            auto H0_GW_all_band = construct_H0_GW_cut(
                meanfield_band, H_KS0_band, vxc_band, exx.exx_is_ik_KS, Vc_all,
                meanfield_band.get_n_spins(), meanfield_band.get_n_kpoints(), n_bands,
                qsgw_band0_unoccupied_keep, qsgw_band0_cut_mode, qsgw_band0_cut_shift_ha);
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
                fn << "QSGW_band_spin_" << i_spin + 1 << "_" << iteration << ".dat";
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
                        const auto &H_state =
                            meanfield_band.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;

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
    Profiler::stop("qsgw_band0");
}
