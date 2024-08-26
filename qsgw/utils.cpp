#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <stdexcept>
#include <iterator>

// 读取 ELSI CSC 文件的函数
void read_elsi_to_csc(const std::string& filePath, std::vector<int>& col_ptr, std::vector<int>& row_idx, std::vector<std::complex<double>>& nnz_val, int& n_basis) {
    std::ifstream inputFile(filePath, std::ios::binary);
    if (!inputFile) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return;
    }

    // 读取文件内容
    inputFile.seekg(0, std::ios::end);
    std::streampos size = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    inputFile.read(buffer.data(), size);
    inputFile.close();

    // 解析 header
    int64_t header[16];
    std::memcpy(header, buffer.data(), 128);

    n_basis = header[3];
    int64_t nnz = header[5];

    // 获取列指针
    int64_t* col_ptr_raw = reinterpret_cast<int64_t*>(buffer.data() + 128);
    col_ptr.assign(col_ptr_raw, col_ptr_raw + n_basis);
    col_ptr.push_back(nnz + 1);

    // 获取行索引
    int32_t* row_idx_raw = reinterpret_cast<int32_t*>(buffer.data() + 128 + n_basis * 8);
    row_idx.assign(row_idx_raw, row_idx_raw + nnz);

    // 获取非零值
    char* nnz_val_raw = buffer.data() + 128 + n_basis * 8 + nnz * 4;
    if (header[2] == 0) {
        // 实数情况
        double* nnz_val_double = reinterpret_cast<double*>(nnz_val_raw);
        for (int64_t i = 0; i < nnz; ++i) {
            nnz_val.push_back(std::complex<double>(nnz_val_double[i], 0.0));
        }
    } else {
        // 复数情况
        double* nnz_val_double = reinterpret_cast<double*>(nnz_val_raw);
        for (int64_t i = 0; i < nnz; ++i) {
            nnz_val.push_back(std::complex<double>(nnz_val_double[2 * i], nnz_val_double[2 * i + 1]));
        }
    }

    // 更改索引
    for (int32_t& idx : row_idx) {
        idx -= 1;
    }
    for (int32_t& ptr : col_ptr) {
        ptr -= 1;
    }
}

void loadMatrix(const std::string& filePath, Eigen::MatrixXcd& matrix) {
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
    std::vector<std::complex<double>> nnz_val;
    int n_basis;

    read_elsi_to_csc(filePath, col_ptr, row_idx, nnz_val, n_basis);

    matrix = Eigen::MatrixXcd::Zero(n_basis, n_basis);
    for (int col = 0; col < n_basis; ++col) {
        for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx) {
            int row = row_idx[idx];
            matrix(row, col) = nnz_val[idx];
        }
    }
}

std::tuple<int, int> read_aims_state_limits(const std::string& aimsout) {
    std::ifstream inputFile(aimsout);
    if (!inputFile) {
        throw std::runtime_error("无法打开文件: " + aimsout);
    }

    std::string line;
    int lb = -1, ub = -1;

    while (std::getline(inputFile, line)) {
        if (line.find("Actual lower/upper states to compute self-energy:") != std::string::npos) {
            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
            lb = std::stoi(tokens[tokens.size() - 2]) - 1;
            ub = std::stoi(tokens.back());
            break;
        }
    }

    if (lb == -1 || ub == -1) {
        throw std::runtime_error("未找到状态限制信息");
    }

    return {lb, ub};
}

double get_chemical_potential(const std::string& aimsout, const std::string& unit) {
    std::ifstream inputFile(aimsout);
    if (!inputFile) {
        throw std::runtime_error("无法打开文件: " + aimsout);
    }

    std::string line;
    double chempot = -1.0;

    while (std::getline(inputFile, line)) {
        if (line.find("Chemical Potential") != std::string::npos) {
            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
            chempot = std::stod(tokens[tokens.size() - 2]);
            break;
        }
    }

    if (chempot == -1.0) {
        throw std::runtime_error("未找到化学势信息");
    }

    if (unit == "ha") {
        chempot /= HA2EV;
    }

    return chempot;
}

std::vector<double> read_self_energy_binary(const std::string& filename) {
    std::ifstream inputFile(filename, std::ios::binary);
    if (!inputFile) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    int header[4];
    inputFile.read(reinterpret_cast<char*>(header), sizeof(header));
    int nspin = header[0], nkpts = header[1], nstates = header[2], nfreq = header[3];

    std::vector<double> omega_imag(nfreq);
    inputFile.seekg(8 * nfreq * 2, std::ios::cur);
    inputFile.read(reinterpret_cast<char*>(omega_imag.data()), nfreq * sizeof(double));

    return omega_imag;
}

int get_n_freq() {
    std::ifstream inputFile("control.in");
    if (!inputFile) {
        throw std::runtime_error("无法打开文件: control.in");
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        if (line.find("frequency_points") != std::string::npos) {
            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
            return std::stoi(tokens[1]);
        }
    }

    throw std::runtime_error("未找到频率点信息");
}

std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>> get_pade_params(const std::vector<double>& freq, const std::vector<std::complex<double>>& selfe, int n_par) {
    // 注意：此处使用了简化的Pade近似方法，实际应用中应使用高精度数学库
    int n_freq = freq.size();
    if (selfe.size() != n_freq || n_par <= 1 || n_par > n_freq) {
        throw std::invalid_argument("输入参数不正确");
    }

    std::vector<std::complex<double>> xtmp(n_par);
    std::vector<std::complex<double>> ytmp(n_par);
    int n_step = n_freq / (n_par - 1);
    int i_dat = 0;
    for (int i_par = 0; i_par < n_par - 1; ++i_par) {
        xtmp[i_par] = {0.0, freq[i_dat]};
        ytmp[i_par] = selfe[i_dat];
        i_dat += n_step;
    }
    xtmp[n_par - 1] = {0.0, freq.back()};
    ytmp[n_par - 1] = selfe.back();

    std::vector<std::vector<std::complex<double>>> g_func(n_par, std::vector<std::complex<double>>(n_par));
    for (int i_par = 0; i_par < n_par; ++i_par) {
        g_func[i_par][0] = ytmp[i_par];
    }
    for (int i_par = 1; i_par < n_par; ++i_par) {
        for (int i_dat = i_par; i_dat < n_par; ++i_dat) {
            g_func[i_dat][i_par] = (g_func[i_par - 1][i_par - 1] - g_func[i_dat][i_par - 1]) /
                                   ((xtmp[i_dat] - xtmp[i_par - 1]) * g_func[i_dat][i_par - 1]);
        }
    }

    std::vector<std::complex<double>> omega_par(n_par);
    std::vector<std::complex<double>> pars(n_par);
    for (int i = 0; i < n_par; ++i) {
        omega_par[i] = xtmp[i];
        pars[i] = g_func[i][i];
    }

    return {omega_par, pars};
}

Eigen::VectorXd get_real_selfe(const std::vector<double>& freq, const std::vector<std::complex<double>>& omega_par, const std::vector<std::complex<double>>& pars, double ref) {
    int n_freq = freq.size();
    Eigen::VectorXd selfe(n_freq);

    for (int i = 0; i < n_freq; ++i) {
        std::complex<double> f(freq[i] - ref, 0);
        std::complex<double> gtmp(1.0, 0.0);
        for (int ipar = pars.size() - 1; ipar > 0; --ipar) {
            gtmp = 1.0 + pars[ipar] * (f - omega_par[ipar - 1]) / gtmp;
        }
        selfe[i] = (pars[0] / gtmp).real();
    }

    return selfe;
}
