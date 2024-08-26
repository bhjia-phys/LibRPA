#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <tuple>

// 常量
const double HA2EV = 27.21138602;
const int MPC_PREC_ACCEPT = 15;

// 函数声明
void read_elsi_to_csc(const std::string& filePath, std::vector<int>& col_ptr, std::vector<int>& row_idx, std::vector<std::complex<double>>& nnz_val, int& n_basis);
void loadMatrix(const std::string& filePath, Eigen::MatrixXcd& matrix);
std::tuple<int, int> read_aims_state_limits(const std::string& aimsout = "aims.out");
double get_chemical_potential(const std::string& aimsout = "aims.out", const std::string& unit = "ev");
std::vector<double> read_self_energy_binary(const std::string& filename = "self_energy_grid.dat");
int get_n_freq();
std::tuple<std::vector<std::complex<double>>, std::vector<std::complex<double>>> get_pade_params(const std::vector<double>& freq, const std::vector<std::complex<double>>& selfe, int n_par);
Eigen::VectorXd get_real_selfe(const std::vector<double>& freq, const std::vector<std::complex<double>>& omega_par, const std::vector<std::complex<double>>& pars, double ref);

#endif // UTILS_H
