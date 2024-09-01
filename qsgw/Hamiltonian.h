#pragma once

#include <vector>
#include <complex>
#include <map>

#include "matrix_m.h"
#include "meanfield.h"  // 假设你有这个头文件定义了 MeanField 类

// 构建关联势函数
std::vector<std::vector<Matz>> build_correlation_potential(
    const std::map<int, std::map<int, std::map<int, std::map<int, std::map<double, std::complex<double>>>>>>& sigc_real_all,
    const std::map<int, std::map<int, std::map<int, double>>>& e_qp_all,
    int n_spins, int n_kpoints, int n_states);

// 构建哈密顿量函数
std::vector<std::vector<Matz>> construct_H0_GW(
    const std::vector<std::vector<Matz>>& H_KS_all,
    const std::vector<std::vector<Matz>>& vxc0_all,
    const std::vector<std::vector<Matz>>& Hexx_all,
    const std::vector<std::vector<Matz>>& Vc_all,
    int n_spins, int n_kpoints, int n_states);

// 对 Hamiltonian 进行对角化并存储本征值和本征矢量
void diagonalize_and_store(MeanField& meanfield, const std::vector<std::vector<Matz>>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension);

// 旋转本征态至原子基下，并存储到 MeanField 对象的 wfc 矩阵中
void store_newMatrix_to_wfc(MeanField &meanfield, const std::vector<std::vector<std::vector<std::vector<double>>>>& newMatrix,
                            int n_spins, int nkpts, int nbands, int n_aos);
