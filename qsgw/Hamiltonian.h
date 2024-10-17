#pragma once

#include <vector>
#include <complex>
#include <map>

#include "matrix_m.h"
#include "meanfield.h"  // 假设你有这个头文件定义了 MeanField 类

// 构建关联势函数
Matz build_correlation_potential_spin_k(
    const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
    int n_states);

// 构建哈密顿量函数
std::map<int, std::map<int, Matz>> construct_H0_GW(
    const std::map<int, std::map<int, Matz>> & H_KS_all,
    const std::map<int, std::map<int, Matz>> & vxc_all,
    const std::map<int, std::map<int, Matz>> & Hexx_all,
    const std::map<int, std::map<int, Matz>> & Vc_all,
    int n_spins, int n_kpoints, int n_states);

// 对 Hamiltonian 进行对角化并存储本征值和本征矢量
void diagonalize_and_store(MeanField& meanfield, const std::map<int, std::map<int, Matz>>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension);
