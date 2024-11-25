#pragma once

#include <vector>
#include <complex>
#include <map>

#include "matrix_m.h"
#include "meanfield.h" 

//G0, for scRPA
std::vector<std::vector<cplxdb>> build_G0(
    MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    int ispin,
    int ikpt,
    int n_states);
// 构建关联势函数
Matz build_correlation_potential_spin_k(
    const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
    int n_states);

Matz calculate_scRPA_exchange_correlation(
    MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    const std::vector<double>& freq_weights,
    const std::map<double, Matz>& sigc_spin_k, 
    const std::vector<std::vector<cplxdb>>& G0,
    int ispin,
    int ikpt, 
    int n_states, 
    double mu, 
    double temperature);

// 构建哈密顿量函数
std::map<int, std::map<int, Matz>> construct_H0_GW(
    MeanField& meanfield,
    const std::map<int, std::map<int, Matz>> & H_KS_all,
    const std::map<int, std::map<int, Matz>> & vxc_all,
    const std::map<int, std::map<int, Matz>> & Hexx_all,
    const std::map<int, std::map<int, Matz>> & Vc_all,
    int n_spins, int n_kpoints, int n_states);

// 对 Hamiltonian 进行对角化并存储本征值和本征矢量
void diagonalize_and_store(MeanField& meanfield, const std::map<int, std::map<int, Matz>>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension);

std::map<int, Matz> FT_R_TO_K(
    MeanField& meanfield, 
    const std::map<int, Matz>& Vr_ispin, 
    const std::vector<Vector3_Order<int>>& Rlist);

std::map<int, Matz> FT_K_TO_R(
    MeanField& meanfield, 
    const std::map<int, Matz>& Vk_ispin, 
    const std::vector<Vector3_Order<int>>& Rlist);
