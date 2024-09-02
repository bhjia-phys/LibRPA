#include "Hamiltonian.h"

#include <iostream>


Matz build_correlation_potential_spin_k(
    const Matz& sigc_spin_k,
    const std::vector<double>& e_qp_spin_k,
    int n_states) {

    Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
    for (int i = 0; i < n_states; ++i)
    {
        for (int j = 0; j < n_states; ++j)
        {
            // 获取 i 和 j 状态对应的准粒子能量
            double freq_i = e_qp_spin_k[i];
            double freq_j = e_qp_spin_k[j];

            // 从自能矩阵中获取对应的元素
            std::complex<double> sigc_i = sigc_spin_k(i, j);
            std::complex<double> sigc_j = sigc_spin_k(j, i);

            // 构建关联势矩阵
            std::complex<double> Vc_ij = 0.5 * (sigc_i + sigc_j);
            Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
        }
    }

    return Vc_spin_k;
}

std::map<int, std::map<int, Matz>> construct_H0_GW(
    const std::map<int, std::map<int, Matz>> & H_KS_all,
    const std::map<int, std::map<int, Matz>> & vxc0_all,
    const std::map<int, std::map<int, Matz>> & Hexx_all,
    const std::map<int, std::map<int, Matz>> & Vc_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化 GW 哈密顿量矩阵
    std::map<int, std::map<int, Matz>> H0_GW_all;

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            // 构建 GW 哈密顿量矩阵
            Matz H0_GW_spin_k = H_KS_all.at(ispin).at(ikpt) - vxc0_all.at(ispin).at(ikpt) +
                                Hexx_all.at(ispin).at(ikpt) + Vc_all.at(ispin).at(ikpt);
            H0_GW_all[ispin][ikpt] = H0_GW_spin_k;
        }
    }

    return H0_GW_all;
}

void diagonalize_and_store(MeanField& meanfield, const std::map<int, std::map<int, Matz>>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension)
{
    int nao = meanfield.get_n_aos();

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            // 取出相应的哈密顿量矩阵
            const auto &h = H0_GW_all.at(ispin).at(ikpt).copy();

            // 对角化哈密顿量
            // MYZ: we don't use Eigen3, so avoid using their classes.
            // For matrix, use the matrix_m template
            std::vector<double> w;
            Matz eigvec_spin_k;
            eigsh(h, w, eigvec_spin_k);

            // 将本征值存储到 MeanField 的 eskb 矩阵
            for (int ib = 0; ib < dimension; ++ib)
            {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = w[ib];
            }

            // TODO: rotate H0_GW eigenvectors and store in meanfield
        }
    }
    std::cout << "所有本征值已存储到 MeanField 对象。" << std::endl;
}
