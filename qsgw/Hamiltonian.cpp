#include "Hamiltonian.h"

#include <iostream>

std::vector<std::vector<Matz>> build_correlation_potential(
    const std::map<int, std::map<int, std::map<int, std::map<int, std::map<double, std::complex<double>>>>>>& sigc_real_all,
    const std::map<int, std::map<int, std::map<int, double>>>& e_qp_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化关联势矩阵
    std::vector<std::vector<Matz>> Vc_all(n_spins);

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
            for (int i = 0; i < n_states; ++i)
            {
                for (int j = 0; j < n_states; ++j)
                {
                    // 获取 i 和 j 状态对应的准粒子能量
                    double freq_i = e_qp_all.at(ispin).at(ikpt).at(i);
                    double freq_j = e_qp_all.at(ispin).at(ikpt).at(j);

                    // 从自能矩阵中获取对应的元素
                    std::complex<double> sigc_i =
                        sigc_real_all.at(ispin).at(ikpt).at(i).at(j).at(freq_i);
                    std::complex<double> sigc_j =
                        sigc_real_all.at(ispin).at(ikpt).at(i).at(j).at(freq_j);

                    // 构建关联势矩阵
                    std::complex<double> Vc_ij = 0.5 * (sigc_i + sigc_j);
                    Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
                }
            }
            Vc_all[ispin].push_back(Vc_spin_k);
        }
    }

    return Vc_all;
}

std::vector<std::vector<Matz>> construct_H0_GW(
    const std::vector<std::vector<Matz>>& H_KS_all,
    const std::vector<std::vector<Matz>>& vxc0_all,
    const std::vector<std::vector<Matz>>& Hexx_all,
    const std::vector<std::vector<Matz>>& Vc_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化 GW 哈密顿量矩阵
    std::vector<std::vector<Matz>> H0_GW_all(n_spins);

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            // 构建 GW 哈密顿量矩阵
            Matz H0_GW_spin_k = H_KS_all[ispin][ikpt] - vxc0_all[ispin][ikpt] +
                                Hexx_all[ispin][ikpt] + Vc_all[ispin][ikpt];
            H0_GW_all[ispin].push_back(H0_GW_spin_k);
        }
    }

    return H0_GW_all;
}

void diagonalize_and_store(MeanField& meanfield, const std::vector<std::vector<Matz>>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension)
{
    int nao = meanfield.get_n_aos();

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            // 取出相应的哈密顿量矩阵
            const auto &h = H0_GW_all[ispin][ikpt].copy();

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
            // FIXME: eigenvectors not stored?
        }
    }
    std::cout << "所有本征值已存储到 MeanField 对象。" << std::endl;
}

void store_newMatrix_to_wfc(MeanField &meanfield, const std::vector<std::vector<std::vector<std::vector<double>>>>& newMatrix,
                            int n_spins, int nkpts, int nbands, int n_aos) {

    // FIXME: Your input newMatrix is of dimension NBANDS*NBANDS, which is not wave function expanded in NAO basis.
    // Eigenvectors should be modified in diagonalize_and_store, not here
    for (int is = 0; is < n_spins; ++is)
    {
        for (int ik = 0; ik < nkpts; ++ik)
        {
            for (int n = 0; n < nbands; ++n)
            {
                for (int i = 0; i < n_aos; ++i)
                {
                    meanfield.get_eigenvectors()[is][ik](n, i) = newMatrix[is][ik][n][i];
                }
            }
        }
    }
    std::cout << "所有转换后的本征矢量已存储到 MeanField 对象。" << std::endl;
}
