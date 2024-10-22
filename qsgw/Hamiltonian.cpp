#include "Hamiltonian.h"

#include <iostream>

//mode A
Matz build_correlation_potential_spin_k(
    const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
    int n_states) {

    Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
    for (int i = 0; i < n_states; ++i)
    {
        for (int j = 0; j < n_states; ++j)
        {
            // 从自能矩阵中获取对应的元素
            std::complex<double> sigc_i = sigc_spin_k[i][j][i];
            std::complex<double> sigc_j = sigc_spin_k[i][j][j];

            // 构建关联势矩阵
            std::complex<double> Vc_ij = 0.5 * (sigc_i + sigc_j);
            Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
        }
    }

    return Vc_spin_k;
}

// //mode B
// Matz build_correlation_potential_spin_k(
//     const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
//     int n_states) {

//     Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
//     for (int i = 0; i < n_states; ++i)
//     {
//         std::complex<double> sigc1_i = sigc_spin_k[i][i][i];
//         std::complex<double> Vc_ii = sigc1_i;
//         Vc_spin_k(i, i) = std::real(Vc_ii);
//         for (int j = 0; j < n_states; ++j)
//         {
//             if(i==j){
//                 continue;
//             }
//             // 从自能矩阵中获取对应的元素
            
//             std::complex<double> sigc2 = sigc_spin_k[i][j][n_states];

//             // 构建关联势矩阵
//             std::complex<double> Vc_ij = sigc2;
//             Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
//         }
//     }

//     return Vc_spin_k;
// }



std::map<int, std::map<int, Matz>> construct_H0_GW(
    const std::map<int, std::map<int, Matz>> & H_KS_all,
    const std::map<int, std::map<int, Matz>> & vxc_all,
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
            Matz H0_GW_spin_k = H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
                                Hexx_all.at(ispin).at(ikpt) + Vc_all.at(ispin).at(ikpt);
            // Matz H0_GW_spin_k = H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
            //                     Hexx_all.at(ispin).at(ikpt) ;
            // // 设置阈值，小于此阈值的非对角元素置为 0
            // const double threshold = 1e-6;
            
            // // 假设 Matz 是 n_states x n_states 的方阵
            // for (int i = 0; i < n_states; ++i) {
            //     for (int j = 0; j < n_states; ++j) {
            //         if (i != j && std::abs(H0_GW_spin_k(i, j).real()) < threshold) {
            //             H0_GW_spin_k(i, j) = 0.0; // 将小于阈值的非对角元实部置为 0
            //         }
            //     }
            // }
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
            const std::string final_banner(90, '-');
            // 对角化哈密顿量，得到 QP 波函数在 KS 表象下的表示
            std::vector<double> w;
            Matz eigvec_KS;
            eigsh(h, w, eigvec_KS);

            // 将本征值存储到 MeanField 的 eskb 矩阵
            for (int ib = 0; ib < dimension; ++ib)
            {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = w[ib];
            }
            // printf("%77s\n", final_banner.c_str());
            // printf("Eigenvectors1:\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &eigenvectors = meanfield.get_eigenvectors()[ispin][ikpt](i, j) ;
            //         printf("%16.6f ", eigenvectors.real()); 
            //     }
            //     printf("\n"); // 换行
            // }
            // printf("%77s\n", final_banner.c_str());
            // printf("\n");
            // Rotate H0_GW eigenvectors and store in meanfield
            Matz wfc(dimension, nao, MAJOR::COL);
            for (int ib = 0; ib < dimension; ++ib)
            {
                for (int iao = 0; iao < nao; iao++)
                {
                    wfc(ib, iao) = meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao);
                }
            }
            auto eigvec_NAO = transpose(eigvec_KS) * wfc;
            
            // // 确保新特征向量方向与原特征向量一致
            // for (int ib = 0; ib < dimension; ++ib)
            // {
            //     double dot_product = 0.0;
            //     for (int iao = 0; iao < nao; ++iao)
            //     {
            //         dot_product += wfc(ib, iao).real() * eigvec_NAO(ib, iao).real();
            //     }

            //     // 如果点积为负数，翻转新特征向量的方向
            //     if (dot_product < 0.0)
            //     {
            //         for (int iao = 0; iao < nao; ++iao)
            //         {
            //             eigvec_NAO(ib, iao) = -eigvec_NAO(ib, iao);
            //         }
            //     }
            // }
            printf("%77s\n", final_banner.c_str());
            printf("Eigenvectors2:\n");
            for (int i = 0; i < meanfield.get_n_bands(); i++) {
                for (int j = 0; j < meanfield.get_n_bands(); j++) {
                    const auto &eigenvectors = meanfield.get_eigenvectors()[ispin][ikpt](i, j) ;
                    printf("%20.16f ", eigenvectors.real()); 
                }
                printf("\n"); // 换行
            }
            printf("%77s\n", final_banner.c_str());
            printf("\n");
            // 将 KS 表示旋转到 NAO 表示

            for (int ib = 0; ib < dimension; ++ib)
            {
                for (int iao = 0; iao < nao; iao++)
                {
                    meanfield.get_eigenvectors()[ispin][ikpt](ib, iao) = eigvec_NAO(ib, iao);
                }
            }
            printf("%77s\n", final_banner.c_str());
            printf("Eigenvectors3:\n");
            for (int i = 0; i < meanfield.get_n_bands(); i++) {
                for (int j = 0; j < meanfield.get_n_bands(); j++) {
                    const auto &eigenvectors = meanfield.get_eigenvectors()[ispin][ikpt](i, j) ;
                    printf("%20.16f ", eigenvectors.real()); 
                }
                printf("\n"); // 换行
            }
            printf("%77s\n", final_banner.c_str());
            printf("\n");
        }
    }
    std::cout << "所有本征值已存储到 MeanField 对象。" << std::endl;
}


