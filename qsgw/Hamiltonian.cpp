#include "Hamiltonian.h"
#include "fermi_energy_occupation.h"
#include <iostream>


// 定义复数类型
using cplxdb = std::complex<double>;

//G0, for scRPA
std::vector<std::vector<cplxdb>> build_G0(
    MeanField& meanfield,
    const std::vector<double>& freq_nodes,
    int ispin,
    int ikpt,
    int n_states) {
    // 获取虚频点列表和能量列表
    std::vector<double> eigenvals(n_states);

    for (int n = 0; n < n_states; ++n) {
        eigenvals[n] = meanfield.get_eigenvals()[ispin](ikpt, n);
    }

    // 创建 G0 矩阵，大小为 n_states x freq_nodes.size()
    std::vector<std::vector<cplxdb>> G0(n_states, std::vector<cplxdb>(freq_nodes.size()));

    for (int n = 0; n < n_states; ++n) {
        for (size_t w = 0; w < freq_nodes.size(); ++w) {
            // 计算 G_n^0(iω) = 1 / (iω - ε_n)
            cplxdb iw(0.0, freq_nodes[w]);  // 虚频 iω
            G0[n][w] = 1.0 / (iw - eigenvals[n]);
        }
    }
    return G0;
}
// // mode A
// Matz build_correlation_potential_spin_k(
//     const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
//     int n_states) {

//     Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
//     for (int i = 0; i < n_states; ++i)
//     {
//         for (int j = 0; j < n_states; ++j)
//         {
//             // 从自能矩阵中获取对应的元素
//             std::complex<double> sigc_i = sigc_spin_k[i][j][i];
//             std::complex<double> sigc_j = sigc_spin_k[i][j][j];

//             // 构建关联势矩阵
//             std::complex<double> Vc_ij = 0.5 * (sigc_i + sigc_j);
//             Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
//         }
//     }

//     return Vc_spin_k;
// }

//mode B
Matz build_correlation_potential_spin_k(
    const std::vector<std::vector<std::vector<cplxdb>>>& sigc_spin_k,
    int n_states) {
    Matz Vc_spin_k(n_states, n_states, MAJOR::COL);
    for (int i = 0; i < n_states; ++i)
    {
        std::complex<double> sigc1_i = sigc_spin_k[i][i][i];
        std::complex<double> Vc_ii = sigc1_i;
        Vc_spin_k(i, i) = std::real(Vc_ii);
        for (int j = 0; j < n_states; ++j)
        {
            if(i!=j){
                std::complex<double> sigc2 = sigc_spin_k[i][j][n_states];
                // 构建关联势矩阵
                std::complex<double> Vc_ij = sigc2;
                Vc_spin_k(i, j) = std::real(Vc_ij);  // 只取实部
            }  
        }
    }
    return Vc_spin_k;
}


// 计算scRPA交换-关联能量矩阵
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
    double temperature) 
{
    Matz V_rpa_ks(n_states, n_states, MAJOR::COL); // 交换-关联能量矩阵，列优先存储
    const double K_B = 3.16681e-6;  // Hartree/K
    const double threshold = 1e-5;  // 设定一个阈值，如果差值太小就不进行除法计算
    
    for (int n = 0; n < n_states; ++n) {
        // 获取填充因子 fn
        double energy_n = meanfield.get_eigenvals()[ispin](ikpt, n);
        double f_n = fermi_dirac(energy_n, mu, temperature) * 2.0 / meanfield.get_n_spins();
        
        for (int m = 0; m < n_states; ++m) {
            cplxdb V_nm = 0.0;
            
            if (n == m) {
                // // 计算 δfn / δεn
                // double df_n_depsilon_n = -f_n * (1.0 - f_n) / (K_B * temperature);

                // // 如果 df_n_depsilon_n 太小，则跳过该计算
                // if (std::abs(df_n_depsilon_n) < threshold) {
                //     V_rpa_ks(n, n) = 0.0;
                //     continue;  // 跳过这个循环的进一步计算
                // }

                // // 累加自能项
                // for (size_t w = 0; w < freq_weights.size(); ++w) {
                //     cplxdb sigc_nn_iw = sigc_spin_k.at(freq_nodes[w])(n, n);
                    
                //     // n = m 的对角元计算
                //     V_nm += freq_weights[w] * sigc_nn_iw * G0[n][w] * G0[n][w];
                // }

                // // 对 δfn / δεn 进行归一化处理
                // V_rpa_ks(n, n) = std::real(V_nm) / df_n_depsilon_n;
                V_rpa_ks(n, n) = 0.0;
            } 
            else {
                // 获取填充因子 fm
                double energy_m = meanfield.get_eigenvals()[ispin](ikpt, m);
                double f_m = fermi_dirac(energy_m, mu, temperature) * 2.0 / meanfield.get_n_spins();
                // double delta_nm = (f_n - f_m)/(energy_n - energy_m);
                double delta_nm = f_n - f_m;
                // 如果 f_n - f_m 太小，则跳过该计算
                if (std::abs(delta_nm) < threshold) {
                    V_rpa_ks(n, m) = 0.0;
                    continue;  // 跳过这个循环的进一步计算
                }

                // 累加交换-关联项
                for (size_t w = 0; w < freq_weights.size(); ++w) {
                    cplxdb sigc_nm_iw = sigc_spin_k.at(freq_nodes[w])(n, m);
                    V_nm += freq_weights[w] * sigc_nm_iw * G0[n][w] * G0[m][w];
                }

                // 归一化并存储
                V_rpa_ks(n, m) = std::real(V_nm * (energy_n - energy_m)) / delta_nm;
            }

            
        }
    }
    return V_rpa_ks;
}



std::map<int, std::map<int, Matz>> construct_H0_GW(
    MeanField& meanfield,
    const std::map<int, std::map<int, Matz>> & H_KS_all,
    const std::map<int, std::map<int, Matz>> & vxc_all,
    const std::map<int, std::map<int, Matz>> & Hexx_all,
    const std::map<int, std::map<int, Matz>> & Vc_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化 GW 哈密顿量矩阵
    std::map<int, std::map<int, Matz>> H0_GW_all;
    double efermi = meanfield.get_efermi();

    for (int ispin = 0; ispin < n_spins; ++ispin)
    {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt)
        {
            Matz Vxc_diff_spin_k = Hexx_all.at(ispin).at(ikpt) + Vc_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt);
            // cut if possible
            // for (int i = 0; i < n_states; ++i){
            //     double energy_i = meanfield.get_eigenvals()[ispin](ikpt, i);
            //     for (int j = 0; j < n_states; ++j){
            //         double energy_j = meanfield.get_eigenvals()[ispin](ikpt, j);
            //         if(energy_i > efermi+1.75||energy_j > efermi+1.75){
            //             Vxc_diff_spin_k(i, j)= 0.0 ;                    
            //         }
            //     }
            // }
            // 构建 GW 哈密顿量矩阵
            Matz H0_GW_spin_k = H_KS_all.at(ispin).at(ikpt) + Vxc_diff_spin_k;
            // Matz H0_GW_spin_k = H_KS_all.at(ispin).at(ikpt) - vxc_all.at(ispin).at(ikpt) +
            //                     Hexx_all.at(ispin).at(ikpt) ;
           
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

            // // 打印哈密顿量矩阵
            // printf("%77s\n", final_banner.c_str());
            // printf("Hamiltonian matrix (h):\n");
            // for (int i = 0; i < dimension; ++i) {
            //     for (int j = 0; j < nao; ++j) {
            //         printf("%20.16f ", h(i, j).real());
            //     }
            //     printf("\n"); // 换行
            // }
            // printf("%77s\n", final_banner.c_str());

            eigsh(h, w, eigvec_KS);

            // // 打印本征值 w
            // printf("Eigenvalues (w):\n");
            // for (int i = 0; i < dimension; ++i) {
            //     printf("%20.16f\n", w[i]);
            // }
            // printf("%77s\n", final_banner.c_str());

            // // 打印本征向量矩阵 eigvec_KS
            // printf("Eigenvectors (eigvec_KS):\n");
            // for (int i = 0; i < dimension; ++i) {
            //     for (int j = 0; j < nao; ++j) {
            //         printf("%20.16f ", eigvec_KS(i, j).real());
            //     }
            //     printf("\n"); // 换行
            // }
            // printf("%77s\n", final_banner.c_str());


            // 将本征值存储到 MeanField 的 eskb 矩阵
            for (int ib = 0; ib < dimension; ++ib)
            {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = w[ib];
            }
            
            Matz wfc(dimension, nao, MAJOR::COL);
            for (int ib = 0; ib < dimension; ++ib)
            {
                for (int iao = 0; iao < nao; iao++)
                {
                    wfc(ib, iao) = meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao);
                }
            }
            auto eigvec_NAO = transpose(eigvec_KS) * wfc;
            
            // printf("%77s\n", final_banner.c_str());
            // printf("Eigenvectors2:\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &eigenvectors = meanfield.get_eigenvectors()[ispin][ikpt](i, j) ;
            //         printf("%20.16f ", eigenvectors.real());
            //     }
            //     printf("\n"); // 换行
            // }
            // printf("%77s\n", final_banner.c_str());
            // printf("\n");
            // 将 KS 表示旋转到 NAO 表示

            for (int ib = 0; ib < dimension; ++ib)
            {
                for (int iao = 0; iao < nao; iao++)
                {
                    meanfield.get_eigenvectors()[ispin][ikpt](ib, iao) = eigvec_NAO(ib, iao);
                }
            }
            // printf("%77s\n", final_banner.c_str());
            // printf("Eigenvectors3:\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &eigenvectors = meanfield.get_eigenvectors()[ispin][ikpt](i, j) ;
            //         printf("%20.16f ", eigenvectors.real()); 
            //     }
            //     printf("\n"); // 换行
            // }
            // printf("%77s\n", final_banner.c_str());
            // printf("\n");
        }
    }
    std::cout << "所有本征值已存储到 MeanField 对象。" << std::endl;
}


