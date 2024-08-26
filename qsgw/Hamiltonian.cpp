#include <vector>
#include <complex>
#include <iostream>
#include <map>
#include <Eigen/Dense>
#include "meanfield.h"  // 假设你有这个头文件定义了 MeanField 类

using namespace Eigen;
using namespace std;

// 定义矩阵结构
using Matrix = std::vector<std::vector<std::complex<double>>>;

// 构建关联势函数
std::vector<Matrix> build_correlation_potential(
    const std::map<int, std::map<int, std::map<int, std::map<int, std::map<double, std::complex<double>>>>>>& sigc_real_all,
    const std::map<int, std::map<int, std::map<int, double>>>& e_qp_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化关联势矩阵
    std::vector<Matrix> Vc_all(n_spins, std::vector<Matrix>(n_kpoints, Matrix(n_states, std::vector<std::complex<double>>(n_states, {0.0, 0.0}))));

    for (int ispin = 0; ispin < n_spins; ++ispin) {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    // 获取 i 和 j 状态对应的准粒子能量
                    double freq_i = e_qp_all.at(ispin).at(ikpt).at(i);
                    double freq_j = e_qp_all.at(ispin).at(ikpt).at(j);
                    
                    // 从自能矩阵中获取对应的元素
                    std::complex<double> sigc_i = sigc_real_all.at(ispin).at(ikpt).at(i).at(j).at(freq_i);
                    std::complex<double> sigc_j = sigc_real_all.at(ispin).at(ikpt).at(i).at(j).at(freq_j);
                    
                    // 构建关联势矩阵
                    std::complex<double> Vc_ij = 0.5 * (sigc_i + sigc_j);
                    Vc_all[ispin][ikpt][i][j] = std::real(Vc_ij);  // 只取实部
                }
            }
        }
    }

    return Vc_all;
}

// 构建哈密顿量函数
std::vector<Matrix> construct_H0_GW(
    const std::vector<Matrix>& H_KS_all,
    const std::vector<Matrix>& vxc0_all,
    const std::vector<Matrix>& Hexx_all,
    const std::vector<Matrix>& Vc_all,
    int n_spins, int n_kpoints, int n_states) {

    // 初始化 GW 哈密顿量矩阵
    std::vector<Matrix> H0_GW_all(n_spins, std::vector<Matrix>(n_kpoints, Matrix(n_states, std::vector<std::complex<double>>(n_states, {0.0, 0.0}))));

    for (int ispin = 0; ispin < n_spins; ++ispin) {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
            for (int i = 0; i < n_states; ++i) {
                for (int j = 0; j < n_states; ++j) {
                    // 构建 GW 哈密顿量矩阵
                    H0_GW_all[ispin][ikpt][i][j] = H_KS_all[ispin][ikpt][i][j]
                                                  - vxc0_all[ispin][ikpt][i][j]
                                                  + Hexx_all[ispin][ikpt][i][j]
                                                  + Vc_all[ispin][ikpt][i][j];
                }
            }
        }
    }

    return H0_GW_all;
}

// 对 Hamiltonian 进行对角化并存储本征值和本征矢量
void diagonalize_and_store(MeanField &meanfield, const std::vector<Matrix>& H0_GW_all,
                           int n_spins, int n_kpoints, int dimension) {
    int nao = meanfield.get_n_aos();

    for (int ispin = 0; ispin < n_spins; ++ispin) {
        for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
            // 取出相应的哈密顿量矩阵
            MatrixXcd Hamiltonian_realfreq = MatrixXcd::Zero(dimension, dimension);
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; i < dimension; ++j) {
                    Hamiltonian_realfreq(i, j) = H0_GW_all[ispin][ikpt][i][j];
                }
            }

            // 对角化哈密顿量
            SelfAdjointEigenSolver<MatrixXd> es(Hamiltonian_realfreq.real());
            VectorXd eigenvalues = es.eigenvalues();
            MatrixXd eigenvectors = es.eigenvectors();

            // 将本征值存储到 MeanField 的 eskb 矩阵
            for (int ib = 0; ib < dimension; ++ib) {
                meanfield.get_eigenvals()[ispin](ikpt, ib) = eigenvalues(ib);
            }

        }
    }
    std::cout << "所有本征值已存储到 MeanField 对象。" << std::endl;
}

// 旋转本征态至原子基下，并存储到 MeanField 对象的 wfc 矩阵中
void store_newMatrix_to_wfc(MeanField &meanfield, const std::vector<std::vector<std::vector<std::vector<double>>>>& newMatrix,
                            int n_spins, int nkpts, int nbands, int n_aos) {

    for (int is = 0; is < n_spins; ++is) {
        for (int ik = 0; ik < nkpts; ++ik) {
            for (int n = 0; n < nbands; ++n) {
                for (int i = 0; i < n_aos; ++i) {
                    meanfield.get_eigenvectors()[is][ik](n, i) = newMatrix[is][ik][n][i];
                }
            }
        }
    }
    std::cout << "所有转换后的本征矢量已存储到 MeanField 对象。" << std::endl;
}
