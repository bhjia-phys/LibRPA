#include <iostream>
#include <cmath>
#include <vector>
#include "meanfield.h"

// 费米分布函数
double fermi_dirac(double energy, double mu, double temperature) {
    const double k_B = 8.617333262145e-5; // 玻尔兹曼常数，单位为eV/K
    return 1.0 / (1.0 + exp((energy - mu) / (k_B * temperature)));
}

// 计算给定化学势下的总占据态
double calculate_total_occupation(const MeanField &mf, double mu, double temperature) {
    double total_occupation = 0.0;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                double occupation = fermi_dirac(energy, mu, temperature);
                total_occupation += occupation;
            }
        }
    }

    return total_occupation;
}

// 调整化学势以匹配总占据数，并计算HOMO和LUMO能级
double calculate_fermi_energy_and_occupations(MeanField &mf, double temperature, double total_electrons) {
    double mu_min = -1e6;
    double mu_max = 1e6;
    double mu = 0.0;
    double tolerance = 1e-8;
    double homo = -1e6, lumo = 1e6;

    // 初始二分法寻找使总占据数等于总电子数的费米能级
    while (true) {
        mu = 0.5 * (mu_min + mu_max);
        double total_occupation = calculate_total_occupation(mf, mu, temperature);

        if (total_occupation > total_electrons) {
            mu_max = mu;
        } else if (total_occupation < total_electrons) {
            mu_min = mu;
        } else {
            // 如果总占据数正好等于总电子数，则停止二分法
            break;
        }
    }

    // 定义mu1和mu2
    double mu1 = mu;
    double mu2 = mu;

    // 对mu1和mu_min进行二分法，确定HOMO
    while (mu1 - mu_min > tolerance) {
        double mu_mid = 0.5 * (mu1 + mu_min);
        double total_occupation = calculate_total_occupation(mf, mu_mid, temperature);

        if (total_occupation < total_electrons) {
            mu_min = mu_mid;
        } else {
            mu1 = mu_mid;
        }
    }
    homo = mu1;

    // 对mu2和mu_max进行二分法，确定LUMO
    while (mu_max - mu2 > tolerance) {
        double mu_mid = 0.5 * (mu2 + mu_max);
        double total_occupation = calculate_total_occupation(mf, mu_mid, temperature);

        if (total_occupation > total_electrons) {
            mu_max = mu_mid;
        } else {
            mu2 = mu_mid;
        }
    }
    lumo = mu_max;

    // 计算费米能级为HOMO和LUMO的平均值
    double fermi_energy = 0.5 * (homo + lumo);

    // 更新占据数
    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                mf.get_weight()[ispin](ikpt, ib) = fermi_dirac(energy, fermi_energy, temperature);
            }
        }
    }

    return fermi_energy;
}
