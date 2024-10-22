#include "fermi_energy_occupation.h"

#include <cmath>
#include <vector>
#include "constants.h"

// 费米分布函数
static double fermi_dirac(double energy, double mu, double temperature)
{
    const double K_B = 3.16681e-6;  // Hartree/K
    // return 1.0 / (1.0 + exp((energy - mu) / (K_B * temperature)));
    if (energy > mu) {
        return 0.0;
    } else {
        return 1.0;
    }
}

// 计算给定化学势下的总占据态
static double calculate_total_occupation(const MeanField &mf, double mu, double temperature) {
    double total_occupation = 0.0;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_occupation += occupation;
            }
        }
    }

    return total_occupation;
}
//
double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons) {
    double tolerance = 1e-4;  
    double total_occupation = 0.0;
    double mu = 0.0;
    double gap = 0.0;
    double vbm = -10000.0;  // 比mu小的最大值
    double cbm = 10000.0;  // 比mu大的最小值

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                mu = energy;
                total_occupation = calculate_total_occupation(mf, mu, temperature);
                // 如果能量比mu小，更新vbm
                if (total_occupation < total_electrons + tolerance) {
                    if (energy > vbm) {
                        vbm = energy;
                    } 
                }
                // 如果能量比mu大，更新cbm
                else{
                    if (energy < cbm) {
                        cbm = energy;
                    } 
                }
            }
        }
    }

    // 最终费米能级取 vbm 和 cbm 的中间值
    mu = (vbm + cbm) * 0.5;
    gap = cbm - vbm ;
    std::cout << "Final VBM: " << vbm << ", CBM: " << cbm << ", Final Fermi level: " << mu << std::endl;
    std::cout << "gap: " << gap * HA2EV << " eV, "<< std::endl;
    return mu;  
}

void update_fermi_energy_and_occupations(MeanField &mf, const double temperature, const double efermi)
{
    double total_electrons1 = 0.0;
    // 更新占据数
    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin)
    {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt)
        {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib)
            {
                const double energy = mf.get_eigenvals()[ispin](ikpt, ib);
                mf.get_weight()[ispin](ikpt, ib) = fermi_dirac(energy, efermi, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_electrons1 += mf.get_weight()[ispin](ikpt, ib);  // 计算总占据数
            }
        }
    }
    // 输出 total_electrons
    std::cout << "Total electrons: " << total_electrons1 << std::endl;
    std::cout << "efermi: " << efermi << std::endl;
    mf.get_efermi() = efermi;
}
