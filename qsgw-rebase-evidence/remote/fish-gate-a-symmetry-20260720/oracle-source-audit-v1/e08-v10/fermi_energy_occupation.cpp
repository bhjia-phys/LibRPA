#include "fermi_energy_occupation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "constants.h"

namespace
{

struct QsgwState
{
    double energy;
    double physical_capacity;
    double storage_capacity;
    int spin;
    int kpoint;
    int band;
};

double qsgw_state_capacity_factor(const MeanField &meanfield)
{
    return 2.0 / static_cast<double>(
                     meanfield.get_n_spins() * meanfield.get_n_soc());
}

void validate_qsgw_kpoint_weights(
    const MeanField &meanfield,
    const std::vector<double> &kpoint_weights,
    const double tolerance)
{
    if (!std::isfinite(tolerance) || !(tolerance > 0.0))
    {
        throw std::invalid_argument(
            "Legacy Scheme-A occupation tolerance must be finite and positive");
    }
    if (kpoint_weights.size() !=
        static_cast<std::size_t>(meanfield.get_n_kpoints()))
    {
        throw std::invalid_argument(
            "Legacy Scheme-A k-point weight count does not match meanfield");
    }
    const double sum =
        std::accumulate(kpoint_weights.begin(), kpoint_weights.end(), 0.0);
    if (!std::isfinite(sum) || std::abs(sum - 1.0) > tolerance)
    {
        throw std::invalid_argument(
            "Legacy Scheme-A k-point weights must sum to one");
    }
    for (const double weight : kpoint_weights)
    {
        if (!std::isfinite(weight) || weight < 0.0)
        {
            throw std::invalid_argument(
                "Legacy Scheme-A k-point weights must be finite and nonnegative");
        }
    }
}

void validate_qsgw_meanfield_layout(
    const MeanField &live_meanfield,
    const MeanField &reference_meanfield)
{
    if (live_meanfield.get_n_spins() !=
            reference_meanfield.get_n_spins() ||
        live_meanfield.get_n_kpoints() !=
            reference_meanfield.get_n_kpoints() ||
        live_meanfield.get_n_bands() !=
            reference_meanfield.get_n_bands() ||
        live_meanfield.get_n_soc() != reference_meanfield.get_n_soc())
    {
        throw std::invalid_argument(
            "Legacy Scheme-A live and reference mean-field layouts differ");
    }
    if (&live_meanfield == &reference_meanfield)
    {
        throw std::invalid_argument(
            "Legacy Scheme-A live and reference mean fields must not alias");
    }
}

} // namespace

std::vector<double> read_qsgw_kpoint_weights(
    const std::string &file_path, const int expected_n_kpoints)
{
    std::ifstream input(file_path);
    if (!input.good())
    {
        throw std::runtime_error(
            "Legacy Scheme-A cannot read k-point weights from " + file_path);
    }
    int period_x = 0;
    int period_y = 0;
    int period_z = 0;
    int n_kpoints = 0;
    int n_ibz = 0;
    if (!(input >> period_x >> period_y >> period_z >> n_kpoints >> n_ibz) ||
        period_x <= 0 || period_y <= 0 ||
        period_z <= 0 || n_kpoints != expected_n_kpoints || n_ibz <= 0 ||
        n_ibz > n_kpoints ||
        n_kpoints > period_x * period_y * period_z)
    {
        throw std::runtime_error(
            "Legacy Scheme-A bz_sampling_out header is inconsistent");
    }
    std::string line;
    std::getline(input, line);
    std::vector<double> weights;
    weights.reserve(static_cast<std::size_t>(n_kpoints));
    for (int expected_index = 1; expected_index <= n_kpoints;
         ++expected_index)
    {
        if (!std::getline(input, line))
        {
            throw std::runtime_error(
                "Legacy Scheme-A bz_sampling_out is missing k-point rows");
        }
        std::istringstream row(line);
        int index = 0;
        double weight = 0.0;
        double kx = 0.0;
        double ky = 0.0;
        double kz = 0.0;
        if (!(row >> index >> weight >> kx >> ky >> kz))
        {
            throw std::runtime_error(
                "Legacy Scheme-A bz_sampling_out contains a malformed row");
        }
        if (index != expected_index || !std::isfinite(weight) ||
            weight < 0.0 || !std::isfinite(kx) || !std::isfinite(ky) ||
            !std::isfinite(kz))
        {
            throw std::runtime_error(
                "Legacy Scheme-A bz_sampling_out contains invalid k-point data");
        }
        weights.push_back(weight);
    }
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1.0e-8)
    {
        throw std::runtime_error(
            "Legacy Scheme-A bz_sampling_out weights do not sum to one");
    }
    return weights;
}

double qsgw_physical_electron_count(
    const MeanField &meanfield,
    const std::vector<double> &kpoint_weights,
    const double tolerance)
{
    validate_qsgw_kpoint_weights(meanfield, kpoint_weights, tolerance);
    const double capacity_factor = qsgw_state_capacity_factor(meanfield);
    double electrons = 0.0;
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            const double storage_capacity =
                capacity_factor *
                kpoint_weights[static_cast<std::size_t>(kpoint)];
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                const double stored =
                    meanfield.get_weight()[spin](kpoint, band);
                if (!std::isfinite(stored) || stored < -tolerance ||
                    stored > storage_capacity + tolerance)
                {
                    throw std::invalid_argument(
                        "Legacy Scheme-A occupation exceeds storage capacity");
                }
                electrons += stored;
            }
        }
    }
    if (!std::isfinite(electrons))
    {
        throw std::invalid_argument(
            "Legacy Scheme-A electron count is non-finite");
    }
    return electrons;
}

QsgwZeroTemperatureOccupationResult update_qsgw_zero_temperature_occupations(
    MeanField &meanfield,
    const MeanField &reference_meanfield,
    const std::vector<double> &kpoint_weights,
    const double total_electrons,
    const double degeneracy_tolerance_ha,
    const double electron_tolerance)
{
    validate_qsgw_meanfield_layout(meanfield, reference_meanfield);
    if (!(degeneracy_tolerance_ha > 0.0) ||
        !(electron_tolerance > 0.0) ||
        !std::isfinite(degeneracy_tolerance_ha) ||
        !std::isfinite(electron_tolerance))
    {
        throw std::invalid_argument(
            "Legacy Scheme-A occupation tolerances must be finite and positive");
    }
    validate_qsgw_kpoint_weights(
        meanfield, kpoint_weights, electron_tolerance);
    const double capacity_factor =
        qsgw_state_capacity_factor(reference_meanfield);
    const double reference_electrons = qsgw_physical_electron_count(
        reference_meanfield, kpoint_weights, electron_tolerance);
    std::vector<QsgwState> states;
    states.reserve(static_cast<std::size_t>(meanfield.get_n_spins()) *
                   meanfield.get_n_kpoints() * meanfield.get_n_bands());
    double total_capacity = 0.0;
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            const double physical_capacity =
                capacity_factor *
                kpoint_weights[static_cast<std::size_t>(kpoint)];
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                const double energy =
                    meanfield.get_eigenvals()[spin](kpoint, band);
                if (!std::isfinite(energy))
                {
                    throw std::invalid_argument(
                        "Legacy Scheme-A eigenvalue is non-finite");
                }
                total_capacity += physical_capacity;
                states.push_back({energy, physical_capacity,
                                  physical_capacity, spin, kpoint, band});
            }
        }
    }
    if (!std::isfinite(total_capacity) ||
        !std::isfinite(total_electrons) || total_electrons < 0.0 ||
        total_electrons > total_capacity + electron_tolerance)
    {
        throw std::invalid_argument(
            "Legacy Scheme-A target electron count is outside capacity");
    }
    if (std::abs(reference_electrons - total_electrons) >
        electron_tolerance)
    {
        throw std::invalid_argument(
            "Legacy Scheme-A target electron count differs from reference occupations");
    }
    std::sort(states.begin(), states.end(),
              [](const QsgwState &lhs, const QsgwState &rhs) {
                  if (lhs.energy != rhs.energy) return lhs.energy < rhs.energy;
                  if (lhs.spin != rhs.spin) return lhs.spin < rhs.spin;
                  if (lhs.kpoint != rhs.kpoint) return lhs.kpoint < rhs.kpoint;
                  return lhs.band < rhs.band;
              });
    std::vector<matrix> updated_weights = meanfield.get_weight();
    for (auto &spin_weights : updated_weights)
    {
        spin_weights.zero_out();
    }
    double remaining = total_electrons;
    for (std::size_t begin = 0; begin < states.size();)
    {
        std::size_t end = begin + 1;
        while (end < states.size() &&
               std::abs(states[end].energy - states[begin].energy) <=
                   degeneracy_tolerance_ha)
        {
            ++end;
        }
        double group_capacity = 0.0;
        for (std::size_t index = begin; index < end; ++index)
        {
            group_capacity += states[index].physical_capacity;
        }
        const double fraction = group_capacity > 0.0
                                    ? std::max(0.0, std::min(
                                          1.0, remaining / group_capacity))
                                    : 0.0;
        for (std::size_t index = begin; index < end; ++index)
        {
            const QsgwState &state = states[index];
            updated_weights[state.spin](state.kpoint, state.band) =
                fraction * state.storage_capacity;
        }
        remaining -= fraction * group_capacity;
        if (remaining < electron_tolerance) remaining = 0.0;
        begin = end;
    }
    if (remaining > electron_tolerance)
    {
        throw std::runtime_error(
            "Legacy Scheme-A occupation filling did not conserve charge");
    }

    QsgwZeroTemperatureOccupationResult result;
    result.vbm = -std::numeric_limits<double>::infinity();
    result.cbm = std::numeric_limits<double>::infinity();
    double lowest = std::numeric_limits<double>::infinity();
    double highest = -std::numeric_limits<double>::infinity();
    for (const QsgwState &state : states)
    {
        const double stored =
            updated_weights[state.spin](state.kpoint, state.band);
        result.electron_count += stored;
        if (state.physical_capacity <= electron_tolerance) continue;
        lowest = std::min(lowest, state.energy);
        highest = std::max(highest, state.energy);
        if (stored > electron_tolerance)
            result.vbm = std::max(result.vbm, state.energy);
        if (stored <
            state.physical_capacity - electron_tolerance)
            result.cbm = std::min(result.cbm, state.energy);
        if (stored > electron_tolerance &&
            stored <
                state.physical_capacity - electron_tolerance)
            result.metallic = true;
    }
    if (!std::isfinite(result.vbm))
        result.vbm = std::isfinite(lowest) ? lowest : states.front().energy;
    if (!std::isfinite(result.cbm))
        result.cbm = std::isfinite(highest) ? highest : states.back().energy;
    result.gap = result.metallic
                     ? 0.0
                     : std::max(0.0, result.cbm - result.vbm);
    result.chemical_potential = result.metallic
                                    ? result.vbm
                                    : 0.5 * (result.vbm + result.cbm);
    if (std::abs(result.electron_count - total_electrons) >
        electron_tolerance)
    {
        throw std::runtime_error(
            "Legacy Scheme-A updated occupations do not conserve charge");
    }
    meanfield.get_weight() = std::move(updated_weights);
    meanfield.get_efermi() = result.chemical_potential;
    return result;
}

// 费米分布
double fermi_dirac(double energy, double mu, double temperature)
{
    // const double K_B = 3.16681e-6;  // Hartree/K
    // return 1.0 / (1.0 + exp((energy - mu) / (K_B * temperature)));
    if (energy <= mu) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// 计算给定化学势下的总占据态
double calculate_total_occupation(const MeanField &mf, double mu, double temperature) {
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
//calculate_local_occupation_for each ispin-ikpoint_0-temperature
static double calculate_local_occupation(const MeanField &mf, double mu, double temperature, int ispin, int ikpt) {
    double local_occupation = 0.0;
    for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
        double energy = mf.get_eigenvals()[ispin](ikpt, ib);
        double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / mf.get_n_spins();
        local_occupation += occupation;
    }
    return local_occupation;
}

// //1
// double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons) {
//     double tolerance = 1e-4;  
//     double total_occupation = 0.0;
//     double mu = 0.0;
//     double gap = 0.0;
//     double vbm = -10000.0;  // 比mu小的最大值
//     double cbm = 10000.0;  // 比mu大的最小值

//     for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
//         for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            
            
//             for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
//                 double energy = mf.get_eigenvals()[ispin](ikpt, ib);
//                 mu = energy;
//                 total_occupation = calculate_total_occupation(mf, mu, temperature);
//                 // 如果能量比mu小，更新vbm
//                 if (total_occupation < total_electrons + tolerance) {
//                     if (energy > vbm) {
//                         vbm = energy;
//                         std::cout << "ikpt1: " << ikpt << std::endl;
//                         std::cout << "ib1: " << ib << std::endl;
//                     } 
//                 }
//                 // 如果能量比mu大，更新cbm
//                 else{
//                     if (energy < cbm) {
//                         cbm = energy;
//                         std::cout << "ikpt2: " << ikpt << std::endl;
//                         std::cout << "ib2: " << ib << std::endl;
//                     } 
//                 }
//             }
//         }
//     }
//     // 最终费米能级取 vbm 和 cbm 的中间值
//     mu = (vbm + cbm) * 0.5;
//     gap = cbm - vbm ;
//     std::cout << "Final VBM: " << vbm * HA2EV<< ", CBM: " << cbm * HA2EV << ", Final Fermi level: " << mu * HA2EV << std::endl;
//     std::cout << "Hamiltonian_gap: " << gap * HA2EV << " eV, "<< std::endl;
//     return mu;  
// }

//calculate_semiconductor_gap
double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons) {
    double tolerance = 1e-5;   
    double mu = 0.0;
    double gap = 0.0;
    double vbm = -10000.0;  // 比mu小的最大值
    double cbm = 10000.0;  // 比mu大的最小值

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            double local_vbm = -10000.0;
            double local_cbm = 10000.0;
            double local_occupation = 0.0;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double local_mu = mf.get_eigenvals()[ispin](ikpt, ib);
                local_occupation = calculate_local_occupation(mf, local_mu, temperature, ispin, ikpt);
                
                // update vbm,cbm;
                if (local_occupation <= total_electrons + tolerance ) {
                    local_vbm = local_mu;                    
                }
                if (local_occupation > total_electrons + tolerance ) {
                    if (local_mu < local_cbm) {
                        local_cbm = local_mu;
                    }   
                }
            }
            if (local_vbm > vbm) {
                vbm = local_vbm;
            } 
            if (local_cbm < cbm){
                cbm = local_cbm;
            }
        }
    }
    mu = (vbm + cbm) * 0.5;
    gap = cbm - vbm ;
    std::cout << "Final VBM: " << vbm * HA2EV<< ", CBM: " << cbm * HA2EV << ", Final Fermi level: " << mu * HA2EV << std::endl;
    std::cout << "Hamiltonian_gap: " << gap * HA2EV << " eV, "<< std::endl;
    return mu;  
}


//calculate_local_occupation_for each ispin-ikpoint_0-temperature
static double calculate_eqp_local_occupation(const MeanField &mf, std::map<int, std::map<int, std::map<int, double>>> e_qp_all, double mu, double temperature, int ispin, int ikpt) {
    double local_occupation = 0.0;
    for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
        double energy = e_qp_all[ispin][ikpt][ib];
        double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / mf.get_n_spins();
        local_occupation += occupation;
    }
    return local_occupation;
}

static double calculate_eqp_total_occupation(const MeanField &mf, std::map<int, std::map<int, std::map<int, double>>> e_qp_all, double mu, double temperature) {
    double total_occupation = 0.0;

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double energy = e_qp_all[ispin][ikpt][ib];
                double occupation = fermi_dirac(energy, mu, temperature) * 2.0 / (mf.get_n_kpoints() * mf.get_n_spins());
                total_occupation += occupation;
            }
        }
    }

    return total_occupation;
}
// //2
// double calculate_eqp_fermi_energy(const MeanField &mf,
//                                   std::map<int, std::map<int, std::map<int, double>>> e_qp_all, 
//                                   double temperature, 
//                                   double total_electrons) {
//     double tolerance = 1e-4;  
//     double total_occupation = 0.0;
//     double mu = 0.0;
//     double gap = 0.0;
//     double vbm = -10000.0;  // 比mu小的最大值
//     double cbm = 10000.0;  // 比mu大的最小值

//     for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
//         for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            
            
//             for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
//                 double energy = e_qp_all[ispin][ikpt][ib];
//                 mu = energy;
//                 total_occupation = calculate_eqp_total_occupation(mf, e_qp_all, mu, temperature);
//                 // 如果能量比mu小，更新vbm
//                 if (total_occupation < total_electrons + tolerance) {
//                     if (energy > vbm) {
//                         vbm = energy;
//                         std::cout << "ikpt3: " << ikpt << std::endl;
//                         std::cout << "ib3: " << ib << std::endl;
//                     } 
//                 }
//                 // 如果能量比mu大，更新cbm
//                 else{
//                     if (energy < cbm) {
//                         cbm = energy;
//                         std::cout << "ikpt4: " << ikpt << std::endl;
//                         std::cout << "ib4: " << ib << std::endl;
//                     } 
//                 }
//             }
//         }
//     }

//     // 最终费米能级取 vbm 和 cbm 的中间值
//     mu = (vbm + cbm) * 0.5;
//     gap = cbm - vbm ;
//     std::cout << "Final eqp_VBM: " << vbm* HA2EV << ", eqp_CBM: " << cbm* HA2EV << ", Final eqp_Fermi level: " << mu * HA2EV<< std::endl;
//     std::cout << "eqp_gap: " << gap * HA2EV << " eV, "<< std::endl;
//     return gap;  
// }
//22
double calculate_eqp_fermi_energy(const MeanField &mf,
                                  std::map<int, std::map<int, std::map<int, double>>> e_qp_all, 
                                  double temperature, 
                                  double total_electrons) {
    double tolerance = 1e-5;                                 
    double mu = 0.0;
    double gap = 0.0;
    double vbm = -10000.0;  // 比mu小的最大值
    double cbm = 10000.0;  // 比mu大的最小值

    for (int ispin = 0; ispin < mf.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < mf.get_n_kpoints(); ++ikpt) {
            double local_vbm = -10000.0;
            double local_cbm = 10000.0;
            double local_occupation = 0.0;
            for (int ib = 0; ib < mf.get_n_bands(); ++ib) {
                double local_mu = e_qp_all[ispin][ikpt][ib];
                local_occupation = calculate_eqp_local_occupation(mf,e_qp_all ,local_mu, temperature, ispin, ikpt);
                std::cout << "local_occupation: " << local_occupation << std::endl;
                // update vbm,cbm;
                if (local_occupation <= total_electrons + tolerance ) {
                    local_vbm = local_mu;                    
                }
                if (local_occupation > total_electrons + tolerance) {
                    if (local_mu < local_cbm) {
                        local_cbm = local_mu;
                    }   
                }
            }
            if (local_vbm > vbm) {
                vbm = local_vbm;
            } 
            if (local_cbm < cbm){
                cbm = local_cbm;
            }
        }
    }

    // 最终费米能级取 vbm 和 cbm 的中间值
    mu = (vbm + cbm) * 0.5;
    gap = cbm - vbm ;
    std::cout << "Final eqp_VBM: " << vbm* HA2EV << ", eqp_CBM: " << cbm* HA2EV << ", Final eqp_Fermi level: " << mu * HA2EV<< std::endl;
    std::cout << "eqp_gap: " << gap * HA2EV << " eV, "<< std::endl;
    return gap;  
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
                total_electrons1 += (mf.get_weight()[ispin](ikpt, ib)*mf.get_n_kpoints());  // 计算总占据数
            }
        }
    }
    total_electrons1 = total_electrons1 / mf.get_n_kpoints();
    // 输出 total_electrons
    std::cout << "Total electrons: " << total_electrons1 << std::endl;
    std::cout << "efermi: " << efermi << std::endl;
    mf.get_efermi() = efermi;
}
