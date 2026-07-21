#pragma once
#include "meanfield.h"

#include <string>
#include <vector>

struct QsgwZeroTemperatureOccupationResult
{
    double chemical_potential = 0.0;
    double electron_count = 0.0;
    double vbm = 0.0;
    double cbm = 0.0;
    double gap = 0.0;
    bool metallic = false;
};

std::vector<double> read_qsgw_kpoint_weights(
    const std::string &file_path, int expected_n_kpoints);
double qsgw_physical_electron_count(
    const MeanField &meanfield,
    const std::vector<double> &kpoint_weights,
    double tolerance = 1.0e-12);
QsgwZeroTemperatureOccupationResult update_qsgw_zero_temperature_occupations(
    MeanField &meanfield,
    const MeanField &reference_meanfield,
    const std::vector<double> &kpoint_weights,
    double total_electrons,
    double degeneracy_tolerance_ha = 1.0e-10,
    double electron_tolerance = 1.0e-12);

double calculate_total_occupation(const MeanField &mf, double mu, double temperature);
double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons);
double calculate_eqp_fermi_energy(const MeanField &mf,
                                  std::map<int, std::map<int, std::map<int, double>>> e_qp_all, 
                                  double temperature, 
                                  double total_electrons) ;
double fermi_dirac(double energy, double mu, double temperature);
void update_fermi_energy_and_occupations(MeanField &meanfield, const double temperature, const double efermi);
