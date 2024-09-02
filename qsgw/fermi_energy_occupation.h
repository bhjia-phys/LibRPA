#pragma once
#include "meanfield.h"

double calculate_fermi_energy(const MeanField &mf, double temperature, double total_electrons);

void update_fermi_energy_and_occupations(MeanField &meanfield, const double temperature, const double efermi);
