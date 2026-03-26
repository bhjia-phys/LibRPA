#pragma once
#include <string>
#include <vector>

#include "dielecmodel.h"

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target);
bool use_iterative_pyatb_headwing_bundle();
std::string get_iterative_pyatb_headwing_bundle_dir();
std::string resolve_active_pyatb_headwing_dir();
void initialize_headwing_velocity_from_input(MeanField &mf);
void refresh_pyatb_headwing_bundle(const MeanField &mf,
                                   const std::vector<Vector3_Order<double>> &kfrac,
                                   const std::string &bundle_dir = "");
