#pragma once
#include <map>
#include <string>
#include <vector>

#include "dielecmodel.h"
#include "matrix_m.h"

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target);
bool use_iterative_pyatb_headwing_bundle();
std::string get_iterative_pyatb_headwing_bundle_dir();
std::string resolve_active_pyatb_headwing_dir();
void initialize_headwing_velocity_from_input(MeanField &mf);
void export_pyatb_state_bundle(
    const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac,
    const std::string &bundle_dir = "",
    const std::map<int, std::map<int, Matz>> *hamiltonians = nullptr, int iteration = -1,
    const std::string &state_label = "kgrid", bool run_rebuild_command = false);
void refresh_pyatb_headwing_bundle(const MeanField &mf,
                                   const std::vector<Vector3_Order<double>> &kfrac,
                                   const std::string &bundle_dir = "",
                                   const std::map<int, std::map<int, Matz>> *hamiltonians = nullptr,
                                   int iteration = -1,
                                   const std::string &state_label = "kgrid");
