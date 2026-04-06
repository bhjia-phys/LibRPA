/*!
 @file pbc.h
 @brief Utilities to deal with periodic boundary conditions
 */
#pragma once
#include <array>
#include <complex>
#include <map>
#include <vector>
#include "vector3_order.h"
#include "matrix3.h"

// TODO: make it into a template
std::vector<Vector3_Order<int>> construct_R_grid(const Vector3_Order<int> &period, bool upper_half = false);

//! Get the index of R in an Rlist. If R is not found in the list, return a negative number
int get_R_index(const std::vector<Vector3_Order<int>> &Rlist, const Vector3_Order<int> &R);

bool is_gamma_point(const Vector3_Order<double> &kpt);
bool is_gamma_point(const Vector3_Order<int> &kpt);

// Find the pair-dependent BvK representative of a lattice translation for the AO pair (I, J).
Vector3_Order<int> collapse_to_pairwise_bvk_R(const std::array<double, 3>& tau_I,
                                              const std::array<double, 3>& tau_J,
                                              const Matrix3& lattice,
                                              const Vector3_Order<int>& period,
                                              const Vector3_Order<int>& reference_R);

// Build the pair-dependent BvK translation set used by AO Fourier interpolation.
std::vector<Vector3_Order<int>> build_pairwise_bvk_R_grid(
    const std::array<double, 3>& tau_I,
    const std::array<double, 3>& tau_J,
    const Matrix3& lattice,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& reference_Rs);

// Compute the AO Fourier interpolation coefficient on a precomputed pair-dependent BvK grid.
std::complex<double> pairwise_bvk_interpolation_coeff(
    const std::vector<Vector3_Order<int>>& pair_bvk_Rs,
    const Vector3_Order<double>& target_k,
    const Vector3_Order<double>& mesh_k);

// Convenience overload that constructs the pair-dependent BvK grid on demand.
std::complex<double> pairwise_bvk_interpolation_coeff(
    const std::array<double, 3>& tau_I,
    const std::array<double, 3>& tau_J,
    const Matrix3& lattice,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& reference_Rs,
    const Vector3_Order<double>& target_k,
    const Vector3_Order<double>& mesh_k);

extern int kv_nmp[3];
//! lattice vectors as a 3D-matrix, each row as a lattice vector. Unit: Bohr
extern Matrix3 latvec;
//! same as latvec, but a nested array for LibRI call
extern std::array<std::array<double, 3>, 3> lat_array;
//! reciprocal lattice vectors as a 3D-matrix, each row as a reciprocal vector. Unit: 2pi/Bohr
extern Matrix3 G;
extern std::vector<Vector3_Order<double>> klist;
extern std::vector<Vector3_Order<double>> klist_ibz;
extern std::vector<Vector3_Order<double>> kfrac_list;
extern std::vector<int> irk_point_id_mapping;
extern map<Vector3_Order<double>, vector<Vector3_Order<double>>> map_irk_ks;
extern Vector3<double> *kvec_c;

//! Return the effective full-BZ k/q-point count represented by `map_irk_ks`.
//! When no irreducible-to-full mapping is available, fall back to `klist.size()`.
int get_full_bz_kpoint_count();
