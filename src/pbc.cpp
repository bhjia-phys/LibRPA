#include "pbc.h"
#include <algorithm>
#include <complex>
#include <limits>
#include <set>

#include "constants.h"

vector<Vector3_Order<int>> construct_R_grid(const Vector3_Order<int> &period, bool upper_half)
{
    // cout<<" begin to construct_R_grid"<<endl;
    vector<Vector3_Order<int>> R_grid;
    R_grid.clear();

    if (upper_half)
    {
        for (int x = -(period.x - 1) / 2; x <= period.x / 2; ++x)
            for (int y = -(period.y - 1) / 2; y <= period.y / 2; ++y)
                for (int z = -(period.z - 1) / 2; z <= period.z / 2; ++z)
                    R_grid.push_back({x, y, z});
    }
    else
    {
        for (int x = -(period.x) / 2; x <= (period.x - 1) / 2; ++x)
            for (int y = -(period.y) / 2; y <= (period.y - 1) / 2; ++y)
                for (int z = -(period.z) / 2; z <= (period.z - 1) / 2; ++z)
                    R_grid.push_back({x, y, z});
    }

    return R_grid;
}

int get_R_index(const vector<Vector3_Order<int>> &Rlist, const Vector3_Order<int> &R)
{
    // v1: manual search
    // for ( int iR = 0; iR != Rlist.size(); iR++ )
    // {
    //     if (Rlist[iR] == R) return iR;
    // }
    // v2: use algorithm
    auto itr = std::find(Rlist.cbegin(), Rlist.cend(), R);
    if ( itr != Rlist.cend()) return distance(Rlist.cbegin(), itr);
    return -1;
}

bool is_gamma_point(const Vector3_Order<double> &kpt)
{
    double thres = 1.0e-5;
    return -thres < kpt.x && kpt.x < thres
        && -thres < kpt.y && kpt.y < thres
        && -thres < kpt.z && kpt.z < thres;
}

bool is_gamma_point(const Vector3_Order<int> &kpt_int)
{
    return kpt_int.x == 0 && kpt_int.y == 0 && kpt_int.z == 0;
}

namespace
{

Vector3<double> to_vector3(const std::array<double, 3>& values)
{
    return {values[0], values[1], values[2]};
}

} // namespace

Vector3_Order<int> collapse_to_pairwise_bvk_R(const std::array<double, 3>& tau_I,
                                              const std::array<double, 3>& tau_J,
                                              const Matrix3& lattice,
                                              const Vector3_Order<int>& period,
                                              const Vector3_Order<int>& reference_R)
{
    const auto tau_I_vec = to_vector3(tau_I);
    const auto tau_J_vec = to_vector3(tau_J);

    double min_norm2 = std::numeric_limits<double>::max();
    Vector3_Order<int> nearest_R = reference_R;
    for (int ix = -1; ix <= 1; ++ix)
    {
        for (int iy = -1; iy <= 1; ++iy)
        {
            for (int iz = -1; iz <= 1; ++iz)
            {
                const Vector3_Order<int> candidate{
                    reference_R.x + ix * period.x,
                    reference_R.y + iy * period.y,
                    reference_R.z + iz * period.z,
                };
                const auto displacement = (tau_I_vec - tau_J_vec - Vector3<double>{
                                                                      static_cast<double>(candidate.x),
                                                                      static_cast<double>(candidate.y),
                                                                      static_cast<double>(candidate.z)})
                                          * lattice;
                const auto norm2 = displacement.norm2();
                if (norm2 < min_norm2)
                {
                    min_norm2 = norm2;
                    nearest_R = candidate;
                }
            }
        }
    }
    return nearest_R;
}

std::vector<Vector3_Order<int>> build_pairwise_bvk_R_grid(
    const std::array<double, 3>& tau_I,
    const std::array<double, 3>& tau_J,
    const Matrix3& lattice,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& reference_Rs)
{
    std::vector<Vector3_Order<int>> pair_bvk_Rs;
    pair_bvk_Rs.reserve(reference_Rs.size());
    std::set<Vector3_Order<int>> unique_Rs;
    for (const auto& reference_R : reference_Rs)
    {
        const auto pair_R =
            collapse_to_pairwise_bvk_R(tau_I, tau_J, lattice, period, reference_R);
        pair_bvk_Rs.push_back(pair_R);
        unique_Rs.insert(pair_R);
    }

    if (unique_Rs.size() != reference_Rs.size())
    {
        throw std::runtime_error(
            "The pair-dependent BvK construction generated duplicated lattice representatives");
    }
    return pair_bvk_Rs;
}

std::complex<double> pairwise_bvk_interpolation_coeff(
    const std::vector<Vector3_Order<int>>& pair_bvk_Rs,
    const Vector3_Order<double>& target_k,
    const Vector3_Order<double>& mesh_k)
{
    if (pair_bvk_Rs.empty())
    {
        throw std::runtime_error(
            "Cannot build an AO Fourier interpolation coefficient from an empty BvK grid");
    }

    std::complex<double> coeff = 0.0;
    const auto delta_k = target_k - mesh_k;
    for (const auto& R : pair_bvk_Rs)
    {
        const auto ang = (delta_k * R) * TWO_PI;
        coeff += std::complex<double>(std::cos(ang), std::sin(ang));
    }
    coeff /= static_cast<double>(pair_bvk_Rs.size());
    return coeff;
}

std::complex<double> pairwise_bvk_interpolation_coeff(
    const std::array<double, 3>& tau_I,
    const std::array<double, 3>& tau_J,
    const Matrix3& lattice,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& reference_Rs,
    const Vector3_Order<double>& target_k,
    const Vector3_Order<double>& mesh_k)
{
    return pairwise_bvk_interpolation_coeff(
        build_pairwise_bvk_R_grid(tau_I, tau_J, lattice, period, reference_Rs), target_k,
        mesh_k);
}

int kv_nmp[3] = {1, 1, 1};
Vector3<double> *kvec_c;
std::vector<Vector3_Order<double>> klist;
std::vector<Vector3_Order<double>> klist_ibz;
std::vector<Vector3_Order<double>> kfrac_list;
std::vector<int> irk_point_id_mapping;
map<Vector3_Order<double>, vector<Vector3_Order<double>>> map_irk_ks;
Matrix3 latvec;
std::array<std::array<double, 3>, 3> lat_array;
Matrix3 G;

int get_full_bz_kpoint_count()
{
    int full_count = 0;
    for (const auto &irk_full : map_irk_ks)
    {
        full_count += static_cast<int>(irk_full.second.size());
    }

    if (full_count > 0)
    {
        return full_count;
    }
    return static_cast<int>(klist.size());
}
