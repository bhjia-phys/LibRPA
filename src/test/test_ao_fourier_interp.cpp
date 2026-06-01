#include "../constants.h"
#include "../pbc.h"

#include <cassert>
#include <complex>
#include <vector>

namespace
{

std::complex<double> direct_pair_bvk_fourier_sum(
    const std::vector<Vector3_Order<int>>& pair_bvk_Rs,
    const std::vector<std::complex<double>>& realspace_values,
    const Vector3_Order<double>& kfrac)
{
    assert(pair_bvk_Rs.size() == realspace_values.size());
    std::complex<double> result = 0.0;
    for (std::size_t iR = 0; iR != pair_bvk_Rs.size(); ++iR)
    {
        const auto ang = (kfrac * pair_bvk_Rs[iR]) * TWO_PI;
        result += std::complex<double>(std::cos(ang), std::sin(ang)) * realspace_values[iR];
    }
    return result;
}

void test_pair_bvk_interpolation_reproduces_offgrid_value()
{
    const Vector3_Order<int> period{4, 1, 1};
    const auto reference_Rs = construct_R_grid(period);
    const std::array<double, 3> tau_I{0.0, 0.0, 0.0};
    const std::array<double, 3> tau_J{0.30, 0.0, 0.0};
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);

    const auto pair_bvk_Rs =
        build_pairwise_bvk_R_grid(tau_I, tau_J, lattice, period, reference_Rs);
    assert(pair_bvk_Rs.size() == reference_Rs.size());

    const std::vector<std::complex<double>> realspace_values{
        {1.0, 0.0},
        {0.2, -0.4},
        {-0.3, 0.1},
        {0.7, 0.2},
    };
    assert(realspace_values.size() == pair_bvk_Rs.size());

    const std::vector<Vector3_Order<double>> mesh_kpoints{
        {0.00, 0.0, 0.0},
        {0.25, 0.0, 0.0},
        {-0.50, 0.0, 0.0},
        {-0.25, 0.0, 0.0},
    };

    std::vector<std::complex<double>> mesh_values(mesh_kpoints.size());
    for (std::size_t ik = 0; ik != mesh_kpoints.size(); ++ik)
    {
        mesh_values[ik] =
            direct_pair_bvk_fourier_sum(pair_bvk_Rs, realspace_values, mesh_kpoints[ik]);
    }

    const Vector3_Order<double> target_k{0.125, 0.0, 0.0};
    std::complex<double> interpolated = 0.0;
    for (std::size_t ik = 0; ik != mesh_kpoints.size(); ++ik)
    {
        interpolated += pairwise_bvk_interpolation_coeff(
                            tau_I, tau_J, lattice, period, reference_Rs, target_k,
                            mesh_kpoints[ik])
                        * mesh_values[ik];
    }

    const auto exact = direct_pair_bvk_fourier_sum(pair_bvk_Rs, realspace_values, target_k);
    assert(std::abs(interpolated - exact) < 1e-12);
}

} // namespace

int main()
{
    test_pair_bvk_interpolation_reproduces_offgrid_value();
    return 0;
}
