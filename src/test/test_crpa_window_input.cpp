#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../core/crpa_window_input.h"

using namespace librpa_int;

namespace
{

void require(bool passed, const char* message)
{
    if (!passed) throw std::runtime_error(message);
}

double distance(const ComplexMatrix& left, const ComplexMatrix& right)
{
    require(left.nr == right.nr && left.nc == right.nc, "matrix dimensions differ");
    double result = 0.0;
    for (int i = 0; i < left.size; ++i) result += std::norm(left.c[i] - right.c[i]);
    return std::sqrt(result);
}

void test_independent_windows_spin_frames_and_spectator()
{
    // Li(s), Co(d), O(p): the Li species must never become a ligand site.
    AtomicBasis basis(std::vector<std::size_t>{1, 5, 3});
    basis.set_l_shells({{0}, {2}, {1}});
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    const std::vector<Vector3_Order<double>> kpoints{{0, 0, 0}, {0.5, 0, 0}};

    ComplexMatrix overlap(9, 9);
    overlap.set_as_identity_matrix();
    const double coupling = 0.23;
    overlap(1, 6) = {0, coupling};
    overlap(6, 1) = {0, -coupling};
    ComplexMatrix inverse_sqrt(9, 9);
    inverse_sqrt.set_as_identity_matrix();
    const double plus = 1.0 / std::sqrt(1.0 + coupling);
    const double minus = 1.0 / std::sqrt(1.0 - coupling);
    inverse_sqrt(1, 1) = inverse_sqrt(6, 6) = (plus + minus) / 2.0;
    inverse_sqrt(1, 6) = std::complex<double>(0, (plus - minus) / 2.0);
    inverse_sqrt(6, 1) = std::conj(inverse_sqrt(1, 6));

    MeanField meanfield(2, 2, 9, 9);
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            ComplexMatrix unitary(9, 9);
            unitary.set_as_identity_matrix();
            const double theta = 0.2 + 0.15 * spin + 0.08 * k;
            unitary(0, 0) = unitary(1, 1) = std::cos(theta);
            unitary(0, 1) = unitary(1, 0) = std::complex<double>(0, std::sin(theta));
            meanfield.get_eigenvectors()[spin][0][k] = transpose(inverse_sqrt * unitary, false);
            for (int band = 0; band < 9; ++band)
                meanfield.get_eigenvals()[spin](k, band) = band == 0 ? -10.0 : band - 5.0;
        }

    CrpaWindowOptions options;
    options.correlated_species = "Co";
    options.ligand_species = "O";
    options.parent_orbitals = "dp";
    options.output_orbitals = "dp";
    options.response_windows_ha = {-3.0, -2.0};
    options.orbital_windows_ha = {-4.0, 3.0};
    const std::vector<ComplexMatrix> overlaps{overlap, overlap};
    auto build = [&](const CrpaWindowOptions& opts)
    {
        return build_crpa_window_input(basis, convention, {0, 1, 2}, {"Li", "Co", "O"}, overlaps,
                                       kpoints, meanfield, opts);
    };
    const auto dp = build(options);
    require(dp.sites.size() == 2 && dp.sites[0].atom_index == 1 && dp.sites[1].atom_index == 2,
            "spectator was incorrectly included in local orbital sites");
    require(dp.response_bands[0][0] == std::vector<int>({2, 3}),
            "response energy selection changed");
    require(dp.orbital_bands[0][0].size() == 8, "orbital window has the wrong rank");
    require(distance(dp.orbitals_spin_k[0][0], dp.orbitals_spin_k[1][0]) > 0.05,
            "distinct spin output frames were collapsed");

    ComplexMatrix identity(8, 8);
    identity.set_as_identity_matrix();
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            const auto& phi = dp.orbitals_spin_k[spin][k];
            require(distance(transpose(phi, true) * overlap * phi, identity) < 1.0e-11,
                    "projected local frame is not metric-orthonormal");
            const auto* wfc = meanfield.find_wfc(spin, 0, k);
            require(distance(project_trial_orbitals_to_band_window(*wfc, overlap, phi,
                                                                   dp.orbital_bands[spin][k]),
                             phi) < 1.0e-11,
                    "output frame escaped its KS orbital window");
        }

    auto different_response = options;
    different_response.response_windows_ha = {0.0, 1.0};
    const auto changed = build(different_response);
    require(changed.response_bands != dp.response_bands,
            "response windows did not change selected states");
    require(distance(changed.orbitals_spin_k[1][1], dp.orbitals_spin_k[1][1]) < 1.0e-13,
            "response selection incorrectly changed output orbitals");

    options.output_orbitals = "eg";
    const auto eg = build(options);
    const auto& parent = dp.orbitals_spin_k[1][0];
    const auto& subset = eg.orbitals_spin_k[1][0];
    for (int row = 0; row < 9; ++row)
    {
        require(std::abs(subset(row, 0) - parent(row, 0)) < 1.0e-12,
                "eg|dp did not retain dz2 from the joint parent frame");
        require(std::abs(subset(row, 1) - parent(row, 3)) < 1.0e-12,
                "eg|dp did not retain dx2-y2 from the joint parent frame");
    }
    CorrelatedSubspace frame(eg.sites, kpoints, {{0, 0, 0}, {1, 0, 0}}, 9, 9, 1);
    for (int k = 0; k < 2; ++k)
    {
        frame.set_S_k(k, overlap);
        frame.set_W_k(k, eg.orbitals_spin_k[1][k]);
    }
    frame.compute_T_R();
    require(frame.compute_t_roundtrip_residual() < 1.0e-12,
            "spin frame Fourier transform failed to preserve projected orbitals");
}

void test_explicit_bands_across_overlapping_energy_ranges()
{
    AtomicBasis basis(std::vector<std::size_t>{5});
    basis.set_l_shells({{2}});
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    const std::vector<Vector3_Order<double>> kpoints{{0, 0, 0}, {0.5, 0, 0}};
    ComplexMatrix overlap(5, 5);
    overlap.set_as_identity_matrix();
    MeanField meanfield(1, 2, 5, 5);
    const int orbital_for_band[] = {1, 2, 4, 0, 3};
    const double energies[2][5] = {{-1, -0.5, 0, 0.3, 1}, {0.2, 0.5, 0.8, 1.2, 1.5}};
    for (int k = 0; k < 2; ++k)
    {
        ComplexMatrix eigenvectors(5, 5);
        for (int band = 0; band < 5; ++band)
        {
            eigenvectors(band, orbital_for_band[band]) = 1;
            meanfield.get_eigenvals()[0](k, band) = energies[k][band];
        }
        meanfield.get_eigenvectors()[0][0][k] = eigenvectors;
    }
    CrpaWindowOptions options;
    options.correlated_species = "V";
    options.parent_orbitals = options.output_orbitals = "t2g";
    options.response_bands = options.orbital_bands = {0, 1, 2};
    auto build = [&](const CrpaWindowOptions& selected)
    {
        return build_crpa_window_input(basis, convention, {0}, {"V"}, {overlap, overlap}, kpoints,
                                       meanfield, selected);
    };
    const auto result = build(options);
    ComplexMatrix expected(5, 3);
    for (int column = 0; column < 3; ++column) expected(orbital_for_band[column], column) = 1;
    for (int k = 0; k < 2; ++k)
    {
        require(result.response_bands[0][k] == std::vector<int>({0, 1, 2}) &&
                    result.orbital_bands[0][k] == std::vector<int>({0, 1, 2}),
                "explicit t2g selection did not retain exactly three original KS states");
        require(distance(result.orbitals_spin_k[0][k], expected) < 1e-12,
                "explicit t2g projection changed the analytic local orbitals");
    }
    const auto energy_selection = crpa_bands_in_energy_windows(meanfield, {-1, 0.8});
    require(energy_selection[0][0].size() == 4 && energy_selection[0][1].size() == 3,
            "fixture does not distinguish energy windows from fixed band selection");
    for (const bool response : {true, false})
    {
        auto ambiguous = options;
        (response ? ambiguous.response_windows_ha : ambiguous.orbital_windows_ha) = {-1, 0.8};
        bool rejected = false;
        try
        {
            build(ambiguous);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        require(rejected, "ambiguous energy and band selection was accepted");
    }
}

}  // namespace

int main()
{
    try
    {
        test_independent_windows_spin_frames_and_spectator();
        test_explicit_bands_across_overlapping_energy_ranges();
        std::cout << "cRPA window orbital numerical test passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
