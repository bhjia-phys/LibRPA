#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../core/crpa_window_input.h"
#include "../core/orbital_weights.h"

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

void test_common_sp_frame_with_cross_shell_overlap()
{
    // Two radial s shells distinguish the selected first shell from a spectator.
    AtomicBasis basis(std::vector<std::size_t>{5});
    basis.set_l_shells({{0, 0, 1}});
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    const double coupling = 0.31;
    ComplexMatrix overlap(5, 5), inverse_sqrt(5, 5);
    overlap.set_as_identity_matrix();
    inverse_sqrt.set_as_identity_matrix();
    overlap(0, 2) = overlap(2, 0) = coupling;
    const double plus = 1.0 / std::sqrt(1.0 + coupling);
    const double minus = 1.0 / std::sqrt(1.0 - coupling);
    inverse_sqrt(0, 0) = inverse_sqrt(2, 2) = (plus + minus) / 2.0;
    inverse_sqrt(0, 2) = inverse_sqrt(2, 0) = (plus - minus) / 2.0;
    MeanField meanfield(1, 1, 5, 5);
    meanfield.get_eigenvectors()[0][0][0] = transpose(inverse_sqrt, false);
    CrpaWindowOptions options;
    options.correlated_species = "Si";
    options.parent_orbitals = options.output_orbitals = "sp";
    options.response_bands = {0, 2};
    options.orbital_bands = {0, 1, 2, 3, 4};
    auto build = [&](const CrpaWindowOptions& selected)
    {
        return build_crpa_window_input(basis, convention, {0}, {"Si"}, {overlap},
                                       {{0, 0, 0}}, meanfield, selected);
    };
    const auto sp = build(options);
    require(sp.sites.size() == 1 && sp.sites[0].n_orbitals == 4,
            "s and p must share one site for cross-shell tensor elements");
    const int rows[] = {0, 2, 3, 4};
    ComplexMatrix expected(5, 4);
    for (int row = 0; row < 5; ++row)
        for (int col = 0; col < 4; ++col) expected(row, col) = inverse_sqrt(row, rows[col]);
    require(distance(sp.orbitals_spin_k[0][0], expected) < 1e-12,
            "joint sp normalization differs from the analytic metric frame");
    ComplexMatrix identity(4, 4);
    identity.set_as_identity_matrix();
    require(distance(transpose(expected, true) * overlap * expected, identity) < 1e-12,
            "sp frame is not orthonormal across shells");
    for (const auto& selection : {std::string("s"), std::string("p")})
    {
        options.output_orbitals = selection;
        const auto subset = build(options);
        const auto& phi = subset.orbitals_spin_k[0][0];
        const int offset = selection == "s" ? 0 : 1;
        for (int row = 0; row < 5; ++row)
            for (int col = 0; col < phi.nc; ++col)
                require(std::abs(phi(row, col) - expected(row, offset + col)) < 1e-12,
                        "s|sp or p|sp was normalized outside its common parent frame");
        require(subset.response_bands == sp.response_bands,
                "output orbital selection changed the response subspace");
    }
}

void test_existing_state_character_and_pdos()
{
    // Two s radial shells share one group, plus a p shell on a second atom.
    // An analytic complex 2x2 metric block gives S^(1/2)C=Q without a KS solve.
    AtomicBasis basis(std::vector<std::size_t>{2, 3});
    basis.set_l_shells({{0, 0}, {1}});
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    const double coupling = 0.31;
    ComplexMatrix overlap(5, 5), root(5, 5), inverse(5, 5);
    overlap.set_as_identity_matrix();
    root.set_as_identity_matrix();
    inverse.set_as_identity_matrix();
    overlap(0, 3) = {0, coupling};
    overlap(3, 0) = {0, -coupling};
    for (int sign : {-1, 1})
    {
        const double value = 1 + sign * coupling;
        const double a = std::sqrt(value) / 2, b = 1 / (2 * std::sqrt(value));
        root(0, 0) += a - 0.5;
        root(3, 3) += a - 0.5;
        root(0, 3) += std::complex<double>(0, sign * a);
        root(3, 0) -= std::complex<double>(0, sign * a);
        inverse(0, 0) += b - 0.5;
        inverse(3, 3) += b - 0.5;
        inverse(0, 3) += std::complex<double>(0, sign * b);
        inverse(3, 0) -= std::complex<double>(0, sign * b);
    }
    MeanField mf(2, 2, 5, 5);
    std::vector<ComplexMatrix> q_values, c_values;
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            ComplexMatrix q(5, 5);
            q.set_as_identity_matrix();
            const double theta = 0.19 + 0.13 * spin + 0.07 * k;
            q(0, 0) = q(3, 3) = std::cos(theta);
            q(0, 3) = q(3, 0) = std::complex<double>(0, std::sin(theta));
            auto c = inverse * q;
            // Independent band phases must not affect either population.
            for (int b = 0; b < 5; ++b)
                for (int ao = 0; ao < 5; ++ao)
                    c(ao, b) *= std::polar(1.0, 0.17 * (1 + spin + k + b));
            mf.get_eigenvectors()[spin][0][k] = transpose(c, false);
            for (int b = 0; b < 5; ++b)
            {
                mf.get_eigenvals()[spin](k, b) = -0.7 + 0.42 * b + 0.05 * spin + 0.03 * k;
                mf.get_weight()[spin](k, b) = b < 2 ? 0.5 : 0;
            }
            q_values.push_back(q);
            c_values.push_back(c);
        }
    const std::vector<double> kw{0.27, 0.73};
    const auto result =
        analyze_orbital_weights(basis, convention, {0, 1}, mf, {overlap, overlap}, kw, 1e-11);
    require(result.groups.size() == 2, "radial shells must be summed into the same atom/l group");
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            const auto index = spin * 2 + k;
            const auto sc = root * q_values[index];
            for (int b = 0; b < 5; ++b)
            {
                const auto& state = result.states[index * 5 + b];
                double grouped[2] = {0, 0};
                for (int ao = 0; ao < 5; ++ao)
                {
                    const double lo = std::norm(q_values[index](ao, b));
                    require(std::abs(state.lowdin_ao[ao] - lo) < 2e-12,
                            "Lowdin weights differ from analytic orthogonal coefficients");
                    const auto c_no_phase = inverse * q_values[index];
                    const double mu = std::real(std::conj(c_no_phase(ao, b)) * sc(ao, b));
                    require(std::abs(state.mulliken_ao[ao] - mu) < 2e-12,
                            "Mulliken weights differ from analytic complex metric contraction");
                    grouped[ao < 2 ? 0 : 1] += lo;
                }
                for (int g = 0; g < 2; ++g)
                    require(std::abs(state.lowdin[g] - grouped[g]) < 2e-12,
                            "orbital grouping lost a radial shell");
            }
            require(
                distance(mf.get_eigenvectors()[spin][0][k], transpose(c_values[index], false)) == 0,
                "orbital analysis changed the original KS coefficients");
        }
    const std::vector<double> grid{-0.45, 0.0, 0.71};
    const double sigma = 0.23;
    std::vector<double> mulliken, lowdin;
    gaussian_orbital_pdos(result, grid, sigma, mulliken, lowdin);
    for (int spin = 0; spin < 2; ++spin)
        for (int e = 0; e < 3; ++e)
        {
            double expected = 0;
            for (int k = 0; k < 2; ++k)
                for (int b = 0; b < 5; ++b)
                {
                    const double delta = (grid[e] - mf.get_eigenvals()[spin](k, b)) / sigma;
                    expected += kw[k] * std::exp(-0.5 * delta * delta) /
                                (sigma * std::sqrt(2 * std::acos(-1.0)));
                }
            const int offset = (spin * 3 + e) * 2;
            require(std::abs(lowdin[offset] + lowdin[offset + 1] - expected) < 2e-12 &&
                        std::abs(mulliken[offset] + mulliken[offset + 1] - expected) < 2e-12,
                    "PDOS does not preserve nonuniform k weights and separate spin spectra");
        }
}

}  // namespace

int main()
{
    try
    {
        test_independent_windows_spin_frames_and_spectator();
        test_explicit_bands_across_overlapping_energy_ranges();
        test_common_sp_frame_with_cross_shell_overlap();
        test_existing_state_character_and_pdos();
        std::cout << "cRPA window orbital numerical test passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
