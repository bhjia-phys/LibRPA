#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <valarray>

#include "../core/dielecmodel.h"
#include "../api/dataset.h"
#include "../core/chi0.h"
#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;
using Z = std::complex<double>;

namespace
{
void close(Z actual, Z expected, double tolerance = 2e-11)
{
    if (!std::isfinite(std::abs(actual - expected)) || std::abs(actual - expected) > tolerance)
    {
        std::cerr << "Optical oracle: actual=" << actual << " expected=" << expected
                  << " error=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

Matz identity(int n)
{
    Matz result(n, n, MAJOR::COL);
    for (int i = 0; i < n; ++i) result(i, i) = 1.0;
    return result;
}

// Independent small dense Gauss-Jordan inverse; no production Schur or LU helper.
Matz inverse(Matz a)
{
    const int n = a.nr();
    auto result = identity(n);
    for (int k = 0; k < n; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(a(i, k)) > std::abs(a(pivot, k))) pivot = i;
        assert(std::abs(a(pivot, k)) > 1e-10);
        for (int j = 0; j < n; ++j)
        {
            std::swap(a(k, j), a(pivot, j));
            std::swap(result(k, j), result(pivot, j));
        }
        const auto diagonal = a(k, k);
        for (int j = 0; j < n; ++j)
        {
            a(k, j) /= diagonal;
            result(k, j) /= diagonal;
        }
        for (int i = 0; i < n; ++i)
        {
            if (i == k) continue;
            const auto factor = a(i, k);
            for (int j = 0; j < n; ++j)
            {
                a(i, j) -= factor * a(k, j);
                result(i, j) -= factor * result(k, j);
            }
        }
    }
    return result;
}

Z determinant3(const Matz &a)
{
    return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
           a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
           a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

Matz ordered_response(const std::array<double, 4> &energies,
                      const std::array<double, 4> &occupations, const std::vector<Matz> &probes,
                      double omega)
{
    Matz result(probes.size(), probes.size(), MAJOR::COL);
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
        {
            if (m == n) continue;
            const auto factor =
                (occupations[m] - occupations[n]) / Z(energies[m] - energies[n], omega);
            for (int a = 0; a < result.nr(); ++a)
                for (int b = 0; b < result.nc(); ++b)
                    result(a, b) += factor * probes[a](m, n) * std::conj(probes[b](m, n));
        }
    return result;
}

#ifdef LIBRPA_USE_LIBRI
void test_complete_optical_wc_has_zero_vacuum()
{
    for (const int naux : {1, 2})
    {
        Dataset ds(MPI_COMM_WORLD);
        ds.basis_wfc.set({1});
        ds.basis_wfc.set_l_shells({{0}});
        ds.basis_aux.set({std::size_t(naux)});
        ds.basis_aux.set_l_shells({std::vector<int>(naux, 0)});
        ds.mf.set(1, 1, 2, 1, 2);
        ds.mf.get_efermi() = 0;
        for (int component = 0; component != 2; ++component)
        {
            auto &c = ds.mf.get_eigenvectors()[0][component][0];
            c.create(2, 1);
            c(component, 0) = 1;
        }
        ds.mf.get_eigenvals()[0](0, 0) = -.7;
        ds.mf.get_eigenvals()[0](0, 1) = .8;
        ds.mf.get_weight()[0](0, 0) = 1;
        ds.mf.get_weight()[0](0, 1) = 0;
        ds.pbc.set_latvec({7.1, 0, 0, 1.3, 5.9, 0, 0, 0, 18});
        ds.pbc.set_kgrids_kvec(1, 1, 1, {0, 0, 0});
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true),
                               MPI_COMM_WORLD, 1);
        ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(2, 1, 2, 1);
        ds.desc_abf.reset_handler(ds.blacs_h);
        ds.desc_abf.init_1b1p(naux, naux, 0, 0);
        ds.tfg.reset(6);
        ds.tfg.generate_minimax(.4, 4., 1e-12, true);
        ds.cs_data.use_libri = true;
        atpair_k_cplx_mat_t epsilon_coulomb, external_coulomb;
        const auto q = ds.pbc.klist.front();
        if (ds.comm_h.is_root())
        {
            auto zero = std::make_shared<std::valarray<double>>(0., naux);
            ds.cs_data.data_libri[0][{0, {0, 0, 0}}] =
                RI::Tensor<double>({std::size_t(naux), 1UL, 1UL}, zero);
            auto ve = std::make_shared<ComplexMatrix>(naux, naux);
            auto vw = std::make_shared<ComplexMatrix>(naux, naux);
            (*ve)(0, 0) = 4;
            (*vw)(0, 0) = 1.8;
            if (naux == 2)
            {
                (*ve)(1, 1) = 1;
                (*ve)(0, 1) = Z(.4, .45);
                (*ve)(1, 0) = std::conj((*ve)(0, 1));
                (*vw)(1, 1) = 2.7;
                (*vw)(0, 1) = (*vw)(1, 0) = .23;
            }
            epsilon_coulomb[0][0][q] = ve;
            external_coulomb[0][0][q] = vw;
        }
        velocity_matrix_t velocity;
        initialize_velocity_matrix(velocity, 1, 1, 2);
        const auto assigned = dispatch_upper_triangular_tasks(
            1, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
            ds.blacs_h.myprow, ds.blacs_h.mypcol);
        const std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
        std::map<Vector3_Order<double>, ComplexMatrix> no_projection;
        for (const bool normalized : {false, true})
        {
            diele_func optics(ds.mf, velocity, ds.pbc.kfrac_list, ds.basis_wfc, ds.basis_aux,
                              ds.tfg.get_freq_nodes(), 1, 2, 1, naux, ds.pbc, ds.comm_h,
                              ds.blacs_h);
            optics.use_soc = true;
            optics.normalize_gamma_cell_weights = normalized;
            optics.init(0, epsilon_coulomb);
            optics.cal_head();
            optics.cal_wing(ds.cs_data, 0, epsilon_coulomb);
            const double weight = optics.gamma_cell_weight_sum();
            if (normalized)
                assert(std::abs(weight - 1) < 2e-14);
            else
                assert(std::abs(weight - 1) > 1e-5);
            for (const int option : {3, 53})
            {
                Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                         ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
                chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
                chi.build(LIBRPA_ROUTING_LIBRI, ds.cs_data, pairs, ds.basis_aux,
                          no_projection, ds.blacs_h);
                const std::vector<Z> activation(ds.tfg.get_n_grids(), 1.);
                auto wc = compute_Wc_freq_q_blacs(
                    chi, epsilon_coulomb, external_coulomb, 1e-12, true, option, activation,
                    &optics, ds.blacs_h, ds.desc_abf, false, ".", false, false, false);
                double error = 0;
                for (const auto &[frequency, qs] : wc)
                    for (const auto &[momentum, block] : qs)
                        for (std::size_t i = 0; i != block.size(); ++i)
                            error = std::max(error, std::abs(block.ptr()[i]));
                MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, ds.comm_h.comm);
                assert(error < 2e-11);
            }
        }
    }
}
#endif

void test_native_head_wings_and_joint_permutation(const BlacsCtxtHandler &blacs)
{
    const std::array<double, 4> energies{-1.2, -0.7, 0.8, 1.3};
    const std::vector<double> frequencies{0.03, 0.7, 2.1};
    Matz coefficients(4, 4, MAJOR::COL);
    for (int i = 0; i < 4; ++i)
        for (int n = 0; n < 4; ++n)
            coefficients(i, n) = 0.5 * std::polar(1.0, 0.5 * PI * i * n + 0.13 * i);
    const std::array<double, 4> gamma_values{0.3, 0.17, 0.17, 0.5};
    Matz gamma_ao(4, 4, MAJOR::COL);
    for (int s = 0; s < 2; ++s)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) gamma_ao(2 * s + i, 2 * s + j) = gamma_values[2 * i + j];
    const auto gamma = coefficients.get_transpose(true) * gamma_ao * coefficients;
    std::vector<Matz> position;
    for (int a = 0; a < 3; ++a) position.emplace_back(4, 4, MAJOR::COL);
    for (int a = 0; a < 3; ++a)
        for (int m = 0; m < 4; ++m)
            for (int n = m + 1; n < 4; ++n)
            {
                position[a](m, n) = Z(0.04 * (1 + a + m + n), 0.03 * (a + 1) * (n - m));
                position[a](n, m) = std::conj(position[a](m, n));
            }
    std::vector<Matz> probes;
    for (const auto &r : position) probes.push_back(Z(0, -1) * r);
    probes.push_back(gamma.copy());

    AtomicBasis basis_wfc({2}), basis_aux({1});
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    pbc.set_kgrids_kvec(1, 1, 1, {0, 0, 0});
    Cs_LRI cs;
    cs.use_libri = true;
    if (global::mpi_comm_global_h.is_root())
    {
        auto tensor = std::make_shared<std::valarray<double>>(4);
        for (int i = 0; i < 4; ++i) (*tensor)[i] = 0.5 * gamma_values[i];
        cs.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor);
    }
    double nonreciprocal_witness = 0.0, nonhermitian_head_witness = 0.0;
    for (bool fractional : {false, true})
    {
        const std::array<double, 4> f = fractional ? std::array<double, 4>{.93, .82, .19, .07}
                                                   : std::array<double, 4>{1, 1, 0, 0};
        for (const auto &permutation :
             {std::array<int, 4>{0, 1, 2, 3}, std::array<int, 4>{3, 0, 2, 1}})
        {
            MeanField mf(1, 1, 4, 2, 2);
            velocity_matrix_t velocity;
            initialize_velocity_matrix(velocity, 1, 1, 4);
            for (int m = 0; m < 4; ++m)
            {
                const int original_m = permutation[m];
                mf.get_eigenvals()[0](0, m) = energies[original_m];
                mf.get_weight()[0](0, m) = f[original_m];
                for (int s = 0; s < 2; ++s)
                {
                    auto &wfc = mf.get_eigenvectors()[0][s][0];
                    if (m == 0) wfc.create(4, 2);
                    for (int i = 0; i < 2; ++i) wfc(m, i) = coefficients(2 * s + i, original_m);
                }
                for (int n = 0; n < 4; ++n)
                    for (int a = 0; a < 3; ++a)
                    {
                        const int original_n = permutation[n];
                        velocity[0][0][a](m, n) =
                            Z(0, energies[original_m] - energies[original_n]) *
                            position[a](original_m, original_n);
                    }
            }
            diele_func optics(mf, velocity, pbc.kfrac_list, basis_wfc, basis_aux, frequencies, 2, 4,
                              1, 1, pbc, global::mpi_comm_global_h, blacs);
            optics.use_soc = true;
            atpair_k_cplx_mat_t empty_coulomb;
            optics.init(0, empty_coulomb);
            optics.cal_head();
#ifdef LIBRPA_USE_LIBRI
            optics.cal_wing(cs, 0, empty_coulomb);
#endif
            std::vector<Z> pair_right(3 * frequencies.size()), pair_left(3 * frequencies.size());
            for (int m = 0; m < 4; ++m)
                for (int n = m + 1; n < 4; ++n)
                {
                    const int om = permutation[m], on = permutation[n];
                    accumulate_lehmann_wing_mu_for_pair(
                        frequencies,
                        {velocity[0][0][0](n, m), velocity[0][0][1](n, m), velocity[0][0][2](n, m)},
                        gamma(om, on), energies[on] - energies[om], f[om] - f[on],
                        pair_right.data(), pair_left.data());
                }
            for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
            {
                const auto exact = ordered_response(energies, f, probes, frequencies[iw]);
                const auto &head = optics.get_head_tensors()[iw];
#ifdef LIBRPA_USE_LIBRI
                const auto &right = optics.get_wing_auxiliary()[iw];
                const auto &left = optics.get_left_wing_auxiliary_adjoint()[iw];
#endif
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        close(head(a, b), (a == b ? 1.0 : 0.0) - 4 * PI * exact(a, b));
                        nonhermitian_head_witness =
                            std::max(nonhermitian_head_witness,
                                     std::abs(head(a, b) - std::conj(head(b, a))));
                    }
                    close(2.0 * pair_right[3 * iw + a], exact(3, a));
                    close(2.0 * std::conj(pair_left[3 * iw + a]), exact(a, 3));
#ifdef LIBRPA_USE_LIBRI
                    close(right(0, a), -std::sqrt(4 * PI) * exact(3, a));
                    close(std::conj(left(0, a)), -std::sqrt(4 * PI) * exact(a, 3));
#endif
                    nonreciprocal_witness =
                        std::max(nonreciprocal_witness,
                                 std::abs(pair_right[3 * iw + a] - pair_left[3 * iw + a]));
                }
            }
#ifdef LIBRPA_USE_LIBRI
            // An independent complex rectangular map must transport BOTH wings.
            const std::array<double, 4> eta_values{0.11, -0.08, -0.08, 0.37};
            Matz eta_ao(4, 4, MAJOR::COL);
            for (int spin = 0; spin != 2; ++spin)
                for (int i = 0; i != 2; ++i)
                    for (int j = 0; j != 2; ++j)
                        eta_ao(2 * spin + i, 2 * spin + j) = eta_values[2 * i + j];
            const auto eta = coefficients.get_transpose(true) * eta_ao * coefficients;
            auto parent_probes = probes;
            parent_probes.push_back(eta);
            AtomicBasis parent_basis({2});
            Cs_LRI parent_cs;
            parent_cs.use_libri = true;
            if (global::mpi_comm_global_h.is_root())
            {
                auto tensor = std::make_shared<std::valarray<double>>(8);
                for (int i = 0; i != 4; ++i)
                {
                    (*tensor)[i] = 0.5 * gamma_values[i];
                    (*tensor)[4 + i] = 0.5 * eta_values[i];
                }
                parent_cs.data_libri[0][{0, {0, 0, 0}}] =
                    RI::Tensor<double>({2UL, 2UL, 2UL}, tensor);
            }
            ComplexMatrix projection(1, 2);
            projection(0, 0) = Z(0.8, 0.13);
            projection(0, 1) = Z(-0.21, 0.37);
            diele_func projected(mf, velocity, pbc.kfrac_list, basis_wfc, basis_aux,
                                  frequencies, 2, 4, 1, 1, pbc,
                                  global::mpi_comm_global_h, blacs);
            projected.use_soc = true;
            projected.init(0, empty_coulomb);
            projected.cal_head();
            projected.cal_projected_wing(parent_cs, parent_basis, projection, 0, empty_coulomb);
            assert(projected.has_projected_parent_wings());
            if (!fractional)
            {
                // A fresh borrowed context has no grid, head or frequency wings.
                // Compare both exact signed endpoint probes with raw ordered AO
                // columns, including a band permutation and complex rectangular U.
                auto endpoint = diele_func::make_borrowed_context(
                    mf, velocity, pbc.kfrac_list, basis_wfc, basis_aux, pbc,
                    global::mpi_comm_global_h, blacs, false, nullptr, nullptr);
                endpoint->use_soc = true;
                const auto grams = endpoint->build_optical_transition_grams(
                    parent_cs, parent_basis, &projection);
                assert(endpoint->get_frequency_nodes().empty());
                assert(endpoint->get_head_tensors().empty() && !endpoint->has_wing());
                Matz exact_head[2]{Matz(3, 3), Matz(3, 3)};
                Matz exact_cross[2]{Matz(1, 3), Matz(1, 3)};
                for (int m = 0; m != 4; ++m)
                    for (int n = 0; n != 4; ++n)
                    {
                        const int om = permutation[m], on = permutation[n];
                        if (f[om] == f[on]) continue;
                        const int side = f[om] > f[on] ? 0 : 1;
                        const auto rho = projection(0, 0) * gamma(om, on) +
                                         projection(0, 1) * eta(om, on);
                        for (int a = 0; a != 3; ++a)
                        {
                            exact_cross[side](0, a) +=
                                rho * std::conj(probes[a](om, on));
                            for (int b = 0; b != 3; ++b)
                                exact_head[side](a, b) +=
                                    probes[a](om, on) * std::conj(probes[b](om, on));
                        }
                    }
                double wrong_conjugation = 0.0;
                for (int side = 0; side != 2; ++side)
                    for (int a = 0; a != 3; ++a)
                    {
                        close(grams.charge_dipole[side](0, a), exact_cross[side](0, a), 2e-12);
                        for (int b = 0; b != 3; ++b)
                            close(grams.dipole[side](a, b), exact_head[side](a, b), 2e-12);
                        wrong_conjugation = std::max(
                            wrong_conjugation,
                            std::abs(grams.charge_dipole[1](0, a) +
                                     std::conj(grams.charge_dipole[0](0, a))));
                    }
                assert(wrong_conjugation > 1e-4);
                for (int m = 0; m != 4; ++m)
                {
                    assert(mf.get_weight()[0](0, m) == f[permutation[m]]);
                    assert(mf.get_eigenvals()[0](0, m) == energies[permutation[m]]);
                }
            }
            for (std::size_t iw = 0; iw != frequencies.size(); ++iw)
            {
                const auto exact = ordered_response(energies, f, parent_probes, frequencies[iw]);
                for (int a = 0; a != 3; ++a)
                {
                    const auto expected_right = -std::sqrt(4 * PI) *
                        (projection(0, 0) * exact(3, a) + projection(0, 1) * exact(4, a));
                    const auto expected_left_adjoint = -std::sqrt(4 * PI) *
                        (projection(0, 0) * std::conj(exact(a, 3)) +
                         projection(0, 1) * std::conj(exact(a, 4)));
                    close(projected.get_wing_auxiliary()[iw](0, a), expected_right);
                    close(projected.get_left_wing_auxiliary_adjoint()[iw](0, a),
                          expected_left_adjoint);
                    for (int b = 0; b != 3; ++b)
                        close(projected.get_head_tensors()[iw](a, b),
                              optics.get_head_tensors()[iw](a, b));
                }
            }
#endif
        }
    }
    assert(nonreciprocal_witness > 1e-4);
    assert(nonhermitian_head_witness > 1e-4);
#ifndef LIBRPA_USE_LIBRI
    if (global::mpi_comm_global_h.is_root())
        std::cout << "Native optical wing contraction skipped: no-LibRI build; "
                     "head, Lehmann pair, inverse and ST controls remain active."
                  << std::endl;
#endif
}

void fill(Matz &local, const ArrayDesc &desc, const Matz &global)
{
    for (int i = 0; i < desc.m_loc(); ++i)
        for (int j = 0; j < desc.n_loc(); ++j)
            local(i, j) = global(desc.indx_l2g_r(i), desc.indx_l2g_c(j));
}

void test_general_optical_inverse(const BlacsCtxtHandler &blacs)
{
    constexpr int n = 4, active = 3;
    Matz u(n, n, MAJOR::COL), ec(n, n, MAJOR::COL), right(n, 3, MAJOR::COL), left(n, 3, MAJOR::COL);
    auto head = identity(3);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            u(i, j) = .5 * std::polar(1.0, .5 * PI * i * j);
            ec(i, j) = Z(i == j ? 1.4 + .2 * i : .03 * (i + 1), .02 * (2 * i - j));
        }
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
        {
            right(i, a) = Z(.025 * (i + a + 1), .014 * (i - a));
            left(i, a) = Z(.017 * (i + 2 * a + 1), -.019 * (i + a));
        }
    head(0, 0) = 1.7;
    head(1, 1) = 1.5;
    head(2, 2) = 1.3;
    head(0, 1) = .06;
    head(1, 0) = -.03;
    std::vector<std::array<double, 3>> directions;
    for (const auto &x :
         {std::array<double, 3>{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 2, 1}, {2, -1, 1}})
    {
        const double norm = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        for (double sign : {1., -1.})
            directions.push_back({sign * x[0] / norm, sign * x[1] / norm, sign * x[2] / norm});
    }
    Matz expected_c(n, n, MAJOR::COL);
    for (const auto &q : directions)
    {
        Matz directional(active, active, MAJOR::COL);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) directional(0, 0) += q[a] * head(a, b) * q[b];
        for (int i = 1; i < active; ++i)
        {
            for (int a = 0; a < 3; ++a)
            {
                directional(i, 0) += right(i, a) * q[a];
                directional(0, i) += std::conj(left(i, a)) * q[a];
            }
            for (int j = 1; j < active; ++j) directional(i, j) = ec(i, j);
        }
        const auto direct = inverse(directional);
        for (int i = 0; i < active; ++i)
            for (int j = 0; j < active; ++j)
                expected_c(i, j) += direct(i, j) / double(directions.size());
    }
    const auto expected = u * expected_c * u.get_transpose(true);
    const auto e = u * ec * u.get_transpose(true), r = u * right, l = u * left;
    for (int block : {1, 2})
    {
        ArrayDesc desc(blacs);
        desc.init(n, n, block, block, 0, 0);
        ArrayDesc dh(blacs);
        dh.init(n, 1, block, block, 0, 0);
        auto ed = init_local_mat<Z>(desc, MAJOR::COL), ud = init_local_mat<Z>(desc, MAJOR::COL);
        auto vd = init_local_mat<Z>(desc, MAJOR::COL), hd = init_local_mat<Z>(dh, MAJOR::COL);
        fill(ed, desc, e);
        fill(ud, desc, u);
        fill(vd, desc, identity(n));
        for (int i = 0; i < dh.m_loc(); ++i)
            for (int j = 0; j < dh.n_loc(); ++j) hd(i, j) = u(dh.indx_l2g_r(i), 0);
        std::vector<double> qx, qy, qz, weights;
        for (const auto &q : directions)
        {
            qx.push_back(q[0]);
            qy.push_back(q[1]);
            qz.push_back(q[2]);
            weights.push_back(1. / directions.size());
        }
        rewrite_eps_abf_space(ed, vd, ud, hd, dh, head, r, l, qx, qy, qz, weights, desc, blacs,
                              active, 0., false, false);
        for (int i = 0; i < desc.m_loc(); ++i)
            for (int j = 0; j < desc.n_loc(); ++j)
                close(ed(i, j), expected(desc.indx_l2g_r(i), desc.indx_l2g_c(j)));
    }
}

void test_sternheimer_two_wings()
{
    ComplexMatrix coulomb(3, 3), response_m(3, 3);
    const std::array<double, 3> sqrtv{3, 2, 1};
    Matz body(2, 2, MAJOR::COL);
    body(0, 0) = -.3;
    body(1, 1) = -.2;
    body(0, 1) = Z(.05, .02);
    body(1, 0) = Z(-.03, .01);
    for (int i = 0; i < 3; ++i) coulomb(i, i) = sqrtv[i] * sqrtv[i];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            response_m(i + 1, j + 1) = sqrtv[i + 1] * sqrtv[j + 1] * body(i, j);
    SternheimerRpaHeadwingInput input;
    input.head = ComplexMatrix(3, 3);
    input.wing_mu = ComplexMatrix(3, 3);
    input.wing_mu_left_adjoint = ComplexMatrix(3, 3);
    for (int a = 0; a < 3; ++a)
    {
        input.head(a, a) = -.4 - .1 * a;
        for (int i = 1; i < 3; ++i)
        {
            input.wing_mu(i, a) = Z(.03 * (i + a), .01 * (i - a)) / sqrtv[i];
            input.wing_mu_left_adjoint(i, a) = Z(-.02 * (i + 2 * a), .015 * (i + a)) / sqrtv[i];
        }
        std::array<double, 3> q{};
        q[a] = 1;
        input.directions.push_back({q, 1. / 3., 0});
    }
    Z expected = 0;
    for (int a = 0; a < 3; ++a)
    {
        auto e = identity(3);
        Z trace = input.head(a, a);
        e(0, 0) -= input.head(a, a);
        for (int i = 1; i < 3; ++i)
        {
            e(i, 0) = -sqrtv[i] * input.wing_mu(i, a);
            e(0, i) = -sqrtv[i] * std::conj(input.wing_mu_left_adjoint(i, a));
            trace += body(i - 1, i - 1);
            for (int j = 1; j < 3; ++j) e(i, j) -= body(i - 1, j - 1);
        }
        expected += (trace + std::log(determinant3(e))) / 3.;
    }
    const auto actual = compute_sternheimer_rpa_frequency_headwing(coulomb, response_m, input, 0,
                                                                   .7, 1., 1., 1e-12);
    close(actual.integrand, expected);
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    {
        BlacsCtxtHandler blacs(MPI_COMM_WORLD);
        blacs.init();
        blacs.set_square_grid();
        test_native_head_wings_and_joint_permutation(blacs);
#ifdef LIBRPA_USE_LIBRI
        test_complete_optical_wc_has_zero_vacuum();
#endif
        test_general_optical_inverse(blacs);
        test_sternheimer_two_wings();
    }
    if (global::mpi_comm_global_h.is_root())
        std::cout << "GENERAL_OPTICAL_LEHMANN_PASS" << std::endl;
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
}
