// Exact finite-model Lindhard input -> native RPA body energy. The oracle
// evaluates 1x1/3x3 determinants explicitly, without the native LU or collector.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "../api/dataset.h"
#include "../core/epsilon.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "librpa.hpp"

using namespace librpa_int;
using Z = std::complex<double>;

namespace
{
struct Spectrum
{
    std::vector<std::vector<double>> energies;
    std::vector<Matz> vectors;
    Spectrum(const int nk, const double bz)
    {
        for (int k = 0; k != nk; ++k)
        {
            Matz h(6, 6, MAJOR::COL);
            h.zero_out();
            const double phase = TWO_PI * k / nk;
            for (int a = 0; a != 3; ++a)
            {
                h(a, a) = bz;
                h(a + 3, a + 3) = -bz;
                h(a, a + 3) = std::polar(1., -TWO_PI * a / 3.);
                h(a + 3, a) = std::conj(h(a, a + 3));
                for (int s = 0; s != 2; ++s)
                {
                    const int b = (a + 1) % 3;
                    const Z hop{.15 + (nk > 1 ? .025 * std::cos(phase) : 0.), s ? -.04 : .04};
                    h(a + 3 * s, b + 3 * s) = hop;
                    h(b + 3 * s, a + 3 * s) = std::conj(hop);
                    if (nk > 1) h(a + 3 * s, a + 3 * s) += .04 * (a - 1) * std::sin(phase);
                }
            }
            std::vector<double> e;
            Matz c;
            eigsh(h, e, c);
            assert(e[3] - e[2] > 1.);
            energies.push_back(std::move(e));
            vectors.push_back(std::move(c));
        }
    }

    Matz chi(const int iq, const double omega) const
    {
        const int nk = energies.size();
        Matz result(3, 3, MAJOR::COL);
        result.zero_out();
        for (int sign : {1, -1})
            for (int k = 0; k != nk; ++k)
            {
                const int target = (k + sign * iq + nk) % nk;
                for (int m = 0; m != 3; ++m)
                    for (int n = 3; n != 6; ++n)
                    {
                        Z d[3]{};
                        for (int a = 0; a != 3; ++a)
                            for (int s = 0; s != 2; ++s)
                                d[a] += std::conj(vectors[k](a + 3 * s, m)) *
                                        vectors[target](a + 3 * s, n);
                        const double delta = energies[target][n] - energies[k][m];
                        for (int a = 0; a != 3; ++a)
                            for (int b = 0; b != 3; ++b)
                            {
                                const Z amplitude = d[a] * std::conj(d[b]);
                                result(a, b) -= (sign == 1 ? amplitude : std::conj(amplitude)) /
                                                (double(nk) * Z(delta, -sign * omega));
                            }
                    }
            }
        return result;
    }
};

Z determinant(const Matz &a)
{
    if (a.nr() == 1) return a(0, 0);
    assert(a.nr() == 3 && a.nc() == 3);
    return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
           a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
           a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
}

Z integrand(const Matz &v, const Matz &chi)
{
    const auto response = v * chi;
    auto dielectric = response.copy();
    dielectric *= -1.;
    Z trace{};
    for (int i = 0; i != v.nr(); ++i)
    {
        trace += response(i, i);
        dielectric(i, i) += 1.;
    }
    return trace + std::log(determinant(dielectric));
}

void run_case(const std::vector<size_t> &groups, const int nk, const double bz,
              const bool complex_gauge, const bool folded, const bool scalar)
{
    const int naux = std::accumulate(groups.begin(), groups.end(), size_t(0));
    assert(naux == 1 || naux == 3);
    assert(!scalar || (nk == 1 && bz == 0. && !complex_gauge));
    assert(!folded || (nk == 3 && naux == 3));
    const Spectrum spectrum(nk, bz);
    Dataset ds(MPI_COMM_WORLD);
    ds.mf.set(1, nk, scalar ? 3 : 6, 3, scalar ? 1 : 2);
    ds.basis_wfc.set(naux == 3 ? groups : std::vector<size_t>{3});
    ds.basis_aux.set(groups);
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    std::vector<double> kvecs(3 * nk, 0.);
    for (int k = 0; k != nk; ++k) kvecs[3 * k] = TWO_PI * k / nk;
    ds.pbc.set_kgrids_kvec(nk, 1, 1, kvecs);
    if (folded)
    {
        // Charge reality pairs q and -q. Full three-dimensional auxiliary
        // gauge rotations preserve their equal real trace/log integrands.
        const auto q0 = ds.pbc.klist[0], q1 = ds.pbc.klist[1], q2 = ds.pbc.klist[2];
        ds.pbc.klist_coul = {q0, q1};
        ds.pbc.map_irk_ks = {{q0, {q0}}, {q1, {q1, q2}}};
        ds.pbc.map_q_weight = {{q0, 1. / 3}, {q1, 2. / 3}};
    }
    ds.tfg.reset(16);
    ds.tfg.generate_GaussLegendre();
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, nk);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(3, 6, 3, 6);
    Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
             ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);

    Matz u(naux, 3, MAJOR::COL);
    u.zero_out();
    if (naux == 1)
        u(0, 0) = 1.;
    else
        for (int a = 0; a != 3; ++a)
            for (int b = 0; b != 3; ++b)
                u(a, b) = complex_gauge
                    ? std::polar(1. / std::sqrt(3.), TWO_PI * a * b / 3. + .31 * b)
                    : Z(a == b ? 1. : 0.);
    Matz v0(3, 3, MAJOR::COL);
    const double values[3][3] = {{1.2, .13, .04}, {.13, .8, -.07}, {.04, -.07, .65}};
    for (int a = 0; a != 3; ++a)
        for (int b = 0; b != 3; ++b) v0(a, b) = values[a][b];
    const auto v = u * v0 * transpose(u, true);
    const auto &offset = ds.basis_aux.get_part_range();
    atpair_k_cplx_mat_t coulomb;
    std::vector<atpair_t> local_pairs;
    for (int i = 0; i != int(groups.size()); ++i)
        for (int j = i; j != int(groups.size()); ++j)
            if ((i * groups.size() + j) % ds.comm_h.nprocs == size_t(ds.comm_h.myid))
            {
                local_pairs.emplace_back(i, j);
                for (const auto &q : chi.active_qpoints())
                {
                    auto block = std::make_shared<ComplexMatrix>(groups[i], groups[j]);
                    for (int a = 0; a != block->nr; ++a)
                        for (int b = 0; b != block->nc; ++b)
                            (*block)(a, b) = v(offset[i] + a, offset[j] + b);
                    coulomb[i][j][q] = std::move(block);
                }
            }
    auto response = [&]() {
        Chi0QMap result;
        for (double frequency : ds.tfg.get_freq_nodes())
            for (const auto &q : chi.active_qpoints())
            {
                const int iq = ds.pbc.get_k_index_full(q);
                const auto exact = u * spectrum.chi(iq, frequency) * transpose(u, true);
                for (int i = 0; i != int(groups.size()); ++i)
                    for (int j = (scalar ? i : 0); j != int(groups.size()); ++j)
                        if ((std::min(i, j) * groups.size() + std::max(i, j)) % ds.comm_h.nprocs ==
                            size_t(ds.comm_h.myid))
                        {
                            ComplexMatrix block(groups[i], groups[j]);
                            for (int a = 0; a != block.nr; ++a)
                                for (int b = 0; b != block.nc; ++b)
                                    block(a, b) = exact(offset[i] + a, offset[j] + b);
                            result[frequency][q][i][j] = std::move(block);
                        }
            }
        return result;
    };
    std::map<Vector3_Order<double>, double> expected_q;
    double expected = 0., wrong_hermitian = 0., imaginary_integrand = 0.;
    for (double frequency : ds.tfg.get_freq_nodes())
        for (int iq = 0; iq != nk; ++iq)
        {
            const auto exact = u * spectrum.chi(iq, frequency) * transpose(u, true);
            const auto value = integrand(v, exact);
            imaginary_integrand = std::max(imaginary_integrand, std::abs(value.imag()));
            const double weight = ds.tfg.find_freq_weight(frequency) / (nk * TWO_PI);
            expected += weight * value.real();
            wrong_hermitian += weight * integrand(v, (exact + transpose(exact, true)) * .5).real();
        }
    for (double frequency : ds.tfg.get_freq_nodes())
        for (const auto &q : chi.active_qpoints())
        {
            const auto exact = u * spectrum.chi(ds.pbc.get_k_index_full(q), frequency) * transpose(u, true);
            expected_q[q] += ds.tfg.find_freq_weight(frequency) * chi.q_weight(q) *
                             integrand(v, exact).real() / TWO_PI;
        }
    auto input = response();
    chi.swap_chi0_q(input);
    const auto actual = compute_RPA_correlation_blacs_2d(chi, coulomb, local_pairs, ds.blacs_h);
    double error = 0., gamma_error = 0.;
    if (ds.comm_h.is_root())
    {
        error = std::abs(actual.value.real() - expected);
        for (const auto &[q, expected_value] : expected_q)
            error = std::max(error, std::abs(actual.qcontrib.at(q).real() - expected_value));
        if (!scalar) error = std::max(error, std::abs(actual.value.imag()));
    }
    if (nk == 1)
    {
        input = response();
        chi.swap_chi0_q(input);
        const auto gamma = compute_RPA_correlation_blacs_2d_gamma_only(chi, coulomb, local_pairs, ds.blacs_h);
        if (ds.comm_h.is_root()) gamma_error = std::abs(gamma.value.real() - expected);
    }
    ds.comm_h.bcast(&error, 1, 0);
    ds.comm_h.bcast(&gamma_error, 1, 0);
    assert(error < 3e-12 && gamma_error < 3e-12);
    if (bz != 0. && nk == 1 && naux == 3) assert(std::abs(expected - wrong_hermitian) > 1e-12);
    if (ds.comm_h.is_root())
        std::cout << std::setprecision(17) << "RPA_SIGNED groups=" << groups.size()
                  << " naux=" << naux << " nk=" << nk << " bz=" << bz
                  << " gauge=" << complex_gauge << " folded=" << folded << " scalar=" << scalar
                  << " energy=" << expected << " error=" << error << " gamma_error=" << gamma_error
                  << " hermitianization_error=" << std::abs(expected - wrong_hermitian)
                  << " imaginary_integrand=" << imaginary_integrand << std::endl;
    ds.free();
}
} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    librpa::init_global(LIBRPA_SWITCH_OFF, "stdout", LIBRPA_SWITCH_OFF);
    for (const auto &groups : {std::vector<size_t>{3}, std::vector<size_t>{1, 2}, std::vector<size_t>{1, 1, 1}})
        for (int nk : {1, 3})
            for (double bz : {0., .21})
                for (bool gauge : {false, true}) run_case(groups, nk, bz, gauge, false, false);
    for (double bz : {0., .21})
        for (bool gauge : {false, true}) run_case({1, 1, 1}, 3, bz, gauge, true, false);
    for (double bz : {0., .21}) run_case({1}, 1, bz, false, false, false);
    for (const auto &groups : {std::vector<size_t>{3}, std::vector<size_t>{1}})
        run_case(groups, 1, 0., false, false, true);
    librpa::finalize_global();
    MPI_Finalize();
}
