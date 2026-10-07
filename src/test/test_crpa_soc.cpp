#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../api/dataset.h"
#include "../core/crpa.h"
#include "../core/crpa_window_input.h"
#include "../core/dielecmodel.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;

namespace
{
using Z = std::complex<double>;

void close(Z actual, Z expected, double tolerance = 2e-8)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > tolerance)
    {
        std::cerr << "SOC actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("spinor cRPA numerical comparison failed");
    }
}

struct Fixture
{
    Dataset ds{MPI_COMM_WORLD};
    std::unique_ptr<Chi0> chi;
    std::unique_ptr<CorrelatedSubspace> frame;
    std::vector<atpair_t> pairs;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    std::array<ComplexMatrix, 2> density{ComplexMatrix(3, 3), ComplexMatrix(3, 3)};
    ComplexMatrix coefficients, coulomb{2, 2};
    std::vector<ComplexMatrix> coefficients_k;
    int components, bands;

    Fixture(bool spinor, double soc_angle, double representation_angle = 0,
            bool dispersive_phase = false)
        : coefficients(spinor ? 6 : 3, spinor ? 6 : 3),
          components(spinor ? 2 : 1), bands(spinor ? 6 : 3)
    {
        ds.basis_wfc.set(std::vector<std::size_t>{3});
        ds.basis_wfc.set_l_shells({{0, 0, 0}});
        ds.basis_aux.set(std::vector<std::size_t>{2});
        ds.mf.set(1, 4, bands, 3, components);
        ds.mf.get_efermi() = 0;
        ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        const double pi = std::acos(-1.0);
        ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true),
                                MPI_COMM_WORLD, 4);
        ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(bands, 3, bands, 3);
        ds.desc_abf.reset_handler(ds.blacs_h);
        ds.desc_abf.init_1b1p(2, 2, 0, 0);
        ds.tfg.reset(24);
        ds.tfg.generate_minimax(0.4, 4.0);
        const double f0[] = {0.4, 0.3, 0.4, 0.3, 0.7, 0.2, 0.4, 0.2, 0.9};
        const double f1[] = {0.9, 0.13, -0.17, 0.13, 0.1, 0.07, -0.17, 0.07, 0.4};
        for (int i = 0; i < 9; ++i)
        {
            density[0].c[i] = f0[i];
            density[1].c[i] = f1[i];
        }
        coulomb(0, 0) = 2;
        coulomb(1, 1) = 1.3;
        coulomb(0, 1) = coulomb(1, 0) = 0.2;

        if (!spinor) coefficients.set_as_identity_matrix();
        else
        {
            // Kramers pairs (0,1), (2,3), (4,5), with complex inter-orbital SOC.
            const double c = std::cos(soc_angle), s = std::sin(soc_angle);
            coefficients(0, 0) = coefficients(3, 1) = c;
            coefficients(4, 0) = coefficients(1, 1) = Z(0, s);
            coefficients(1, 2) = coefficients(4, 3) = c;
            coefficients(3, 2) = coefficients(0, 3) = Z(0, s);
            coefficients(2, 4) = coefficients(5, 5) = 1;
            // Change the spin representation of both KS and output columns.
            const auto original = coefficients;
            const double u = std::cos(representation_angle), v = std::sin(representation_angle);
            for (int i = 0; i < 3; ++i)
                for (int n = 0; n < bands; ++n)
                {
                    coefficients(i, n) = u * original(i, n) + Z(0, v) * original(3 + i, n);
                    coefficients(3 + i, n) = Z(0, v) * original(i, n) + u * original(3 + i, n);
                }
        }
        const double energies[] = {-0.7, 0.3, 1.1};
        SiteOrbitalGroup site{"spinor-test", 0, bands, 0, {}};
        frame = std::make_unique<CorrelatedSubspace>(
            std::vector<SiteOrbitalGroup>{site}, ds.pbc.kfrac_list, ds.pbc.Rlist,
            3 * components, bands, 1);
        ComplexMatrix overlap(3 * components, 3 * components);
        overlap.set_as_identity_matrix();
        for (int k = 0; k < 4; ++k)
        {
            auto ck = coefficients;
            if (dispersive_phase)
                for (int s = 0; s < components; ++s)
                    for (int n = 0; n < bands; ++n)
                        ck(3 * s + 1, n) *= std::exp(Z(0, pi * k / 2));
            coefficients_k.push_back(ck);
            for (int s = 0; s < components; ++s)
            {
                ComplexMatrix wfc(bands, 3);
                for (int n = 0; n < bands; ++n)
                    for (int i = 0; i < 3; ++i) wfc(n, i) = ck(3 * s + i, n);
                ds.mf.get_eigenvectors()[0][s][k] = wfc;
            }
            for (int n = 0; n < bands; ++n)
            {
                ds.mf.get_eigenvals()[0](k, n) = energies[n / components];
                ds.mf.get_weight()[0](k, n) = (n < components ? 1.0 : 0.0) *
                                              (spinor ? 0.25 : 0.5);
            }
            frame->set_S_k(k, overlap);
            frame->set_W_k(k, coefficients);
            frame->build_spin_k(0, k, transpose(coefficients, false));
            if (ds.comm_h.is_root())
                ds.vq[0][0][ds.pbc.klist[k]] = std::make_shared<ComplexMatrix>(coulomb);
        }
        frame->compute_T_R();
        if (ds.comm_h.is_root())
        {
            ds.cs_data.use_libri = true;
            auto values = std::make_shared<std::valarray<double>>(0.0, 18);
            for (int mu = 0; mu < 2; ++mu)
                for (int i = 0; i < 9; ++i) (*values)[9 * mu + i] = 0.5 * density[mu].c[i].real();
            ds.cs_data.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 3, 3}, values);
        }
        const auto assigned = dispatch_upper_triangular_tasks(
            1, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
            ds.blacs_h.myprow, ds.blacs_h.mypcol);
        pairs.assign(assigned.begin(), assigned.end());
        chi = std::make_unique<Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc,
                                     ds.symmetry_context, ds.tfg, ds.scfk_blacs_ctxt,
                                     ds.desc_wfc_kb_full, false, false);
        chi->gf_threshold = chi->libri_threshold_C = chi->libri_threshold_G = 0;
    }

    Chi0::BandSelection selection() const
    {
        std::vector<unsigned char> mask(bands, 0);
        for (int n = 0; n < 2 * components; ++n) mask[n] = 1;
        return Chi0::BandSelection(1, std::vector<std::vector<unsigned char>>(4, mask));
    }

    Z vertex(int mu, int n, int m) const
    {
        Z result{};
        for (int s = 0; s < components; ++s)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result += std::conj(coefficients(3 * s + i, n)) * density[mu](i, j) *
                              coefficients(3 * s + j, m);
        return result;
    }

    Z transition(int mu, int n, int m, int k, int kq) const
    {
        Z result{};
        for (int s = 0; s < components; ++s)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result += std::conj(coefficients_k[k](3 * s + i, n)) * density[mu](i, j) *
                              coefficients_k[kq](3 * s + j, m);
        return result;
    }

    ComplexMatrix response(double frequency, bool excluded, int iq = 0) const
    {
        ComplexMatrix result(2, 2);
        for (int k = 0; k < 4; ++k)
        for (int n = 0; n < bands; ++n)
            for (int m = 0; m < bands; ++m)
            {
                if (excluded && (n >= 2 * components || m >= 2 * components)) continue;
                const double occupation_difference = (n < components ? 1.0 : 0.0) -
                                                     (m < components ? 1.0 : 0.0);
                if (occupation_difference == 0) continue;
                const int kq = (k + iq) % 4;
                const double gap = ds.mf.get_eigenvals()[0](kq, m) - ds.mf.get_eigenvals()[0](k, n);
                const Z factor = (components == 1 ? 2.0 : 1.0) * occupation_difference /
                                 (4.0 * (Z(0, frequency) - gap));
                for (int mu = 0; mu < 2; ++mu)
                    for (int nu = 0; nu < 2; ++nu)
                        result(mu, nu) += factor * transition(mu, n, m, k, kq) *
                                        std::conj(transition(nu, n, m, k, kq));
            }
        return result;
    }

    void compare_response(bool excluded)
    {
        if (excluded) chi->set_band_selection(selection());
        else chi->clear_band_selection();
        chi->build(LIBRPA_ROUTING_LIBRI, ds.cs_data, pairs, ds.basis_aux, no_shrink, ds.blacs_h);
        for (double frequency : ds.tfg.get_freq_nodes())
            for (int iq = 0; iq < 4; ++iq)
            {
                const auto& q = ds.pbc.klist[iq];
                ComplexMatrix local(2, 2), total(2, 2);
                const auto f = chi->get_chi0_q().find(frequency);
                if (f != chi->get_chi0_q().end())
                {
                    const auto it = f->second.find(q);
                    if (it != f->second.end() && it->second.count(0) && it->second.at(0).count(0))
                        local = it->second.at(0).at(0);
                }
                MPI_Allreduce(local.c, total.c, 4, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
                const auto exact = response(frequency, excluded, iq);
                for (int i = 0; i < 4; ++i) close(total.c[i], exact.c[i]);
            }
        chi->clear_band_selection();
    }

    ComplexMatrix screen(const ComplexMatrix& response) const
    {
        ComplexMatrix dielectric(2, 2);
        dielectric.set_as_identity_matrix();
        dielectric -= coulomb * response;
        const Z determinant = dielectric(0, 0) * dielectric(1, 1) -
                              dielectric(0, 1) * dielectric(1, 0);
        ComplexMatrix inverse(2, 2);
        inverse(0, 0) = dielectric(1, 1) / determinant;
        inverse(1, 1) = dielectric(0, 0) / determinant;
        inverse(0, 1) = -dielectric(0, 1) / determinant;
        inverse(1, 0) = -dielectric(1, 0) / determinant;
        return inverse * coulomb;
    }

    CrpaResult run()
    {
        CrpaContext context{*chi, ds.cs_data, ds.basis_aux, pairs, no_shrink,
                            ds.vq, ds.blacs_h, ds.desc_abf};
        return compute_crpa_onsite(context, selection(), {frame.get()});
    }

    void compare_tensors(const CrpaResult& result) const
    {
        for (std::size_t iw = 0; iw < result.frequencies.size(); ++iw)
        {
            const double frequency = result.frequencies[iw];
            std::array<ComplexMatrix, 3> kernels{coulomb, ComplexMatrix(2, 2), ComplexMatrix(2, 2)};
            for (int iq = 0; iq < 4; ++iq)
            {
                const auto p0 = response(frequency, false, iq), pd = response(frequency, true, iq);
                kernels[1] += 0.25 * screen(p0 - pd);
                kernels[2] += 0.25 * screen(p0);
            }
            const auto& tensor = result.tensors.at(0);
            const std::array<const ComplexMatrix*, 3> actual{&tensor.bare, &tensor.u[iw], &tensor.w[iw]};
            for (int a = 0; a < bands; ++a)
                for (int b = 0; b < bands; ++b)
                    for (int c = 0; c < bands; ++c)
                        for (int d = 0; d < bands; ++d)
                            for (int family = 0; family < 3; ++family)
                            {
                                Z expected{};
                                for (int mu = 0; mu < 2; ++mu)
                                    for (int nu = 0; nu < 2; ++nu)
                                        expected += std::conj(vertex(mu, b, a)) * kernels[family](mu, nu) *
                                                    vertex(nu, c, d);
                                close((*actual[family])(a * bands + b, c * bands + d), expected);
                            }
        }
    }
};

void test_complex_soc_response()
{
    Fixture scalar(false, 0), spinor(true, 0.37), rotated(true, 0.37, 0.61);
    Fixture periodic(true, 0.37, 0, true);
    scalar.compare_response(false);
    spinor.compare_response(false);
    rotated.compare_response(false);
    periodic.compare_response(false);
}
}  // namespace

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    try
    {
        test_complex_soc_response();
    }
    catch (const std::exception &failure)
    {
        std::cerr << failure.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_crpa_soc: passed\n";
}
