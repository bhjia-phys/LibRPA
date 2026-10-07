#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../api/dataset.h"
#include "../api/dataset_helper.h"
#include "../core/crpa.h"
#include "../core/crpa_window_input.h"
#include "../core/dielecmodel.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"

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
          components(spinor ? 2 : 1),
          bands(spinor ? 6 : 3)
    {
        ds.basis_wfc.set(std::vector<std::size_t>{3});
        ds.basis_wfc.set_l_shells({{0, 0, 0}});
        ds.basis_aux.set(std::vector<std::size_t>{2});
        ds.mf.set(1, 4, bands, 3, components);
        ds.mf.get_efermi() = 0;
        ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        const double pi = std::acos(-1.0);
        ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD,
                                4);
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

        if (!spinor)
            coefficients.set_as_identity_matrix();
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
        frame = std::make_unique<CorrelatedSubspace>(std::vector<SiteOrbitalGroup>{site},
                                                     ds.pbc.kfrac_list, ds.pbc.Rlist,
                                                     3 * components, bands, 1);
        ComplexMatrix overlap(3 * components, 3 * components);
        overlap.set_as_identity_matrix();
        for (int k = 0; k < 4; ++k)
        {
            auto ck = coefficients;
            if (dispersive_phase)
                for (int s = 0; s < components; ++s)
                    for (int n = 0; n < bands; ++n) ck(3 * s + 1, n) *= std::exp(Z(0, pi * k / 2));
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
                ds.mf.get_weight()[0](k, n) = (n < components ? 1.0 : 0.0) * (spinor ? 0.25 : 0.5);
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
        const auto assigned = dispatch_upper_triangular_tasks(1, ds.blacs_h.myid, ds.blacs_h.nprows,
                                                              ds.blacs_h.npcols, ds.blacs_h.myprow,
                                                              ds.blacs_h.mypcol);
        pairs.assign(assigned.begin(), assigned.end());
        chi = std::make_unique<Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                                     ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
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
                    const double occupation_difference =
                        (n < components ? 1.0 : 0.0) - (m < components ? 1.0 : 0.0);
                    if (occupation_difference == 0) continue;
                    const int kq = (k + iq) % 4;
                    const double gap =
                        ds.mf.get_eigenvals()[0](kq, m) - ds.mf.get_eigenvals()[0](k, n);
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
        if (excluded)
            chi->set_band_selection(selection());
        else
            chi->clear_band_selection();
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
        const Z determinant =
            dielectric(0, 0) * dielectric(1, 1) - dielectric(0, 1) * dielectric(1, 0);
        ComplexMatrix inverse(2, 2);
        inverse(0, 0) = dielectric(1, 1) / determinant;
        inverse(1, 1) = dielectric(0, 0) / determinant;
        inverse(0, 1) = -dielectric(0, 1) / determinant;
        inverse(1, 0) = -dielectric(1, 0) / determinant;
        return inverse * coulomb;
    }

    CrpaResult run()
    {
        CrpaContext context{*chi,      ds.cs_data, ds.basis_aux, pairs,
                            no_shrink, ds.vq,      ds.blacs_h,   ds.desc_abf};
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
            const std::array<const ComplexMatrix*, 3> actual{&tensor.bare, &tensor.u[iw],
                                                             &tensor.w[iw]};
            for (int a = 0; a < bands; ++a)
                for (int b = 0; b < bands; ++b)
                    for (int c = 0; c < bands; ++c)
                        for (int d = 0; d < bands; ++d)
                            for (int family = 0; family < 3; ++family)
                            {
                                Z expected{};
                                for (int mu = 0; mu < 2; ++mu)
                                    for (int nu = 0; nu < 2; ++nu)
                                        expected += std::conj(vertex(mu, b, a)) *
                                                    kernels[family](mu, nu) * vertex(nu, c, d);
                                close((*actual[family])(a * bands + b, c * bands + d), expected);
                            }
        }
    }
};

void test_zero_soc_limit()
{
    Fixture scalar(false, 0), spinor(true, 0);
    scalar.compare_response(false);
    scalar.compare_response(true);
    spinor.compare_response(false);
    spinor.compare_response(true);
    const auto reference = scalar.run(), packed = spinor.run();
    scalar.compare_tensors(reference);
    spinor.compare_tensors(packed);
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
            for (int c = 0; c < 6; ++c)
                for (int d = 0; d < 6; ++d)
                    for (std::size_t iw = 0; iw < reference.frequencies.size(); ++iw)
                    {
                        const bool same_components = a % 2 == b % 2 && c % 2 == d % 2;
                        const int row = (a / 2) * 3 + b / 2, col = (c / 2) * 3 + d / 2;
                        for (int family = 0; family < 3; ++family)
                        {
                            const auto& s = reference.tensors[0];
                            const auto& p = packed.tensors[0];
                            const auto& x =
                                family == 0 ? s.bare : (family == 1 ? s.u[iw] : s.w[iw]);
                            const auto& y =
                                family == 0 ? p.bare : (family == 1 ? p.u[iw] : p.w[iw]);
                            close(y(a * 6 + b, c * 6 + d), same_components ? x(row, col) : Z{});
                        }
                    }
}

struct MagneticModel
{
    Matz hamiltonian{6, 6, MAJOR::COL};
    Matz eigenvectors;
    std::vector<double> eigenvalues;
    std::vector<double> positions;
    std::vector<int> operations;
    std::vector<int> antiunitary{0, 0, 0, 1, 1, 1};
    const double lattice[9]{6.0, 0.0, 0.0, 3.0, 3.0 * std::sqrt(3.0), 0.0, 0.0, 0.0, 8.0};
    const double reciprocal[9]{TWO_PI / 6.0,
                               -TWO_PI / (6.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               TWO_PI / (3.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               0.0,
                               TWO_PI / 8.0};

    MagneticModel()
    {
        hamiltonian.zero_out();
        for (int atom = 0; atom != 3; ++atom)
        {
            const double angle = TWO_PI * atom / 3.0;
            positions.insert(positions.end(), {0.8 * std::cos(angle), 0.8 * std::sin(angle), 0.0});
            hamiltonian(atom, atom + 3) = std::polar(1.0, -angle);
            hamiltonian(atom + 3, atom) = std::conj(hamiltonian(atom, atom + 3));
            for (int spinor = 0; spinor != 2; ++spinor)
            {
                const int next = (atom + 1) % 3;
                const Z hopping{0.15, spinor == 0 ? 0.04 : -0.04};
                hamiltonian(atom + 3 * spinor, next + 3 * spinor) = hopping;
                hamiltonian(next + 3 * spinor, atom + 3 * spinor) = std::conj(hopping);
            }
        }
        eigsh(hamiltonian, eigenvalues, eigenvectors);
        if (eigenvalues[3] - eigenvalues[2] <= 1.0)
            throw std::runtime_error("magnetic model must be gapped");

        // Row fractional C3, C3^2 and their horizontal-mirror partners.
        const int rotations[3][9]{{1, 0, 0, 0, 1, 0, 0, 0, 1},
                                  {-1, 1, 0, -1, 0, 0, 0, 0, 1},
                                  {0, -1, 0, 1, -1, 0, 0, 0, 1}};
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
                for (int index = 0; index != 9; ++index)
                    operations.push_back(index == 8 && mirror ? -1 : rotations[rotation][index]);

        // Independent Cartesian Hamiltonian covariance, including Theta Mz.
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
            {
                Matz action(6, 6, MAJOR::COL);
                action.zero_out();
                const double angle = TWO_PI * rotation / 3.0 + mirror * PI;
                for (int atom = 0; atom != 3; ++atom)
                    for (int spinor = 0; spinor != 2; ++spinor)
                    {
                        const Z u = std::polar(1.0, (spinor == 0 ? -0.5 : 0.5) * angle);
                        const int target_atom = (atom + rotation) % 3;
                        if (mirror)
                            action(target_atom + 3 * (1 - spinor), atom + 3 * spinor) =
                                (spinor == 0 ? -1.0 : 1.0) * std::conj(u);
                        else
                            action(target_atom + 3 * spinor, atom + 3 * spinor) = u;
                    }
                for (int i = 0; i != 6; ++i)
                    for (int j = 0; j != 6; ++j)
                    {
                        Z transformed = 0.0;
                        for (int a = 0; a != 6; ++a)
                            for (int b = 0; b != 6; ++b)
                                transformed +=
                                    action(i, a) *
                                    (mirror ? std::conj(hamiltonian(a, b)) : hamiltonian(a, b)) *
                                    std::conj(action(j, b));
                        close(transformed, hamiltonian(i, j), 1e-12);
                    }
            }
    }
};

void test_magnetic_qstar_screening()
{
    // Complex periodic Kramers state: the antiunitary q member is actually
    // restored, while both selected GF branches retain all original k points.
    Fixture f(true, 0.37, 0, true);
    const auto full = f.run();
    LibrpaOptions options;
    librpa_init_options(&options);
    options.use_symmetry_rpa = LIBRPA_SWITCH_ON;
    int legacy_rejected = 0, all_legacy_rejected = 0;
    try
    {
        initialize_ds_chi0(f.ds, options, true);
    }
    catch (const std::exception&)
    {
        legacy_rejected = 1;
    }
    MPI_Allreduce(&legacy_rejected, &all_legacy_rejected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_legacy_rejected)
        throw std::runtime_error("SOC cRPA accepted an implicit time-reversal declaration");
    f.ds.basis_aux.set_l_shells({{0, 0}});
    auto& ctx = f.ds.symmetry_context;
    ctx.set_crystal_structure(f.ds.pbc.latvec, f.ds.pbc.G, {{0, 0}}, {{0, {0, 0, 0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.build_periodic_mappings(f.ds.pbc, f.ds.pbc.Rlist);
    ctx.set_available();
    const auto implicit_view = build_symmetry_qpoint_view(ctx, f.ds.pbc, true);
    if (implicit_view.restore_mode != SymmetryQPointRestoreMode::NONE ||
        implicit_view.representatives.size() != 4)
        throw std::runtime_error("a legacy identity-only table silently enabled time reversal");
    SymmetrySpinOperation identity;
    identity.spin_source = SymmetrySpinActionSource::DerivedFromSpatialSOC;
    ctx.set_symmetry_spin_operations({identity}, true);
    ctx.build_periodic_mappings(f.ds.pbc, f.ds.pbc.Rlist);
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    ctx.build_rsh_rotations(convention, 0);
    ctx.build_kstar_member_rotations(0);
    ctx.set_available();
    f.chi = std::make_unique<Chi0>(f.ds.mf, f.ds.basis_wfc, f.ds.basis_aux, f.ds.pbc, ctx, f.ds.tfg,
                                   f.ds.scfk_blacs_ctxt, f.ds.desc_wfc_kb_full, false, true);
    f.chi->qstar_only_symmetry = true;
    f.chi->gf_threshold = f.chi->libri_threshold_C = f.chi->libri_threshold_G = 0;
    if (f.chi->active_qpoints().size() != 3)
        throw std::runtime_error("grey-group cRPA did not reduce four q points to three");
    bool has_antiunitary = false;
    for (const auto& star : ctx.kstars)
        for (const auto& member : star.members) has_antiunitary |= member.time_reversal;
    if (!has_antiunitary) throw std::runtime_error("antiunitary cRPA member was not exercised");

    // A deliberately star-dependent KS selector must fail on every MPI rank,
    // without silently replacing the original transition model.
    auto invalid = f.selection();
    invalid[0][3][0] ^= 1;
    int rejected = 0, all_rejected = 0;
    try
    {
        f.chi->set_band_selection(invalid);
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    MPI_Allreduce(&rejected, &all_rejected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_rejected)
        throw std::runtime_error("noncovariant cRPA selector was not collectively rejected");

    for (bool excluded : {false, true})
    {
        if (excluded)
            f.chi->set_band_selection(f.selection());
        else
            f.chi->clear_band_selection();
        f.chi->build(LIBRPA_ROUTING_LIBRI, f.ds.cs_data, f.pairs, f.ds.basis_aux, f.no_shrink,
                     f.ds.blacs_h);
        for (double frequency : f.ds.tfg.get_freq_nodes())
            for (const auto& q : f.chi->active_qpoints())
            {
                const int iq =
                    std::distance(f.ds.pbc.klist.begin(),
                                  std::find(f.ds.pbc.klist.begin(), f.ds.pbc.klist.end(), q));
                ComplexMatrix local(2, 2), total(2, 2);
                const auto& response = f.chi->get_chi0_q();
                const auto fi = response.find(frequency);
                if (fi != response.end())
                {
                    const auto qi = fi->second.find(q);
                    if (qi != fi->second.end() && qi->second.count(0) && qi->second.at(0).count(0))
                        local = qi->second.at(0).at(0);
                }
                MPI_Allreduce(local.c, total.c, total.size, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                              MPI_COMM_WORLD);
                const auto expected = f.response(frequency, excluded, iq);
                for (int i = 0; i < total.size; ++i) close(total.c[i], expected.c[i]);
            }
    }
    const auto reduced = f.run();
    double maximum = 0;
    for (std::size_t node = 0; node < full.frequencies.size(); ++node)
        for (const auto& pair : {std::pair{&full.tensors[0].u[node], &reduced.tensors[0].u[node]},
                                 std::pair{&full.tensors[0].w[node], &reduced.tensors[0].w[node]}})
            for (int i = 0; i < pair.first->size; ++i)
            {
                maximum = std::max(maximum, std::abs(pair.first->c[i] - pair.second->c[i]));
                close(pair.first->c[i], pair.second->c[i], 2e-11);
            }
    if (f.ds.comm_h.is_root())
        std::cout << "SOC magnetic q-star: 4 -> 3 q, antiunitary restore, all tensor max="
                  << maximum << '\n';
}

void test_noncollinear_magnetic_crpa()
{
    MagneticModel model;
    Dataset ds(MPI_COMM_WORLD);
    ds.basis_wfc.set(std::vector<std::size_t>{1, 1, 1});
    ds.basis_wfc.set_l_shells({{0}, {0}, {0}});
    ds.basis_aux.set(std::vector<std::size_t>{1, 1, 1});
    ds.basis_aux.set_l_shells({{0}, {0}, {0}});
    ds.mf.set(1, 9, 6, 3, 2);
    ds.mf.get_efermi() = (model.eigenvalues[2] + model.eigenvalues[3]) * 0.5;
    ds.pbc.set_latvec_and_G(std::vector<double>(model.lattice, model.lattice + 9),
                            std::vector<double>(model.reciprocal, model.reciprocal + 9));
    std::vector<double> kvec;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
        {
            const Vector3_Order<double> cart{Vector3_Order<double>{x / 3.0, y / 3.0, 0} * ds.pbc.G};
            // The public BvK input includes 2pi, whereas internal pbc.G does not.
            kvec.insert(kvec.end(), {TWO_PI * cart.x, TWO_PI * cart.y, TWO_PI * cart.z});
        }
    ds.pbc.set_kgrids_kvec(3, 3, 1, kvec);
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 9);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds.desc_abf.reset_handler(ds.blacs_h);
    ds.desc_abf.init_1b1p(3, 3, 0, 0);
    ds.tfg.reset(24);
    ds.tfg.generate_minimax(0.4, 4.0);
    std::vector<SiteOrbitalGroup> sites;
    ComplexMatrix trial(6, 6), overlap(6, 6);
    overlap.set_as_identity_matrix();
    for (int a = 0; a < 3; ++a)
    {
        sites.push_back({"magnetic-site-" + std::to_string(a), 2 * a, 2, a, {}});
        trial(a, 2 * a) = trial(a + 3, 2 * a + 1) = 1;
        ds.atoms.types[a] = 0;
        const Vector3_Order<double> cart{model.positions[3 * a], model.positions[3 * a + 1], 0};
        const Vector3_Order<double> frac{cart * ds.pbc.latvec.Inverse()};
        ds.atoms.coords_frac[a] = Vector3<double>{frac.x, frac.y, frac.z};
    }
    CorrelatedSubspace frame(sites, ds.pbc.kfrac_list, ds.pbc.Rlist, 6, 6, 1);
    for (int k = 0; k < 9; ++k)
    {
        for (int spin = 0; spin < 2; ++spin)
        {
            ComplexMatrix c(6, 3);
            for (int band = 0; band < 6; ++band)
                for (int a = 0; a < 3; ++a) c(band, a) = model.eigenvectors(a + 3 * spin, band);
            ds.mf.get_eigenvectors()[0][spin][k] = c;
        }
        for (int band = 0; band < 6; ++band)
        {
            ds.mf.get_eigenvals()[0](k, band) = model.eigenvalues[band];
            ds.mf.get_weight()[0](k, band) = band < 3 ? 1.0 / 9 : 0;
        }
        frame.set_S_k(k, overlap);
        frame.set_W_k(k, trial);
        if (ds.comm_h.is_root())
            for (int a = 0; a < 3; ++a)
                for (int b = a; b < 3; ++b)
                {
                    auto block = std::make_shared<ComplexMatrix>(1, 1);
                    (*block)(0, 0) = a == b ? 1.2 : 0.08;
                    ds.vq[a][b][ds.pbc.klist[k]] = block;
                }
    }
    frame.compute_T_R();
    if (ds.comm_h.is_root())
    {
        ds.cs_data.use_libri = true;
        for (int a = 0; a < 3; ++a)
        {
            auto value = std::make_shared<std::valarray<double>>(0.5, 1);
            ds.cs_data.data_libri[a][{a, {0, 0, 0}}] = RI::Tensor<double>({1, 1, 1}, value);
        }
    }
    const auto assigned =
        dispatch_upper_triangular_tasks(3, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
                                        ds.blacs_h.myprow, ds.blacs_h.mypcol);
    std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    std::map<Vector3_Order<double>, ComplexMatrix> transform;
    Chi0::BandSelection selection(1,
                                  std::vector<std::vector<unsigned char>>(9, {1, 1, 1, 1, 0, 0}));
    const auto calculate = [&](bool symmetry)
    {
        Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
                 ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, symmetry);
        chi.qstar_only_symmetry = true;
        chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
        CrpaContext context{chi,       ds.cs_data, ds.basis_aux, pairs,
                            transform, ds.vq,      ds.blacs_h,   ds.desc_abf};
        return std::pair{compute_crpa_onsite(context, selection, {&frame}),
                         chi.active_qpoints().size()};
    };
    const auto full = calculate(false);
    ds.basis_convention.bloch_phase = -1;
    ds.basis_convention.bloch_ratom = 0;
    ds.basis_convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    ds.basis_convention.coeff_m_negative = ds.basis_convention.coeff_m_positive =
        LIBRPA_RSH_COEFF_1_M;
    for (int i = 0; i < 6; ++i)
    {
        std::array<double, 9> rotation;
        for (int j = 0; j < 9; ++j) rotation[j] = model.operations[9 * i + j];
        ds.spg_symops.push_back({rotation, {0, 0, 0}, true});
        SymmetrySpinOperation op;
        op.spatial_id = i;
        op.antiunitary = model.antiunitary[i];
        op.spin_source = SymmetrySpinActionSource::DerivedFromSpatialSOC;
        ds.spg_spin_ops.push_back(op);
    }
    ds.spg_spin_ops_explicit = true;
    ds.spg_grey_group = false;
    initialize_symmetry_context(ds, true);
    const auto reduced = calculate(true);
    if (ds.symmetry_context.spin_operations.size() != 6 || reduced.second >= full.second)
        throw std::runtime_error("explicit magnetic table was widened or did not reduce q points");
    double maximum = 0;
    for (int site = 0; site < full.first.tensors.size(); ++site)
        for (int node = 0; node < full.first.frequencies.size(); ++node)
            for (const auto pair : {std::pair{&full.first.tensors[site].u[node],
                                              &reduced.first.tensors[site].u[node]},
                                    std::pair{&full.first.tensors[site].w[node],
                                              &reduced.first.tensors[site].w[node]}})
                for (int i = 0; i < pair.first->size; ++i)
                {
                    maximum = std::max(maximum, std::abs(pair.first->c[i] - pair.second->c[i]));
                    close(pair.first->c[i], pair.second->c[i], 2e-11);
                }
    if (ds.comm_h.is_root())
        std::cout << "Noncollinear magnetic cRPA: " << full.second << " -> " << reduced.second
                  << " q; 3 unitary + 3 antiunitary; complete U/W max=" << maximum << '\n';
}

void test_soc_and_spin_representation()
{
    Fixture original(true, 0.37), rotated(true, 0.37, 0.61);
    original.compare_response(false);
    original.compare_response(true);
    rotated.compare_response(false);
    rotated.compare_response(true);
    const auto first = original.run(), second = rotated.run();
    original.compare_tensors(first);
    rotated.compare_tensors(second);
    for (std::size_t iw = 0; iw < first.frequencies.size(); ++iw)
        for (int i = 0; i < first.tensors[0].bare.size; ++i)
        {
            close(first.tensors[0].bare.c[i], second.tensors[0].bare.c[i], 1e-12);
            close(first.tensors[0].u[iw].c[i], second.tensors[0].u[iw].c[i], 1e-11);
            close(first.tensors[0].w[iw].c[i], second.tensors[0].w[iw].c[i], 1e-11);
        }

    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    ComplexMatrix spatial(3, 3);
    spatial.set_as_identity_matrix();
    CrpaWindowOptions options;
    options.correlated_species = "X";
    options.parent_orbitals = options.output_orbitals = "s";
    options.response_bands = {0, 1, 2, 3};
    options.orbital_bands = {0, 1, 4, 5};
    const auto built = build_crpa_window_input(original.ds.basis_wfc, convention, {0}, {"X"},
                                               std::vector<ComplexMatrix>(4, spatial),
                                               original.ds.pbc.kfrac_list, original.ds.mf, options);
    if (built.sites[0].n_orbitals != 2 ||
        built.sites[0].orbital_labels != std::vector<std::string>{"s:up", "s:down"})
        throw std::runtime_error("SOC output labels/count are inconsistent");
    // Analytic spectral projection gives precisely the first Kramers pair.
    for (const auto& phi : built.orbitals_spin_k.at(0))
        for (int i = 0; i < 6; ++i)
            for (int a = 0; a < 2; ++a) close(phi(i, a), original.coefficients(i, a), 1e-12);

    // An independently specified nonorthogonal AO change: C=R*C0,
    // S=R^(-dagger)R^(-1). The complete B projects back to the original trials.
    ComplexMatrix r(3, 3), inverse(3, 3);
    r.set_as_identity_matrix();
    inverse.set_as_identity_matrix();
    r(0, 1) = 0.2;
    r(1, 2) = 0.3;
    inverse(0, 1) = -0.2;
    inverse(1, 2) = -0.3;
    inverse(0, 2) = 0.06;
    const auto metric = transpose(inverse, true) * inverse;
    auto nonorthogonal = original.ds.mf;
    for (int k = 0; k < 4; ++k)
        for (int s = 0; s < 2; ++s)
            nonorthogonal.get_eigenvectors()[0][s][k] =
                nonorthogonal.get_eigenvectors()[0][s][k] * transpose(r, false);
    options.orbital_bands = {0, 1, 2, 3, 4, 5};
    const auto changed = build_crpa_window_input(
        original.ds.basis_wfc, convention, {0}, {"X"}, std::vector<ComplexMatrix>(4, metric),
        original.ds.pbc.kfrac_list, nonorthogonal, options);
    for (const auto& phi : changed.orbitals_spin_k.at(0))
        for (int i = 0; i < 6; ++i)
            for (int a = 0; a < 2; ++a) close(phi(i, a), i == 3 * a ? 1.0 : 0.0, 1e-12);
}

void test_soc_bloch_phases()
{
    Fixture periodic(true, 0.37, 0, true);
    periodic.compare_response(false);
    periodic.compare_response(true);
    periodic.compare_tensors(periodic.run());
}

void test_constrained_soc_optical()
{
    Fixture f(true, 0.37);
    const double pi = std::acos(-1.0);
    initialize_velocity_matrix(f.ds.velocity_matrix, 1, 4, f.bands);
    // Hermitian imaginary AO velocity operators, odd under time reversal.
    for (int k = 0; k < 4; ++k)
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int n = 0; n < f.bands; ++n)
                for (int m = 0; m < f.bands; ++m)
                    for (int s = 0; s < 2; ++s)
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                f.ds.velocity_matrix[0][k][alpha](n, m) +=
                                    std::conj(f.coefficients(3 * s + i, n)) *
                                    Z(0, 0.025 * (alpha + 1) * (j - i)) *
                                    f.coefficients(3 * s + j, m);
    const auto& frequencies = f.ds.tfg.get_freq_nodes();
    const auto make_optical = [&](const MeanField& mf, bool constrained)
    {
        auto optical = std::make_unique<diele_func>(mf, f.ds.velocity_matrix, f.ds.pbc.kfrac_list,
                                                    f.ds.basis_wfc, f.ds.basis_aux, frequencies, 3,
                                                    6, 1, 2, f.ds.pbc, f.ds.comm_h, f.ds.blacs_h);
        optical->use_soc = true;
        if (constrained) optical->set_excluded_band_pairs(f.selection());
        optical->normalize_gamma_cell_weights = true;
        optical->init(0, f.ds.vq);
        optical->cal_head();
        optical->cal_wing(f.ds.cs_data, 0, f.ds.vq);
        return optical;
    };
    const auto full = make_optical(f.ds.mf, false);
    const auto retained = make_optical(f.ds.mf, true);
    double excluded_signal = 0;
    for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
        for (int which = 0; which < 2; ++which)
        {
            ComplexMatrix head(3, 3), wing(2, 3);
            head.set_as_identity_matrix();
            for (int k = 0; k < 4; ++k)
                for (int n = 0; n < 2; ++n)
                    for (int m = 2; m < 6; ++m)
                    {
                        if (which && m < 4) continue;
                        const double delta =
                            f.ds.mf.get_eigenvals()[0](k, m) - f.ds.mf.get_eigenvals()[0](k, n);
                        const double denominator =
                            delta * delta + frequencies[iw] * frequencies[iw];
                        for (int alpha = 0; alpha < 3; ++alpha)
                        {
                            const auto& va = f.ds.velocity_matrix[0][k][alpha];
                            for (int beta = 0; beta < 3; ++beta)
                                head(alpha, beta) += 4 * pi * 0.5 * va(m, n) *
                                                     f.ds.velocity_matrix[0][k][beta](n, m) /
                                                     (delta * denominator);
                            for (int mu = 0; mu < 2; ++mu)
                                wing(mu, alpha) -= 2 * std::sqrt(4 * pi) * 0.25 *
                                                   std::conj(f.vertex(mu, n, m) * va(m, n)) /
                                                   denominator;
                        }
                    }
            const auto& actual = which ? retained : full;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    close(actual->get_dielectric_head(iw)(i, j), head(i, j), 2e-12);
                    excluded_signal = std::max(excluded_signal,
                                               std::abs(full->get_dielectric_head(iw)(i, j) -
                                                        retained->get_dielectric_head(iw)(i, j)));
                }
            for (int mu = 0; mu < 2; ++mu)
                for (int alpha = 0; alpha < 3; ++alpha)
                    close(actual->get_wing_mu(iw)(mu, alpha), wing(mu, alpha), 2e-12);
        }
    if (excluded_signal < 1e-4) throw std::runtime_error("optical Pd control was trivial");

    // A zero-response vacuum must give the retained identity after cell averaging.
    auto vacuum_mf = f.ds.mf;
    vacuum_mf.get_weight()[0].zero_out();
    const auto vacuum = make_optical(vacuum_mf, false);
    auto identity = init_local_mat<Z>(f.ds.desc_abf, MAJOR::COL);
    for (int i = 0; i < identity.nr(); ++i)
        for (int j = 0; j < identity.nc(); ++j)
            identity(i, j) = f.ds.desc_abf.indx_l2g_r(i) == f.ds.desc_abf.indx_l2g_c(j) ? 1.0 : 0.0;
    auto inverse = identity.copy();
    vacuum->rewrite_eps_abf_space(inverse, 0, identity, identity, f.ds.desc_abf, 2, 0, false,
                                  false);
    for (int i = 0; i < inverse.nr(); ++i)
        for (int j = 0; j < inverse.nc(); ++j) close(inverse(i, j), identity(i, j), 2e-12);
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
        test_zero_soc_limit();
        test_soc_and_spin_representation();
        test_soc_bloch_phases();
        test_magnetic_qstar_screening();
        test_noncollinear_magnetic_crpa();
        test_constrained_soc_optical();
    }
    catch (const std::exception& failure)
    {
        std::cerr << failure.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_crpa_soc: passed\n";
}
