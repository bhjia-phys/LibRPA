// Full native SOC charge response against an independent finite Lindhard sum.
// No dielectric solver, Hermitian collector, or GW continuation is used here.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <valarray>
#include <vector>

#include "../api/dataset.h"
#include "../api/dataset_helper.h"
#include "../api/gw_decay_bound.h"
#include "../api/instance_manager.h"
#include "../core/chi0.h"
#include "../io/global_io.h"
#include "../math/matrix_m.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "../utils/dev_options.h"
#include "librpa.hpp"

using namespace librpa_int;
using Complex = std::complex<double>;

namespace
{

struct Spectrum
{
    std::vector<std::vector<double>> energies;
    std::vector<Matz> vectors;
};

Spectrum make_spectrum(const int nk, const double bz, const bool rotate_spin)
{
    Spectrum result;
    Matz spin_rotation(6, 6, MAJOR::COL);
    spin_rotation.zero_out();
    const double angle = rotate_spin ? 0.71 : 0.0;
    const auto off_diagonal = std::polar(std::sin(angle / 2), 0.37);
    for (int a = 0; a != 3; ++a)
    {
        spin_rotation(a, a) = spin_rotation(a + 3, a + 3) = std::cos(angle / 2);
        spin_rotation(a, a + 3) = -std::conj(off_diagonal);
        spin_rotation(a + 3, a) = off_diagonal;
    }
    for (int k = 0; k != nk; ++k)
    {
        Matz h(6, 6, MAJOR::COL);
        h.zero_out();
        const double phase = TWO_PI * k / nk;
        for (int a = 0; a != 3; ++a)
        {
            h(a, a) = bz;
            h(a + 3, a + 3) = -bz;
            h(a, a + 3) = std::polar(1.0, -TWO_PI * a / 3);
            h(a + 3, a) = std::conj(h(a, a + 3));
            for (int s = 0; s != 2; ++s)
            {
                const int next = (a + 1) % 3;
                // The periodic case has a nonconstant occupied projector and
                // an odd-k orbital term; its q/-q responses need not coincide.
                const Complex hop{0.15 + (nk > 1 ? 0.025 * std::cos(phase) : 0.0),
                                  s == 0 ? 0.04 : -0.04};
                h(a + 3 * s, next + 3 * s) = hop;
                h(next + 3 * s, a + 3 * s) = std::conj(hop);
                if (nk > 1) h(a + 3 * s, a + 3 * s) += 0.04 * (a - 1) * std::sin(phase);
            }
        }
        h = spin_rotation * h * transpose(spin_rotation, true);
        std::vector<double> eigenvalues;
        Matz eigenvectors;
        eigsh(h, eigenvalues, eigenvectors);
        assert(eigenvalues[3] - eigenvalues[2] > 1.0);
        result.energies.push_back(std::move(eigenvalues));
        result.vectors.push_back(std::move(eigenvectors));
    }
    return result;
}

// Independent charge-vertex Lehmann sum at +iw. The backward transition has
// the opposite momentum; it is not a same-q conjugation of the forward vertex.
Matz lindhard(const Spectrum &spectrum, const int iq, const double omega, const int mesh_y = 1)
{
    const int nk = spectrum.energies.size();
    const int mesh_x = nk / mesh_y;
    Matz response(3, 3, MAJOR::COL);
    response.zero_out();
    for (int sign : {1, -1})
        for (int k = 0; k != nk; ++k)
        {
            const int target = ((k / mesh_y + sign * (iq / mesh_y) + mesh_x) % mesh_x) * mesh_y +
                               (k % mesh_y + sign * (iq % mesh_y) + mesh_y) % mesh_y;
            for (int m = 0; m != 3; ++m)
                for (int n = 3; n != 6; ++n)
                {
                    Complex vertex[3]{};
                    for (int a = 0; a != 3; ++a)
                        for (int spin = 0; spin != 2; ++spin)
                            vertex[a] += std::conj(spectrum.vectors[k](a + 3 * spin, m)) *
                                         spectrum.vectors[target](a + 3 * spin, n);
                    const double delta = spectrum.energies[target][n] - spectrum.energies[k][m];
                    for (int a = 0; a != 3; ++a)
                        for (int b = 0; b != 3; ++b)
                        {
                            const Complex amplitude = vertex[a] * std::conj(vertex[b]);
                            response(a, b) -= (sign == 1 ? amplitude : std::conj(amplitude)) /
                                              (double(nk) * Complex{delta, -sign * omega});
                        }
                }
        }
    return response;
}

Matz projection(const int rows, const double qfrac)
{
    Matz u(rows, 3, MAJOR::COL);
    // Complex orthonormal rows with an additional q-dependent column phase.
    // Rectangular cases deliberately project to two independent charge probes.
    for (int a = 0; a != rows; ++a)
        for (int b = 0; b != 3; ++b)
            u(a, b) = std::polar(1.0 / std::sqrt(3.0),
                                 TWO_PI * a * b / 3 + 0.29 * (b + 1) + 0.43 * b * qfrac);
    return u;
}

std::vector<Complex> run_case(const int natom, const int nk, const double bz,
                              const bool rotate_spin, const int shrink_mode,
                              const int naux, const bool split_q = false)
{
    assert(natom == 1 || natom == 3);
    assert(natom == 1 || naux == 3);
    assert(shrink_mode != 0 || naux == 3);
    const auto spectrum = make_spectrum(nk, bz, rotate_spin);
    Dataset ds(MPI_COMM_WORLD);
    const std::vector<std::size_t> original_basis = natom == 1
        ? std::vector<std::size_t>{3} : std::vector<std::size_t>{1, 1, 1};
    ds.basis_wfc.set(original_basis);
    AtomicBasis original_aux;
    original_aux.set(original_basis);
    ds.basis_aux.set(natom == 1 ? std::vector<std::size_t>{std::size_t(naux)} : original_basis);
    ds.mf.set(1, nk, 6, 3, 2);
    double occupied_max = -std::numeric_limits<double>::infinity();
    double empty_min = std::numeric_limits<double>::infinity();
    for (const auto &energies : spectrum.energies)
    {
        occupied_max = std::max(occupied_max, energies[2]);
        empty_min = std::min(empty_min, energies[3]);
    }
    assert(empty_min - occupied_max > 1.0);
    ds.mf.get_efermi() = 0.5 * (occupied_max + empty_min);
    ds.pbc.set_latvec_and_G({8, 0, 0, 0, 8, 0, 0, 0, 8},
                            {TWO_PI / 8, 0, 0, 0, TWO_PI / 8, 0, 0, 0, TWO_PI / 8});
    std::vector<double> kcart;
    for (int k = 0; k != nk; ++k) kcart.insert(kcart.end(), {TWO_PI * k / (8 * nk), 0, 0});
    ds.pbc.set_kgrids_kvec(nk, 1, 1, kcart);
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, nk);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds.tfg.reset(24);
    ds.tfg.generate_minimax(0.4, 4.0);
    for (int k = 0; k != nk; ++k)
    {
        for (int spin = 0; spin != 2; ++spin)
        {
            ComplexMatrix coefficients(6, 3);
            for (int n = 0; n != 6; ++n)
                for (int a = 0; a != 3; ++a)
                    coefficients(n, a) = spectrum.vectors[k](a + 3 * spin, n);
            ds.mf.get_eigenvectors()[0][spin][k] = coefficients;
        }
        for (int n = 0; n != 6; ++n)
        {
            ds.mf.get_eigenvals()[0](k, n) = spectrum.energies[k][n];
            ds.mf.get_weight()[0](k, n) = n < 3 ? 1.0 / nk : 0.0;
        }
    }
    ds.cs_data.use_libri = true;
    if (ds.comm_h.is_root())
        for (int atom = 0; atom != natom; ++atom)
        {
            const std::size_t size = original_basis[atom];
            RI::Tensor<double> coefficients({size, size, size});
            // Two equal RI centers give the exact local density vertex.
            for (std::size_t a = 0; a != size; ++a) coefficients(a, a, a) = 0.5;
            ds.cs_data.data_libri[atom][{atom, {0, 0, 0}}] = std::move(coefficients);
        }
    const auto assigned = dispatch_upper_triangular_tasks(
        natom, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
        ds.blacs_h.myprow, ds.blacs_h.mypcol);
    const std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    Chi0 response(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                  ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
    assert(response.full_frequency_matrix());
    response.gf_threshold = response.libri_threshold_C = response.libri_threshold_G = 0;
    // Exercise byte-chunk collection, including reversed-pair requests.
    response.libri_collect_max_bytes = natom == 3 ? 64 : 0;
    std::map<Vector3_Order<double>, ComplexMatrix> shrink;
    if (shrink_mode)
        for (const auto &q : response.active_qpoints())
        {
            const auto u = projection(naux, (ds.pbc.latvec * q).x);
            ComplexMatrix matrix(naux, 3);
            for (int a = 0; a != naux; ++a)
                for (int b = 0; b != 3; ++b) matrix(a, b) = u(a, b);
            shrink.emplace(q, std::move(matrix));
        }
    const bool old_delayed = global::dev_opts.use_delayed_ft_shrink;
    const bool old_split = global::dev_opts.use_chi0_q_uhap_split;
    global::dev_opts.use_delayed_ft_shrink = shrink_mode == 2;
    global::dev_opts.use_chi0_q_uhap_split = split_q;
    response.build(LIBRPA_ROUTING_LIBRI, ds.cs_data, pairs, original_aux, shrink, ds.blacs_h);
    global::dev_opts.use_delayed_ft_shrink = old_delayed;
    global::dev_opts.use_chi0_q_uhap_split = old_split;

    const auto &qs = response.active_qpoints();
    const int nf = ds.tfg.get_n_grids(), nq = qs.size();
    const int elements = nf * nq * naux * naux;
    std::vector<Complex> values(elements, 0.0);
    std::vector<int> owners(elements, 0);
    for (const auto &[omega, qmap] : response.get_chi0_q())
        for (const auto &[q, atoms] : qmap)
        {
            const auto found = std::find(qs.begin(), qs.end(), q);
            assert(found != qs.end());
            const int f = ds.tfg.get_freq_index(omega), iq = found - qs.begin();
            for (const auto &[a, cols] : atoms)
                for (const auto &[b, matrix] : cols)
                    for (int i = 0; i != matrix.nr; ++i)
                        for (int j = 0; j != matrix.nc; ++j)
                        {
                            const int row = ds.basis_aux.get_global_index(a, i);
                            const int col = ds.basis_aux.get_global_index(b, j);
                            const int index = ((f * nq + iq) * naux + row) * naux + col;
                            values[index] += matrix(i, j);
                            ++owners[index];
                        }
        }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), elements, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, ds.comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), elements, MPI_INT, MPI_SUM, ds.comm_h.comm);
    for (const int count : owners) assert(count == 1);
    double error = 0.0, odd_norm = 0.0;
    for (int f = 0; f != nf; ++f)
        for (int iq = 0; iq != nq; ++iq)
        {
            const double qfrac = (ds.pbc.latvec * qs[iq]).x;
            const int qindex = (int(std::llround(nk * qfrac)) % nk + nk) % nk;
            auto exact = lindhard(spectrum, qindex, ds.tfg.get_freq_nodes()[f]);
            if (shrink_mode)
            {
                const auto u = projection(naux, qfrac);
                exact = u * exact * transpose(u, true);
            }
            for (int a = 0; a != naux; ++a)
                for (int b = 0; b != naux; ++b)
                {
                    const int index = ((f * nq + iq) * naux + a) * naux + b;
                    error = std::max(error, std::abs(values[index] - exact(a, b)));
                    odd_norm = std::max(odd_norm, std::abs(exact(a, b) - std::conj(exact(b, a))));
                }
        }
    if (ds.comm_h.is_root())
        std::cout << "Signed Chi0: natom=" << natom << " nk=" << nk << " Bz=" << bz
                  << " spin_rotation=" << rotate_spin << " shrink=" << shrink_mode
                  << " naux=" << naux << " q_split=" << split_q
                  << " Lindhard_error=" << error << " antiHerm_norm=" << odd_norm << std::endl;
    assert(error < 2e-8);
    if (bz != 0.0) assert(odd_norm > 1e-6);
    if (bz == 0.0 && nk == 1) assert(odd_norm < 1e-12);

    if (natom == 1 && nk == 3 && shrink_mode == 1 && naux == 2)
    {
        // The exact endpoint branch must retain the loaded subspace and both
        // signed sides before a complex q-dependent rectangular projection.
        Chi0 endpoint(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
                      ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
        endpoint.gf_threshold = endpoint.libri_threshold_C = endpoint.libri_threshold_G = 0;
        endpoint.nbands_G = 5;
        endpoint.libri_collect_max_bytes = 64;
        auto endpoint_u = shrink;
        for (const auto &[representative, members] : endpoint.qpoint_view().members)
            for (const auto &q : members)
            {
                const auto u = projection(naux, (ds.pbc.latvec * q).x);
                ComplexMatrix matrix(naux, 3);
                for (int a = 0; a != naux; ++a)
                    for (int b = 0; b != 3; ++b) matrix(a, b) = u(a, b);
                endpoint_u[q] = std::move(matrix);
            }
        const auto grams = endpoint.build_transition_grams(ds.cs_data, pairs, original_aux,
                                                           endpoint_u, ds.blacs_h);
        assert(grams.qpoints.size() == std::size_t(nk));
        std::vector<Complex> actual(2 * nk * naux * naux, 0.0);
        std::vector<int> counts(actual.size(), 0);
        for (int side = 0; side != 2; ++side)
        {
            const auto &qmap = side == 0 ? grams.positive : grams.negative;
            for (int iq = 0; iq != nk; ++iq)
                for (const auto &[I, columns] : qmap.at(grams.qpoints[iq]))
                    for (const auto &[J, block] : columns)
                        for (int a = 0; a != block.nr; ++a)
                            for (int b = 0; b != block.nc; ++b)
                            {
                                const int index = ((side * nk + iq) * naux + a) * naux + b;
                                actual[index] += block(a, b);
                                ++counts[index];
                            }
        }
        MPI_Allreduce(MPI_IN_PLACE, actual.data(), actual.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                      ds.comm_h.comm);
        MPI_Allreduce(MPI_IN_PLACE, counts.data(), counts.size(), MPI_INT, MPI_SUM, ds.comm_h.comm);
        for (const int count : counts) assert(count == 1);
        double endpoint_error = 0.0, endpoint_norm = 0.0;
        for (int iq = 0; iq != nk; ++iq)
        {
            const double qfrac = (ds.pbc.latvec * grams.qpoints[iq]).x;
            const int qindex = (int(std::llround(nk * qfrac)) % nk + nk) % nk;
            Matz expected[2]{Matz(3, 3, MAJOR::COL), Matz(3, 3, MAJOR::COL)};
            for (auto &matrix : expected) matrix.zero_out();
            for (int k = 0; k != nk; ++k)
                for (int m = 0; m != 5; ++m)
                    for (int n = 0; n != 5; ++n)
                    {
                        if ((m < 3) == (n < 3)) continue;
                        Complex vertex[3]{};
                        for (int a = 0; a != 3; ++a)
                            for (int spin = 0; spin != 2; ++spin)
                                vertex[a] += std::conj(spectrum.vectors[k](a + 3 * spin, m)) *
                                             spectrum.vectors[(k + qindex) % nk](a + 3 * spin, n);
                        const int side = m < 3 ? 0 : 1;
                        for (int a = 0; a != 3; ++a)
                            for (int b = 0; b != 3; ++b)
                                expected[side](a, b) +=
                                    vertex[a] * std::conj(vertex[b]) / double(nk);
                    }
            const auto u = projection(naux, qfrac);
            for (int side = 0; side != 2; ++side)
            {
                const auto projected = u * expected[side] * transpose(u, true);
                for (int a = 0; a != naux; ++a)
                    for (int b = 0; b != naux; ++b)
                    {
                        const int index = ((side * nk + iq) * naux + a) * naux + b;
                        endpoint_error =
                            std::max(endpoint_error, std::abs(actual[index] - projected(a, b)));
                        endpoint_norm = std::max(endpoint_norm, std::abs(projected(a, b)));
                    }
            }
        }
        assert(endpoint_error < 2e-12 && endpoint_norm > 1e-6);
        if (ds.comm_h.is_root())
            std::cout << "Signed endpoint Gram: retained=5/6 rectangular_U(q) error="
                      << endpoint_error << " norm=" << endpoint_norm << std::endl;
    }

    // cRPA subtraction must retain lower blocks and the diagonal odd part too.
    Chi0QMap doubled = response.get_chi0_q();
    for (auto &[omega, qmap] : doubled)
        for (auto &[q, atoms] : qmap)
            for (auto &[a, cols] : atoms)
                for (auto &[b, matrix] : cols)
                    for (int i = 0; i != matrix.size; ++i) matrix.c[i] *= 2;
    const Chi0QMap original = response.get_chi0_q();
    response.replace_chi0_q_by_difference(doubled);
    for (const auto &[omega, qmap] : original)
        for (const auto &[q, atoms] : qmap)
            for (const auto &[a, cols] : atoms)
                for (const auto &[b, matrix] : cols)
                {
                    const auto &difference = response.get_chi0_q().at(omega).at(q).at(a).at(b);
                    for (int i = 0; i != matrix.size; ++i)
                        assert(std::abs(matrix.c[i] - difference.c[i]) < 1e-14);
                }
    return values;
}

void compare(const std::vector<Complex> &a, const std::vector<Complex> &b)
{
    assert(a.size() == b.size());
    double error = 0.0;
    for (std::size_t i = 0; i != a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i]));
    assert(error < 2e-10);
}

// Exercise the real Gram/bound entry, but compare its energy metadata with an
// exhaustive ordered-band oracle. Energies are deliberately unsorted; the SOC
// case loads only five of six spinor states, and its retained two-band window
// has occupied-only, empty-only and mixed sectors on different k frames.
void check_decay_metadata(const int nspin, const int nspinor, const int used,
                          const int excluded, const int mu_endpoint)
{
    const int nk = 3, nao = 3, nb = nspinor == 2 ? 5 : 3;
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(nspin, nk, nb, nao, nspinor);
    handler.set_ao_basis_wfc({3}, {{0, 0, 0}});
    handler.set_ao_basis_aux({1}, {{0}});
    const double lattice[9]{6, 0, 0, 0, 7, 0, 0, 0, 8};
    const double reciprocal[9]{TWO_PI / 6, 0, 0, 0, TWO_PI / 7, 0, 0, 0, TWO_PI / 8};
    handler.set_latvec_and_G(lattice, reciprocal);
    // Reordered mesh, including reciprocal-lattice-shifted representatives.
    handler.set_kgrids_kvec(3, 1, 1,
                           {5 * TWO_PI / 18, 0, 0, TWO_PI / 6, 0, 0, -TWO_PI / 9, 0, 0});
    const auto ds = api::get_dataset_instance(handler);
    initialize_ds_atpairs_local(*ds, LIBRPA_ROUTING_LIBRI);
    const double table[3][5]{{9.1, -2.2, -0.4, 1.9, -1.1},
                              {-1.3, 1.5, 2.6, 7.5, -1.9},
                              {-0.8, 2.0, -2.8, 1.7, -8.5}};
    std::vector<double> energy(nspin * nk * nb), occupation(energy.size());
    double eo = -1e10, eu = 1e10;
    for (int s = 0; s != nspin; ++s)
        for (int k = 0; k != nk; ++k)
        {
            std::vector<Complex> up(nb * nao), down(nb * nao);
            for (int b = 0; b != nb; ++b)
            {
                const int index = (s * nk + k) * nb + b;
                energy[index] = table[k][b] + 0.23 * s;
                occupation[index] = energy[index] < 0 ? 2.0 / (nspin * nspinor) : 0.0;
                if (occupation[index] > 0)
                    eo = std::max(eo, energy[index]);
                else
                    eu = std::min(eu, energy[index]);
                for (int a = 0; a != nao; ++a)
                {
                    const double phase = TWO_PI * a * b / (nao * nspinor) + 0.19 * k * a;
                    up[b * nao + a] = std::polar(1.0 / std::sqrt(nao * nspinor), phase);
                    down[b * nao + a] = std::polar(1.0 / std::sqrt(nao * nspinor),
                                                  phase + TWO_PI * nao * b / (nao * nspinor));
                }
            }
            if (nspinor == 2)
                handler.set_wfc_spinor_packed(k, nb, nao, up.data(), down.data());
            else
                handler.set_wfc_packed(s, k, nb, nao, up.data());
        }
    const double mu = mu_endpoint < 0 ? eo : mu_endpoint > 0 ? eu : (eo + eu) / 2;
    handler.set_wg_ekb_efermi(nspin, nk, nb, occupation.data(), energy.data(), mu);
    if (ds->comm_h.is_root())
    {
        const int origin[3]{0, 0, 0};
        const double coefficients[9]{0.3, 0.02, 0, 0.02, 0.4, -0.03, 0, -0.03, 0.5};
        handler.set_lri_coeff(LIBRPA_ROUTING_LIBRI, 0, 0, nao, nao, 1, origin, coefficients);
    }
    for (int q = 0; q != nk; ++q)
        for (const auto &[I, J] : ds->atpairs_local)
        {
            const Complex v{1.0};
            handler.set_aux_cut_coulomb_k_atom_pair_packed(q, I, J, 1, 1, &v, 0.0);
        }
    librpa::Options options;
    options.use_symmetry_rpa = options.use_shrink_abfs = options.use_fullcoul_eps = LIBRPA_SWITCH_OFF;
    options.gf_threshold = options.libri_chi0_threshold_C = options.libri_chi0_threshold_G = 0;
    options.n_bands_chi0 = used;
    options.n_bands_exclude_chi0 = excluded;
    options.n_bands_sigc = -7; // Any negative count still means all loaded bands.
    options.n_bands_exclude_sigc = 1;
    const auto bound = compute_ds_gw_body_decay_bound(*ds, options);
    const int retained = used < 0 ? nb : used;
    std::array<double, 2> tail{};
    for (int s = 0; s != nspin; ++s)
        for (int k = 0; k != nk; ++k)
            for (int b = 1; b != nb; ++b)
            {
                const int index = (s * nk + k) * nb + b;
                const int side = occupation[index] > 0 ? 1 : 0;
                tail[side] = std::max(tail[side], std::abs(energy[index] - mu));
            }
    assert(bound.gap == eu - eo && bound.electron_tail == tail);
    double error = 0.0, asymmetry = 0.0, full_window_difference = 0.0;
    for (const auto &q : bound.grams.qpoints)
    {
        const Vector3_Order<double> qfrac{ds->pbc.latvec * q};
        std::array<double, 2> exact{}, full_window{};
        for (int k = 0; k != nk; ++k)
        {
            int kp = -1;
            for (int j = 0; j != nk; ++j)
            {
                const auto difference = ds->pbc.kfrac_list[k] + qfrac - ds->pbc.kfrac_list[j];
                if (std::abs(difference.x - std::round(difference.x)) < 1e-8) kp = j;
            }
            assert(kp >= 0);
            for (int s = 0; s != nspin; ++s)
                for (int m = 0; m != nb; ++m)
                    for (int n = 0; n != nb; ++n)
                    {
                        const int i = (s * nk + k) * nb + m, j = (s * nk + kp) * nb + n;
                        if ((occupation[i] > 0) == (occupation[j] > 0)) continue;
                        const int side = occupation[i] > 0 ? 0 : 1;
                        const double delta = std::abs(energy[j] - energy[i]);
                        full_window[side] = std::max(full_window[side], delta);
                        if (m >= excluded && m < retained && n >= excluded && n < retained)
                            exact[side] = std::max(exact[side], delta);
                    }
        }
        const auto &actual = bound.q_bounds.at(q).transition_max;
        for (int side = 0; side != 2; ++side)
        {
            error = std::max(error, std::abs(actual[side] - exact[side]));
            full_window_difference =
                std::max(full_window_difference, std::abs(full_window[side] - exact[side]));
        }
        asymmetry = std::max(asymmetry, std::abs(exact[0] - exact[1]));
    }
    assert(error < 2e-12);
    if (nspinor == 2 && used == 3)
        assert(asymmetry > 0.1 && full_window_difference > 1.0);
    if (ds->comm_h.is_root())
        std::cout << "Exact decay metadata: spins=" << nspin << " spinor=" << nspinor
                  << " loaded=" << nb << " used=" << used << " excluded=" << excluded
                  << " mu_endpoint=" << mu_endpoint << " error=" << error
                  << " signed_asymmetry=" << asymmetry
                  << " wrong_full_window_difference=" << full_window_difference << std::endl;
}

Matz crystal_hamiltonian(const int kx, const int ky, const double bz)
{
    Matz h(6, 6, MAJOR::COL);
    h.zero_out();
    const double dispersion = 0.035 * (std::cos(TWO_PI * kx / 3) +
        std::cos(TWO_PI * ky / 3) + std::cos(TWO_PI * (kx - ky) / 3));
    for (int a = 0; a != 3; ++a)
    {
        h(a, a) = bz;
        h(a + 3, a + 3) = -bz;
        h(a, a + 3) = std::polar(1.0, -TWO_PI * a / 3);
        h(a + 3, a) = std::conj(h(a, a + 3));
        for (int s = 0; s != 2; ++s)
        {
            const int next = (a + 1) % 3;
            const Complex hopping{0.15 + dispersion, s == 0 ? 0.04 : -0.04};
            h(a + 3 * s, next + 3 * s) = hopping;
            h(next + 3 * s, a + 3 * s) = std::conj(hopping);
        }
    }
    return h;
}

int mod3(const int value) { return (value % 3 + 3) % 3; }

std::array<int, 2> crystal_target(int x, int y, const int rotation, const bool anti)
{
    // Theta M_y: q -> (-qx, qy-qx); then Cartesian C3 rotations.
    if (anti) { y -= x; x = -x; }
    for (int r = 0; r != rotation; ++r)
    {
        const int next_x = -y;
        y = x - y;
        x = next_x;
    }
    return {mod3(x), mod3(y)};
}

int crystal_star_count(const bool with_anti)
{
    std::set<std::array<int, 2>> remaining;
    for (int x = 0; x != 3; ++x)
        for (int y = 0; y != 3; ++y) remaining.insert({x, y});
    int count = 0;
    while (!remaining.empty())
    {
        const auto k = *remaining.begin();
        ++count;
        for (int a = 0; a != (with_anti ? 2 : 1); ++a)
            for (int r = 0; r != 3; ++r)
                remaining.erase(crystal_target(k[0], k[1], r, a));
    }
    return count;
}

// Genuine crystal reduction: nine full k points, five C3 stars or four stars
// after adding the VALID magnetic antiunitary Theta M_y. Unlike Theta M_z,
// Theta M_y preserves Bz. It acts as site reflection times bare K on spinors.
std::vector<Complex> run_crystal_case(const bool symmetry, const bool ibz,
                                      const bool with_anti,
                                      const std::vector<Complex> *full_reference)
{
    assert(!ibz || symmetry);
    constexpr double bz = 0.21;
    Spectrum spectrum;
    double covariance_error = 0.0;
    for (int x = 0; x != 3; ++x)
        for (int y = 0; y != 3; ++y)
        {
            const auto h = crystal_hamiltonian(x, y, bz);
            for (int a = 0; a != (with_anti ? 2 : 1); ++a)
                for (int r = 0; r != 3; ++r)
                {
                    Matz action(6, 6, MAJOR::COL);
                    action.zero_out();
                    for (int atom = 0; atom != 3; ++atom)
                        for (int spin = 0; spin != 2; ++spin)
                            action(mod3((a ? -atom : atom) + r) + 3 * spin, atom + 3 * spin) =
                                std::polar(1.0, (spin == 0 ? -0.5 : 0.5) * TWO_PI * r / 3);
                    auto operand = h.copy();
                    if (a)
                        for (int i = 0; i != 6; ++i)
                            for (int j = 0; j != 6; ++j) operand(i, j) = std::conj(operand(i, j));
                    const auto target = crystal_target(x, y, r, a);
                    const auto expected = crystal_hamiltonian(target[0], target[1], bz);
                    const auto actual = action * operand * transpose(action, true);
                    for (int i = 0; i != 6; ++i)
                        for (int j = 0; j != 6; ++j)
                            covariance_error = std::max(covariance_error, std::abs(actual(i, j) - expected(i, j)));
                }
            std::vector<double> energies;
            Matz vectors;
            eigsh(h, energies, vectors);
            spectrum.energies.push_back(std::move(energies));
            spectrum.vectors.push_back(std::move(vectors));
        }
    assert(covariance_error < 1e-12);
    const int nstars = crystal_star_count(with_anti);
    assert(nstars == (with_anti ? 4 : 5));
    const int nloaded = ibz ? nstars : 9;
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, nloaded, 6, 3, 2);
    handler.set_ao_basis_wfc({1, 1, 1}, {{0}, {0}, {0}});
    handler.set_ao_basis_aux({1, 1, 1}, {{0}, {0}, {0}});
    const double lattice[9]{6, 0, 0, 3, 3 * std::sqrt(3.0), 0, 0, 0, 8};
    const double reciprocal[9]{TWO_PI / 6, -TWO_PI / (6 * std::sqrt(3.0)), 0,
                               0, TWO_PI / (3 * std::sqrt(3.0)), 0, 0, 0, TWO_PI / 8};
    handler.set_latvec_and_G(lattice, reciprocal);
    std::vector<double> positions;
    for (int a = 0; a != 3; ++a)
        positions.insert(positions.end(), {0.8 * std::cos(TWO_PI * a / 3),
                                           0.8 * std::sin(TWO_PI * a / 3), 0});
    handler.set_atoms({0, 0, 0}, positions);
    handler.set_basis_convention(-1, 0, LIBRPA_ANGULAR_ORDER_ABS_PM,
                                 LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M);
    const int rotations[3][9]{{1, 0, 0, 0, 1, 0, 0, 0, 1},
                              {-1, 1, 0, -1, 0, 0, 0, 0, 1},
                              {0, -1, 0, 1, -1, 0, 0, 0, 1}};
    const int mirror[9]{1, 0, 0, 1, -1, 0, 0, 0, 1};
    std::vector<int> operations, antiunitary;
    for (int a = 0; a != (with_anti ? 2 : 1); ++a)
        for (int r = 0; r != 3; ++r)
        {
            antiunitary.push_back(a);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                {
                    int entry = rotations[r][3 * i + j];
                    if (a)
                    {
                        entry = 0;
                        for (int k = 0; k != 3; ++k)
                            entry += mirror[3 * i + k] * rotations[r][3 * k + j];
                    }
                    operations.push_back(entry);
                }
        }
    handler.set_symmetry_spin_operations(antiunitary.size(), 1, operations.data(), nullptr,
                                         antiunitary.data(), nullptr, 2, 0);
    std::vector<double> kcart;
    for (int x = 0; x != 3; ++x)
        for (int y = 0; y != 3; ++y)
            for (int alpha = 0; alpha != 3; ++alpha)
                kcart.push_back((x * reciprocal[alpha] + y * reciprocal[3 + alpha]) / 3);
    handler.set_kgrids_kvec(3, 3, 1, kcart);
    const auto ds = api::get_dataset_instance(handler);
    initialize_ds_atpairs_local(*ds, LIBRPA_ROUTING_LIBRI);
    if (ibz)
    {
        initialize_symmetry_context(*ds, true);
        assert(ds->symmetry_context.kstars.size() == std::size_t(nstars));
        std::vector<double> representatives;
        std::vector<std::vector<Vector3_Order<double>>> stars;
        for (const auto &star : ds->symmetry_context.kstars)
        {
            const auto cart = star.k_ibz * ds->pbc.G;
            representatives.insert(representatives.end(), {TWO_PI * cart.x, TWO_PI * cart.y, TWO_PI * cart.z});
            stars.emplace_back();
            for (const auto &member : star.members) stars.back().push_back(member.k_bz * ds->pbc.G);
        }
        ds->pbc.set_irreducible_kgrids_kvec(3, 3, 1, representatives, stars);
    }
    std::vector<double> energies(6 * nloaded), occupations(6 * nloaded, 0.0);
    double occupied_max = -1e10, empty_min = 1e10;
    for (const auto &values : spectrum.energies)
    {
        occupied_max = std::max(occupied_max, values[2]);
        empty_min = std::min(empty_min, values[3]);
    }
    assert(empty_min - occupied_max > 1.0);
    for (int k = 0; k != nloaded; ++k)
    {
        const auto frac = ds->pbc.kfrac_list.at(k);
        const int index = 3 * mod3(std::llround(3 * frac.x)) + mod3(std::llround(3 * frac.y));
        std::vector<Complex> up(18), down(18);
        for (int band = 0; band != 6; ++band)
        {
            energies[6 * k + band] = spectrum.energies[index][band];
            occupations[6 * k + band] = band < 3 ? nloaded * ds->pbc.weight_k.at(k) : 0.0;
            for (int a = 0; a != 3; ++a)
            {
                up[3 * band + a] = spectrum.vectors[index](a, band);
                down[3 * band + a] = spectrum.vectors[index](a + 3, band);
            }
        }
        handler.set_wfc_spinor_packed(k, 6, 3, up.data(), down.data());
    }
    handler.set_wg_ekb_efermi(1, nloaded, 6, occupations.data(), energies.data(),
                             0.5 * (occupied_max + empty_min));
    std::size_t restored_blocks = 0, anti_blocks = 0;
    if (symmetry)
    {
        initialize_symmetry_context(*ds, true);
        assert(ds->symmetry_context.count_kstar_members() == 9);
        assert(ds->symmetry_context.count_irreducible_blocks() < 81);
        assert(rspace_symmetry_has_complete_band_space(ds->mf, -1));
        for (const auto &[pair, sectors] : ds->symmetry_context.rspace_sector_stars)
            for (const auto &[R, members] : sectors)
                for (const auto &member : members)
                {
                    ++restored_blocks;
                    if (symmetry_rspace_restore_member_is_antiunitary(ds->symmetry_context, member))
                        ++anti_blocks;
                }
        assert(restored_blocks == 81);
        if (with_anti) assert(anti_blocks > 0);
    }
    const int origin[3]{0, 0, 0};
    const double coefficient = 0.5;
    for (int a = 0; a != 3; ++a)
        if (a % ds->comm_h.nprocs == ds->comm_h.myid)
            handler.set_lri_coeff(LIBRPA_ROUTING_LIBRI, a, a, 1, 1, 1, origin, &coefficient);
    ds->scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds->comm_h.nprocs, true), MPI_COMM_WORLD, nloaded);
    ds->desc_wfc_kb_full = ds->scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds->tfg.reset(24);
    ds->tfg.generate_minimax(0.4, 4.0);
    Chi0 response(ds->mf, ds->basis_wfc, ds->basis_aux, ds->pbc, ds->symmetry_context,
                  ds->tfg, ds->scfk_blacs_ctxt, ds->desc_wfc_kb_full, false, symmetry);
    response.gf_threshold = response.libri_threshold_C = response.libri_threshold_G = 0;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    response.build(LIBRPA_ROUTING_LIBRI, ds->cs_data, ds->atpairs_local, ds->basis_aux,
                   no_shrink, ds->blacs_h);
    const int nf = ds->tfg.get_n_grids(), elements = nf * 9 * 9;
    std::vector<Complex> values(elements, 0.0);
    std::vector<int> owners(elements, 0);
    const auto q_index = [&](const Vector3_Order<double> &q) {
        const auto frac = ds->pbc.latvec * q;
        return 3 * mod3(std::llround(3 * frac.x)) + mod3(std::llround(3 * frac.y));
    };
    for (const auto &[omega, qs] : response.get_chi0_q())
        for (const auto &[q, rows] : qs)
            for (const auto &[i, cols] : rows)
                for (const auto &[j, block] : cols)
                {
                    assert(block.nr == 1 && block.nc == 1);
                    const int index = (ds->tfg.get_freq_index(omega) * 9 + q_index(q)) * 9 + 3 * i + j;
                    values[index] += block(0, 0);
                    ++owners[index];
                }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), elements, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, ds->comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), elements, MPI_INT, MPI_SUM, ds->comm_h.comm);
    double error = 0.0, full_difference = 0.0, odd_norm = 0.0;
    for (const auto &q : response.active_qpoints())
        for (int f = 0; f != nf; ++f)
        {
            const int iq = q_index(q);
            const auto exact = lindhard(spectrum, iq, ds->tfg.get_freq_nodes()[f], 3);
            for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                {
                    const int index = (f * 9 + iq) * 9 + 3 * i + j;
                    assert(owners[index] == 1);
                    error = std::max(error, std::abs(values[index] - exact(i, j)));
                    odd_norm = std::max(odd_norm, std::abs(exact(i, j) - std::conj(exact(j, i))));
                    if (full_reference)
                        full_difference = std::max(full_difference, std::abs(values[index] - full_reference->at(index)));
                }
        }
    if (ds->comm_h.is_root())
        std::cout << "Crystal signed Chi0: symmetry=" << symmetry << " IBZ=" << ibz
                  << " anti=" << with_anti << " loaded_k=" << nloaded
                  << " response_q=" << response.active_qpoints().size()
                  << " restored_R_blocks=" << restored_blocks << " anti_R_blocks=" << anti_blocks
                  << " H_covariance=" << covariance_error << " Lindhard_error=" << error
                  << " full_grid_difference=" << full_difference << " antiHerm_norm=" << odd_norm << std::endl;
    assert(error < 2e-8 && full_difference < 2e-10 && odd_norm > 1e-6);
    if (!symmetry)
    {
        // Charge-reality completion in the original real auxiliary basis is
        // independent of physical TR. It lets later reduced runs compare at
        // every requested q against the native full-k/no-crystal calculation.
        for (int f = 0; f != nf; ++f)
            for (int q = 0; q != 9; ++q)
                for (int entry = 0; entry != 9; ++entry)
                {
                    const int index = (f * 9 + q) * 9 + entry;
                    if (owners[index] == 1) continue;
                    const int opposite = 3 * mod3(-(q / 3)) + mod3(-(q % 3));
                    const int partner = (f * 9 + opposite) * 9 + entry;
                    assert(owners[partner] == 1);
                    values[index] = std::conj(values[partner]);
                }
    }
    return values;
}

} // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    check_decay_metadata(1, 2, 3, 1, -1);
    check_decay_metadata(1, 2, 3, 1, 1);
    check_decay_metadata(1, 2, 2, 2, 0);
    check_decay_metadata(2, 1, -9, 0, 0);
    for (const double bz : {0.0, 0.21})
        compare(run_case(3, 1, bz, false, 0, 3), run_case(3, 1, bz, true, 0, 3));
    // One atom exercises a genuinely non-Hermitian atom-diagonal block and
    // MPI ranks with no atom-pair ownership.
    run_case(1, 1, 0.21, true, 0, 3);
    for (const int naux : {2, 3})
        compare(run_case(1, 1, 0.21, true, 1, naux),
                run_case(1, 1, 0.21, true, 2, naux));
    compare(run_case(3, 3, 0.21, false, 0, 3, false),
            run_case(3, 3, 0.21, false, 0, 3, true));
    compare(run_case(3, 3, 0.21, false, 1, 3), run_case(3, 3, 0.21, false, 2, 3));
    compare(run_case(1, 3, 0.21, false, 1, 2), run_case(1, 3, 0.21, false, 2, 2));
    const auto crystal_full = run_crystal_case(false, false, true, nullptr);
    for (const bool with_anti : {false, true})
    {
        run_crystal_case(true, false, with_anti, &crystal_full);
        run_crystal_case(true, true, with_anti, &crystal_full);
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
}
