#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "../core/epsilon.h"
#include "../core/coulmat.h"
#include "../core/qpoint_view.h"
#include "../io/global_io.h"
#include "../math/lebedev_laikov.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;

namespace
{

void assert_complex_close(const std::complex<double> &actual, const std::complex<double> &expected,
                          const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        assert(false);
    }
}

void test_gamma_s_head_vector(const BlacsCtxtHandler &blacs_h)
{
    const std::vector<double> frequencies{0.1, 1.0}, dielectric{12.0, 4.0};
    assert(interpolate_dielec_func(5, frequencies, dielectric, frequencies) == dielectric);
    assert(interpolate_dielec_func(0, frequencies, dielectric, frequencies) == dielectric);
    AtomicBasis basis(std::vector<std::size_t>{5, 5});
    basis.set_l_shells({{0, 1, 0}, {0, 1, 0}});
    const std::vector<double> expected{0.5, 0, 0, 0, -0.5, 0.5, 0, 0, 0, -0.5};
    matrix_m<std::complex<double>> dense(10, 10, MAJOR::COL);
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i)
            dense(i, j) = (i == j ? (expected[i] == 0 ? 20.0 : 1.0) : 0.0)
                          + 6.0 * expected[i] * expected[j];
    // The full leading mode is p-like; nonzero s-p wings also make the desired
    // zero-padded vector different from any full Coulomb eigenvector.
    dense(0, 1) = dense(1, 0) = 0.2;
    const auto serial = compute_gamma_s_head_vector(basis, dense);
    ArrayDesc desc(blacs_h);
    desc.init(10, 10, 2, 2, 0, 0);
    auto local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int j = 0; j < desc.n_loc(); ++j)
        for (int i = 0; i < desc.m_loc(); ++i)
        {
            const int gi = desc.indx_l2g_r(i), gj = desc.indx_l2g_c(j);
            // Distributed Coulomb assembly only guarantees the upper triangle.
            local(i, j) = gi <= gj ? dense(gi, gj) : std::complex<double>{99.0, 0.0};
        }
    const auto parallel = compute_gamma_s_head_vector(basis, local, &desc);
    for (const auto &actual : {serial, parallel})
    {
        double overlap = 0.0, norm = 0.0;
        for (int i = 0; i < 10; ++i)
        {
            overlap += actual[i] * expected[i];
            norm += actual[i] * actual[i];
            if (expected[i] == 0.0) assert(actual[i] == 0.0);
        }
        assert(std::abs(norm - 1.0) < 1e-12);
        assert(std::abs(std::abs(overlap) - 1.0) < 1e-12);
    }
    for (bool missing_metadata : {false, true})
    {
        AtomicBasis invalid(std::vector<std::size_t>{3});
        if (!missing_metadata) invalid.set_l_shells({{1}});
        matrix_m<std::complex<double>> matrix(3, 3, MAJOR::COL);
        bool rejected = false;
        try { compute_gamma_s_head_vector(invalid, matrix); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
}

void test_head_only_trace_logdet_can_use_reduced_response(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(3, 3, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>{0.02 * (i + 1) + 0.01 * (j + 1), 0.002 * (i - j)};
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);

    const matrix_m<std::complex<double>> dense_response(
        std::vector<std::vector<std::complex<double>>>{
            {{0.03, 0.0}, {0.04, -0.002}, {0.05, -0.004}},
            {{0.05, 0.002}, {0.06, 0.0}, {0.07, -0.002}},
            {{0.07, 0.004}, {0.08, 0.002}, {0.09, 0.0}}},
        MAJOR::COL);
    std::complex<double> trace = 0.0;
    for (int i = 0; i != 3; ++i)
    {
        trace += dense_response(i, i);
    }
    const auto a = 1.0 - dense_response(0, 0);
    const auto b = -dense_response(0, 1);
    const auto c = -dense_response(0, 2);
    const auto d = -dense_response(1, 0);
    const auto e = 1.0 - dense_response(1, 1);
    const auto f = -dense_response(1, 2);
    const auto g = -dense_response(2, 0);
    const auto h = -dense_response(2, 1);
    const auto i = 1.0 - dense_response(2, 2);
    const auto det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const auto expected = trace + std::log(det);

    assert_complex_close(actual, expected, 1e-12);

    ArrayDesc desc_full(blacs_h);
    desc_full.init_square_blk(5, 5, 0, 0);
    auto full_response = init_local_mat<std::complex<double>>(desc_full, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc_full.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc_full.indx_g2l_c(j);
            if (jlo < 0) continue;
            full_response(ilo, jlo) = dense_response(i, j);
        }
    }
    const auto full_actual =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(full_response, desc_full);

    assert_complex_close(full_actual, actual, 1e-12);
}

void test_trace_logdet_accounts_for_complex_row_pivot(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(2, 2, 0, 0);

    // response = I - A, where A = [[1, 2i], [-2i, 5]] is positive definite
    // with det(A) = 1. PZGETRF must pivot the first row because |-2i| > |1|.
    const std::complex<double> response_dense[2][2] = {{{0.0, 0.0}, {0.0, -2.0}},
                                                       {{0.0, 2.0}, {-4.0, 0.0}}};
    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 2; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 2; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo >= 0) response(ilo, jlo) = response_dense[i][j];
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);
    assert_complex_close(actual, {-4.0, 0.0}, 1e-12);
}

void add_scalar_wq_block(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const atom_t atom_i,
    const atom_t atom_j,
    const Vector3_Order<double> &q,
    const std::complex<double> value)
{
    auto &block = wq[atom_i][atom_j][q];
    block = matrix_m<std::complex<double>>(1, 1, MAJOR::ROW);
    block(0, 0) = value;
}

librpa_int::symmetry_atom_block_matrix_map_t scalar_wq_to_blocks(
    const std::map<atom_t, std::map<atom_t, std::complex<double>>> &values)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    for (const auto &[atom_i, row] : values)
    {
        for (const auto &[atom_j, value] : row)
        {
            blocks[atom_i][atom_j] = ComplexMatrix(1, 1);
            blocks[atom_i][atom_j](0, 0) = value;
        }
    }
    return blocks;
}

void add_scalar_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const Vector3_Order<double> &q,
    const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, block] : row)
        {
            add_scalar_wq_block(wq, atom_i, atom_j, q, block(0, 0));
        }
    }
}

PeriodicBoundaryData make_wq_full_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0};
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_wq_reduced_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0};
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}};
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);
    return pbc;
}

SymmetryContext make_two_atom_inversion_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}, {1, 0}},
        {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};

    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    inversion.translation = {0.0, 0.0, 0.0};

    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

void assert_wq_rspace_maps_close(
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &actual,
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &expected)
{
    for (const auto &[atom_i, expected_row] : expected)
    {
        assert(actual.count(atom_i) != 0);
        for (const auto &[atom_j, expected_Rs] : expected_row)
        {
            assert(actual.at(atom_i).count(atom_j) != 0);
            for (const auto &[R, expected_block] : expected_Rs)
            {
                assert(actual.at(atom_i).at(atom_j).count(R) != 0);
                const auto &actual_block = actual.at(atom_i).at(atom_j).at(R);
                if (std::abs(actual_block(0, 0) - expected_block(0, 0)) >= 1e-12)
                {
                    std::cerr << "atom_pair=(" << atom_i << "," << atom_j << ") R=("
                              << R.x << "," << R.y << "," << R.z << ")" << std::endl;
                }
                assert_complex_close(actual_block(0, 0), expected_block(0, 0), 1e-12);
            }
        }
    }
}

void test_wq_to_wr_symmetry_reduced_q_matches_full_bz()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
    add_scalar_wq_blocks(wq_sym, q_gamma_sym, gamma_blocks);
    add_scalar_wq_blocks(wq_sym, q_rep_sym, rep_blocks);
    const double symmetry_collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
    {
        for (auto &[atom_j, q_blocks] : row)
        {
            for (auto &[q, block] : q_blocks)
            {
                block *= symmetry_collective_scale;
            }
        }
    }

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

/*!
 * Single-atom s-orbital toy with an explicit antiunitary-only "magnetic"
 * table {E, Theta}: the q-star of q=1/3 gains the member -1/3 through Theta
 * (k -> -k, no spatial rotation), so W(-1/3) must be the plain complex
 * conjugate of W(1/3). Verifies that the member time_reversal flag produced
 * by the explicit spin-operation table drives the conjugation in the
 * symmetry W(q)->W(R) accumulation.
 */
SymmetryContext make_single_atom_antiunitary_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}},
        {{0, {0.0, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};
    ctx.set_rspace_operations({identity});

    SymmetrySpinOperation spin_unitary;  // E, identity spin
    SymmetrySpinOperation spin_anti;     // Theta: k -> -k, conjugation
    spin_anti.antiunitary = true;
    ctx.set_symmetry_spin_operations({spin_unitary, spin_anti}, false);

    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

void test_wq_to_wr_antiunitary_member_conjugates()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_single_atom_antiunitary_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1});
    basis_abf.set_l_shells({{0}});

    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({{0, {{0, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({{0, {{0, {2.1, -0.7}}}}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    add_scalar_wq_blocks(wq_sym, q_gamma_sym, gamma_blocks);
    add_scalar_wq_blocks(wq_sym, q_rep_sym, rep_blocks);
    const double symmetry_collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
    {
        for (auto &[atom_j, q_blocks] : row)
        {
            for (auto &[q, block] : q_blocks)
            {
                block *= symmetry_collective_scale;
            }
        }
    }

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
    // Theta maps q=1/3 to q=-1/3 with conjugation and no orbital rotation.
    const auto tr_blocks = scalar_wq_to_blocks({{0, {{0, {2.1, 0.7}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), tr_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

void test_wq_to_wr_symmetry_collective_handles_empty_local_rank()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        const auto gamma_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
            {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks({
            {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
            {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
            {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(actual.empty());
    }
}

Matz dense_wq_from_scalar_blocks(const librpa_int::symmetry_atom_block_matrix_map_t &blocks,
                                 const ArrayDesc &desc)
{
    Matz mat(desc.m_loc(), desc.n_loc(), MAJOR::COL);
    for (int i_local = 0; i_local < desc.m_loc(); ++i_local)
    {
        const int atom_i = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local < desc.n_loc(); ++j_local)
        {
            const int atom_j = desc.indx_l2g_c(j_local);
            mat(i_local, j_local) = blocks.at(static_cast<atom_t>(atom_i))
                                        .at(static_cast<atom_t>(atom_j))(0, 0);
        }
    }
    return mat;
}

void assert_dense_wq_rspace_maps_close(
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &actual,
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &expected)
{
    for (const auto &[freq, expected_Rs] : expected)
    {
        assert(actual.count(freq) != 0);
        for (const auto &[R, expected_mat] : expected_Rs)
        {
            assert(actual.at(freq).count(R) != 0);
            const auto &actual_mat = actual.at(freq).at(R);
            assert(actual_mat.nr() == expected_mat.nr() && actual_mat.nc() == expected_mat.nc());
            double max_abs = 0.0;
            for (int i = 0; i < expected_mat.nr(); ++i)
            {
                for (int j = 0; j < expected_mat.nc(); ++j)
                {
                    max_abs = std::max(max_abs, std::abs(actual_mat(i, j) - expected_mat(i, j)));
                }
            }
            if (max_abs >= 1e-12)
            {
                std::cerr << "freq=" << freq << " R=(" << R.x << "," << R.y << "," << R.z
                          << ") max_abs=" << max_abs << std::endl;
                for (int i = 0; i < expected_mat.nr(); ++i)
                {
                    for (int j = 0; j < expected_mat.nc(); ++j)
                    {
                        std::cerr << "  (" << i << "," << j << ") actual="
                                  << actual.at(freq).at(R)(i, j) << " expected="
                                  << expected_mat(i, j) << " diff="
                                  << actual_mat(i, j) - expected_mat(i, j) << std::endl;
                    }
                }
            }
            assert(max_abs < 1e-12);
        }
    }
}

void test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);
    const auto initial_view = build_symmetry_qpoint_view(ctx, pbc_sym, true);
    assert(initial_view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    ArrayDesc ad_Wc(blacs_h);
    ad_Wc.init(2, 2, 2, 2, 0, 0); // includes empty local owners at four ranks

    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    const auto gamma = pbc_sym.klist[0], plus = pbc_sym.klist[1];
    const auto &members = initial_view.members.at(plus);
    const auto minus = *std::find_if(members.begin(), members.end(),
                                    [&](const auto &q) { return q != plus; });
    constexpr double freq = 0.25;

    for (const auto major : {MAJOR::COL, MAJOR::ROW})
    {
        const auto local = [&](const symmetry_atom_block_matrix_map_t &blocks)
        {
            Matz result(ad_Wc.m_loc(), ad_Wc.n_loc(), major);
            for (int i = 0; i != ad_Wc.m_loc(); ++i)
                for (int j = 0; j != ad_Wc.n_loc(); ++j)
                    result(i, j) = blocks.at(ad_Wc.indx_l2g_r(i))
                                       .at(ad_Wc.indx_l2g_c(j))(0, 0);
            return result;
        };
        const auto assert_raw = [&](const Matz &matrix,
                                    const symmetry_atom_block_matrix_map_t &blocks)
        {
            assert(matrix.major() == major);
            assert(matrix.nr() == ad_Wc.m_loc() && matrix.nc() == ad_Wc.n_loc());
            for (int i = 0; i != ad_Wc.m_loc(); ++i)
                for (int j = 0; j != ad_Wc.n_loc(); ++j)
                    assert(matrix(i, j) == blocks.at(ad_Wc.indx_l2g_r(i))
                                                .at(ad_Wc.indx_l2g_c(j))(0, 0));
        };
        for (const bool reversed_source : {false, true})
        {
            auto view = initial_view;
            const auto source = reversed_source ? minus : plus;
            if (reversed_source)
            {
                std::replace(view.representatives.begin(), view.representatives.end(), plus, minus);
                view.members[minus] = view.members.at(plus);
                view.members.erase(plus);
            }
            std::map<double, std::map<Vector3_Order<double>, Matz>> wq_sym, wq_full;
            wq_sym[freq][gamma] = local(gamma_blocks);
            wq_sym[freq][source] = local(reversed_source ? inversion_minus_blocks : rep_blocks);
            wq_full[freq][pbc_full.klist[0]] = local(gamma_blocks);
            wq_full[freq][pbc_full.klist[1]] = local(rep_blocks);
            wq_full[freq][pbc_full.klist[2]] = local(inversion_minus_blocks);

            // The three full-q matrices above are explicit physical inputs,
            // independent of the symmetry restore and its local memory layout.
            std::map<double, std::map<Vector3_Order<int>, Matz>> expected;
            for (const auto &R : pbc_full.Rlist)
            {
                auto &matrix = expected[freq][R];
                matrix = Matz(ad_Wc.m_loc(), ad_Wc.n_loc(), MAJOR::COL);
                for (int i = 0; i != ad_Wc.m_loc(); ++i)
                    for (int j = 0; j != ad_Wc.n_loc(); ++j)
                        for (const auto &[q, sample] : wq_full.at(freq))
                            matrix(i, j) += sample(i, j)
                                * std::exp(cplxdb(0, -TWO_PI * (q * (R * pbc_full.latvec)))) / 3.;
            }
            const auto full_actual = FT_Wc_freq_q(
                global::mpi_comm_global_h, wq_full, pbc_full, false);
            const auto actual = FT_Wc_freq_q(
                global::mpi_comm_global_h, wq_sym, pbc_sym, false,
                &view, &ctx, &basis_abf, &ad_Wc);
            assert_dense_wq_rspace_maps_close(full_actual, expected);
            assert_dense_wq_rspace_maps_close(actual, expected);
            for (const auto &R : pbc_full.Rlist)
            {
                assert(full_actual.at(freq).at(R).major() == major);
                assert(actual.at(freq).at(R).major() == major);
            }
            assert(wq_sym.size() == 1 && wq_sym.at(freq).size() == 2);
            assert(wq_full.size() == 1 && wq_full.at(freq).size() == 3);
            assert_raw(wq_sym.at(freq).at(gamma), gamma_blocks);
            assert_raw(wq_sym.at(freq).at(source),
                       reversed_source ? inversion_minus_blocks : rep_blocks);
            assert_raw(wq_full.at(freq).at(pbc_full.klist[0]), gamma_blocks);
            assert_raw(wq_full.at(freq).at(pbc_full.klist[1]), rep_blocks);
            assert_raw(wq_full.at(freq).at(pbc_full.klist[2]), inversion_minus_blocks);
        }
    }
}

void test_dynamic_antiunitary_wq_uses_transpose(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc = make_wq_reduced_pbc();
    const auto full_pbc = make_wq_full_pbc();
    auto context = make_single_atom_antiunitary_context(pbc);
    const auto initial_view = build_symmetry_qpoint_view(context, pbc, true);
    AtomicBasis basis(std::vector<std::size_t>{2});
    basis.set_l_shells({{0, 0}});
    ArrayDesc descriptor(blacs_h);
    descriptor.init(2, 2, 2, 2, 0, 0); // includes empty local owners at four ranks
    const Matz representative({{{1.1, .3}, {.4, .7}}, {{-.2, .1}, {.9, -.2}}}, MAJOR::COL);
    const auto partner = representative.get_transpose(false);
    const auto gamma = pbc.klist[0], plus = pbc.klist[1];
    const auto &members = initial_view.members.at(plus);
    const auto minus = *std::find_if(members.begin(), members.end(),
                                    [&](const auto &q) { return q != plus; });
    double local_error = 0.0;
    for (const auto major : {MAJOR::COL, MAJOR::ROW})
    {
        const auto local = [&](const Matz &dense)
        {
            Matz result(descriptor.m_loc(), descriptor.n_loc(), major);
            for (int i = 0; i != descriptor.m_loc(); ++i)
                for (int j = 0; j != descriptor.n_loc(); ++j)
                    result(i, j) = dense(descriptor.indx_l2g_r(i), descriptor.indx_l2g_c(j));
            return result;
        };
        for (const bool reversed_source : {false, true})
        {
            auto view = initial_view;
            const auto source = reversed_source ? minus : plus;
            if (reversed_source)
            {
                std::replace(view.representatives.begin(), view.representatives.end(), plus, minus);
                view.members[minus] = view.members.at(plus);
                view.members.erase(plus);
            }
            std::map<Vector3_Order<double>, Matz> input;
            input[gamma] = Matz(descriptor.m_loc(), descriptor.n_loc(), major);
            input[source] = local(reversed_source ? partner : representative);
            const auto output = restore_symmetry_dense_wq_map(input, pbc, view, context, basis,
                                                              descriptor);
            assert(output.size() == 3);
            for (const auto &q : {plus, minus})
            {
                const auto &expected = q == plus ? representative : partner;
                const auto &matrix = output.at(q);
                assert(matrix.major() == major);
                assert(matrix.nr() == descriptor.m_loc() && matrix.nc() == descriptor.n_loc());
                for (int i = 0; i != descriptor.m_loc(); ++i)
                    for (int j = 0; j != descriptor.n_loc(); ++j)
                        local_error = std::max(local_error, std::abs(matrix(i, j)
                            - expected(descriptor.indx_l2g_r(i), descriptor.indx_l2g_c(j))));
            }
            assert(output.at(gamma).major() == major);

            std::map<double, std::map<Vector3_Order<double>, Matz>> wq;
            constexpr double frequency = .25;
            wq[frequency] = input;
            const auto realspace = FT_Wc_freq_q(global::mpi_comm_global_h, wq, pbc, false,
                                               &view, &context, &basis, &descriptor);
            for (const auto &R : full_pbc.Rlist)
            {
                const auto &matrix = realspace.at(frequency).at(R);
                assert(matrix.major() == major);
                for (int i = 0; i != descriptor.m_loc(); ++i)
                    for (int j = 0; j != descriptor.n_loc(); ++j)
                    {
                        const int row = descriptor.indx_l2g_r(i), col = descriptor.indx_l2g_c(j);
                        const auto expected = (
                            representative(row, col) * std::exp(cplxdb(
                                0, -TWO_PI * (full_pbc.klist[1] * (R * full_pbc.latvec))))
                            + partner(row, col) * std::exp(cplxdb(
                                0, -TWO_PI * (full_pbc.klist[2] * (R * full_pbc.latvec))))) / 3.;
                        local_error = std::max(local_error, std::abs(matrix(i, j) - expected));
                    }
            }
            assert(input.size() == 2 && wq.at(frequency).size() == 2);
            for (const auto *raw : {&input, &wq.at(frequency)})
            {
                assert(raw->at(gamma).major() == major && raw->at(source).major() == major);
                for (int i = 0; i != descriptor.m_loc(); ++i)
                    for (int j = 0; j != descriptor.n_loc(); ++j)
                    {
                        const int row = descriptor.indx_l2g_r(i), col = descriptor.indx_l2g_c(j);
                        assert(raw->at(gamma)(i, j) == cplxdb(0, 0));
                        assert(raw->at(source)(i, j)
                               == (reversed_source ? partner : representative)(row, col));
                    }
            }
        }
    }
    double error = 0.0;
    MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    assert(error < 1e-12);
    // This is a transformation-law control on a general matrix. It does not
    // assert that a pure-Theta material can have a nonreciprocal charge kernel.
    assert(std::abs(partner(0, 1) - std::conj(representative(0, 1))) > .5);
}

void test_atom_block_frequency_antiunitary_transport()
{
    const auto full = make_wq_full_pbc(), reduced = make_wq_reduced_pbc();
    const auto context = make_single_atom_antiunitary_context(reduced);
    AtomicBasis basis({2});
    basis.set_l_shells({{0, 0}});
    const Matz sample({{{1.1, .3}, {.4, .7}}, {{-.2, .1}, {.9, -.2}}}, MAJOR::ROW);
    using Blocks = atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old;
    Blocks representatives;
    const auto &comm = global::mpi_comm_global_h;
    if (comm.is_root())
    {
        representatives[0][0][reduced.klist[0]] = Matz(2, 2, MAJOR::ROW);
        representatives[0][0][reduced.klist[1]] = sample.copy();
    }
    const TFGrids grids;
    const auto result = FT_Wc_q2R(comm, basis, context, representatives, grids, reduced,
                                  reduced.Rlist, true, "", true, true);
    double local_error = 0;
    for (const auto &[left, rights] : result)
        for (const auto &[right, blocks] : rights)
            for (const auto &[R, block] : blocks)
                for (int row = 0; row < 2; ++row)
                    for (int col = 0; col < 2; ++col)
                    {
                        const auto phase = [&](int iq) {
                            return std::exp(cplxdb(0, -TWO_PI * (full.klist[iq] * (R * full.latvec)))) / 3.;
                        };
                        // Algebraic dynamic antiunitary law; not a claim that
                        // a pure-Theta material has a nonreciprocal kernel.
                        const auto expected = sample(row, col) * phase(1) + sample(col, row) * phase(2);
                        local_error = std::max(local_error, std::abs(block(row, col) - expected));
                    }
    double error = 0;
    comm.allreduce(&local_error, &error, 1, MPI_MAX);
    assert(error < 1e-12);
}

void test_signed_screened_interaction_inverse(const BlacsCtxtHandler &blacs_h)
{
    TFGrids grids(24);
    grids.generate_minimax(.4286910870734395, 3.021313235509797, 1e-12, true);
    constexpr double delta = .8;
    double local_error = 0., local_old_error = 0.;
    for (const int nk : {1, 3})
        for (const int block_size : {1, 2})
            for (const bool complex_auxiliary_gauge : {false, true})
        {
            PeriodicBoundaryData pbc;
            pbc.set_latvec({1., 0., 0., 0., 1., 0., 0., 0., 1.});
            std::vector<double> points;
            for (int q = 0; q != nk; ++q)
                points.insert(points.end(), {TWO_PI * q / nk, 0., 0.});
            pbc.set_kgrids_kvec(nk, 1, 1, points);
            ArrayDesc descriptor(blacs_h);
            descriptor.init(2, 2, block_size, block_size, 0, 0);
            const auto residue = [](double k)
            {
                const double angle = TWO_PI * k;
                const cplxdb u[2]{{1., 0.}, {.2 + .13 * std::cos(angle),
                                            .17 + .12 * std::sin(angle)}};
                Matz a(2, 2, MAJOR::COL);
                for (int i = 0; i != 2; ++i)
                    for (int j = 0; j != 2; ++j) a(i, j) = u[i] * std::conj(u[j]);
                return a;
            };
            Matz gauge({{{std::cos(.31), 0.}, {0., std::sin(.31)}},
                         {{0., std::sin(.31)}, {std::cos(.31), 0.}}}, MAJOR::COL);
            const auto in_gauge = [&](const Matz &matrix)
            {
                return complex_auxiliary_gauge ? gauge * matrix * transpose(gauge, true)
                                               : matrix.copy();
            };
            // The complex-gauge case has complex F(R,iw). It distinguishes
            // the exact negative-R adjoint from a transpose justified only
            // after enforcing entrywise charge reality in a real basis.
            std::map<double, std::map<Vector3_Order<double>, Matz>> input;
            for (const auto omega : grids.get_freq_nodes())
                for (const auto &q : pbc.klist)
                {
                    auto a = in_gauge(residue(q.x)), b = in_gauge(residue(-q.x).get_transpose(false));
                    auto matrix = init_local_mat<cplxdb>(descriptor, MAJOR::COL);
                    for (int i = 0; i != descriptor.m_loc(); ++i)
                        for (int j = 0; j != descriptor.n_loc(); ++j)
                        {
                            const int row = descriptor.indx_l2g_r(i), col = descriptor.indx_l2g_c(j);
                            matrix(i, j) = a(row, col) / cplxdb(delta, -omega) +
                                           b(row, col) / cplxdb(delta, omega);
                        }
                    input[omega][q] = std::move(matrix);
                }
            auto old_input = input;
            const auto legacy = CT_FT_Wc_freq_q(global::mpi_comm_global_h, old_input, pbc,
                                                grids, true, false, 0, -1, false, ".",
                                                &descriptor);
            const auto actual = CT_FT_Wc_freq_q(global::mpi_comm_global_h, input, pbc,
                                                grids, true, false, 0, -1, false, ".",
                                                &descriptor, nullptr, nullptr, nullptr, true);
            assert(input.empty());
            AtomicBasis atom_basis(std::vector<std::size_t>{1, 1});
            std::map<double, atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old>
                atom_input;
            const auto &comm = global::mpi_comm_global_h;
            for (const auto omega : grids.get_freq_nodes())
                for (const auto &q : pbc.klist)
                {
                    const auto a = in_gauge(residue(q.x));
                    const auto b = in_gauge(residue(-q.x).get_transpose(false));
                    for (int i = 0; i != 2; ++i)
                        for (int j = 0; j != 2; ++j)
                            if ((2 * std::min(i, j) + std::max(i, j)) % comm.nprocs == comm.myid)
                            {
                                Matz block(1, 1, MAJOR::ROW);
                                block(0, 0) = a(i, j) / cplxdb(delta, -omega) +
                                              b(i, j) / cplxdb(delta, omega);
                                atom_input[omega][i][j][q] = std::move(block);
                            }
                }
            const auto atom_output = CT_Wc_freq2time_q(comm, atom_basis, atom_input, grids,
                                                        nk, pbc.Rlist, pbc.klist, true);
            for (const auto &[tau, atoms] : atom_output)
                for (const auto &[i, rights] : atoms)
                    for (const auto &[j, matrices] : rights)
                    {
                        assert(i <= j); // time-only Hermitian contract after general inversion
                        for (const auto &[q, value] : matrices)
                            local_error = std::max(local_error,
                                std::abs(value(0, 0) - in_gauge(residue(q.x))(i, j) *
                                                       std::exp(-delta * tau)));
                    }
            for (const auto tau : grids.get_time_nodes())
                for (const auto &r : pbc.Rlist)
                {
                    Matz exact(2, 2, MAJOR::COL);
                    for (const auto &q : pbc.klist)
                        exact += in_gauge(residue(q.x)) * (std::polar(std::exp(-delta * tau) / nk,
                                                           -TWO_PI * q.x * r.x));
                    for (int i = 0; i != descriptor.m_loc(); ++i)
                        for (int j = 0; j != descriptor.n_loc(); ++j)
                        {
                            const auto expected = exact(descriptor.indx_l2g_r(i), descriptor.indx_l2g_c(j));
                            local_error = std::max(local_error, std::abs(actual.at(tau).at(r)(i, j) - expected));
                            local_old_error = std::max(local_old_error, std::abs(legacy.at(tau).at(r)(i, j) - expected));
                        }
                }
        }
    double errors[2]{local_error, local_old_error}, maxima[2]{};
    MPI_Allreduce(errors, maxima, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (global::mpi_comm_global_h.is_root())
        std::cout << "Signed W inverse: error=" << maxima[0] << " old_cosine_error=" << maxima[1] << std::endl;
    assert(maxima[0] < 1e-6 && maxima[1] > .01);
}

void test_dense_wc_real_ct_matches_legacy_real_part()
{
    const auto pbc = make_wq_full_pbc();
    TFGrids tfg(2);
    tfg.generate_evenspaced_tf(0.1, 0.2, 0.3, 0.4);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    const auto freq_nodes = tfg.get_freq_nodes();
    for (std::size_t ifreq = 0; ifreq != freq_nodes.size(); ++ifreq)
    {
        for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
        {
            Matz mat(2, 2, MAJOR::COL);
            for (std::size_t i = 0; i != mat.size(); ++i)
            {
                const double real = 1.0 + ifreq + 0.1 * i;
                const double imag = 0.2 + 0.05 * i;
                if (iq == 0)
                    mat.ptr()[i] = {real, 0.0};
                else if (iq == 1)
                    mat.ptr()[i] = {real, imag};
                else
                    mat.ptr()[i] = {real, -imag};
            }
            input[freq_nodes[ifreq]][pbc.klist[iq]] = std::move(mat);
        }
    }

    auto legacy_input = input;
    auto real_input = input;
    const auto legacy = librpa_int::CT_FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, legacy_input, pbc, tfg, true);
    const auto actual = librpa_int::CT_FT_Wc_freq_q_real(
        librpa_int::global::mpi_comm_global_h, real_input, pbc, tfg);
    assert(legacy_input.empty());
    assert(real_input.empty());

    for (const auto &[tau, expected_Rs] : legacy)
    {
        assert(actual.count(tau) != 0);
        for (const auto &[R, expected] : expected_Rs)
        {
            assert(actual.at(tau).count(R) != 0);
            const auto &result = actual.at(tau).at(R);
            assert(result.major() == expected.major());
            assert(result.size() == expected.size());
            for (std::size_t i = 0; i != result.size(); ++i)
                assert(std::abs(result.ptr()[i] - expected.ptr()[i].real()) < 1e-12);
        }
    }
}

void test_dense_wc_real_ct_projects_large_imaginary_residual()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    TFGrids tfg(1);
    tfg.generate_evenspaced_tf(0.1, 0.1, 0.2, 0.1);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    Matz gamma(1, 1, MAJOR::COL);
    gamma(0, 0) = {1.0, 1e-4};
    input[tfg.get_freq_nodes().front()][pbc.klist.front()] = std::move(gamma);

    const auto projected = librpa_int::CT_FT_Wc_freq_q_real(
        librpa_int::global::mpi_comm_global_h, input, pbc, tfg);
    assert(input.empty());
    assert(std::abs(projected.at(tfg.get_time_nodes().front()).at({0, 0, 0})(0, 0) - 1.0) <
           1e-12);
}

namespace q_alias_control
{
using Z=std::complex<double>;
using QB=atom_mapping<std::map<Vector3_Order<double>,Matz>>::pair_t_old;
static const std::array<Vector3_Order<double>,2> positions{{{.25,0,0},{.6,.1,0}}};
static double mx(const ComplexMatrix&a){double r=0;for(int i=0;i<a.size;++i)r=std::max(r,std::abs(a.c[i]));return r;}
static Matrix3 lattice(bool unit){return unit?Matrix3(1,0,0,0,1,0,0,0,1):Matrix3(2.3,.4,0,.2,3.7,.3,.1,.2,4.1);}
static PeriodicBoundaryData full_pbc(bool unit){
 auto a=lattice(unit);PeriodicBoundaryData p;p.set_latvec({a.e11,a.e12,a.e13,a.e21,a.e22,a.e23,a.e31,a.e32,a.e33});
 std::vector<double> ks;for(int i=0;i<3;++i){auto q=Vector3_Order<double>{i/3.,0,0}*p.G;ks.insert(ks.end(),{TWO_PI*q.x,TWO_PI*q.y,TWO_PI*q.z});}
 p.set_kgrids_kvec(3,1,1,ks);return p;
}
static PeriodicBoundaryData reduced_pbc(bool unit){
 auto p=full_pbc(unit);auto q=Vector3_Order<double>{1./3,0,0}*p.G;
 p.set_irreducible_kgrids_kvec(3,1,1,{0,0,0,TWO_PI*q.x,TWO_PI*q.y,TWO_PI*q.z},{{{0,0,0}},{q,-q}});return p;
}
static SymmetryContext context(const PeriodicBoundaryData&p,int ratom){
 SymmetryContext c;c.set_crystal_structure(p.latvec,p.G,{{0,0},{1,0}},{{0,positions[0]},{1,positions[1]}});
 SymmetryOperation e;e.rotation.Identity();e.translation={0,0,0};c.set_rspace_operations({e});
 SymmetrySpinOperation identity,theta;theta.antiunitary=true;c.set_symmetry_spin_operations({identity,theta},false);c.set_available();
 c.build_periodic_mappings(p,p.Rlist);c.build_rsh_rotations({-1,ratom,LIBRPA_ANGULAR_ORDER_NATURAL,LIBRPA_RSH_COEFF_1_M,LIBRPA_RSH_COEFF_1_M},0);c.build_kstar_member_rotations(0);return c;
}
static Vector3_Order<double> shift(const PeriodicBoundaryData&p,int variant){
 if(variant==1)return Vector3_Order<double>{2e-6,-1e-6,.5e-6}*p.G;
 if(variant==2||variant==4||variant==5)return Vector3_Order<double>{1,0,0}*p.G;
 if(variant==3)return {1,0,0};
 return {0,0,0};
}
static Z phase(int atom,int ratom){return std::polar(1.,TWO_PI*ratom*positions[atom].x);}
static ComplexMatrix square(int iq){ComplexMatrix a(2,2);a(0,0)=iq?2.1:1.5;a(1,1)=iq?1.4:2.2;a(0,1)=iq?Z(-.7,.5):Z(.4,0);a(1,0)=std::conj(a(0,1));return a;}
static ComplexMatrix sewn_square(const ComplexMatrix&a,int ratom,int variant){
 auto b=a;if(variant==2)for(int i=0;i<2;++i)for(int j=0;j<2;++j)b(i,j)=phase(i,ratom)*a(i,j)*std::conj(phase(j,ratom));return b;
}
// Both V(R) and W(R) use the localized-cell frame. Reciprocal restoration
// and unfolding retain the raw operator frame before the Fourier boundary.
// Use the literal fractional mesh and independently de-embed every sample.
static ComplexMatrix expected_cell_R(const Vector3_Order<int>& r, bool false_point, int ratom)
{
    ComplexMatrix result(2, 2);
    const auto zero = square(0), positive = square(1);
    for (int iq = 0; iq != (false_point ? 1 : 3); ++iq)
    {
        const Vector3_Order<double> qfrac{iq / 3.0, 0.0, 0.0};
        const auto fourier = std::polar(1.0 / 3.0, -TWO_PI * (qfrac * r));
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
            {
                auto raw = iq == 0 ? zero(i, j) : positive(i, j);
                // The -1/3 partner is stored at +2/3, with reciprocal
                // sewing exp[2*pi*i*eta*(tau_i-tau_j)] after conjugation.
                if (iq == 2)
                    raw = std::polar(1.0, TWO_PI * ratom * (positions[i].x - positions[j].x))
                        * std::conj(positive(i, j));
                const auto to_cell = std::polar(
                    1.0, TWO_PI * ratom * (qfrac * (positions[j] - positions[i])));
                result(i, j) += fourier * to_cell * raw;
            }
    }
    return result;
}
struct Outcome{bool accepted=false;double error=0,norm=0;std::string exception;};
#ifdef LIBRPA_USE_LIBRI
static Outcome unfold(const BlacsCtxtHandler&blacs,bool unit,int ratom,int variant){
 auto p=full_pbc(unit);auto c=context(p,ratom);const TFGrids grids;AtomicBasis active(std::vector<std::size_t>{1,1}),parent(std::vector<std::size_t>{2,1});
 MeanField mf(1,3,2,2);KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1,blacs.nprocs),MPI_COMM_WORLD,3);
 ArrayDesc desc(blacs);desc.init(2,2,1,1,0,0);Chi0 helper(mf,active,active,p,c,grids,kctx,desc,false,false);
 std::map<Vector3_Order<double>,ComplexMatrix> u,expected;QB w;const auto d=shift(p,variant);
 for(int k=0;k<3;++k){ComplexMatrix a(2,3),s(2,3),b(2,2),e(3,3);
  for(int i=0;i<2;++i)for(int j=0;j<3;++j){a(i,j)=Z(.4+.1*i+.07*j+.03*k,.13*(i-j)+.09*k);s(i,j)=a(i,j);}
  if(variant==2||variant==5)for(int i=0;i<2;++i)for(int j=0;j<3;++j)s(i,j)=phase(i,ratom)*a(i,j)*std::conj(phase(j<2?0:1,ratom));
  b(0,0)={1.2+.1*k,.3};b(0,1)={-.2,.4};b(1,0)={.7,-.1};b(1,1)={1.1,-.2-.1*k};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int mu=0;mu<2;++mu)for(int nu=0;nu<2;++nu)e(i,j)+=std::conj(a(mu,i))*b(mu,nu)*a(nu,j);
  u[p.klist[k]+((variant==4)?Vector3_Order<double>{0,0,0}:d)]=s;expected[p.klist[k]]=e;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)if((std::min(i,j)*2+std::max(i,j))%blacs.nprocs==blacs.myid){Matz block(1,1,MAJOR::ROW);block(0,0)=b(i,j);if(variant==4||variant==5)block(0,0)*=phase(i,ratom)*std::conj(phase(j,ratom));w[i][j][p.klist[k]+((variant==4||variant==5)?d:Vector3_Order<double>{0,0,0})]=block;}
 }
 Outcome o;
 try{helper.unfold_abfs_Wc_q(u,w,p.klist,parent,blacs,true);o.accepted=true;
  for(const auto&[i,js]:w)for(const auto&[j,qs]:js)for(const auto&[q,b]:qs){
   int ik=-1;Vector3_Order<int> fold;
   for(int k=0;k<3;++k){auto delta=p.latvec*(q-p.klist[k]);if(nearly_integer_vector(delta,1e-8)){ik=k;fold={int(std::lround(delta.x)),int(std::lround(delta.y)),int(std::lround(delta.z))};}}
   if(ik<0||b.nr()!=parent[i]||b.nc()!=parent[j]){o.error=INFINITY;continue;}
   const auto&target=expected.at(p.klist[ik]);
   for(int a=0;a<b.nr();++a)for(int z=0;z<b.nc();++z){auto value=target(parent.get_global_index(i,a),parent.get_global_index(j,z));
    value*=std::polar(1.,TWO_PI*ratom*(fold.x*(positions[i].x-positions[j].x)+fold.y*(positions[i].y-positions[j].y)+fold.z*(positions[i].z-positions[j].z)));
    o.error=std::max(o.error,std::abs(b(a,z)-value));o.norm=std::max(o.norm,std::abs(b(a,z)));}
  }
 }catch(const std::exception&e){o.exception=e.what();}
 return o;
}
#endif
static Outcome transform(const BlacsCtxtHandler&blacs,bool unit,int ratom,int variant,int route){
 auto p=reduced_pbc(unit);auto c=context(p,ratom);AtomicBasis basis(std::vector<std::size_t>{1,1});basis.set_l_shells({{0},{0}});const TFGrids grids;
 QB w;atpair_k_cplx_mat_t v;const auto d=shift(p,variant);ArrayDesc desc(blacs);desc.init(2,2,1,1,0,0);
 std::map<double,std::map<Vector3_Order<double>,Matz>> dense;
 for(int k=0;k<2;++k){auto a=square(k);auto key=p.klist[k];if(k==1)key+=d;auto s=sewn_square(a,ratom,k==1?variant:0);if(variant==3&&k==1)s*=Z(4,0);
  for(int i=0;i<2;++i)for(int j=0;j<2;++j){Matz b(1,1,MAJOR::ROW);b(0,0)=s(i,j)/double(blacs.nprocs);w[i][j][key]=b;v[i][j][key]=std::make_shared<ComplexMatrix>(1,1);(*v[i][j][key])(0,0)=b(0,0);}
  auto b=init_local_mat<Z>(desc,MAJOR::COL);for(int i=0;i<b.nr();++i)for(int j=0;j<b.nc();++j){int x=desc.indx_l2g_r(i),y=desc.indx_l2g_c(j);if(x<2&&y<2)b(i,j)=s(x,y);}dense[.25][key]=b;
 }
 Outcome o;
 try{
  if(route==0){auto result=FT_Wc_q2R(blacs.comm_h(),basis,c,w,grids,p,p.Rlist,false,"",true);
   for(const auto&[i,js]:result)for(const auto&[j,rs]:js)for(const auto&[r,b]:rs){auto e=expected_cell_R(r,variant==3,ratom);o.error=std::max(o.error,std::abs(b(0,0)-e(i,j)));o.norm=std::max(o.norm,std::abs(b(0,0)));}
  }else if(route==1){bool replicated=false;auto result=FT_Vq(blacs.comm_h(),basis,c,v,p,true,true,&replicated);
   for(const auto&[i,js]:result)for(const auto&[j,rs]:js)for(const auto&[r,b]:rs){auto e=expected_cell_R(r,variant==3,ratom);o.error=std::max(o.error,std::abs((*b)(0,0)-e(i,j).real()));o.norm=std::max(o.norm,std::abs((*b)(0,0)));}
  }else{const auto view=build_symmetry_qpoint_view(c,p,true);auto result=FT_Wc_freq_q(blacs.comm_h(),dense,p,false,&view,&c,&basis,&desc);
   for(const auto&[f,rs]:result)for(const auto&[r,b]:rs){auto e=expected_cell_R(r,variant==3,ratom);for(int i=0;i<b.nr();++i)for(int j=0;j<b.nc();++j){int x=desc.indx_l2g_r(i),y=desc.indx_l2g_c(j);if(x<2&&y<2){o.error=std::max(o.error,std::abs(b(i,j)-e(x,y)));o.norm=std::max(o.norm,std::abs(b(i,j)));}}}
  }o.accepted=true;
 }catch(const std::exception&e){o.exception=e.what();}
 return o;
}

void test(const BlacsCtxtHandler& blacs)
{
    const std::vector<std::tuple<bool,int,int>> cases{{false,0,0},{false,0,1},{false,0,2},{false,0,3},{false,1,0},{false,1,2},{false,-1,2},{true,1,0},{true,1,2},{false,1,4},{false,1,5},{true,1,4},{true,1,5}};

    double max_error = 0.;
    for (auto [unit, ratom, variant] : cases)
        for (int route = 0; route < 4; ++route)
        {
            if (variant > 3 && route > 0) continue;
#ifndef LIBRPA_USE_LIBRI
            if (route == 0) continue;
#endif
            Outcome value;
#ifdef LIBRPA_USE_LIBRI
            if (route == 0) value = unfold(blacs, unit, ratom, variant);
            else
#endif
                value = transform(blacs, unit, ratom, variant, route - 1);
            const bool false_point = variant == 3;
            int bad = false_point
                ? (value.accepted && (route == 0 || value.error >= 2e-12))
                : (!value.accepted || value.error >= 2e-12);
            int any_bad = 0;
            MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, blacs.comm());
            if (any_bad)
            {
                std::cerr << "q alias operator frame: unit=" << unit << " ratom=" << ratom
                          << " variant=" << variant << " route=" << route
                          << " accepted=" << value.accepted << " error=" << value.error
                          << " exception=" << value.exception << std::endl;
            }
            assert(!any_bad);
            if (value.accepted) max_error = std::max(max_error, value.error);
        }
    double error = 0.;
    MPI_Allreduce(&max_error, &error, 1, MPI_DOUBLE, MPI_MAX, blacs.comm());
    if (blacs.myid == 0) std::cout << "Reciprocal-q operator-frame residual: " << error << std::endl;
}
} // namespace q_alias_control

#ifdef LIBRPA_USE_LIBRI

Matz serial_inverse(const Matz &input)
{
    const int n = input.nr();
    auto factor = input.copy();
    Matz inverse(n, n, MAJOR::COL);
    for (int i = 0; i != n; ++i) inverse(i, i) = 1.0;
    for (int k = 0; k != n; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i != n; ++i)
            if (std::abs(factor(i, k)) > std::abs(factor(pivot, k))) pivot = i;
        assert(std::abs(factor(pivot, k)) > 1e-12);
        for (int j = 0; j != n; ++j)
        {
            std::swap(factor(k, j), factor(pivot, j));
            std::swap(inverse(k, j), inverse(pivot, j));
        }
        const auto diagonal = factor(k, k);
        for (int j = 0; j != n; ++j)
        {
            factor(k, j) /= diagonal;
            inverse(k, j) /= diagonal;
        }
        for (int i = 0; i != n; ++i)
        {
            if (i == k) continue;
            const auto multiplier = factor(i, k);
            for (int j = 0; j != n; ++j)
            {
                factor(i, j) -= multiplier * factor(k, j);
                inverse(i, j) -= multiplier * inverse(k, j);
            }
        }
    }
    return inverse;
}

double rectangular_gamma_volume_ratio()
{
    // The fixture's 2x1x1 mesh in a unit reciprocal cube has Gamma box
    // half-widths (1/4, 1/2, 1/2) and volume 1/2. Finite angular quadrature
    // does not give exactly 1 even at zero response; retain that independent
    // head residual rather than changing the production quadrature.
    const auto grid = lebedev_laikov::grid(5810);
    double volume = 0.0;
    for (std::size_t i = 0; i != grid.weights.size(); ++i)
    {
        const double inverse_radius = std::max(
            {4.0 * std::abs(grid.x[i]), 2.0 * std::abs(grid.y[i]), 2.0 * std::abs(grid.z[i])});
        volume +=
            4.0 * PI * grid.weights[i] / (3.0 * inverse_radius * inverse_radius * inverse_radius);
    }
    return volume / 0.5;
}

void verify_gamma_wc_retained_identity(const BlacsCtxtHandler &blacs_h, const bool rank_deficient,
                                       const bool zero_response, const int dielectric_option,
                                       const bool use_cholesky, const int block_size = 1)
{
    constexpr int n = 4;
    const int retained = rank_deficient ? n - 1 : n;
    const std::array<double, n> eigenvalues{4.0, 2.0, 1.0, rank_deficient ? -0.2 : 0.5};
    const Matz eigenvectors({{{0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}},
                             {{0.5, 0.0}, {-0.5, 0.0}, {0.5, 0.0}, {-0.5, 0.0}},
                             {{0.5, 0.0}, {0.5, 0.0}, {-0.5, 0.0}, {-0.5, 0.0}},
                             {{0.5, 0.0}, {-0.5, 0.0}, {-0.5, 0.0}, {0.5, 0.0}}},
                            MAJOR::COL);
    Matz diagonal(n, n, MAJOR::COL), sqrt_diagonal(n, n, MAJOR::COL);
    Matz cut_root_channels(n, n, MAJOR::COL), response_channels(n, n, MAJOR::COL);
    const double cut_diagonal[n]{1.2, 1.1, 1.4, 0.9};
    for (int i = 0; i != n; ++i)
    {
        diagonal(i, i) = eigenvalues[i];
        sqrt_diagonal(i, i) = std::sqrt(std::max(0.0, eigenvalues[i]));
        cut_root_channels(i, i) = cut_diagonal[i];
    }
    // The positive cut root couples retained and discarded bare-Coulomb
    // channels. Projecting the cut interaction too would fail this control.
    cut_root_channels(1, 3) = cut_root_channels(3, 1) = 0.21;
    cut_root_channels(2, 3) = cut_root_channels(3, 2) = -0.13;
    if (!zero_response)
    {
        response_channels(1, 1) = -0.07;
        response_channels(2, 2) = -0.09;
        response_channels(3, 3) = -0.11;
        response_channels(1, 2) = {0.012, 0.009};
        response_channels(2, 1) = {0.012, -0.009};
        response_channels(2, 3) = {0.006, -0.005};
        response_channels(3, 2) = {0.006, 0.005};
    }
    const auto adjoint = transpose(eigenvectors, true);
    const auto bare = eigenvectors * diagonal * adjoint;
    const auto cut_root = eigenvectors * cut_root_channels * adjoint;
    const auto cut = cut_root * cut_root;
    const auto response = eigenvectors * response_channels * adjoint;
    auto dielectric_channels = sqrt_diagonal * response_channels * sqrt_diagonal;
    dielectric_channels *= -1.0;
    for (int i = 0; i != n; ++i) dielectric_channels(i, i) += 1.0;
    const auto inverse = serial_inverse(dielectric_channels);

    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(2, 1, 1, {0.0, 0.0, 0.0, PI, 0.0, 0.0});
    AtomicBasis wfc_basis({2}), aux_basis({n});
    MeanField mf(1, 2, 2, 2);
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
    }
    TFGrids grids(1);
    grids.generate_evenspaced_tf(0.1, 0.1, 0.25, 0.1);
    KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1, blacs_h.nprocs), MPI_COMM_WORLD, 2);
    ArrayDesc desc_wfc(blacs_h), desc_aux(blacs_h);
    desc_wfc.init(2, 2, 1, 1, 0, 0);
    desc_aux.init(n, n, block_size, block_size, 0, 0);
    const SymmetryContext context;
    Chi0 chi0(mf, wfc_basis, aux_basis, pbc, context, grids, kctx, desc_wfc, false, false);
    atpair_k_cplx_mat_t bare_blocks, cut_blocks;
    using ResponseMap =
        std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>;
    // Inject fixed numerical response data in this non-const fixture so the
    // complete production Wc path is exercised without coupling this test to
    // Green-function contraction or a new production input API.
    auto &response_blocks = const_cast<ResponseMap &>(chi0.get_chi0_q());
    if (blacs_h.myid == 0)
    {
        for (const auto &q : pbc.klist)
        {
            auto &v = bare_blocks[0][0][q];
            auto &vc = cut_blocks[0][0][q];
            v = std::make_shared<ComplexMatrix>(n, n);
            vc = std::make_shared<ComplexMatrix>(n, n);
            auto &chi = response_blocks[grids.get_freq_nodes().front()][q][0][0];
            chi.create(n, n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != n; ++j)
                {
                    (*v)(i, j) = bare(i, j);
                    (*vc)(i, j) = cut(i, j);
                    chi(i, j) = response(i, j);
                }
        }
    }
    velocity_matrix_t velocity;
    initialize_velocity_matrix(velocity, 1, 2, 2);
    diele_func headwing(mf, velocity, pbc.kfrac_list, wfc_basis, aux_basis, grids.get_freq_nodes(),
                        2, 2, 1, n, pbc, global::mpi_comm_global_h, blacs_h);
    headwing.init(0.0, bare_blocks);
    headwing.cal_head();
    headwing.init_wing(0.0, bare_blocks);
    headwing.set_0_wing();
    const auto actual =
        compute_Wc_freq_q_blacs(chi0, bare_blocks, cut_blocks, 0.0, true, dielectric_option,
                                std::vector<std::complex<double>>{1.0}, &headwing, blacs_h,
                                desc_aux, false, ".", use_cholesky, false, false);

    for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
    {
        auto expected_channels = inverse.copy();
        if (iq == 0 && dielectric_option == 3)
        {
            // Here velocity and both wings vanish: the optical head is I
            // and decouples from the finite body. Matched bare-head correlation
            // subtraction gives inverse head 1 before subtracting its identity.
            expected_channels(0, 0) = 1.0;
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != n; ++j)
                    if (i >= retained || j >= retained) expected_channels(i, j) = 0.0;
            for (int i = 0; i != retained; ++i) expected_channels(i, i) -= 1.0;
        }
        else
            for (int i = 0; i != n; ++i) expected_channels(i, i) -= 1.0;
        const auto expected = cut_root * eigenvectors * expected_channels * adjoint * cut_root;
        const auto &result = actual.at(grids.get_freq_nodes().front()).at(pbc.klist[iq]);
        double local_error = 0.0;
        for (int i = 0; i != desc_aux.m_loc(); ++i)
            for (int j = 0; j != desc_aux.n_loc(); ++j)
            {
                const auto actual_value = result(i, j);
                const auto expected_value = expected(desc_aux.indx_l2g_r(i),
                                                     desc_aux.indx_l2g_c(j));
                const double error = std::abs(actual_value - expected_value);
                // Reduce a failure on any rank before asserting, including non-finite data.
                if (!std::isfinite(actual_value.real()) || !std::isfinite(actual_value.imag()) ||
                    !std::isfinite(expected_value.real()) ||
                    !std::isfinite(expected_value.imag()) || !std::isfinite(error))
                    local_error = std::numeric_limits<double>::infinity();
                else
                    local_error = std::max(local_error, error);
            }
        double max_error = 0.0;
        MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (blacs_h.myid == 0)
            std::cout << "Gamma Wc dense control: rank=" << retained
                      << " zero_response=" << zero_response << " option=" << dielectric_option
                      << " cholesky=" << use_cholesky << " block_size=" << block_size
                      << " iq=" << iq << " max_error=" << max_error << std::endl;
        assert(max_error < 1e-10);
    }
}

void test_gamma_wc_retained_identity(const BlacsCtxtHandler &blacs_h)
{
    for (const bool rank_deficient : {true, false})
        for (const bool zero_response : {true, false})
            verify_gamma_wc_retained_identity(blacs_h, rank_deficient, zero_response, 3, false);
    verify_gamma_wc_retained_identity(blacs_h, true, false, 3, true);
    verify_gamma_wc_retained_identity(blacs_h, true, false, 1, false);
    // The filtered eigenvector starts inside a block, rather than at its edge.
    verify_gamma_wc_retained_identity(blacs_h, true, false, 3, false, 2);
}

void test_full_frequency_unfold_and_fourier(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc = make_wq_full_pbc();
    const SymmetryContext context;
    const TFGrids grids;
    const auto &comm = blacs_h.comm_h();
    for (const int natom : {1, 2})
    {
        AtomicBasis active(natom == 1 ? std::vector<std::size_t>{2}
                                     : std::vector<std::size_t>{1, 1});
        AtomicBasis parent(natom == 1 ? std::vector<std::size_t>{3}
                                     : std::vector<std::size_t>{2, 1});
        MeanField mf(1, 3, 2, 2);
        KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1, blacs_h.nprocs), MPI_COMM_WORLD, 3);
        ArrayDesc desc(blacs_h);
        desc.init(2, 2, 1, 1, 0, 0);
        Chi0 helper(mf, active, active, pbc, context, grids, kctx, desc, false, false);
        std::map<Vector3_Order<double>, ComplexMatrix> projections, expected_q;
        atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old blocks;
        for (int iq = 0; iq < 3; ++iq)
        {
            const auto q = pbc.klist[iq];
            ComplexMatrix U(2, 3), W(2, 2), expected(3, 3);
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 3; ++col)
                    U(row, col) = cplxdb(.4 + .1 * row + .07 * col, .13 * (row - col) + .09 * iq);
            W(0, 0) = {1.2 + .1 * iq, .3}; W(0, 1) = {-.2, .4};
            W(1, 0) = {.7, -.1}; W(1, 1) = {1.1, -.2 - .1 * iq};
            // Independent four-index contraction, including independent lower data.
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int mu = 0; mu < 2; ++mu)
                        for (int nu = 0; nu < 2; ++nu)
                            expected(a, b) += std::conj(U(mu, a)) * W(mu, nu) * U(nu, b);
            projections[q] = U;
            expected_q[q] = expected;
            for (int left = 0; left < natom; ++left)
                for (int right = 0; right < natom; ++right)
                {
                    const int owner = (std::min(left, right) * natom + std::max(left, right)) % comm.nprocs;
                    if (comm.myid != owner) continue;
                    Matz block(active[left], active[right], MAJOR::ROW);
                    for (int row = 0; row < block.nr(); ++row)
                        for (int col = 0; col < block.nc(); ++col)
                            block(row, col) = W(active.get_global_index(left, row), active.get_global_index(right, col));
                    blocks[left][right][q] = std::move(block);
                }
        }
        helper.unfold_abfs_Wc_q(projections, blocks, pbc.klist, parent, blacs_h, true);
        double local_error = 0;
        for (const auto &[left, rights] : blocks)
            for (const auto &[right, qs] : rights)
                for (const auto &[q, block] : qs)
                    for (int row = 0; row < block.nr(); ++row)
                        for (int col = 0; col < block.nc(); ++col)
                            local_error = std::max(local_error, std::abs(block(row, col) -
                                expected_q.at(q)(parent.get_global_index(left, row), parent.get_global_index(right, col))));
        const auto realspace = FT_Wc_q2R(comm, parent, context, blocks, grids, pbc, pbc.Rlist,
                                         true, "", false, true);
        int local_entries = 0, entries = 0;
        for (const auto &[left, rights] : realspace)
            for (const auto &[right, rs] : rights)
                for (const auto &[R, block] : rs)
                    for (int row = 0; row < block.nr(); ++row)
                        for (int col = 0; col < block.nc(); ++col)
                        {
                            cplxdb expected = 0;
                            for (const auto &[q, sample] : expected_q)
                                expected += sample(parent.get_global_index(left, row), parent.get_global_index(right, col)) *
                                    std::exp(cplxdb(0, -TWO_PI * (q * (R * pbc.latvec)))) / 3.;
                            local_error = std::max(local_error, std::abs(block(row, col) - expected));
                            ++local_entries;
                        }
        double error = 0;
        comm.allreduce(&local_error, &error, 1, MPI_MAX);
        comm.allreduce(&local_entries, &entries, 1, MPI_SUM);
        assert(entries == 27 && error < 1e-12);
    }
}

#endif

}  // namespace

int main(int argc, char *argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io(false, "stdout", false);

    {
        BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
        blacs_h.init();
        blacs_h.set_square_grid();

        test_gamma_s_head_vector(blacs_h);
        test_head_only_trace_logdet_can_use_reduced_response(blacs_h);
        test_trace_logdet_accounts_for_complex_row_pivot(blacs_h);
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_antiunitary_member_conjugates();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_dynamic_antiunitary_wq_uses_transpose(blacs_h);
        test_atom_block_frequency_antiunitary_transport();
        test_signed_screened_interaction_inverse(blacs_h);
        q_alias_control::test(blacs_h);
        test_dense_wc_real_ct_matches_legacy_real_part();
        test_dense_wc_real_ct_projects_large_imaginary_residual();
#ifdef LIBRPA_USE_LIBRI
        test_gamma_wc_retained_identity(blacs_h);
        test_full_frequency_unfold_and_fourier(blacs_h);
#endif
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
