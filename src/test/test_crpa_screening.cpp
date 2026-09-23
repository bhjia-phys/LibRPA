#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../api/dataset.h"
#include "../core/crpa.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;

namespace
{
using Complex = std::complex<double>;

void close(Complex actual, Complex expected, double tolerance)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > tolerance)
    {
        std::cerr << "screening actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("cRPA screening numerical mismatch");
    }
}

void test_distributed_complex_contraction()
{
    Dataset ds{MPI_COMM_WORLD};
    constexpr int naux = 5, nleft = 2, nright = 3;
    ArrayDesc descriptor(ds.blacs_h);
    descriptor.init_square_blk(naux, naux, 0, 0);
    auto local_kernel = init_local_mat<Complex>(descriptor, MAJOR::COL);
    ComplexMatrix kernel(naux, naux), left(nleft * nleft, naux), right(nright * nright, naux);
    for (int i = 0; i < naux; ++i)
        for (int j = 0; j < naux; ++j)
        {
            kernel(i, j) = Complex(0.12 * (i + 1) * (j + 1) + (i == j ? 2.0 : 0.0), 0.17 * (i - j));
            const int row = descriptor.indx_g2l_r(i), column = descriptor.indx_g2l_c(j);
            if (row >= 0 && column >= 0) local_kernel(row, column) = kernel(i, j);
        }
    for (int pair = 0; pair < left.nr; ++pair)
        for (int i = 0; i < naux; ++i)
            left(pair, i) = Complex(0.13 * (pair + 1) + 0.07 * i, 0.09 * (pair - 2 * i));
    for (int pair = 0; pair < right.nr; ++pair)
        for (int i = 0; i < naux; ++i)
            right(pair, i) = Complex(0.11 * (pair + 2) - 0.06 * i, 0.08 * (2 * pair + i));
    const auto actual =
        contract_crpa_blacs(left, right, local_kernel, descriptor, nleft, nright, ds.comm_h);
    for (int a = 0; a < nleft; ++a)
        for (int b = 0; b < nleft; ++b)
            for (int c = 0; c < nright; ++c)
                for (int d = 0; d < nright; ++d)
                {
                    Complex expected = 0;
                    for (int i = 0; i < naux; ++i)
                        for (int j = 0; j < naux; ++j)
                            expected += std::conj(left(b * nleft + a, i)) * kernel(i, j) *
                                        right(c * nright + d, j);
                    close(actual(a * nleft + b, c * nright + d), expected, 2e-12);
                }
}

void test_native_screening_pipeline(bool negative_mode)
{
    Dataset ds{MPI_COMM_WORLD};
    const std::size_t naux = negative_mode ? 2 : 1;
    ds.basis_wfc.set(std::vector<std::size_t>{2});
    ds.basis_aux.set(std::vector<std::size_t>{naux});
    ds.mf.set(1, 4, 2, 2, 1);
    ds.mf.get_efermi() = 0;
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    const double pi = std::acos(-1.0);
    ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 4);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(2, 2, 2, 2);
    ds.desc_abf.reset_handler(ds.blacs_h);
    ds.desc_abf.init_1b1p(naux, naux, 0, 0);
    ds.tfg.reset(16);
    ds.tfg.generate_minimax(0.4, 4.0);

    SiteOrbitalGroup site;
    site.label = "mixed-complex-orbital";
    site.atom_index = 0;
    site.orb_start = 0;
    site.n_orbitals = 1;
    CorrelatedSubspace frame({site}, ds.pbc.kfrac_list, ds.pbc.Rlist, 2, 2, 1);
    for (int k = 0; k < 4; ++k)
    {
        ComplexMatrix eigenvectors(2, 2), overlap(2, 2), trial(2, 1);
        eigenvectors.set_as_identity_matrix();
        overlap.set_as_identity_matrix();
        trial(0, 0) = 1.0 / std::sqrt(2.0);
        trial(1, 0) = Complex(0.0, 1.0 / std::sqrt(2.0));
        ds.mf.get_eigenvectors()[0][0][k] = eigenvectors;
        ds.mf.get_eigenvals()[0](k, 0) = -0.5;
        ds.mf.get_eigenvals()[0](k, 1) = 0.5;
        ds.mf.get_weight()[0](k, 0) = 0.5;
        ds.mf.get_weight()[0](k, 1) = 0;
        frame.set_S_k(k, overlap);
        frame.set_W_k(k, trial);
        frame.build_spin_k(0, k, eigenvectors);
        if (ds.comm_h.is_root())
        {
            auto v = std::make_shared<ComplexMatrix>(naux, naux);
            if (negative_mode)
            {
                // Eigenvalues 2 and -0.4 in a rotated auxiliary basis.
                (*v)(0, 0) = (*v)(1, 1) = 0.8;
                (*v)(0, 1) = (*v)(1, 0) = 1.2;
            }
            else
                (*v)(0, 0) = 2.0;
            ds.vq[0][0][ds.pbc.klist[k]] = v;
        }
    }
    frame.compute_T_R();
    ds.cs_data.use_libri = true;
    if (ds.comm_h.is_root())
    {
        auto data = std::make_shared<std::valarray<double>>(0.0, 4 * naux);
        const double retained[] = {0.15, 0.2, 0.2, 0.4};
        const double discarded[] = {0.3, 0.05, 0.05, 0.1};
        for (int i = 0; i < 4; ++i)
        {
            (*data)[i] =
                negative_mode ? (retained[i] + discarded[i]) / std::sqrt(2.0) : retained[i];
            if (negative_mode) (*data)[4 + i] = (retained[i] - discarded[i]) / std::sqrt(2.0);
        }
        ds.cs_data.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({naux, 2, 2}, data);
    }
    const auto assigned =
        dispatch_upper_triangular_tasks(1, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
                                        ds.blacs_h.myprow, ds.blacs_h.mypcol);
    const std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
             ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
    chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    CrpaContext context{chi,       ds.cs_data, ds.basis_aux, pairs,
                        no_shrink, ds.vq,      ds.blacs_h,   ds.desc_abf};
    Chi0::BandSelection selection(1, std::vector<std::vector<unsigned char>>(4, {1, 0}));
    for (const bool all_bands : {false, true})
    {
        for (auto &row : selection[0]) row[1] = all_bands;
        const auto result = compute_crpa_onsite(context, selection, {&frame});
        if (result.tensors.size() != 1 || result.frequencies.size() != 16)
            throw std::runtime_error("native cRPA pipeline returned incomplete tensors");
        const auto &tensor = result.tensors.front();
        constexpr double bare = 0.55 * 0.55 * 2.0;
        close(tensor.bare(0, 0), bare, 1e-12);
        for (std::size_t iw = 0; iw < result.frequencies.size(); ++iw)
        {
            const double omega = result.frequencies[iw];
            const double screened = bare / (1.0 + 1.28 / (1.0 + omega * omega));
            close(tensor.w[iw](0, 0), screened, 2e-5);
            close(tensor.u[iw](0, 0), all_bands ? bare : screened, 2e-5);
        }
    }
}
}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    test_distributed_complex_contraction();
    test_native_screening_pipeline(false);
    test_native_screening_pipeline(true);
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_crpa_screening: passed\n";
}
