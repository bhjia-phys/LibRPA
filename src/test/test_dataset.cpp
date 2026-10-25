#include "../api/dataset.h"
#include "../core/meanfield_mpi.h"
#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"

#include "../mpi/global_mpi.h"
#include "../io/global_io.h"
#include "../utils/constants.h"
#include "librpa_enums.h"
#include "librpa_input.h"
#include "librpa_options.h"
#include "mpi_test_config.h"
#include "testutils.h"

template <typename T, typename = void>
struct has_mf_headwing : std::false_type
{};

template <typename T>
struct has_mf_headwing<T, std::void_t<decltype(std::declval<T>().mf_headwing)>> : std::true_type
{};

template <typename T, typename = void>
struct has_kfrac_headwing_list : std::false_type
{};

template <typename T>
struct has_kfrac_headwing_list<T, std::void_t<decltype(std::declval<T>().kfrac_headwing_list)>> : std::true_type
{};

static_assert(!has_mf_headwing<librpa_int::Dataset>::value,
              "Dataset should use mf for analytic head/wing instead of mf_headwing");
static_assert(!has_kfrac_headwing_list<librpa_int::Dataset>::value,
              "Dataset should use pbc.kfrac_list for analytic head/wing instead of kfrac_headwing_list");

static std::vector<double> sc222_kvecs()
{
    const double half = librpa_int::TWO_PI * 0.5;
    return {
        0.0, 0.0, 0.0, 0.0, 0.0, half, 0.0, half, 0.0, 0.0, half, half,
        half, 0.0, 0.0, half, 0.0, half, half, half, 0.0, half, half, half,
    };
}

static void test_set_comm_blacs_coul_np4()
{
    using namespace librpa_int;

    std::vector<size_t> nbs{2, 3};
    Dataset ds(MPI_COMM_WORLD);
    ds.basis_aux.set(nbs);
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    ds.pbc.set_kgrids_kvec(2, 2, 2, sc222_kvecs());
    // Irreducible map not set, thus the irreducible sector is just the full BZ.
    const int n_aux = ds.basis_aux.nb_total;
    const int m = n_aux, n = n_aux;
    const int mb = (m + 1) / 2, nb = (n + 1) / 2;
    const int nkpts = ds.pbc.klist.size();
    int lbrow, ubrow, lbcol, ubcol;

    // A column-major 2x2 process grid, all q-points on it
    lbrow = 0, ubrow = m;
    lbcol = 0, ubcol = n;
    if (ds.comm_h.myid % 2 == 0) ubrow = mb;
    else lbrow = mb;
    if (ds.comm_h.myid / 2 == 0) ubcol = nb;
    else lbcol = nb;
    ds.vq_lbrow = lbrow;
    ds.vq_ubrow = ubrow;
    ds.vq_lbcol = lbcol;
    ds.vq_ubcol = ubcol;
    for (const auto &k: ds.pbc.klist)
    {
        ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL);
    }
    ds.initialize_comm_blacs_coul();
    assert(ds.blacs_coul_intra_q_h.layout == CTXT_LAYOUT::C);
    assert(ds.blacs_coul_intra_q_h.nprows == 2);
    assert(ds.blacs_coul_intra_q_h.npcols == 2);
    assert(ds.comm_coul_inter_q_h.nprocs == 1);
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    // A row-major 2x2 process grid, all q-points on it
    lbrow = 0, ubrow = m;
    lbcol = 0, ubcol = n;
    if (ds.comm_h.myid / 2 == 0) ubrow = mb;
    else lbrow = mb;
    if (ds.comm_h.myid % 2 == 0) ubcol = nb;
    else lbcol = nb;
    ds.vq_lbrow = lbrow;
    ds.vq_ubrow = ubrow;
    ds.vq_lbcol = lbcol;
    ds.vq_ubcol = ubcol;
    for (const auto &k: ds.pbc.klist)
    {
        ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL);
    }
    ds.initialize_comm_blacs_coul();
    assert(ds.blacs_coul_intra_q_h.layout == CTXT_LAYOUT::R);
    assert(ds.blacs_coul_intra_q_h.nprows == 2);
    assert(ds.blacs_coul_intra_q_h.npcols == 2);
    assert(ds.comm_coul_inter_q_h.nprocs == 1);
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    // 1x1 process grid (major never mind), q-points distributed on 4 tasks
    lbrow = 0, ubrow = m;
    lbcol = 0, ubcol = n;
    ds.vq_lbrow = lbrow;
    ds.vq_ubrow = ubrow;
    ds.vq_lbcol = lbcol;
    ds.vq_ubcol = ubcol;
    for (int ik = 0; ik < nkpts; ik++)
    {
        if (ik % ds.comm_h.nprocs == ds.comm_h.myid)
        {
            const auto &k = ds.pbc.klist[ik];
            ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL);
        }
    }
    ds.initialize_comm_blacs_coul();
    assert(ds.blacs_coul_intra_q_h.nprows == 1);
    assert(ds.blacs_coul_intra_q_h.npcols == 1);
    assert(ds.comm_coul_inter_q_h.nprocs == 4);
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    // 1x1 process grid (major never mind), only on process 0 (not optimal distribution, but possible)
    if (ds.comm_h.myid == 0)
    {
        lbrow = 0, ubrow = m;
        lbcol = 0, ubcol = n;
        ds.vq_lbrow = lbrow;
        ds.vq_ubrow = ubrow;
        ds.vq_lbcol = lbcol;
        ds.vq_ubcol = ubcol;
        for (int ik = 0; ik < nkpts; ik++)
        {
            const auto &k = ds.pbc.klist[ik];
            ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL);
        }
    }
    ds.initialize_comm_blacs_coul();
    if (ds.comm_h.myid == 0)
    {
        assert(ds.blacs_coul_intra_q_h.nprows == 1);
        assert(ds.blacs_coul_intra_q_h.npcols == 1);
        assert(ds.comm_coul_inter_q_h.nprocs == 1);
    }
    else
    {
        assert(!ds.blacs_coul_intra_q_h.is_initialized());
        assert(!ds.comm_coul_inter_q_h.is_initialized());
        assert(!ds.comm_coul_intra_q_h.is_initialized());
        assert(!ds.comm_coul_h.is_initialized());
    }
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    // 1x2 process grid (major never mind), 2 BLACS each with 2 q-points
    lbrow = 0, ubrow = m;
    lbcol = 0, ubcol = n;
    if (ds.comm_h.myid % 2 == 0) ubcol = nb;
    else lbcol = nb;
    ds.vq_lbrow = lbrow;
    ds.vq_ubrow = ubrow;
    ds.vq_lbcol = lbcol;
    ds.vq_ubcol = ubcol;

    for (int ik = 0; ik < nkpts; ik++)
    {
        if ((ik % ds.comm_h.nprocs) / 2 == ds.comm_h.myid / 2)
        {
            const auto &k = ds.pbc.klist[ik];
            ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL);
        }
    }
    ds.initialize_comm_blacs_coul();
    assert(ds.blacs_coul_intra_q_h.nprows == 1);
    assert(ds.blacs_coul_intra_q_h.npcols == 2);
    assert(ds.comm_coul_inter_q_h.nprocs == 2);
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    ds.free();
}

static void test_redistribute_blacs2ap_np4(const std::vector<size_t> &nbs)
{
    using namespace librpa_int;
    Dataset ds(MPI_COMM_WORLD);
    ds.basis_aux.set(nbs);
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    initialize_ds_atpairs_local(ds, LIBRPA_ROUTING_LIBRI);
    const int n_aux = ds.basis_aux.nb_total;
    ds.desc_abf.reset_handler(ds.blacs_h);
    ds.desc_abf.init_1b1p(n_aux, n_aux, 0, 0);
    ds.pbc.set_kgrids_kvec(2, 2, 2, sc222_kvecs());
    const int m = n_aux, n = n_aux;
    const int nkpts = ds.pbc.klist.size();
    int lbrow, ubrow, lbcol, ubcol;

    // 1x1 process grid (major never mind), q-points distributed on 4 tasks
    lbrow = 0, ubrow = m;
    lbcol = 0, ubcol = n;
    ds.vq_lbrow = lbrow;
    ds.vq_ubrow = ubrow;
    ds.vq_lbcol = lbcol;
    ds.vq_ubcol = ubcol;
    for (int ik = 0; ik < nkpts; ik++)
    {
        if (ik % ds.comm_h.nprocs == ds.comm_h.myid)
        {
            const auto &k = ds.pbc.klist[ik];
            ds.vq_block_loc[k] = Matz(ubrow - lbrow, ubcol - lbcol, MAJOR::COL).randomize();
        }
    }
    ds.initialize_comm_blacs_coul();
    assert(ds.blacs_coul_intra_q_h.nprows == 1);
    assert(ds.blacs_coul_intra_q_h.npcols == 1);
    assert(ds.comm_coul_inter_q_h.nprocs == 4);
    ds.redistribute_coulomb_blacs2ap();
    global::ofs_myid << "ds.vq.size() " << ds.vq.size() << " ds.atpairs_local.size() " << ds.atpairs_local.size() << std::endl; 
    size_t n_pairs = 0;
    for (const auto &[_, Jqmap]: ds.vq)
    {
        n_pairs += Jqmap.size();
        for (const auto &[_, qmap]: Jqmap)
        {
            assert(qmap.size() == as_size(nkpts));
        }
    }
    assert(n_pairs == ds.atpairs_local.size());
    ds.finalize_comm_blacs_coul();
    ds.vq_block_loc.clear();

    ds.free();
}

static void test_redistribute_eigvecs_kpara_np4()
{
    using namespace librpa_int;

    constexpr int n_spins = 2;
    constexpr int n_spinor = 2;
    constexpr int n_kpoints = 4;
    constexpr int n_states = 3;
    constexpr int n_aos = 3;
    const std::vector<int> initial_owner{1, 0, 3, 1};
    const std::vector<int> expected_owner{0, 2, 0, 2};

    Dataset ds(MPI_COMM_WORLD);
    ds.mf.set(n_spins, n_kpoints, n_states, n_aos, n_spinor);
    ds.scfk_blacs_ctxt.init({2, 2}, MPI_COMM_WORLD, n_kpoints);

    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
        {
            for (int ik = 0; ik != n_kpoints; ++ik)
            {
                if (ds.comm_h.myid != initial_owner[ik]) continue;
                ComplexMatrix wfc(n_states, n_aos);
                for (int i = 0; i != n_states; ++i)
                {
                    for (int j = 0; j != n_aos; ++j)
                    {
                        const double value = 1000.0 * ispin + 100.0 * ispinor +
                                             10.0 * ik + i + 0.1 * j;
                        wfc(i, j) = {value, -value};
                    }
                }
                ds.mf.get_eigenvectors()[ispin][ispinor][ik] = std::move(wfc);
            }
        }
    }

    ds.redistribute_eigvecs_kpara();
    ds.redistribute_eigvecs_kpara();

    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
        {
            for (int ik = 0; ik != n_kpoints; ++ik)
            {
                const auto *wfc = ds.mf.find_wfc(ispin, ispinor, ik);
                if (ds.comm_h.myid == expected_owner[ik])
                {
                    assert(wfc != nullptr);
                    assert(wfc->nr == n_states);
                    assert(wfc->nc == n_aos);
                    for (int i = 0; i != n_states; ++i)
                    {
                        for (int j = 0; j != n_aos; ++j)
                        {
                            const double value = 1000.0 * ispin + 100.0 * ispinor +
                                                 10.0 * ik + i + 0.1 * j;
                            assert(fequal((*wfc)(i, j), std::complex<double>(value, -value)));
                        }
                    }
                }
                else
                {
                    assert(wfc == nullptr);
                }
            }
        }
    }

    ds.free();
}

static void test_redistribute_eigvecs_kpara_2d_np4()
{
    using namespace librpa_int;

    constexpr int n_spins = 1;
    constexpr int n_spinor = 2;
    constexpr int n_kpoints = 3;
    constexpr int n_states = 193;
    constexpr int n_aos = 263;
    // Exactly representable dyadic values keep this copy/layout check
    // independent of SIMD/FMA rounding in separate fill/verification loops.
    const std::vector<int> initial_owner{3, 0, 1};

    Dataset ds(MPI_COMM_WORLD);
    ds.mf.set(n_spins, n_kpoints, n_states, n_aos, n_spinor);
    ds.scfk_blacs_ctxt.init({2, 2}, MPI_COMM_WORLD, n_kpoints);
    const int block_ao =
        get_capped_blacs_block_size(n_aos, wfc_gemm_block_size_opt, ds.scfk_blacs_ctxt.blacs_h);
    const int block_state =
        get_capped_blacs_block_size(n_states, wfc_gemm_block_size_opt,
                                    ds.scfk_blacs_ctxt.blacs_h);
    ds.desc_wfc_kb = ds.scfk_blacs_ctxt.create_array_desc(
        n_aos, n_states, block_ao, block_state);
    ds.desc_wfc_kb_full =
        ds.scfk_blacs_ctxt.create_array_desc(n_aos, n_states, n_aos, n_states);

    for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
    {
        for (int ik = 0; ik != n_kpoints; ++ik)
        {
            if (ds.comm_h.myid != initial_owner[ik]) continue;
            auto &wfc = ds.mf.get_eigenvectors()[0][ispinor][ik];
            wfc.create(n_states, n_aos);
            for (int ib = 0; ib != n_states; ++ib)
                for (int iao = 0; iao != n_aos; ++iao)
                {
                    const double value = 100.0 * ispinor + 10.0 * ik + ib + iao / 1024.0;
                    wfc(ib, iao) = {value, -value};
                }
        }
    }

    ds.redistribute_eigvecs_kpara_2d();
    ds.redistribute_eigvecs_kpara_2d();
    assert(ds.eigvecs_kpara_2d_ready());
    assert(ds.desc_wfc_kb.mb() == block_ao);
    assert(ds.desc_wfc_kb.nb() == block_state);
    assert(ds.desc_wfc_kb.mb() <= wfc_gemm_block_size_opt &&
           ds.desc_wfc_kb.nb() <= wfc_gemm_block_size_opt);

    for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
    {
        for (int ik = 0; ik != n_kpoints; ++ik)
        {
            const auto *wfc = ds.mf.find_wfc(0, ispinor, ik);
            if (!ds.scfk_blacs_ctxt.owns_kpoint(ik))
            {
                assert(wfc == nullptr);
                continue;
            }
            assert(wfc != nullptr);
            assert(wfc->nr == ds.desc_wfc_kb.n_loc());
            assert(wfc->nc == ds.desc_wfc_kb.m_loc());
            for (int jloc = 0; jloc != ds.desc_wfc_kb.n_loc(); ++jloc)
            {
                const int ib = ds.desc_wfc_kb.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != ds.desc_wfc_kb.m_loc(); ++iloc)
                {
                    const int iao = ds.desc_wfc_kb.indx_l2g_r(iloc);
                    const double value = 100.0 * ispinor + 10.0 * ik + ib + iao / 1024.0;
                    assert(fequal((*wfc)(jloc, iloc),
                                  std::complex<double>(value, -value)));
                }
            }
        }
    }
    ds.free();
}

static void test_redistribute_eigvecs_1b1p_to_opt_np4()
{
    using namespace librpa_int;

    constexpr int n_spinor = 2;
    constexpr int n_kpoints = 3;
    constexpr int n_states = 193;
    constexpr int n_aos = 263;
    // Exactly representable dyadic values keep this copy/layout check
    // independent of SIMD/FMA rounding in separate fill/verification loops.
    KPointBlacsParallelContext kctx({2, 2}, MPI_COMM_WORLD, n_kpoints);
    const auto desc_io = kctx.create_array_desc(n_aos, n_states);
    const int block_ao =
        get_capped_blacs_block_size(n_aos, wfc_gemm_block_size_opt, kctx.blacs_h);
    const int block_state =
        get_capped_blacs_block_size(n_states, wfc_gemm_block_size_opt, kctx.blacs_h);
    const auto desc_opt =
        kctx.create_array_desc(n_aos, n_states, block_ao, block_state);
    assert(desc_io.mb() != desc_opt.mb() || desc_io.nb() != desc_opt.nb());

    MeanField mf(1, n_kpoints, n_states, n_aos, n_spinor);
    for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
    {
        for (const int ik : kctx.kpoints_local())
        {
            auto &wfc = mf.get_eigenvectors()[0][ispinor][ik];
            wfc.create(desc_io.n_loc(), desc_io.m_loc(), false);
            for (int jloc = 0; jloc != desc_io.n_loc(); ++jloc)
            {
                const int ib = desc_io.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_io.m_loc(); ++iloc)
                {
                    const int iao = desc_io.indx_l2g_r(iloc);
                    const double value = 1000.0 * ispinor + 100.0 * ik + ib + iao / 1024.0;
                    wfc(jloc, iloc) = {value, -value};
                }
            }
        }
    }

    redistribute_meanfield_eigvecs_kblacs(
        mf, kctx, desc_io, desc_opt, "test 1b1p input");
    for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
    {
        for (int ik = 0; ik != n_kpoints; ++ik)
        {
            const auto *wfc = mf.find_wfc(0, ispinor, ik);
            if (!kctx.owns_kpoint(ik))
            {
                assert(wfc == nullptr);
                continue;
            }
            assert(wfc != nullptr);
            assert(wfc->nr == desc_opt.n_loc());
            assert(wfc->nc == desc_opt.m_loc());
            for (int jloc = 0; jloc != desc_opt.n_loc(); ++jloc)
            {
                const int ib = desc_opt.indx_l2g_c(jloc);
                for (int iloc = 0; iloc != desc_opt.m_loc(); ++iloc)
                {
                    const int iao = desc_opt.indx_l2g_r(iloc);
                    const double value = 1000.0 * ispinor + 100.0 * ik + ib + iao / 1024.0;
                    assert(fequal((*wfc)(jloc, iloc),
                                  std::complex<double>(value, -value)));
                }
            }
        }
    }
}

static void setup_spinor_dataset(librpa_int::Dataset &ds)
{
    using namespace librpa_int;

    ds.mf.set(1, 1, 2, 1, 2);
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    ds.pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    ds.atoms.types = {{0, 0}};
    ds.atoms.coords_frac = {{0, {0.0, 0.0, 0.0}}};
    ds.basis_wfc.set({1});
    ds.basis_aux.set({1});
    ds.basis_wfc.set_l_shells({{0}});
    ds.basis_aux.set_l_shells({{0}});
    ds.basis_convention = {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL,
                           LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M};
    ds.spg_symops = {SpaceGroupSymOp::IDENTITY,
                    {Matrix3(1, 0, 0, 0, -1, 0, 0, 0, -1), {0, 0, 0}}};
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 1);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(1, 2, 1, 2);
}

static void test_spinor_symmetry_speedup_rejected()
{
    using namespace librpa_int;

    auto rejects = [](auto init, auto set_symmetry_flag) {
        Dataset ds(MPI_COMM_WORLD);
        setup_spinor_dataset(ds);
        LibrpaOptions opts;
        librpa_init_options(&opts);
        set_symmetry_flag(opts);

        bool threw = false;
        try
        {
            init(ds, opts);
        }
        catch (const std::runtime_error& error)
        {
            threw = std::string(error.what()).find("explicit spin-space") != std::string::npos;
        }
        ds.free();
        return threw;
    };

    assert(rejects(initialize_ds_exx, [](LibrpaOptions &opts) {
        opts.use_symmetry_exx = LIBRPA_SWITCH_ON;
    }));
    assert(rejects(initialize_ds_chi0, [](LibrpaOptions &opts) {
        opts.use_symmetry_rpa = LIBRPA_SWITCH_ON;
    }));
    assert(rejects(initialize_ds_g0w0, [](LibrpaOptions &opts) {
        opts.use_symmetry_gw = LIBRPA_SWITCH_ON;
    }));
}

static void test_set_symmetry_spin_operations_api()
{
    using namespace librpa_int;

    const int rotmats_ei[18] = {1, 0, 0, 0, 1, 0, 0, 0, 1,
                                -1, 0, 0, 0, -1, 0, 0, 0, -1};

    // Explicit antiunitary table fills both the spatial and the spin metadata;
    // the legacy entry point clears the explicit table again.
    {
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        const int antiunitary[2] = {0, 1};
        librpa_set_symmetry_spin_operations(h, 2, 1, rotmats_ei, nullptr,
                                            antiunitary, nullptr, 0, 1);
        auto pds = api::get_dataset_instance(h);
        assert(pds->spg_symops.size() == 2);
        assert(pds->spg_symops[1].rotation
               == Matrix3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0));
        assert(pds->spg_spin_ops_explicit);
        assert(pds->spg_grey_group);
        assert(pds->spg_spin_ops.size() == 2);
        const std::array<std::complex<double>, 4> identity_u{1.0, 0.0, 0.0, 1.0};
        assert(pds->spg_spin_ops[0].spatial_id == 0);
        assert(!pds->spg_spin_ops[0].antiunitary);
        assert(pds->spg_spin_ops[0].spin_source == SymmetrySpinActionSource::Identity);
        assert(pds->spg_spin_ops[0].spin_u == identity_u);
        assert(pds->spg_spin_ops[1].spatial_id == 1);
        assert(pds->spg_spin_ops[1].antiunitary);

        librpa_set_symmetry_operations(h, 2, 1, rotmats_ei, nullptr);
        assert(!pds->spg_spin_ops_explicit);
        assert(pds->spg_spin_ops.empty());
        assert(pds->spg_grey_group);
        librpa_destroy_handler(h);
    }

    // DerivedFromSpatialSOC without spin_u is accepted: the context
    // reconstructs U_s = U[det(Q) Q] from the Cartesian axial rotation.
    {
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        const auto pds = librpa_int::api::get_dataset_instance(h);
        setup_spinor_dataset(*pds);
        const int rotmats_c2x[18] = {1, 0, 0, 0, 1, 0, 0, 0, 1,
                                     1, 0, 0, 0, -1, 0, 0, 0, -1};
        librpa_set_symmetry_spin_operations(h, 2, 1, rotmats_c2x, nullptr,
                                            nullptr, nullptr, 2, 0);
        assert(pds->spg_spin_ops.size() == 2);
        assert(pds->spg_spin_ops[1].spin_source
               == librpa_int::SymmetrySpinActionSource::DerivedFromSpatialSOC);
        initialize_symmetry_context(*pds, true);
        const auto& spin = pds->symmetry_context.spin_operations.at(1).spin_u;
        assert(std::abs(spin[0]) < 1e-12 && std::abs(spin[3]) < 1e-12);
        assert(std::abs(std::abs(spin[1]) - 1.0) < 1e-12);
        assert(std::abs(std::abs(spin[2]) - 1.0) < 1e-12);
        librpa_destroy_handler(h);
    }

    // A non-unitary spin_u is rejected.
    {
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        const double bad_u[8] = {2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        bool threw = false;
        try
        {
            librpa_set_symmetry_spin_operations(h, 2, 1, rotmats_ei, nullptr,
                                                nullptr, bad_u, 1, 0);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        assert(threw);
        librpa_destroy_handler(h);
    }

    // ExplicitSpinSpace requires a non-null spin_u.
    {
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        bool threw = false;
        try
        {
            librpa_set_symmetry_spin_operations(h, 2, 1, rotmats_ei, nullptr,
                                                nullptr, nullptr, 1, 0);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        assert(threw);
        librpa_destroy_handler(h);
    }

    // grey_group=1 expands the table to 2N entries with the unitary block
    // first when the symmetry context is initialized from the dataset.
    {
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        librpa_set_symmetry_spin_operations(h, 2, 1, rotmats_ei, nullptr,
                                            nullptr, nullptr, 0, 1);
        auto pds = api::get_dataset_instance(h);
        pds->pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        pds->pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
        initialize_symmetry_context(*pds, false);
        const auto& ctx = pds->symmetry_context;
        assert(ctx.has_explicit_spin_operations);
        assert(ctx.spin_operations.size() == 4);
        assert(ctx.spin_operations[0].spatial_id == 0);
        assert(!ctx.spin_operations[0].antiunitary);
        assert(ctx.spin_operations[1].spatial_id == 1);
        assert(!ctx.spin_operations[1].antiunitary);
        assert(ctx.spin_operations[2].spatial_id == 0);
        assert(ctx.spin_operations[2].antiunitary);
        assert(ctx.spin_operations[3].spatial_id == 1);
        assert(ctx.spin_operations[3].antiunitary);
        librpa_destroy_handler(h);
    }
}

static void test_disabled_component_symmetry_keeps_shared_context()
{
    using namespace librpa_int;

    Dataset ds(MPI_COMM_WORLD);
    ds.mf.set(1, 1, 1, 1, 1);
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    ds.pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, 4, true), MPI_COMM_WORLD, 1);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(1, 1, 1, 1);
    // Prepare a physically matching shared frame before installing the cache
    // sentinel. Disabled components must preserve a valid existing context.
    ds.symmetry_context.set_crystal_structure(
        ds.pbc.latvec, ds.pbc.G, ds.atoms.types, ds.atoms.coords_frac);
    ds.symmetry_context.set_available();
    SymmetryKStar sentinel_star;
    sentinel_star.star_index = 17;
    ds.symmetry_context.kstars.push_back(sentinel_star);

    LibrpaOptions opts;
    librpa_init_options(&opts);
    opts.use_symmetry_exx = LIBRPA_SWITCH_OFF;
    opts.use_symmetry_rpa = LIBRPA_SWITCH_OFF;
    opts.use_symmetry_gw = LIBRPA_SWITCH_OFF;
    assert(opts.output_exx_ks_mat_k == LIBRPA_SWITCH_OFF);
    assert(opts.istate_output_mat_start == 0);
    assert(opts.istate_output_mat_end == -1);
    opts.istate_output_mat_end = 1;
    assert(opts.n_bands_exclude_chi0 == 0);
    assert(opts.n_bands_exclude_sigc == 0);
    opts.n_bands_exclude_chi0 = 1;

    initialize_ds_exx(ds, opts);
    initialize_ds_chi0(ds, opts);
    initialize_ds_g0w0(ds, opts);

    assert(opts.n_bands_sigc == -1);
    assert(ds.p_g0w0->nbands_G == -1);
    for (const int n_bands : {0, 1, -1})
    {
        opts.n_bands_sigc = n_bands;
        initialize_ds_g0w0(ds, opts);
        assert(ds.p_g0w0->nbands_G == n_bands);
    }

    assert(ds.p_chi0->n_bands_exclude == 1);
    assert(ds.p_g0w0->n_bands_exclude == 0);
    opts.n_bands_exclude_chi0 = 0;
    opts.n_bands_exclude_sigc = 1;
    initialize_ds_chi0(ds, opts);
    initialize_ds_g0w0(ds, opts);
    assert(ds.p_chi0->n_bands_exclude == 0);
    assert(ds.p_g0w0->n_bands_exclude == 1);
    for (int invalid : {-1, 2})
    {
        opts.n_bands_exclude_chi0 = invalid;
        opts.n_bands_exclude_sigc = invalid;
        for (auto init : {initialize_ds_chi0, initialize_ds_g0w0})
        {
            bool rejected = false;
            try { init(ds, opts); }
            catch (const std::runtime_error &) { rejected = true; }
            assert(rejected);
        }
    }
    assert(ds.p_g0w0->istate_output_mat_start == 0);
    assert(ds.p_g0w0->istate_output_mat_end == 1);
    assert(ds.symmetry_context.available);
    assert(ds.symmetry_context.kstars.size() == 1);
    assert(ds.symmetry_context.kstars.front().star_index == 17);
    ds.free();
}

static void test_redistribute_band_eigvecs_kpara_np4()
{
    using namespace librpa_int;

    constexpr int n_spins = 1;
    constexpr int n_spinor = 1;
    constexpr int n_kpoints = 4;
    constexpr int n_states = 2;
    constexpr int n_aos = 3;
    const std::vector<int> initial_owner{3, 0, 1, 2};
    const std::vector<int> expected_owner{0, 1, 2, 3};

    Dataset ds(MPI_COMM_WORLD);
    ds.mf_band.set(n_spins, n_kpoints, n_states, n_aos, n_spinor);

    for (int ik = 0; ik != n_kpoints; ++ik)
    {
        if (ds.comm_h.myid != initial_owner[ik]) continue;
        ComplexMatrix wfc(n_states, n_aos);
        for (int i = 0; i != n_states; ++i)
        {
            for (int j = 0; j != n_aos; ++j)
            {
                const double value = 10.0 * ik + i + 0.1 * j;
                wfc(i, j) = {value, -value};
            }
        }
        ds.mf_band.get_eigenvectors()[0][0][ik] = std::move(wfc);
    }

    ds.redistribute_band_eigvecs_kpara();
    ds.redistribute_band_eigvecs_kpara();

    assert(ds.bandk_blacs_ctxt.is_initialized());
    assert(ds.bandk_blacs_ctxt.n_kpoints() == n_kpoints);
    for (int ik = 0; ik != n_kpoints; ++ik)
    {
        const auto *wfc = ds.mf_band.find_wfc(0, 0, ik);
        if (ds.comm_h.myid == expected_owner[ik])
        {
            assert(wfc != nullptr);
            assert(wfc->nr == n_states);
            assert(wfc->nc == n_aos);
            for (int i = 0; i != n_states; ++i)
            {
                for (int j = 0; j != n_aos; ++j)
                {
                    const double value = 10.0 * ik + i + 0.1 * j;
                    assert(fequal((*wfc)(i, j), std::complex<double>(value, -value)));
                }
            }
        }
        else
        {
            assert(wfc == nullptr);
        }
    }

    ds.free();
}

static void test_redistribute_band_eigvecs_kpara_2d_np4()
{
    using namespace librpa_int;
    constexpr int n_kpoints = 4;
    constexpr int n_states = 193;
    constexpr int n_aos = 263;
    // Exactly representable dyadic values keep this copy/layout check
    // independent of SIMD/FMA rounding in separate fill/verification loops.

    Dataset ds(MPI_COMM_WORLD);
    ds.mf_band.set(1, n_kpoints, n_states, n_aos, 1);
    for (int ik = 0; ik != n_kpoints; ++ik)
    {
        if (ds.comm_h.myid != (3 - ik)) continue;
        auto &wfc = ds.mf_band.get_eigenvectors()[0][0][ik];
        wfc.create(n_states, n_aos);
        for (int ib = 0; ib != n_states; ++ib)
            for (int iao = 0; iao != n_aos; ++iao)
                wfc(ib, iao) = {10.0 * ik + ib, 0.1 * iao};
    }
    ds.redistribute_band_eigvecs_kpara_2d();
    assert(ds.band_eigvecs_kpara_2d_ready());
    assert(ds.desc_band_wfc_kb.mb() == wfc_gemm_block_size_opt);
    assert(ds.desc_band_wfc_kb.nb() == wfc_gemm_block_size_opt);
    for (int ik = 0; ik != n_kpoints; ++ik)
    {
        const auto *wfc = ds.mf_band.find_wfc(0, 0, ik);
        if (!ds.bandk_blacs_ctxt.owns_kpoint(ik))
        {
            assert(wfc == nullptr);
            continue;
        }
        assert(wfc != nullptr);
        assert(wfc->nr == ds.desc_band_wfc_kb.n_loc());
        assert(wfc->nc == ds.desc_band_wfc_kb.m_loc());
        for (int jloc = 0; jloc != ds.desc_band_wfc_kb.n_loc(); ++jloc)
        {
            const int ib = ds.desc_band_wfc_kb.indx_l2g_c(jloc);
            for (int iloc = 0; iloc != ds.desc_band_wfc_kb.m_loc(); ++iloc)
            {
                const int iao = ds.desc_band_wfc_kb.indx_l2g_r(iloc);
                assert(fequal((*wfc)(jloc, iloc),
                              std::complex<double>(10.0 * ik + ib, 0.1 * iao)));
            }
        }
    }
    ds.free();
}

// Public setters own complete spin/k matrices. Block-cyclic tiles are
// created only by native redistribution, including legitimate empty ranks.
static void test_dense_wfc_setter_packing_and_ownership_np4()
{
    using namespace librpa_int;
    constexpr int n_states = 4, n_aos = 2, n_values = n_states * n_aos;
    for (int mode = 0; mode != 8; ++mode)
    {
        const bool band = mode >= 4;
        const bool spinor = mode % 4 >= 2;
        LibrpaHandler* h = librpa_create_handler(MPI_COMM_WORLD);
        librpa_set_scf_dimension(h, 1, 1, n_states, n_aos, spinor ? 2 : 1);
        if (band)
        {
            const double kfrac[3] = {0, 0, 0};
            const double occ[4] = {1, 1, 0, 0}, eig[4] = {-1, -1, 1, 1};
            librpa_set_band_kvec(h, 1, kfrac);
            librpa_set_band_occ_eigval(h, 1, 1, n_states, occ, eig);
        }
        auto ds = api::get_dataset_instance(h);
        auto &mf = band ? ds->mf_band : ds->mf;
        double re[2][n_values], im[2][n_values], packed[2][2 * n_values];
        for (int sigma = 0; sigma != 2; ++sigma)
            for (int i = 0; i != n_values; ++i)
            {
                re[sigma][i] = 2 * sigma + i * 0.125;
                im[sigma][i] = -sigma - i * 0.0625;
                packed[sigma][2 * i] = re[sigma][i];
                packed[sigma][2 * i + 1] = im[sigma][i];
            }
        const auto set = [&](int rows, int cols) {
            switch (mode)
            {
                case 0: librpa_set_wfc(h, 0, 0, rows, cols, re[0], im[0]); break;
                case 1: librpa_set_wfc_packed(h, 0, 0, rows, cols, packed[0]); break;
                case 2: librpa_set_wfc_spinor(h, 0, rows, cols, re[0], im[0], re[1], im[1]); break;
                case 3: librpa_set_wfc_spinor_packed(h, 0, rows, cols, packed[0], packed[1]); break;
                case 4: librpa_set_wfc_band(h, 0, 0, rows, cols, re[0], im[0]); break;
                case 5: librpa_set_wfc_band_packed(h, 0, 0, rows, cols, packed[0]); break;
                case 6: librpa_set_wfc_band_spinor(h, 0, rows, cols, re[0], im[0], re[1], im[1]); break;
                case 7: librpa_set_wfc_band_spinor_packed(h, 0, rows, cols, packed[0], packed[1]); break;
            }
        };
        // Start on rank 3, not on the eventual one-k BLACS root. The other
        // ranks omit the setter rather than pretending to supply empty tiles.
        if (ds->comm_h.myid == 3)
        {
            set(n_states, n_aos);
            for (const auto shape : {std::pair<int, int>{1, 1}, {0, n_aos},
                                     {n_states, 1}, {n_states + 1, n_aos}})
            {
                bool rejected = false;
                try { set(shape.first, shape.second); }
                catch (const std::runtime_error &) { rejected = true; }
                assert(rejected);
            }
            // Rejection must leave the complete valid matrix intact.
            for (int sigma = 0; sigma != (spinor ? 2 : 1); ++sigma)
            {
                const auto *wfc = mf.find_wfc(0, sigma, 0);
                assert(wfc && wfc->nr == n_states && wfc->nc == n_aos);
                for (int i = 0; i != n_values; ++i)
                    assert(wfc->c[i] == std::complex<double>(re[sigma][i], im[sigma][i]));
            }
        }
        if (band) ds->redistribute_band_eigvecs_kpara_2d();
        else ds->redistribute_eigvecs_kpara_2d();
        const auto &desc = band ? ds->desc_band_wfc_kb : ds->desc_wfc_kb;
        for (int sigma = 0; sigma != (spinor ? 2 : 1); ++sigma)
        {
            const auto *wfc = mf.find_wfc(0, sigma, 0);
            assert(wfc && wfc->nr == desc.n_loc() && wfc->nc == desc.m_loc());
            for (int jloc = 0; jloc != desc.n_loc(); ++jloc)
                for (int iloc = 0; iloc != desc.m_loc(); ++iloc)
                {
                    const int i = desc.indx_l2g_c(jloc) * n_aos + desc.indx_l2g_r(iloc);
                    assert((*wfc)(jloc, iloc) == std::complex<double>(re[sigma][i], im[sigma][i]));
                }
        }
        librpa_destroy_handler(h);
    }
}

int main (int argc, char *argv[])
{
    using namespace librpa_int;
    using namespace librpa_int::global;
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);

    int size_global = get_mpi_size(MPI_COMM_WORLD);
    if (size_global != 4) throw std::runtime_error("test imposes 4 MPI processes");

    init_global_mpi(MPI_COMM_WORLD);
    init_global_io();

    test_set_comm_blacs_coul_np4();
    test_dense_wfc_setter_packing_and_ownership_np4();

    test_redistribute_blacs2ap_np4({2, 3});
    test_redistribute_blacs2ap_np4({10, 4, 5});
    test_redistribute_eigvecs_kpara_np4();
    test_redistribute_eigvecs_kpara_2d_np4();
    test_redistribute_eigvecs_1b1p_to_opt_np4();
    test_redistribute_band_eigvecs_kpara_np4();
    test_redistribute_band_eigvecs_kpara_2d_np4();
    test_spinor_symmetry_speedup_rejected();
    test_set_symmetry_spin_operations_api();
    test_disabled_component_symmetry_keeps_shared_context();

    finalize_global_io();
    finalize_global_mpi();

    MPI_Finalize();
    return 0;
}
