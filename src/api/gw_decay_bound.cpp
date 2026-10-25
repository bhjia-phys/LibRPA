#include "gw_decay_bound.h"

#include "../utils/error.h"
#include "dataset_helper.h"

#include <cmath>
#include <string>

namespace librpa_int
{
GwBodyDecayBound compute_ds_gw_body_decay_bound(Dataset &ds, const LibrpaOptions &opts)
{
    ds.synchronize_input_changes(&opts);
    if (opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
        ds.redistribute_eigvecs_kpara_2d();
    else
        ds.replicate_eigvecs();
    // The ordinary factory rejects this locally. Diagnose it collectively at
    // this collective diagnostic entry before any rank can enter the kernel.
    int invalid_exclusion =
        opts.n_bands_exclude_chi0 < 0 || opts.n_bands_exclude_chi0 > ds.mf.get_n_bands();
    MPI_Allreduce(MPI_IN_PLACE, &invalid_exclusion, 1, MPI_INT, MPI_MAX, ds.comm_h.comm);
    if (invalid_exclusion)
        throw LIBRPA_RUNTIME_ERROR("GW body decay prepass: chi0 exclusion exceeds loaded bands");
    initialize_ds_atpairs_local(ds, LIBRPA_ROUTING_LIBRI);
    ds.redistribute_coulomb_blacs2ap();
    auto chi = make_ds_chi0(ds, opts);
    const bool shrink = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    const bool project_parent = shrink && opts.use_shrink_chi == LIBRPA_SWITCH_ON;
    const auto &cs = shrink && !project_parent ? ds.cs_data_shrink : ds.cs_data;
    const auto &source_basis = shrink && !project_parent ? ds.basis_aux_shrink : ds.basis_aux;
    std::map<Vector3_Order<double>, ComplexMatrix> empty_projection;
    auto &projection = project_parent ? ds.sinvS : empty_projection;
    const auto &coulomb = opts.use_fullcoul_eps == LIBRPA_SWITCH_ON ? ds.vq : ds.vq_cut;
    return compute_gw_body_decay_bound(*chi, cs, ds.atpairs_local, source_basis, projection,
                                       coulomb, ds.blacs_h, opts.sqrt_coulomb_threshold,
                                       opts.n_bands_sigc, opts.n_bands_exclude_sigc,
                                       opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON,
                                       opts.use_elpa_sqrt_coulomb == LIBRPA_SWITCH_ON);
}

GwOpticalDecayBound compute_ds_gw_optical_decay_bound(Dataset &ds, const LibrpaOptions &opts)
{
    ds.synchronize_input_changes(&opts);
    const auto &comm = ds.comm_h;
    const auto reject_collectively = [&](const std::string &message)
    {
        int invalid = !message.empty();
        MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, comm.comm);
        if (invalid)
            throw LIBRPA_RUNTIME_ERROR((message.empty()
                                           ? "invalid optical decay input on another rank"
                                           : message));
    };
    const int loaded = ds.mf.get_n_bands();
    std::string error;
    if (opts.use_2d_dielectric == LIBRPA_SWITCH_ON)
        error = "optical decay prepass currently requires finite3D, not strict2D scaling";
    if (opts.n_bands_exclude_chi0 != 0 ||
        (opts.n_bands_chi0 >= 0 && opts.n_bands_chi0 != loaded))
        error = "optical bound requires common all-loaded body/head windows; normal analytic "
                "head/wing corrections do not follow the body band cut";
    if (opts.replace_w_head != LIBRPA_SWITCH_ON ||
        opts.use_scalapack_gw_wc != LIBRPA_SWITCH_ON ||
        (opts.option_dielect_func != 3 && opts.option_dielect_func != 53))
        error = "optical bound requires the analytic full head/wing option3/53 ScaLAPACK route";
    if (ds.velocity_matrix.empty()) error = "optical bound requires matched KS velocity inputs";
    if (opts.libri_chi0_threshold_C != 0.0)
        error = "optical bound requires unpruned common body/optical RI vertices (C threshold0)";
    if (ds.mf.get_n_spinor() != 2 &&
        ds.mf.get_n_kpoints() != ds.pbc.get_n_cells_bvk())
        error = "optical decay prepass has no common-column scalar-IBZ normalization acceptance";
    reject_collectively(error);

    // The existing body prepass supplies full-q exact endpoints and unchanged
    // integer/gap/window/pruning/backend guards. It never creates a TF grid.
    auto body = compute_ds_gw_body_decay_bound(ds, opts);
    const bool shrink = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    const bool project = shrink && opts.use_shrink_chi == LIBRPA_SWITCH_ON;
    const auto &active_basis = shrink ? ds.basis_aux_shrink : ds.basis_aux;
    const auto &source_basis = shrink && !project ? ds.basis_aux_shrink : ds.basis_aux;
    const auto &cs = shrink && !project ? ds.cs_data_shrink : ds.cs_data;
    const ComplexMatrix *projection = nullptr;
    if (project)
    {
        const auto found = ds.sinvS.find(Vector3_Order<double>{0.0, 0.0, 0.0});
        if (found == ds.sinvS.end()) error = "projected optical bound requires replicated Gamma U";
        else projection = &found->second;
    }
    reject_collectively(error);
    const bool use_symmetry = opts.use_symmetry_rpa == LIBRPA_SWITCH_ON ||
                              opts.use_symmetry_gw == LIBRPA_SWITCH_ON;
    std::unique_ptr<diele_func> context;
    try
    {
        if (use_symmetry)
        {
            initialize_symmetry_context(ds, true);
            require_symmetry_shell_layouts(ds, "optical decay prepass");
        }
        const auto *resident = ds.p_headwing.get();
        const bool selected_direct = bool(ds.direct_full_bz_optical_input) ||
                                     (resident && resident->has_direct_full_bz_headwing_inputs());
        if (ds.optical_input_frame_stale &&
            (selected_direct || ds.optical_meanfield_input))
            throw LIBRPA_RUNTIME_ERROR("selected optical prepass input belongs to an earlier frame");
        const MeanField *selected_reference = ds.optical_meanfield_input.get();
        if (!selected_reference && resident)
            selected_reference = &static_cast<const diele_func &>(*resident).get_meanfield_df();
        if (selected_direct || ds.optical_meanfield_input)
        {
            // This bound requires a certified common electronic frame. Keep
            // its existing strict eligibility checks when a derived resident
            // was invalidated; never silently replace selected direct inputs
            // with a symmetry-restored target.
            if (!selected_reference)
                throw LIBRPA_RUNTIME_ERROR(
                    "direct optical prepass needs a prepared current active frame; prepare optical first");
            const auto &reference = *selected_reference;
            if (reference.get_n_spins() != ds.mf.get_n_spins() ||
                reference.get_n_spinor() != ds.mf.get_n_spinor() ||
                reference.get_n_kpoints() != ds.mf.get_n_kpoints() ||
                reference.get_n_states() != ds.mf.get_n_states() ||
                reference.get_n_aos() != ds.mf.get_n_aos())
                throw LIBRPA_RUNTIME_ERROR("resident direct optical frame dimensions are stale");
            for (int s = 0; s != ds.mf.get_n_spins(); ++s)
                for (int k = 0; k != ds.mf.get_n_kpoints(); ++k)
                {
                    double current_k_weight = 1.0 / ds.mf.get_n_kpoints();
                    if (ds.mf.get_n_kpoints() != ds.pbc.get_n_cells_bvk())
                        current_k_weight = double(find_symmetry_kstar_for_ibz_kpoint(
                            ds.symmetry_context, ds.pbc.kfrac_list.at(k)).members.size()) /
                            ds.pbc.get_n_cells_bvk();
                    for (int b = 0; b != loaded; ++b)
                    {
                        if (std::abs(reference.get_eigenvals()[s](k, b) -
                                     ds.mf.get_eigenvals()[s](k, b)) > 1e-12)
                            throw LIBRPA_RUNTIME_ERROR("resident direct optical energy frame is stale");
                        // The owned optical copy uses active-grid f/N_loaded;
                        // the current dataset carries geometric-star weights.
                        if (std::abs(reference.get_weight()[s](k, b) * ds.mf.get_n_kpoints() -
                                     ds.mf.get_weight()[s](k, b) / current_k_weight) > 1e-8)
                            throw LIBRPA_RUNTIME_ERROR("resident direct optical occupations are stale");
                    }
                    for (int component = 0; component != ds.mf.get_n_spinor(); ++component)
                    {
                        const auto *current = ds.mf.find_wfc(s, component, k);
                        const auto *stored = reference.find_wfc(s, component, k);
                        if ((current == nullptr) != (stored == nullptr) ||
                            (current && (current->nr != stored->nr || current->nc != stored->nc)))
                            throw LIBRPA_RUNTIME_ERROR(
                                "cannot certify resident direct optical WFC ownership; rebuild context");
                        if (current)
                            for (int i = 0; i != current->size; ++i)
                                if (std::abs(current->c[i] - stored->c[i]) > 1e-12)
                                    throw LIBRPA_RUNTIME_ERROR(
                                        "resident direct optical KS gauge differs; rebuild context");
                    }
                }
        }
        const auto &desc_wfc = ds.eigvecs_kpara_2d_ready() ? ds.desc_wfc_kb : ds.desc_wfc_kb_full;
        context = diele_func::make_borrowed_context(
            ds.mf, ds.velocity_matrix, ds.pbc.kfrac_list, ds.basis_wfc, active_basis, ds.pbc,
            comm, ds.blacs_h, opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON,
            &ds.scfk_blacs_ctxt, &desc_wfc, resident);
        if (ds.direct_full_bz_optical_input)
            context->set_direct_full_bz_headwing_inputs(ds.direct_full_bz_optical_input);
        context->use_soc = ds.mf.get_n_spinor() > 1;
        context->set_symmetry_context(ds.symmetry_context);
        if (use_symmetry && ds.symmetry_context.available && !ds.symmetry_context.kstars.empty())
        {
            context->use_symmetry = true;
            context->atom_nw = ds.basis_wfc.get_atom_nb_map();
            context->coord_frac = ds.symmetry_context.input_coord_frac;
        }
    }
    catch (const std::exception &failure)
    {
        error = failure.what();
    }
    reject_collectively(error);
    auto optical = context->build_optical_transition_grams(cs, source_basis, projection);
    const auto &coulomb = opts.use_fullcoul_eps == LIBRPA_SWITCH_ON ? ds.vq : ds.vq_cut;
    return augment_gw_body_decay_bound(std::move(body), std::move(optical), active_basis, ds.pbc,
                                       comm, coulomb, ds.blacs_h, opts.sqrt_coulomb_threshold,
                                       opts.option_dielect_func);
}
}  // namespace librpa_int
