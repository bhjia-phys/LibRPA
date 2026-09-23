#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "../core/crpa.h"
#include "../core/crpa_window_input.h"
#include "dataset_helper.h"
#include "instance_manager.h"
#include "librpa_crpa.h"

struct LibrpaCrpaResult
{
    librpa_int::CrpaResult values;
    std::vector<std::vector<std::vector<int>>> response_bands;
    std::vector<std::vector<std::vector<int>>> orbital_bands;
};

namespace
{
void collective_error(const librpa_int::MpiCommHandler& comm, const std::string& error)
{
    const int failed = !error.empty();
    int any_failed = 0;
    comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed)
        throw LIBRPA_RUNTIME_ERROR("cRPA input: " +
                                   (error.empty() ? "another MPI rank failed validation" : error));
}
}  // namespace

void librpa_init_crpa_input(LibrpaCrpaInput* input)
{
    if (!input) throw LIBRPA_INVALID_ARGUMENT("null cRPA input");
    *input = {};
    input->parent_orbitals = "d";
    input->output_orbitals = "d";
    input->gram_abs = input->gram_rel = input->s_rel = 1.0e-12;
    input->gram_cond_max = 1.0e12;
    input->s_abs = input->residual_tol = input->t_roundtrip_tol = 1.0e-10;
    input->s_cond_max = 1.0e10;
}

void librpa_get_crpa_kgrid(const LibrpaHandler* h, int n_kpoints, double* fractional_k)
{
    const auto ds = librpa_int::api::get_dataset_instance(h);
    if (!fractional_k || n_kpoints <= 0 ||
        ds->pbc.kfrac_list.size() != static_cast<std::size_t>(n_kpoints))
        throw LIBRPA_INVALID_ARGUMENT("cRPA k-grid query has inconsistent dimensions");
    for (int k = 0; k < n_kpoints; ++k)
    {
        fractional_k[3 * k] = ds->pbc.kfrac_list[k].x;
        fractional_k[3 * k + 1] = ds->pbc.kfrac_list[k].y;
        fractional_k[3 * k + 2] = ds->pbc.kfrac_list[k].z;
    }
}

LibrpaCrpaResult* librpa_compute_crpa_window(const LibrpaHandler* h, const LibrpaOptions* options,
                                             const LibrpaCrpaInput* input)
{
    using namespace librpa_int;
    auto ds = api::get_dataset_instance(h);
    CrpaWindowOptions window;
    std::vector<std::string> species;
    std::vector<ComplexMatrix> overlap;
    std::string error;
    try
    {
        if (!options || !input) throw LIBRPA_INVALID_ARGUMENT("missing cRPA input or options");
        if (options->use_symmetry_rpa == LIBRPA_SWITCH_ON ||
            options->use_kpara_scf_eigvec == LIBRPA_SWITCH_ON || ds->mf.get_n_spinor() != 1 ||
            (options->parallel_routing != LIBRPA_ROUTING_LIBRI &&
             options->parallel_routing != LIBRPA_ROUTING_AUTO))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA requires scalar, full-grid LIBRI and replicated eigenvectors");
        if (options->replace_w_head == LIBRPA_SWITCH_ON || options->option_dielect_func != 0 ||
            options->tfgrids_type != LIBRPA_TFGRID_MINIMAX || options->n_bands_chi0 >= 0 ||
            options->use_fullcoul_eps != LIBRPA_SWITCH_ON ||
            options->use_fullcoul_wc != LIBRPA_SWITCH_ON)
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA requires native minimax, all loaded bands, full/full Coulomb and no head "
                "correction");
        const bool shrink = options->use_shrink_abfs == LIBRPA_SWITCH_ON;
        if (shrink && (options->use_shrink_chi != LIBRPA_SWITCH_ON ||
                       !ds->basis_aux_shrink.initialized() || ds->sinvS.empty()))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA compressed basis requires use_shrink_chi and loaded auxiliary transforms");
        if (!shrink && options->use_shrink_chi == LIBRPA_SWITCH_ON)
            throw LIBRPA_INVALID_ARGUMENT("use_shrink_chi requires use_shrink_abfs for cRPA");
        if (!std::isfinite(options->sqrt_coulomb_threshold) ||
            options->sqrt_coulomb_threshold < 0.0)
            throw LIBRPA_INVALID_ARGUMENT("cRPA Coulomb threshold must be finite and nonnegative");
        if (input->n_species <= 0 || !input->species_labels || !input->correlated_species ||
            !input->parent_orbitals || !input->output_orbitals || !input->overlap_k_ri ||
            input->n_kpoints != ds->mf.get_n_kpoints() || input->n_aos != ds->mf.get_n_aos())
            throw LIBRPA_INVALID_ARGUMENT("incomplete cRPA species/window/overlap input");
        for (int type = 0; type < input->n_species; ++type)
        {
            if (!input->species_labels[type]) throw LIBRPA_INVALID_ARGUMENT("null species label");
            species.emplace_back(input->species_labels[type]);
        }
        window.correlated_species = input->correlated_species;
        window.ligand_species = input->ligand_species ? input->ligand_species : "";
        window.parent_orbitals = input->parent_orbitals;
        window.output_orbitals = input->output_orbitals;
        const auto copy_array = [](int count, const auto* source, auto& destination)
        {
            if (count < 0 || (count > 0 && !source))
                throw LIBRPA_INVALID_ARGUMENT("invalid cRPA selection array");
            if (count > 0) destination.assign(source, source + count);
        };
        copy_array(input->n_response_edges, input->response_windows_ha, window.response_windows_ha);
        copy_array(input->n_orbital_edges, input->orbital_windows_ha, window.orbital_windows_ha);
        copy_array(input->n_response_bands, input->response_bands, window.response_bands);
        copy_array(input->n_orbital_bands, input->orbital_bands, window.orbital_bands);
        window.gram_abs = input->gram_abs;
        window.gram_rel = input->gram_rel;
        window.gram_cond_max = input->gram_cond_max;
        window.s_abs = input->s_abs;
        window.s_rel = input->s_rel;
        window.s_cond_max = input->s_cond_max;
        window.residual_tol = input->residual_tol;
        window.t_roundtrip_tol = input->t_roundtrip_tol;
        const std::size_t block_size = static_cast<std::size_t>(input->n_aos) * input->n_aos;
        for (int k = 0; k < input->n_kpoints; ++k)
        {
            overlap.emplace_back(input->n_aos, input->n_aos);
            for (std::size_t i = 0; i < block_size; ++i)
            {
                const auto offset = 2 * (static_cast<std::size_t>(k) * block_size + i);
                overlap.back().c[i] = {input->overlap_k_ri[offset],
                                       input->overlap_k_ri[offset + 1]};
            }
        }
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    collective_error(ds->comm_h, error);

    CrpaWindowBuildResult built;
    std::vector<std::unique_ptr<CorrelatedSubspace>> frames;
    Chi0::BandSelection selection;
    error.clear();
    try
    {
        std::vector<int> atom_types(ds->basis_wfc.n_atoms);
        for (std::size_t atom = 0; atom < atom_types.size(); ++atom)
            atom_types[atom] = ds->atoms.types.at(atom);
        built = build_crpa_window_input(ds->basis_wfc, ds->basis_convention, atom_types, species,
                                        overlap, ds->pbc.kfrac_list, ds->mf, window);
        selection.resize(ds->mf.get_n_spins());
        for (int spin = 0; spin < ds->mf.get_n_spins(); ++spin)
        {
            auto frame = std::make_unique<CorrelatedSubspace>(
                built.sites, ds->pbc.kfrac_list, ds->pbc.Rlist, ds->mf.get_n_aos(),
                ds->mf.get_n_bands(), 1, window.gram_abs, window.gram_rel, window.gram_cond_max,
                window.s_abs, window.s_rel, window.s_cond_max, window.residual_tol,
                window.t_roundtrip_tol);
            selection[spin].resize(ds->mf.get_n_kpoints());
            for (int k = 0; k < ds->mf.get_n_kpoints(); ++k)
            {
                frame->set_S_k(k, overlap[k]);
                frame->set_W_k(k, built.orbitals_spin_k[spin][k]);
                selection[spin][k].assign(ds->mf.get_n_bands(), 0);
                for (const int band : built.response_bands[spin][k]) selection[spin][k][band] = 1;
            }
            frame->compute_T_R();
            frames.push_back(std::move(frame));
        }
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    collective_error(ds->comm_h, error);

    const auto& opts = *options;
    initialize_ds_global_ddla(*ds, opts);
    initialize_ds_tfgrids(*ds, opts);
    initialize_ds_atpairs_local(*ds, LIBRPA_ROUTING_LIBRI);
    ds->redistribute_coulomb_blacs2ap();
    initialize_ds_chi0(*ds, opts);
    const bool shrink = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    CrpaContext context{*ds->p_chi0,   ds->cs_data,
                        ds->basis_aux, ds->atpairs_local,
                        ds->sinvS,     ds->vq,
                        ds->blacs_h,   shrink ? ds->desc_abf_shrink : ds->desc_abf,
                        shrink,        opts.sqrt_coulomb_threshold};
    std::vector<CorrelatedSubspace*> frame_views;
    for (const auto& frame : frames) frame_views.push_back(frame.get());
    auto result = std::make_unique<LibrpaCrpaResult>();
    result->values = compute_crpa_onsite(context, selection, frame_views);
    result->response_bands = std::move(built.response_bands);
    result->orbital_bands = std::move(built.orbital_bands);
    return result.release();
}

void librpa_delete_crpa_result(LibrpaCrpaResult* result) { delete result; }

int librpa_crpa_result_size(const LibrpaCrpaResult* result)
{
    if (!result) throw LIBRPA_INVALID_ARGUMENT("null cRPA result");
    const auto count = result->values.tensors.size() * result->values.frequencies.size();
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("cRPA tensor count exceeds the public integer range");
    return static_cast<int>(count);
}

void librpa_crpa_result_tensor(const LibrpaCrpaResult* result, int index, LibrpaCrpaTensor* tensor)
{
    if (!tensor || index < 0 || index >= librpa_crpa_result_size(result))
        throw LIBRPA_INVALID_ARGUMENT("invalid cRPA tensor query");
    const auto nf = result->values.frequencies.size();
    const auto& record = result->values.tensors[index / nf];
    const auto& site = result->values.sites.at(record.site_index);
    tensor->atom_index = site.atom_index;
    tensor->spin_left = record.spin_left;
    tensor->spin_right = record.spin_right;
    tensor->n_orbitals = site.n_orbitals;
    tensor->frequency_ha = result->values.frequencies[index % nf];
    tensor->bare_ri = reinterpret_cast<const double*>(record.bare.c);
    tensor->partially_screened_ri = reinterpret_cast<const double*>(record.u.at(index % nf).c);
    tensor->fully_screened_ri = reinterpret_cast<const double*>(record.w.at(index % nf).c);
}

void librpa_crpa_result_bands(const LibrpaCrpaResult* result, int response_window, int spin,
                              int kpoint, int* count, const int** bands)
{
    if (!result || !count || !bands || spin < 0 || kpoint < 0)
        throw LIBRPA_INVALID_ARGUMENT("invalid cRPA window query");
    const auto& row =
        (response_window ? result->response_bands : result->orbital_bands).at(spin).at(kpoint);
    *count = static_cast<int>(row.size());
    *bands = row.data();
}
