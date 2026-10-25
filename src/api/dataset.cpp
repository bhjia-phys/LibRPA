#include "dataset.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <cstring>
#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "../core/utils_atomic_basis_blacs.h"
#include "../core/meanfield_mpi.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/profiler.h"
#include "../mpi/collective_stage.h"
#include "../io/stl_io_helper.h"

namespace librpa_int
{

namespace
{

constexpr std::size_t input_transfer_chunk =
    (128ULL * 1024ULL * 1024ULL) / sizeof(cplxdb);

void send_complete_wfc(const cplxdb *data, std::size_t size, int destination,
                        int tag, const MpiCommHandler &world);
void receive_complete_wfc(cplxdb *data, std::size_t size, int source,
                           int tag, const MpiCommHandler &world);

// Only newly supplied copies (pending updates), or the extant physical copies
// during a representation conversion, participate in this agreement.  Old
// replicated values must never be mistaken for new update owners.
int select_complete_wfc_source(const ComplexMatrix *local, int states, int aos,
                                 const MpiCommHandler &world,
                                 const std::string &label, bool required,
                                 int preferred = -1)
{
    const int present = local != nullptr;
    int owners = 0;
    world.allreduce(&present, &owners, 1, MPI_SUM);
    if (!owners)
    {
        if (required) throw LIBRPA_INVALID_ARGUMENT(label + ": missing complete wave-function input");
        return -1;
    }
    const int candidate = present ? (world.myid == preferred ? -1 : world.myid)
                                  : world.nprocs;
    int selected = -1;
    world.allreduce(&candidate, &selected, 1, MPI_MIN);
    const int source = selected == -1 ? preferred : selected;
    const std::size_t size = static_cast<std::size_t>(states) * aos;
    collective_all_stage(world, label + " complete-matrix validation", [&]
    {
        if (states <= 0 || aos <= 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw LIBRPA_INVALID_ARGUMENT("Complete wave-function input dimensions exceed its matrix index range");
        if (!local) return;
        if (local->nr != states || local->nc != aos ||
            static_cast<std::size_t>(local->size) != size || !local->c)
            throw LIBRPA_INVALID_ARGUMENT("Wave-function input is not a complete band-by-AO matrix");
        for (std::size_t i = 0; i < size; ++i)
            if (!std::isfinite(local->c[i].real()) || !std::isfinite(local->c[i].imag()))
                throw LIBRPA_INVALID_ARGUMENT("Complete wave-function input contains nonfinite entries");
    });
    if (owners > 1)
    {
        std::vector<cplxdb> chunk;
        collective_all_stage(world, label + " comparison workspace", [&]
        { chunk.resize(std::min(input_transfer_chunk, size)); });
        int unequal = 0;
        for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
        {
            const auto count = std::min(input_transfer_chunk, size - offset);
            if (world.myid == source) std::copy_n(local->c + offset, count, chunk.data());
            world.bcast(chunk.data(), static_cast<int>(count), source);
            if (local)
                for (std::size_t i = 0; i < count; ++i)
                    if (local->c[offset + i] != chunk[i]) unequal = 1;
        }
        int any_unequal = 0;
        world.allreduce(&unequal, &any_unequal, 1, MPI_MAX);
        if (any_unequal)
            throw LIBRPA_INVALID_ARGUMENT(label + ": conflicting complete wave-function copies");
    }
    return source;
}

void erase_wfc_key(MeanField &field, int spin, int component, int k)
{
    auto spin_data = field.get_eigenvectors().find(spin);
    if (spin_data == field.get_eigenvectors().end()) return;
    auto component_data = spin_data->second.find(component);
    if (component_data == spin_data->second.end()) return;
    component_data->second.erase(k);
    if (component_data->second.empty()) spin_data->second.erase(component_data);
    if (spin_data->second.empty()) field.get_eigenvectors().erase(spin_data);
}

void redistribute_meanfield_eigvecs_kpara(MeanField &field,
                                          const MpiCommHandler &world,
                                          const KPointBlacsParallelContext &context,
                                          const std::string &label,
                                          const int tag,
                                          std::set<std::array<int, 3>> &root_keys)
{
    int metadata[5] = {field.get_n_spins(), field.get_n_spinor(),
                       field.get_n_kpoints(), field.get_n_states(), field.get_n_aos()};
    int minimum[5], maximum[5];
    MPI_Allreduce(metadata, minimum, 5, MPI_INT, MPI_MIN, world.comm);
    MPI_Allreduce(metadata, maximum, 5, MPI_INT, MPI_MAX, world.comm);
    for (int i = 0; i < 5; ++i)
        if (minimum[i] != maximum[i])
            throw LIBRPA_RUNTIME_ERROR(label + " mean-field metadata differs across ranks");
    collective_all_stage(world, label + " k-root input layout", [&]
    {
        if (!field.initialized() || !context.is_initialized() ||
            context.n_kpoints() != field.get_n_kpoints())
            throw LIBRPA_RUNTIME_ERROR(label + " has no compatible k-group input layout");
    });
    const std::size_t size = static_cast<std::size_t>(field.get_n_states()) * field.get_n_aos();
    for (int spin = 0; spin < field.get_n_spins(); ++spin)
        for (int component = 0; component < field.get_n_spinor(); ++component)
            for (int k = 0; k < field.get_n_kpoints(); ++k)
            {
                const int destination = context.kpoint_blacs_root_global_rank(k);
                const auto *local = field.find_wfc(spin, component, k);
                const std::string key_label = label + " spin/component/k=" +
                    std::to_string(spin) + "/" + std::to_string(component) + "/" + std::to_string(k);
                const int source = select_complete_wfc_source(
                    local, field.get_n_states(), field.get_n_aos(), world, key_label, true, destination);
                ComplexMatrix incoming;
                ComplexMatrix *target = nullptr;
                const std::array<int, 3> key{spin, component, k};
                const bool already_root = root_keys.count(key) != 0;
                try
                {
                    collective_all_stage(world, key_label + " k-root workspace", [&]
                    {
                        if (destination < 0 || destination >= world.nprocs)
                            throw LIBRPA_RUNTIME_ERROR("Invalid wave-function k-root owner");
                        if (world.myid == destination)
                        {
                            target = &field.get_eigenvectors()[spin][component][k];
                            if (source != destination)
                                incoming.create(field.get_n_states(), field.get_n_aos(), false);
                        }
                        root_keys.insert(key);
                    });
                }
                catch (...)
                {
                    if (!already_root) root_keys.erase(key);
                    if (world.myid == destination && !local) erase_wfc_key(field, spin, component, k);
                    throw;
                }
                if (source != destination)
                {
                    if (world.myid == source) send_complete_wfc(local->c, size, destination, tag, world);
                    else if (world.myid == destination)
                        receive_complete_wfc(incoming.c, size, source, tag, world);
                }
                if (world.myid == destination)
                {
                    if (source != destination) *target = std::move(incoming);
                }
                else erase_wfc_key(field, spin, component, k);
            }
}

bool any_change(std::uint32_t changes, InputChange mask)
{
    return (changes & static_cast<std::uint32_t>(mask)) != 0;
}

bool exx_recipe_differs(const LibrpaOptions &left, const LibrpaOptions &right,
                          int atoms, int cells)
{
    const auto resolve = [&](const LibrpaOptions &opts)
    {
        return opts.parallel_routing == LIBRPA_ROUTING_AUTO
                   ? decide_auto_routing(atoms, opts.nfreq * cells)
                   : opts.parallel_routing;
    };
    return resolve(left) != resolve(right) || left.parallel_routing != right.parallel_routing ||
           left.use_kpara_scf_eigvec != right.use_kpara_scf_eigvec ||
           left.use_fullcoul_exx != right.use_fullcoul_exx ||
           left.use_shrink_abfs != right.use_shrink_abfs ||
           left.use_symmetry_exx != right.use_symmetry_exx ||
           left.libri_exx_threshold_C != right.libri_exx_threshold_C ||
           left.libri_exx_threshold_D != right.libri_exx_threshold_D ||
           left.libri_exx_threshold_V != right.libri_exx_threshold_V;
}

bool tf_recipe_differs(const LibrpaOptions &left, const LibrpaOptions &right)
{
    return left.tfgrids_type != right.tfgrids_type || left.nfreq != right.nfreq ||
           left.tfgrids_freq_min != right.tfgrids_freq_min ||
           left.tfgrids_freq_interval != right.tfgrids_freq_interval ||
           left.tfgrids_freq_max != right.tfgrids_freq_max ||
           left.tfgrids_time_min != right.tfgrids_time_min ||
           left.tfgrids_time_interval != right.tfgrids_time_interval ||
           left.minimax_emin != right.minimax_emin ||
           left.minimax_emax != right.minimax_emax ||
           left.minimax_regulation != right.minimax_regulation;
}

bool optical_recipe_differs(const LibrpaOptions &left, const LibrpaOptions &right)
{
    return tf_recipe_differs(left, right) ||
           left.use_kpara_scf_eigvec != right.use_kpara_scf_eigvec ||
           left.replace_w_head != right.replace_w_head ||
           left.option_dielect_func != right.option_dielect_func ||
           left.use_2d_dielectric != right.use_2d_dielectric ||
           left.use_symmetry_rpa != right.use_symmetry_rpa ||
           left.use_symmetry_gw != right.use_symmetry_gw ||
           left.use_shrink_abfs != right.use_shrink_abfs ||
           left.use_shrink_chi != right.use_shrink_chi ||
           left.sqrt_coulomb_threshold != right.sqrt_coulomb_threshold;
}

bool response_recipe_differs(const LibrpaOptions &left, const LibrpaOptions &right)
{
    const bool source_changed = left.read_sigc_mat_rf != right.read_sigc_mat_rf ||
        (right.read_sigc_mat_rf == LIBRPA_SWITCH_ON &&
         (std::strcmp(left.restart_from_dir, right.restart_from_dir) != 0 ||
          (right.restart_from_dir[0] == '\0' &&
           std::strcmp(left.output_dir, right.output_dir) != 0)));
    return tf_recipe_differs(left, right) || optical_recipe_differs(left, right) ||
           left.parallel_routing != right.parallel_routing ||
           left.use_fullcoul_eps != right.use_fullcoul_eps ||
           left.use_fullcoul_wc != right.use_fullcoul_wc ||
           left.use_scalapack_gw_wc != right.use_scalapack_gw_wc ||
           left.use_cholesky_gw_wc != right.use_cholesky_gw_wc ||
           left.use_elpa_sqrt_coulomb != right.use_elpa_sqrt_coulomb ||
           left.use_gpu_replace_scalapack != right.use_gpu_replace_scalapack ||
           left.n_bands_chi0 != right.n_bands_chi0 ||
           left.n_bands_sigc != right.n_bands_sigc ||
           left.n_bands_exclude_chi0 != right.n_bands_exclude_chi0 ||
           left.n_bands_exclude_sigc != right.n_bands_exclude_sigc ||
           left.gf_threshold != right.gf_threshold ||
           left.libri_chi0_threshold_C != right.libri_chi0_threshold_C ||
           left.libri_chi0_threshold_G != right.libri_chi0_threshold_G ||
           left.libri_g0w0_threshold_C != right.libri_g0w0_threshold_C ||
           left.libri_g0w0_threshold_G != right.libri_g0w0_threshold_G ||
           left.libri_g0w0_threshold_Wc != right.libri_g0w0_threshold_Wc ||
           source_changed;
}

bool target_recipe_differs(const LibrpaOptions &left, const LibrpaOptions &right)
{
    return left.option_bvk_remap != right.option_bvk_remap ||
           left.output_gw_sigc_ks_mat_kf != right.output_gw_sigc_ks_mat_kf ||
           left.output_exx_ks_mat_k != right.output_exx_ks_mat_k ||
           left.istate_output_mat_start != right.istate_output_mat_start ||
           left.istate_output_mat_end != right.istate_output_mat_end;
}

void send_complete_wfc(const cplxdb *data, std::size_t size, int destination,
                        int tag, const MpiCommHandler &world)
{
    for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
        MPI_Send(data + offset,
                 static_cast<int>(std::min(input_transfer_chunk, size - offset)),
                 mpi_datatype<cplxdb>::value, destination, tag, world.comm);
}

void receive_complete_wfc(cplxdb *data, std::size_t size, int source,
                           int tag, const MpiCommHandler &world)
{
    for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
        MPI_Recv(data + offset,
                 static_cast<int>(std::min(input_transfer_chunk, size - offset)),
                 mpi_datatype<cplxdb>::value, source, tag, world.comm,
                 MPI_STATUS_IGNORE);
}

} // namespace

Dataset::Dataset(MPI_Comm comm, const bool input_blacs_matloc_row_major)
    : input_blacs_matloc_row_major_(input_blacs_matloc_row_major),
      comm_blacs_coul_initialized_(false),
      coul_blacs2ap_redistributed_(false),
      eigvecs_kpara_redistributed_(false),
      eigvecs_kpara_2d_redistributed_(false),
      band_eigvecs_kpara_2d_redistributed_(false),
      comm_h(comm, true),
      blacs_h(comm),
      scfk_blacs_ctxt(),
      bandk_blacs_ctxt(),
      comm_coul_h(),
      comm_coul_inter_q_h(),
      comm_coul_intra_q_h(),
      blacs_coul_intra_q_h(),
      desc_coul_intra_q(),
      desc_wfc(),
      desc_wfc_kb(),
      desc_wfc_kb_full(),
      desc_band_wfc_kb(),
      desc_band_wfc_kb_full(),
      desc_abf(),
      desc_abf_shrink(),
      atpairs_local(), atpairs_unique_all(),
      Rs_local(),
      basis_wfc(),
      basis_aux(),
      basis_aux_shrink(),
      basis_convention(),
      spg_symops(),
      symmetry_context(),
      atoms(),
      pbc(),
      kfrac_band_list(),
      sinvS(),
      mf(),
      velocity_matrix(),
      mf_band(),
      band_data_id(0),
      is_band_data_set(false),
      is_band_calc_done(false),
      tfg(),
      cs_data(),
      cs_data_shrink(),
      vq(), vq_cut(),
      vq_lbrow(-1),  vq_ubrow(-1), vq_lbcol(-1), vq_ubcol(-1),
      vq_block_loc(), vq_cut_block_loc(),
      epsmacs_imagfreq(),
      omegas_imagfreq(),
      p_headwing(nullptr),
      p_exx(nullptr),
      p_chi0(nullptr),
      p_g0w0(nullptr)
{
    basis_wfc.label = "WFC";
    basis_aux.label = "AUX";
    basis_aux_shrink.label = "AUXSHRINK";
    blacs_h.init();
    // TODO more flexible process grid initialization
    blacs_h.set_square_grid();
}

void Dataset::free()
{
    // Numerical objects may borrow these descriptors/contexts.  Destroy them
    // before releasing the communicators they describe.
    p_headwing.reset();
    p_exx.reset();
    p_chi0.reset();
    p_g0w0.reset();
    finalize_comm_blacs_coul();
    desc_wfc_kb.reset_handler();
    desc_wfc_kb_full.reset_handler();
    desc_band_wfc_kb.reset_handler();
    desc_band_wfc_kb_full.reset_handler();
    bandk_blacs_ctxt.finalize();
    scfk_blacs_ctxt.finalize();
}

void Dataset::note_input_change(InputChange change) noexcept
{
    pending_input_changes_ |= static_cast<std::uint32_t>(change);
}

bool Dataset::has_pending_optical_input_changes() const noexcept
{
    return any_change(pending_input_changes_,
                      InputChange::Spectrum | InputChange::Occupations |
                      InputChange::ScfWavefunctions | InputChange::Ri |
                      InputChange::Coulomb | InputChange::AuxProjection |
                      InputChange::Velocity | InputChange::ExternalDielectric |
                      InputChange::Structure | InputChange::DirectOptical);
}

void Dataset::retain_optical_input()
{
    if (!p_headwing) return;
    if (!direct_full_bz_optical_input)
        direct_full_bz_optical_input = p_headwing->get_direct_full_bz_headwing_inputs();
    if (!optical_meanfield_input && p_headwing->get_meanfield_input())
    {
        auto layout = p_headwing->get_meanfield_input_layout();
        optical_meanfield_layout = std::move(layout);
        optical_meanfield_input = p_headwing->get_meanfield_input();
    }
    if (!(strict_2d_auxiliary_monopole_norm_squared > 0.0) &&
        p_headwing->strict_2d_coulomb_head_configured())
        strict_2d_auxiliary_monopole_norm_squared =
            p_headwing->get_strict_2d_auxiliary_monopole_norm_squared();
    // An explicitly selected optical_meanfield_input is already an independent
    // shared physical owner.  The ordinary derived mf snapshot is not promoted
    // to an external input here.
}

void Dataset::invalidate_input_dependents(std::uint32_t changes,
                                           bool exx_recipe_changed,
                                           bool optical_recipe_changed,
                                           bool response_recipe_changed,
                                           bool target_recipe_changed)
{
    const bool changed_density_or_vertices = any_change(
        changes, InputChange::Occupations | InputChange::ScfWavefunctions |
                 InputChange::Ri | InputChange::Coulomb |
                 InputChange::AuxProjection | InputChange::Structure);
    const bool changed_response = changed_density_or_vertices || any_change(
        changes, InputChange::Spectrum | InputChange::Velocity |
                 InputChange::ExternalDielectric | InputChange::DirectOptical);
    if (changed_density_or_vertices || exx_recipe_changed) p_exx.reset();
    if (changed_response || response_recipe_changed)
    {
        p_chi0.reset();
        p_g0w0.reset();
    }
    if (changed_response || optical_recipe_changed)
    {
        retain_optical_input();
        p_headwing.reset();
        // Legacy output vectors can alias a caller's output buffer during
        // this handoff.  Keep their storage stable; consumers select physical
        // external samples or publish new analytic values after preparation.
    }
    if (any_change(changes, InputChange::Structure)) symmetry_context.clear();
    if (changed_response || exx_recipe_changed || response_recipe_changed ||
        target_recipe_changed || any_change(changes, InputChange::Band))
    {
        is_band_calc_done = false;
        if (p_exx) p_exx->reset_kspace();
        if (p_g0w0) p_g0w0->reset_kspace();
    }
}

ComplexMatrix &Dataset::dense_wfc_input(bool band, int spin, int component, int k)
{
    auto &field = band ? mf_band : mf;
    if (!field.initialized() || spin < 0 || spin >= field.get_n_spins() ||
        component < 0 || component >= field.get_n_spinor() ||
        k < 0 || k >= field.get_n_kpoints())
        throw LIBRPA_INVALID_ARGUMENT("Invalid complete wave-function input key");
    note_input_change(band ? InputChange::Band : InputChange::ScfWavefunctions);
    const bool distributed = band ? (band_eigvecs_kpara_redistributed_ || band_eigvecs_replicated_ ||
                                      !band_root_dense_keys_.empty() || !band_replicated_keys_.empty())
                                  : (eigvecs_kpara_redistributed_ || eigvecs_replicated_ ||
                                      !scf_root_dense_keys_.empty() || !scf_replicated_keys_.empty());
    if (distributed)
        return (band ? pending_band_dense_updates_ : pending_scf_dense_updates_)
            [std::array<int, 3>{spin, component, k}];
    return field.get_eigenvectors()[spin][component][k];
}

void Dataset::note_coulomb_2d_update(bool cut, int iq)
{
    if (iq < 0 || iq >= static_cast<int>(pbc.klist.size()))
        throw LIBRPA_INVALID_ARGUMENT("Invalid 2D Coulomb input q index");
    note_input_change(InputChange::Coulomb);
    pending_coulomb_2d_updates_.insert({cut, iq});
}

void Dataset::synchronize_input_changes(const LibrpaOptions *opts)
{
    constexpr unsigned int exx_ready_bit = 1u << 11;
    constexpr unsigned int exx_absent_bit = 1u << 12;
    constexpr unsigned int exx_incomplete_bit = 1u << 13;
    constexpr unsigned int gw_ready_bit = 1u << 14;
    constexpr unsigned int gw_absent_bit = 1u << 15;
    constexpr unsigned int gw_incomplete_bit = 1u << 16;
    constexpr unsigned int target_recipe_bit = 1u << 26;
    constexpr unsigned int exx_recipe_bit = 1u << 27;
    constexpr unsigned int optical_recipe_bit = 1u << 28;
    constexpr unsigned int response_recipe_bit = 1u << 29;
    constexpr unsigned int with_options_bit = 1u << 30;
    constexpr unsigned int without_options_bit = 1u << 31;
    constexpr unsigned int physical_mask = (1u << 11) - 1;
    unsigned int local_changes = pending_input_changes_ |
        (opts ? with_options_bit : without_options_bit);
    local_changes |= !p_exx ? exx_absent_bit
                            : (p_exx->is_rspace_built() ? exx_ready_bit : exx_incomplete_bit);
    local_changes |= !p_g0w0 ? gw_absent_bit
                             : (p_g0w0->is_rspace_built() ? gw_ready_bit : gw_incomplete_bit);
    if (opts && have_compute_options_)
    {
        if (exx_recipe_differs(last_compute_options_, *opts, atoms.size(),
                                pbc.get_n_cells_bvk())) local_changes |= exx_recipe_bit;
        if (optical_recipe_differs(last_compute_options_, *opts)) local_changes |= optical_recipe_bit;
        if (response_recipe_differs(last_compute_options_, *opts)) local_changes |= response_recipe_bit;
        if (target_recipe_differs(last_compute_options_, *opts)) local_changes |= target_recipe_bit;
    }
    unsigned int global_changes = 0;
    MPI_Allreduce(&local_changes, &global_changes, 1, MPI_UNSIGNED, MPI_BOR, comm_h.comm);
    if ((global_changes & with_options_bit) && (global_changes & without_options_bit))
        throw LIBRPA_INVALID_ARGUMENT("Collective input synchronization needs the same options participation");
    const std::uint32_t changes = global_changes & physical_mask;
    const bool exx_changed = (global_changes & exx_recipe_bit) != 0;
    const bool optical_changed = (global_changes & optical_recipe_bit) != 0;
    const bool response_changed = (global_changes & response_recipe_bit) != 0;
    const bool target_changed = (global_changes & target_recipe_bit) != 0;
    const bool discard_exx = (global_changes & exx_incomplete_bit) ||
        ((global_changes & exx_ready_bit) && (global_changes & exx_absent_bit));
    const bool discard_gw = (global_changes & gw_incomplete_bit) ||
        ((global_changes & gw_ready_bit) && (global_changes & gw_absent_bit));
    if (discard_exx) p_exx.reset();
    if (discard_gw) p_g0w0.reset();
    if (discard_exx || discard_gw) is_band_calc_done = false;
    // Invalidate before transfers or a shared-grid reset can fail.  A failed
    // handoff keeps the local dirty marks and remaining input payloads.
    if (changes || exx_changed || optical_changed || response_changed || target_changed)
        collective_all_stage(comm_h, "Invalidate changed input results", [&]
        { invalidate_input_dependents(changes, exx_changed, optical_changed,
                                        response_changed, target_changed); });
    if (any_change(changes, InputChange::ScfWavefunctions)) apply_dense_wfc_updates(false);
    if (any_change(changes, InputChange::Band)) apply_dense_wfc_updates(true);
    if (any_change(changes, InputChange::Coulomb)) apply_coulomb_2d_updates();

    if (opts)
    {
        auto routing = opts->parallel_routing;
        if (routing == LIBRPA_ROUTING_AUTO)
            routing = decide_auto_routing(atoms.size(), opts->nfreq * pbc.get_n_cells_bvk());
        const bool need_tiles = opts->use_kpara_scf_eigvec == LIBRPA_SWITCH_ON &&
                                routing == LIBRPA_ROUTING_LIBRI;
        if (opts->use_kpara_scf_eigvec != LIBRPA_SWITCH_ON)
        {
            // Fresh raw input may still be incomplete during TF/head-only
            // preparation; actual OFF coefficient consumers call the explicit
            // replication hook.  Completed layouts can be converted here.
            if (eigvecs_kpara_redistributed_ || eigvecs_replicated_)
                replicate_eigvecs();
            // Raw band input can legitimately be incomplete during SCF work.
            // The OFF band getter explicitly calls replicate_band_eigvecs().
            if (band_eigvecs_kpara_redistributed_ || band_eigvecs_replicated_)
                replicate_band_eigvecs();
        }
        else if (!need_tiles)
        {
            if (eigvecs_kpara_2d_redistributed_) collect_wfc_to_kroots(false);
            if (band_eigvecs_kpara_2d_redistributed_) collect_wfc_to_kroots(true);
        }
        last_compute_options_ = *opts;
        have_compute_options_ = true;
    }
    pending_input_changes_ = 0;
}

void Dataset::apply_dense_wfc_updates(bool band)
{
    auto &pending = band ? pending_band_dense_updates_ : pending_scf_dense_updates_;
    const int local_count = static_cast<int>(pending.size());
    int total_count = 0;
    comm_h.allreduce(&local_count, &total_count, 1, MPI_SUM);
    if (!total_count) return;
    auto &field = band ? mf_band : mf;
    auto &context = band ? bandk_blacs_ctxt : scfk_blacs_ctxt;
    const auto &source_desc = band ? desc_band_wfc_kb_full : desc_wfc_kb_full;
    const auto &target_desc = band ? desc_band_wfc_kb : desc_wfc_kb;
    const bool tiled = band ? band_eigvecs_kpara_2d_redistributed_
                            : eigvecs_kpara_2d_redistributed_;
    const bool replicated = band ? (band_eigvecs_replicated_ || !band_replicated_keys_.empty())
                                  : (eigvecs_replicated_ || !scf_replicated_keys_.empty());
    int metadata[7] = {field.get_n_spins(), field.get_n_spinor(),
                       field.get_n_kpoints(), field.get_n_states(),
                       field.get_n_aos(), tiled, replicated};
    int minimum[7], maximum[7];
    MPI_Allreduce(metadata, minimum, 7, MPI_INT, MPI_MIN, comm_h.comm);
    MPI_Allreduce(metadata, maximum, 7, MPI_INT, MPI_MAX, comm_h.comm);
    for (int i = 0; i < 7; ++i)
        if (minimum[i] != maximum[i])
            throw LIBRPA_RUNTIME_ERROR("Pending wave-function input metadata differs across ranks");
    collective_all_stage(comm_h, "Pending wave-function input layout", [&]
    {
        if (!field.initialized() || (replicated && tiled) ||
            (!replicated && (!context.is_initialized() ||
                            context.n_kpoints() != field.get_n_kpoints())))
            throw LIBRPA_RUNTIME_ERROR("Pending wave-function input has no compatible coefficient layout");
        if (tiled && (!source_desc.is_initialized() || !target_desc.is_initialized() ||
            source_desc.ictxt() != context.blacs_h.ictxt ||
            target_desc.ictxt() != context.blacs_h.ictxt ||
            source_desc.m() != field.get_n_aos() ||
            source_desc.n() != field.get_n_states() ||
            target_desc.m() != field.get_n_aos() ||
            target_desc.n() != field.get_n_states() ||
            source_desc.mb() != field.get_n_aos() ||
            source_desc.nb() != field.get_n_states() ||
            source_desc.irsrc() != 0 || source_desc.icsrc() != 0))
            throw LIBRPA_RUNTIME_ERROR("Pending wave-function descriptors do not match the live input");
        if (static_cast<std::size_t>(field.get_n_states()) * field.get_n_aos() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw LIBRPA_INVALID_ARGUMENT("Complete wave-function input exceeds its matrix index range");
        for (const auto &[key, matrix] : pending)
            if (key[0] < 0 || key[0] >= field.get_n_spins() ||
                key[1] < 0 || key[1] >= field.get_n_spinor() ||
                key[2] < 0 || key[2] >= field.get_n_kpoints())
                throw LIBRPA_INVALID_ARGUMENT("Pending wave-function key is incompatible with the current mean field");
    });
    const std::size_t size = static_cast<std::size_t>(field.get_n_states()) * field.get_n_aos();
    // The globally ordered (spin,component,k) scan is small metadata, never a
    // concatenated copy of all wavefunctions.
    for (int spin = 0; spin < field.get_n_spins(); ++spin)
        for (int component = 0; component < field.get_n_spinor(); ++component)
            for (int k = 0; k < field.get_n_kpoints(); ++k)
            {
                const std::array<int, 3> key{spin, component, k};
                auto found = pending.find(key);
                const int present = found != pending.end();
                const auto &root_keys = band ? band_root_dense_keys_ : scf_root_dense_keys_;
                const auto &replica_keys = band ? band_replicated_keys_ : scf_replicated_keys_;
                const bool key_replicated = replicated || replica_keys.count(key) != 0;
                const bool key_tiled = tiled && root_keys.count(key) == 0 && !key_replicated;
                const int destination = key_replicated ? -1 : context.kpoint_blacs_root_global_rank(k);
                const int source = select_complete_wfc_source(
                    present ? &found->second : nullptr, field.get_n_states(), field.get_n_aos(),
                    comm_h, "Pending input spin/component/k=" + std::to_string(spin) + "/" +
                        std::to_string(component) + "/" + std::to_string(k), false, destination);
                if (source < 0) continue;
                if (key_replicated)
                {
                    ComplexMatrix incoming;
                    ComplexMatrix *target = nullptr;
                    collective_all_stage(comm_h, "Replicated wave-function update workspace", [&]
                    {
                        target = &field.get_eigenvectors()[spin][component][k];
                        if (comm_h.myid != source)
                            incoming.create(field.get_n_states(), field.get_n_aos(), false);
                    });
                    for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
                    {
                        cplxdb *buffer = comm_h.myid == source ? found->second.c + offset
                                                              : incoming.c + offset;
                        comm_h.bcast(buffer, static_cast<int>(std::min(input_transfer_chunk, size - offset)), source);
                    }
                    if (comm_h.myid == source) *target = std::move(found->second);
                    else *target = std::move(incoming);
                    if (present) pending.erase(found);
                    continue;
                }
                ComplexMatrix full, local;
                ComplexMatrix *committed = nullptr;
                collective_all_stage(comm_h, "Pending wave-function transfer workspace", [&]
                {
                    if (destination < 0 || destination >= comm_h.nprocs)
                        throw LIBRPA_RUNTIME_ERROR("Pending wave-function has invalid target owner");
                    if (comm_h.myid == destination && source != destination)
                        full.create(field.get_n_states(), field.get_n_aos(), false);
                    if (context.owns_kpoint(k))
                    {
                        if (key_tiled)
                        {
                            local.create(target_desc.n_loc(), target_desc.m_loc(), false);
                            committed = &field.get_eigenvectors()[spin][component][k];
                        }
                        else if (comm_h.myid == destination)
                            committed = &field.get_eigenvectors()[spin][component][k];
                    }
                });
                if (source != destination)
                {
                    if (comm_h.myid == source)
                        send_complete_wfc(found->second.c, size, destination,
                                            band ? 19324 : 19323, comm_h);
                    else if (comm_h.myid == destination)
                        receive_complete_wfc(full.c, size, source,
                                               band ? 19324 : 19323, comm_h);
                }
                collective_all_stage(comm_h, "Pending wave-function scatter", [&]
                {
                    if (!context.owns_kpoint(k)) return;
                    if (key_tiled)
                    {
                        cplxdb dummy = 0.0;
                        const cplxdb *input = &dummy;
                        if (comm_h.myid == destination)
                            input = source == destination ? found->second.c : full.c;
                        ScalapackConnector::pgemr2d_f(
                            field.get_n_aos(), field.get_n_states(), input, 1, 1,
                            source_desc.desc, local.c ? local.c : &dummy, 1, 1,
                            target_desc.desc, context.blacs_h.ictxt);
                    }
                });
                // All map slots were allocated before communication.  These
                // moves replace only this key and do not allocate a WFC clone.
                if (committed)
                {
                    if (key_tiled) *committed = std::move(local);
                    else if (source == destination) *committed = std::move(found->second);
                    else *committed = std::move(full);
                }
                if (!key_tiled && comm_h.myid != destination)
                    erase_wfc_key(field, spin, component, k);
                if (present) pending.erase(found);
            }
}

void Dataset::collect_wfc_to_kroots(bool band)
{
    auto &field = band ? mf_band : mf;
    auto &context = band ? bandk_blacs_ctxt : scfk_blacs_ctxt;
    const auto &source_desc = band ? desc_band_wfc_kb : desc_wfc_kb;
    const auto &target_desc = band ? desc_band_wfc_kb_full : desc_wfc_kb_full;
    auto &root_keys = band ? band_root_dense_keys_ : scf_root_dense_keys_;
    collective_all_stage(comm_h, "Dense wave-function restoration layout", [&]
    {
        if (!context.is_initialized() || !source_desc.is_initialized() ||
            !target_desc.is_initialized() || context.n_kpoints() != field.get_n_kpoints())
            throw LIBRPA_RUNTIME_ERROR("Cannot restore dense coefficients from an invalid tiled layout");
    });
    for (int spin = 0; spin < field.get_n_spins(); ++spin)
        for (int component = 0; component < field.get_n_spinor(); ++component)
            for (int k = 0; k < field.get_n_kpoints(); ++k)
            {
                const std::array<int, 3> key{spin, component, k};
                if (root_keys.count(key)) continue;
                ComplexMatrix full;
                ComplexMatrix *committed = nullptr;
                try
                {
                    collective_all_stage(comm_h, "Dense wave-function restoration workspace", [&]
                    {
                        root_keys.insert(key);
                        if (!context.owns_kpoint(k)) return;
                        const auto *source = field.find_wfc(spin, component, k);
                        const auto count = static_cast<std::size_t>(source_desc.m_loc()) * source_desc.n_loc();
                        if ((count && !source) || (source &&
                            (source->nr != source_desc.n_loc() || source->nc != source_desc.m_loc() ||
                             static_cast<std::size_t>(source->size) != count)))
                            throw LIBRPA_RUNTIME_ERROR("Cannot restore an incomplete coefficient tile");
                        full.create(target_desc.n_loc(), target_desc.m_loc(), false);
                        if (context.comm_blacs_h.is_root())
                            committed = &field.get_eigenvectors()[spin][component][k];
                    });
                    collective_all_stage(comm_h, "Dense wave-function restoration", [&]
                    {
                        if (!context.owns_kpoint(k)) return;
                        cplxdb dummy = 0.0;
                        const auto *source = field.find_wfc(spin, component, k);
                        ScalapackConnector::pgemr2d_f(
                            field.get_n_aos(), field.get_n_states(),
                            source && source->c ? source->c : &dummy, 1, 1, source_desc.desc,
                            full.c ? full.c : &dummy, 1, 1, target_desc.desc, context.blacs_h.ictxt);
                    });
                }
                catch (...)
                {
                    root_keys.erase(key);
                    throw;
                }
                if (context.owns_kpoint(k))
                {
                    if (committed) *committed = std::move(full);
                    else
                    {
                        auto spin_data = field.get_eigenvectors().find(spin);
                        if (spin_data != field.get_eigenvectors().end())
                        {
                            auto component_data = spin_data->second.find(component);
                            if (component_data != spin_data->second.end()) component_data->second.erase(k);
                        }
                    }
                }
            }
    root_keys.clear();
    if (band)
    {
        band_eigvecs_replicated_ = false;
        band_eigvecs_kpara_2d_redistributed_ = false;
        band_eigvecs_kpara_redistributed_ = true;
    }
    else
    {
        eigvecs_replicated_ = false;
        eigvecs_kpara_2d_redistributed_ = false;
        eigvecs_kpara_redistributed_ = true;
    }
}

void Dataset::collect_eigvecs_kroots()
{
    if (eigvecs_kpara_2d_redistributed_) collect_wfc_to_kroots(false);
    if (!eigvecs_kpara_redistributed_ || !scf_replicated_keys_.empty() ||
        !scf_root_dense_keys_.empty())
        redistribute_meanfield_eigvecs_kpara(
            mf, comm_h, scfk_blacs_ctxt, "Native velocity SCF", 19325, scf_root_dense_keys_);
    eigvecs_replicated_ = false;
    eigvecs_kpara_redistributed_ = true;
    scf_root_dense_keys_.clear();
    scf_replicated_keys_.clear();
}

void Dataset::restore_root_dense_keys_to_tiles(bool band)
{
    auto &field = band ? mf_band : mf;
    auto &context = band ? bandk_blacs_ctxt : scfk_blacs_ctxt;
    const auto &source_desc = band ? desc_band_wfc_kb_full : desc_wfc_kb_full;
    const auto &target_desc = band ? desc_band_wfc_kb : desc_wfc_kb;
    auto &root_keys = band ? band_root_dense_keys_ : scf_root_dense_keys_;
    for (auto key_it = root_keys.begin(); key_it != root_keys.end();)
    {
        const auto key = *key_it;
        const int spin = key[0], component = key[1], k = key[2];
        ComplexMatrix tile;
        ComplexMatrix *target = nullptr;
        collective_all_stage(comm_h, "Root-to-tile coefficient workspace", [&]
        {
            if (!context.is_initialized() || !source_desc.is_initialized() ||
                !target_desc.is_initialized() || source_desc.ictxt() != context.blacs_h.ictxt ||
                target_desc.ictxt() != context.blacs_h.ictxt ||
                source_desc.m() != field.get_n_aos() || source_desc.n() != field.get_n_states() ||
                target_desc.m() != field.get_n_aos() || target_desc.n() != field.get_n_states())
                throw LIBRPA_RUNTIME_ERROR("Root-to-tile wave-function descriptors do not match the input");
            if (!context.owns_kpoint(k)) return;
            const auto *source = field.find_wfc(spin, component, k);
            const auto count = static_cast<std::size_t>(source_desc.m_loc()) * source_desc.n_loc();
            if (count && (!source || source->nr != source_desc.n_loc() ||
                source->nc != source_desc.m_loc() || static_cast<std::size_t>(source->size) != count))
                throw LIBRPA_RUNTIME_ERROR("Root-to-tile transfer lacks a complete source matrix");
            if (count)
                for (std::size_t i = 0; i < count; ++i)
                    if (!std::isfinite(source->c[i].real()) || !std::isfinite(source->c[i].imag()))
                        throw LIBRPA_INVALID_ARGUMENT("Root-to-tile source contains nonfinite coefficients");
            tile.create(target_desc.n_loc(), target_desc.m_loc(), false);
            target = &field.get_eigenvectors()[spin][component][k];
        });
        collective_all_stage(comm_h, "Root-to-tile coefficient transfer", [&]
        {
            if (!context.owns_kpoint(k)) return;
            cplxdb dummy = 0.0;
            const auto *source = field.find_wfc(spin, component, k);
            const auto count = static_cast<std::size_t>(source_desc.m_loc()) * source_desc.n_loc();
            ScalapackConnector::pgemr2d_f(
                field.get_n_aos(), field.get_n_states(),
                count ? source->c : &dummy, 1, 1, source_desc.desc,
                tile.c ? tile.c : &dummy, 1, 1, target_desc.desc, context.blacs_h.ictxt);
        });
        if (target) *target = std::move(tile);
        key_it = root_keys.erase(key_it);
    }
}

void Dataset::replicate_eigvecs()
{
    replicate_dense_wfc(false);
}

void Dataset::replicate_band_eigvecs()
{
    replicate_dense_wfc(true);
}

void Dataset::replicate_dense_wfc(bool band)
{
    auto &field = band ? mf_band : mf;
    const bool replicated = band ? band_eigvecs_replicated_ : eigvecs_replicated_;
    const bool tiled = band ? band_eigvecs_kpara_2d_redistributed_
                            : eigvecs_kpara_2d_redistributed_;
    int metadata[7] = {field.get_n_spins(), field.get_n_spinor(), field.get_n_kpoints(),
                       field.get_n_states(), field.get_n_aos(), replicated, tiled};
    int minimum[7], maximum[7];
    MPI_Allreduce(metadata, minimum, 7, MPI_INT, MPI_MIN, comm_h.comm);
    MPI_Allreduce(metadata, maximum, 7, MPI_INT, MPI_MAX, comm_h.comm);
    for (int i = 0; i < 7; ++i)
        if (minimum[i] != maximum[i])
            throw LIBRPA_RUNTIME_ERROR("Dense replication metadata differs across ranks");
    if (!field.initialized())
        throw LIBRPA_RUNTIME_ERROR("Dense replication requires initialized mean-field dimensions");
    const auto &partial_roots = band ? band_root_dense_keys_ : scf_root_dense_keys_;
    const auto &partial_replicas = band ? band_replicated_keys_ : scf_replicated_keys_;
    if (replicated && partial_roots.empty() && partial_replicas.empty()) return;
    if (tiled) collect_wfc_to_kroots(band);
    const std::size_t size = static_cast<std::size_t>(field.get_n_states()) * field.get_n_aos();
    const std::string label = band ? "Band replicated-dense input" : "SCF replicated-dense input";
    auto &replica_keys = band ? band_replicated_keys_ : scf_replicated_keys_;
    auto &root_keys = band ? band_root_dense_keys_ : scf_root_dense_keys_;
    for (int spin = 0; spin < field.get_n_spins(); ++spin)
        for (int component = 0; component < field.get_n_spinor(); ++component)
            for (int k = 0; k < field.get_n_kpoints(); ++k)
            {
                const std::array<int, 3> key{spin, component, k};
                const bool already_replicated = replica_keys.count(key) != 0;
                auto *local = field.find_wfc(spin, component, k);
                const std::string key_label = label + " spin/component/k=" +
                    std::to_string(spin) + "/" + std::to_string(component) + "/" + std::to_string(k);
                const int source = select_complete_wfc_source(
                    local, field.get_n_states(), field.get_n_aos(), comm_h, key_label, true);
                ComplexMatrix incoming;
                ComplexMatrix *target = local;
                try
                {
                    collective_all_stage(comm_h, key_label + " replication workspace", [&]
                    {
                        if (!local)
                        {
                            incoming.create(field.get_n_states(), field.get_n_aos(), false);
                            target = &field.get_eigenvectors()[spin][component][k];
                        }
                        replica_keys.insert(key);
                    });
                }
                catch (...)
                {
                    if (!local) erase_wfc_key(field, spin, component, k);
                    if (!already_replicated) replica_keys.erase(key);
                    throw;
                }
                if (!local) *target = std::move(incoming);
                for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
                    comm_h.bcast(target->c + offset,
                                 static_cast<int>(std::min(input_transfer_chunk, size - offset)), source);
            }
    replica_keys.clear();
    root_keys.clear();
    // This is the final representation required by non-kparallel serial density
    // and KS consumers, not a root-only matrix mislabeled as replicated input.
    if (band)
    {
        band_eigvecs_replicated_ = true;
        band_eigvecs_kpara_redistributed_ = false;
        band_eigvecs_kpara_2d_redistributed_ = false;
    }
    else
    {
        eigvecs_replicated_ = true;
        eigvecs_kpara_redistributed_ = false;
        eigvecs_kpara_2d_redistributed_ = false;
    }
}

void Dataset::apply_coulomb_2d_updates()
{
    const int local_count = static_cast<int>(pending_coulomb_2d_updates_.size());
    int total_count = 0;
    comm_h.allreduce(&local_count, &total_count, 1, MPI_SUM);
    if (!total_count) return;
    if (!coul_blacs2ap_redistributed_)
    {
        // Initial source blocks still belong to the established BLACS-to-AP
        // redistribution.  They are not post-consumption replacement patches.
        pending_coulomb_2d_updates_.clear();
        return;
    }
    const int nq = static_cast<int>(pbc.klist.size());
    const int naux = static_cast<int>(basis_aux.nb_total);
    collective_all_stage(comm_h, "Coulomb replacement metadata", [&]
    {
        if (!basis_aux.initialized() || naux <= 0 || nq <= 0)
            throw LIBRPA_RUNTIME_ERROR("Coulomb replacement needs initialized auxiliary/q metadata");
        for (const auto &[cut, iq] : pending_coulomb_2d_updates_)
            if (iq < 0 || iq >= nq)
                throw LIBRPA_INVALID_ARGUMENT("Coulomb replacement q index is out of range");
    });
    for (int channel = 0; channel < 2; ++channel)
        for (int iq = 0; iq < nq; ++iq)
        {
            const bool cut = channel != 0;
            const auto key = std::make_pair(cut, iq);
            const int local = pending_coulomb_2d_updates_.count(key) != 0;
            int any = 0;
            comm_h.allreduce(&local, &any, 1, MPI_MAX);
            if (!any) continue;
            const auto q = pbc.klist[iq];
            auto &blocks = cut ? vq_cut_block_loc : vq_block_loc;
            auto &target = cut ? vq_cut : vq;
            std::array<int, 5> own{local, vq_lbrow, vq_ubrow, vq_lbcol, vq_ubcol};
            std::vector<int> regions;
            collective_all_stage(comm_h, "Coulomb replacement region workspace", [&]
            { regions.resize(static_cast<std::size_t>(comm_h.nprocs) * own.size()); });
            comm_h.allgather(own.data(), own.size(), regions.data(), own.size());
            collective_all_stage(comm_h, "Coulomb replacement regions", [&]
            {
                for (int rank = 0; rank < comm_h.nprocs; ++rank)
                {
                    const int *region = regions.data() + 5 * rank;
                    if (!region[0]) continue;
                    if (region[1] < 0 || region[2] < region[1] || region[2] > naux ||
                        region[3] < 0 || region[4] < region[3] || region[4] > naux)
                        throw LIBRPA_INVALID_ARGUMENT("Coulomb replacement rectangle is outside the auxiliary basis");
                    for (int previous = 0; previous < rank; ++previous)
                    {
                        const int *other = regions.data() + 5 * previous;
                        if (other[0] && std::max(region[1], other[1]) < std::min(region[2], other[2]) &&
                            std::max(region[3], other[3]) < std::min(region[4], other[4]))
                            throw LIBRPA_INVALID_ARGUMENT("Overlapping 2D Coulomb replacement regions");
                    }
                }
                if (!local) return;
                const int rows = own[2] - own[1], columns = own[4] - own[3];
                const auto found = blocks.find(q);
                if (rows && columns && (found == blocks.end() || found->second.nr() != rows ||
                    found->second.nc() != columns || found->second.major() != MAJOR::COL))
                    throw LIBRPA_INVALID_ARGUMENT("Coulomb replacement buffer does not match its input rectangle");
                if (found != blocks.end())
                    for (std::size_t i = 0; i < found->second.size(); ++i)
                        if (!std::isfinite(found->second.ptr()[i].real()) ||
                            !std::isfinite(found->second.ptr()[i].imag()))
                            throw LIBRPA_INVALID_ARGUMENT("Coulomb replacement contains nonfinite entries");
            });
            // A rectangle is an overwrite, including exact zeros.  Only one
            // input block is communicated at a time; no full q matrix or
            // duplicated old Coulomb source is reconstructed.
            for (int source = 0; source < comm_h.nprocs; ++source)
            {
                const int *region = regions.data() + 5 * source;
                if (!region[0]) continue;
                const int rows = region[2] - region[1], columns = region[4] - region[3];
                const std::size_t size = static_cast<std::size_t>(rows) * columns;
                if (!size) continue;
                std::vector<cplxdb> buffer;
                std::map<atpair_t, ComplexMatrix *> local_targets;
                collective_all_stage(comm_h, "Coulomb replacement target workspace", [&]
                {
                    buffer.resize(std::min(input_transfer_chunk, size));
                    std::set<atpair_t> pairs;
                    for (const auto &[i, partners] : target)
                        for (const auto &[j, values] : partners) pairs.insert({i, j});
                    const auto owners = atpairs_unique_all.find(comm_h.myid);
                    if (owners != atpairs_unique_all.end())
                        pairs.insert(owners->second.begin(), owners->second.end());
                    for (const auto &[i, j] : pairs)
                    {
                        if (i < 0 || j < 0 || i >= basis_aux.n_atoms || j >= basis_aux.n_atoms)
                            throw LIBRPA_RUNTIME_ERROR("Coulomb atom-pair ownership is incompatible with the auxiliary basis");
                        const int row0 = static_cast<int>(basis_aux.get_global_index(i, 0));
                        const int col0 = static_cast<int>(basis_aux.get_global_index(j, 0));
                        const int ni = static_cast<int>(basis_aux[i]);
                        const int nj = static_cast<int>(basis_aux[j]);
                        if (std::max(row0, region[1]) >= std::min(row0 + ni, region[2]) ||
                            std::max(col0, region[3]) >= std::min(col0 + nj, region[4])) continue;
                        auto &matrix = target[i][j][q];
                        if (!matrix) matrix = std::make_shared<ComplexMatrix>(ni, nj);
                        if (matrix->nr != ni || matrix->nc != nj)
                            throw LIBRPA_RUNTIME_ERROR("Coulomb atom-pair buffer has incompatible dimensions");
                        local_targets.emplace(atpair_t{i, j}, matrix.get());
                    }
                });
                const Matz *input = nullptr;
                if (comm_h.myid == source) input = &blocks.at(q);
                for (std::size_t offset = 0; offset < size; offset += input_transfer_chunk)
                {
                    const std::size_t count = std::min(input_transfer_chunk, size - offset);
                    if (input) std::copy_n(input->ptr() + offset, count, buffer.data());
                    comm_h.bcast(buffer.data(), static_cast<int>(count), source);
                    for (std::size_t element = 0; element < count; ++element)
                    {
                        const std::size_t flat = offset + element;
                        const int row = region[1] + static_cast<int>(flat % rows);
                        const int col = region[3] + static_cast<int>(flat / rows);
                        const auto atom_row = basis_aux.get_local_index(row);
                        const auto atom_col = basis_aux.get_local_index(col);
                        const auto found = local_targets.find({atom_row.first, atom_col.first});
                        if (found != local_targets.end())
                            (*found->second)(atom_row.second, atom_col.second) = buffer[element];
                    }
                }
            }
            if (local) blocks.erase(q);
            pending_coulomb_2d_updates_.erase(key);
        }
}

void Dataset::reset_coulomb_input_layout()
{
    collective_all_stage(comm_h, "Reset incompatible Coulomb input", [&]
    {
        if (p_exx || p_chi0 || p_g0w0 || p_headwing)
            throw LIBRPA_RUNTIME_ERROR(
                "Destroy dependent numerical objects before resetting Coulomb input layout");
    });
    note_input_change(InputChange::Coulomb);
    finalize_comm_blacs_coul();
    coul_blacs2ap_redistributed_ = false;
    pending_coulomb_2d_updates_.clear();
    vq.clear();
    vq_cut.clear();
    vq_block_loc.clear();
    vq_cut_block_loc.clear();
    vq_lbrow = vq_ubrow = vq_lbcol = vq_ubcol = -1;
}

void Dataset::reset_scf_eigvecs_kpara_state()
{
    pending_scf_dense_updates_.clear();
    scf_root_dense_keys_.clear();
    scf_replicated_keys_.clear();
    eigvecs_replicated_ = false;
    eigvecs_kpara_redistributed_ = false;
    eigvecs_kpara_2d_redistributed_ = false;
    desc_wfc_kb.reset_handler();
    desc_wfc_kb_full.reset_handler();
    if (scfk_blacs_ctxt.is_initialized()) scfk_blacs_ctxt.finalize();
}

void Dataset::invalidate_compute_objects()
{
    symmetry_context.clear();
    // p_headwing.reset();
    p_exx.reset();
    p_chi0.reset();
    p_g0w0.reset();
    is_band_calc_done = false;
}

void Dataset::initialize_comm_blacs_coul()
{
    using std::cout;
    using std::endl;
    using global::ofs_myid;
    // Already initialized, do not do it again since 2D blocks may have been released.
    // To run it again, manually finalize it first
    if (comm_blacs_coul_initialized_) return;
    // Check if local Coulomb matrices are available on one of the processes
    const int has_coul = (vq_block_loc.size() > 0 || vq_cut_block_loc.size() > 0);
    int has_coul_somewhere;
    comm_h.allreduce(&has_coul, &has_coul_somewhere, 1, MPI_MAX);
    if (has_coul_somewhere == 0)
    {
        // No process has Coulomb local blocks, data not parsed
        // NOTE: this immediate return may be changed in the future,
        //       depending on how we want to use the block-cyclic local matrices.
        return;
    }

    global::profiler.start(__FUNCTION__);
    if (!this->basis_aux.initialized())
    {
        throw LIBRPA_RUNTIME_ERROR("Auxiliary basis must be set first");
    }

    // Below we try to restore the two-level parallel distribution of matrices from input
    MPI_Comm comm_coul;
    MPI_Comm_split(comm_h.comm, has_coul, comm_h.myid, &comm_coul);
    if (has_coul)
        comm_coul_h.reset_comm(comm_coul, true);
    else
        MPI_Comm_free(&comm_coul);

    ofs_myid << "Initializing Coulomb BLACS communicators, context and array descriptor" << endl;
    if (comm_coul_h.is_initialized())
    {
        MPI_Comm comm_intra_q, comm_inter_q;
        const auto &block_loc = vq_block_loc.size() > 0? vq_block_loc : vq_cut_block_loc;
        std::vector<int> iqs;
        for (const auto &[q, _]: block_loc)
            iqs.emplace_back(this->pbc.get_k_index_ibz(q));
        // Assuming q-points are correctly grouped among the processes,
        // we use the first q-point index to identify different groups
        std::sort(iqs.begin(), iqs.end());
        MPI_Comm_split(comm_coul_h.comm, iqs[0], comm_coul_h.myid, &comm_intra_q);
        comm_coul_intra_q_h.reset_comm(comm_intra_q, true);
        ofs_myid << "Coulomb intra-q comm/myid/nprocs: "
                 << comm_coul_intra_q_h.comm << " "
                 << comm_coul_intra_q_h.myid << " "
                 << comm_coul_intra_q_h.nprocs << endl;
        MPI_Comm_split(comm_coul_h.comm, comm_coul_intra_q_h.myid, comm_coul_h.myid, &comm_inter_q);
        comm_coul_inter_q_h.reset_comm(comm_inter_q, true);
        ofs_myid << "Coulomb inter-q comm/myid/nprocs: "
                 << comm_coul_inter_q_h.comm << " "
                 << comm_coul_inter_q_h.myid << " "
                 << comm_coul_inter_q_h.nprocs << endl;
    }
    // Till now, the communicators are initialized.
    // Now try to restore the BLACS context and the array descriptor
    if (comm_coul_intra_q_h.is_initialized())
    {
        const int myid = comm_coul_intra_q_h.myid;
        const int nprocs = comm_coul_intra_q_h.nprocs;
        std::vector<int> lbrow_all(nprocs, 0), lbcol_all(nprocs, 0);
        lbrow_all[myid] = vq_lbrow;
        lbcol_all[myid] = vq_lbcol;
        comm_coul_intra_q_h.allreduce(MPI_IN_PLACE, lbrow_all.data(), nprocs, MPI_SUM);
        comm_coul_intra_q_h.allreduce(MPI_IN_PLACE, lbcol_all.data(), nprocs, MPI_SUM);
        ofs_myid << "Collected lbrow: " << lbrow_all << endl;
        ofs_myid << "Collected lbcol: " << lbcol_all << endl;
        // Get number of rows and columns, as well as the process which stores the head of the matrix
        int nprows = 0, npcols = 0, myid_src = 0;
        for (int pid = 0; pid < nprocs; pid++)
        {
            if (lbrow_all[pid] == 0)
            {
                npcols++;
                if (lbcol_all[pid] == 0) myid_src = pid;
            }
            if (lbcol_all[pid] == 0) nprows++;
        }
        // Checking the boundary of the first two processes for the 2D layout
        CTXT_LAYOUT layout = CTXT_LAYOUT::R;
        if (nprocs > 1)
        {
            if (lbrow_all[0] != lbrow_all[1]) layout = CTXT_LAYOUT::C;
        }
        ofs_myid << "Coulomb BLACS context nprows/npcols/layout: " << nprows << " " << npcols;
        if (layout == CTXT_LAYOUT::C)
            ofs_myid << " column-major" << endl;
        else
            ofs_myid << " row-major" << endl;
        blacs_coul_intra_q_h.reset_comm(comm_coul_intra_q_h.comm, true);
        blacs_coul_intra_q_h.set_grid(nprows, npcols, layout);
        // Now we have restored the BLACS communicator

        // Set up the array descriptor.
        // Decide the row and column blocks size
        int nbrows = vq_ubrow - vq_lbrow;
        int nbcols = vq_ubcol - vq_lbcol;
        int mb, nb;
        comm_coul_intra_q_h.allreduce(&nbrows, &mb, 1, MPI_MAX);
        comm_coul_intra_q_h.allreduce(&nbcols, &nb, 1, MPI_MAX);
        ofs_myid << "Coulomb array descriptor block size (mb/nb): " << mb << " " << nb << endl;
        int irsrc = 0, icsrc = 0;
        blacs_coul_intra_q_h.get_pcoord(myid_src, irsrc, icsrc);
        ofs_myid << "Coulomb array descriptor source pid (ir/ic): " << irsrc << " " << icsrc << endl;
        desc_coul_intra_q.reset_handler(blacs_coul_intra_q_h);
        const int n_aux = this->basis_aux.nb_total;
        desc_coul_intra_q.init(n_aux, n_aux, mb, nb, irsrc, icsrc);
    }

    ofs_myid << "comm_coul_h.is_initialized() = " << std::boolalpha << comm_coul_h.is_initialized() << std::endl;

    if (comm_coul_h.is_initialized())
    {
        if (comm_coul_h.is_root())
        {
            cout << "BLACS environment of 2D Coulomb matrices is restored" << endl;
        }
        comm_coul_h.barrier();
        for (int pid = 0; pid < comm_coul_h.nprocs; pid++)
        {
            // if (pid == comm_coul_h.myid)
            // {
            //     std::cout << blacs_coul_intra_q_h.info() << std::endl;
            // }
            comm_coul_h.barrier();
        }
    }

    comm_blacs_coul_initialized_ = true;
    global::profiler.stop(__FUNCTION__);
}

void Dataset::redistribute_coulomb_blacs2ap()
{
    using std::endl;
    using global::ofs_myid;
    using global::profiler;

    // Already redistributed, or blacs data has been parsed
    if (coul_blacs2ap_redistributed_) return;
    initialize_comm_blacs_coul();
    profiler.start(__FUNCTION__);
    // Step 1: redistribute to global blacs_h if
    // - some process does not have block data
    // - there are more than one comm_coul_inter_q_h, meaning Nabs x Nabs matrix is distributed only on a subset of all processes
    // - the shape and layout of blacs_coul_intra_q_h is different from blacs_h
    // - the descriptor is different from the global one
    // Otherwise, blacs_coul_intra_q_h is equivalent to blacs_h, no need to redistribute
    const int same_shape_blacs = blacs_h.nprows == blacs_coul_intra_q_h.nprows &&
                                 blacs_h.npcols == blacs_coul_intra_q_h.npcols &&
                                 blacs_h.layout == blacs_coul_intra_q_h.layout;
    int same_shape_blacs_all;
    comm_h.allreduce(&same_shape_blacs, &same_shape_blacs_all, 1, MPI_MIN);
    const int same_desc = desc_abf.m() == desc_coul_intra_q.m() &&
                          desc_abf.n() == desc_coul_intra_q.n() &&
                          desc_abf.mb() == desc_coul_intra_q.mb() &&
                          desc_abf.nb() == desc_coul_intra_q.nb() &&
                          desc_abf.irsrc() == desc_coul_intra_q.irsrc() &&
                          desc_abf.icsrc() == desc_coul_intra_q.icsrc();
    int same_desc_all;
    comm_h.allreduce(&same_desc, &same_desc_all, 1, MPI_MIN);

    const int n_comms_q = comm_coul_inter_q_h.nprocs;
    int n_comms_q_min, n_comms_q_max;
    comm_h.allreduce(&n_comms_q, &n_comms_q_min, 1, MPI_MIN);
    comm_h.allreduce(&n_comms_q, &n_comms_q_max, 1, MPI_MAX);
    // No block matrix data parsed, no need to redistribute
    if (n_comms_q_max == 0)
    {
        profiler.stop(__FUNCTION__);
        return;
    }

    const int n_aux = basis_aux.nb_total;
    // Major for communication with p?gemr2d
    const MAJOR major_comm = MAJOR::COL;
    std::map<Vector3_Order<double>, Matz> vq_redist;
    std::map<Vector3_Order<double>, Matz> vq_cut_redist;
    if (n_comms_q_min == 0 || n_comms_q_max > 1 || !same_shape_blacs_all || !same_desc_all)
    {
        auto mat_local = init_local_mat<cplxdb>(desc_abf, major_comm);
        for (int i_comm_q = 0; i_comm_q < n_comms_q_max; i_comm_q++)
        {
            std::vector<int> desc(9, n_aux);
            desc[6] = 0;
            desc[7] = 0;
            if (comm_coul_inter_q_h.nprocs > 0 && comm_coul_inter_q_h.myid == i_comm_q)
            {
                // source, copy the descriptor
                desc = std::vector<int>(desc_coul_intra_q.desc, desc_coul_intra_q.desc + 9);
            }
            else
            {
                // not source, use a dummy descriptor: dtype = 1, ctxt = -1;
                desc[0] = 1;
                desc[1] = -1;
            }
            std::vector<std::pair<std::map<Vector3_Order<double>, Matz>&, std::map<Vector3_Order<double>, Matz>&>>
                vq_src_dist_pair{ { vq_block_loc, vq_redist}, {vq_cut_block_loc, vq_cut_redist} };
            for (auto [vq_src, vq_dst]: vq_src_dist_pair)
            {
                // Get the number and coordinates of q-points for this communicator
                int nq_this = vq_src.size();
                comm_coul_inter_q_h.bcast(&nq_this, 1, i_comm_q);
                // Let processes out of comm_coul know what are going to be processed
                int pid_comm_coul_root_in_global = 0;
                if (comm_coul_h.is_initialized() && comm_coul_h.myid == 0)
                    pid_comm_coul_root_in_global = comm_h.myid;
                comm_h.allreduce(MPI_IN_PLACE, &pid_comm_coul_root_in_global, 1, MPI_MAX);
                if (n_comms_q_min == 0) comm_h.bcast(&nq_this, 1, pid_comm_coul_root_in_global);
                if (nq_this < 1) continue;
                ofs_myid << "vq_src nq_this for i_inter_q " << i_comm_q << " = " << nq_this << endl; 
                ofs_myid << "Array desc for source         : " << desc << endl;
                ofs_myid << "Array desc for dest (desc_abf): " << desc_abf.info_desc() << endl;
                std::vector<double> qs(nq_this * 3);
                std::map<int, Matz> mat_q;
                Matz mat_comm(0, 0, major_comm);
                if (i_comm_q == comm_coul_inter_q_h.myid)
                {
                    int iq = 0;
                    for (const auto &[q, mat]: vq_src)
                    {
                        qs[iq * 3] = q.x;
                        qs[iq * 3 + 1] = q.y;
                        qs[iq * 3 + 2] = q.z;
                        mat_q[iq] = mat;
                        iq++;
                    }
                }
                else
                {
                    // Dummy matrix
                    for (int iq = 0; iq < nq_this; iq++)
                        mat_q[iq] = Matz(0, 0, major_comm);
                }
                comm_coul_inter_q_h.bcast(qs.data(), nq_this * 3, i_comm_q);
                if (n_comms_q_min == 0) comm_h.bcast(qs.data(), nq_this * 3, pid_comm_coul_root_in_global);
                ofs_myid << "broadcasting coulomb matrices on q-points: " << qs << endl; 
                for (int iq = 0; iq < nq_this; iq++)
                {
                    const auto &mat_lo_src = mat_q.at(iq);
                    ScalapackConnector::pgemr2d(n_aux, n_aux, mat_lo_src.ptr(), 1, 1, desc.data(),
                                                mat_local.ptr(), 1, 1, desc_abf.desc,
                                                blacs_h.ictxt);
                    Vector3_Order<double> q{qs[iq*3], qs[iq*3+1], qs[iq*3+2]};
                    vq_dst[q] = mat_local.copy();
                    mat_q.erase(iq);
                }
            }
            // Do the same for cut Coulomb
            // int nq_cut = vq_cut_block_loc.size();
        }
    }
    else
    {
        vq_redist = std::move(vq_block_loc);
        vq_cut_redist = std::move(vq_cut_block_loc);
    }
    // Till now, every process has Coulomb matrices at all q-points, but only one BLACS block
    // Consistency check
    ofs_myid << "vq_redist size after pzgemr2d    : " << vq_redist.size() << endl;
    ofs_myid << "vq_cut_redist size after pzgemr2d: " << vq_cut_redist.size() << endl;
    assert(vq_redist.size() == 0 || vq_redist.size() == this->pbc.klist_coul.size());
    assert(vq_cut_redist.size() == 0 || vq_cut_redist.size() == this->pbc.klist_coul.size());

    // Step 2: redistribute BLACS 2D layout to atom-pair layout, and copy from column-major Matz to ComplexMatrix
    IndexScheduler sched;
    ofs_myid << "atpairs_unique_all:" << this->atpairs_unique_all << endl;
    sched.init(this->atpairs_unique_all, this->basis_aux, this->basis_aux, desc_abf, false);
    std::vector<std::pair<std::map<Vector3_Order<double>, Matz> &, atpair_k_cplx_mat_t &>>
        vq_src_dst_pair{{vq_redist, vq}, {vq_cut_redist, vq_cut}};
    // both bare and cut coulomb
    for (auto &[vq_src, vq_dst]: vq_src_dst_pair)
    {
        for (auto it_q = vq_src.begin(); it_q != vq_src.end();)
        {
            const auto &q = it_q->first;
            const auto &mat_loc = it_q->second;
            auto IJmap = get_ap_map_from_blacs_dist_scheduler(mat_loc, sched, basis_aux, basis_aux, desc_abf);
            for (auto it_ap = IJmap.begin(); it_ap != IJmap.end();)
            {
                const auto IJ = it_ap->first;
                const auto I = IJ.first;
                const auto n_I = basis_aux[I];
                const auto J = IJ.second;
                const auto n_J = basis_aux[J];
                auto &cmat_new = vq_dst[I][J][q];
                cmat_new = std::make_shared<ComplexMatrix>(n_I, n_J);
                const size_t n = n_I * n_J;
                auto &matz = it_ap->second;
                assert(as_size(matz.nr()) == n_I);
                assert(as_size(matz.nc()) == n_J);
                matz.swap_to_row_major();
                memcpy(cmat_new->c, matz.ptr(), n * sizeof(cplxdb));
                assert(as_size(matz.nr()) == n_I);
                assert(as_size(matz.nc()) == n_J);
                assert(as_size(matz.size()) == n);
                it_ap = IJmap.erase(it_ap);
            }
            it_q = vq_src.erase(it_q);
        }
    }

    vq_block_loc.clear();
    vq_cut_block_loc.clear();
    pending_coulomb_2d_updates_.clear();
    coul_blacs2ap_redistributed_ = true;
    profiler.stop(__FUNCTION__);
}

void Dataset::redistribute_eigvecs_kpara()
{
    if (eigvecs_kpara_2d_redistributed_) collect_wfc_to_kroots(false);
    if (eigvecs_kpara_redistributed_ && scf_root_dense_keys_.empty() &&
        scf_replicated_keys_.empty()) return;
    using global::profiler;
    profiler.start(__FUNCTION__);
    redistribute_meanfield_eigvecs_kpara(mf, comm_h, scfk_blacs_ctxt, "SCF", 19317, scf_root_dense_keys_);
    eigvecs_replicated_ = false;
    eigvecs_kpara_redistributed_ = true;
    scf_root_dense_keys_.clear();
    scf_replicated_keys_.clear();
    profiler.stop(__FUNCTION__);
}

void Dataset::redistribute_band_eigvecs_kpara()
{
    if (band_eigvecs_kpara_2d_redistributed_) collect_wfc_to_kroots(true);
    if (band_eigvecs_kpara_redistributed_ && band_root_dense_keys_.empty() &&
        band_replicated_keys_.empty()) return;
    using global::profiler;
    profiler.start(__FUNCTION__);
    initialize_band_kblacs_wfc_layout();
    redistribute_meanfield_eigvecs_kpara(mf_band, comm_h, bandk_blacs_ctxt, "band-path", 19318, band_root_dense_keys_);
    band_eigvecs_replicated_ = false;
    band_eigvecs_kpara_redistributed_ = true;
    band_root_dense_keys_.clear();
    band_replicated_keys_.clear();
    profiler.stop(__FUNCTION__);
}

void Dataset::initialize_band_kblacs_wfc_layout()
{
    const int n_kpoints = mf_band.get_n_kpoints();
    if (n_kpoints <= 0)
        throw LIBRPA_RUNTIME_ERROR(
            "band-path mean-field dimensions must be set before initializing k-BLACS");

    if (!bandk_blacs_ctxt.is_initialized() || bandk_blacs_ctxt.n_kpoints() != n_kpoints)
    {
        if (bandk_blacs_ctxt.is_initialized())
        {
            desc_band_wfc_kb.reset_handler();
            desc_band_wfc_kb_full.reset_handler();
            bandk_blacs_ctxt.finalize();
        }
        KPointBlacsProcessShape bandk_blacs_shape;
        bandk_blacs_shape.favor_square_blacs_grid = true;
        bandk_blacs_ctxt.init(bandk_blacs_shape, comm_h.comm, n_kpoints);
    }
    const int block_ao = get_capped_blacs_block_size(
        mf_band.get_n_aos(), wfc_gemm_block_size_opt, bandk_blacs_ctxt.blacs_h);
    const int block_band = get_capped_blacs_block_size(
        mf_band.get_n_states(), wfc_gemm_block_size_opt, bandk_blacs_ctxt.blacs_h);
    desc_band_wfc_kb = bandk_blacs_ctxt.create_array_desc(
        mf_band.get_n_aos(), mf_band.get_n_states(), block_ao, block_band);
    desc_band_wfc_kb_full = bandk_blacs_ctxt.create_array_desc(
        mf_band.get_n_aos(), mf_band.get_n_states(), mf_band.get_n_aos(),
        mf_band.get_n_states());
}

void Dataset::reset_band_eigvecs_kpara_state()
{
    pending_band_dense_updates_.clear();
    band_root_dense_keys_.clear();
    band_replicated_keys_.clear();
    band_eigvecs_replicated_ = false;
    band_eigvecs_kpara_redistributed_ = false;
    band_eigvecs_kpara_2d_redistributed_ = false;
    desc_band_wfc_kb.reset_handler();
    desc_band_wfc_kb_full.reset_handler();
    if (bandk_blacs_ctxt.is_initialized()) bandk_blacs_ctxt.finalize();
}

void Dataset::redistribute_eigvecs_kpara_2d()
{
    if (!eigvecs_kpara_2d_redistributed_)
    {
        redistribute_meanfield_eigvecs_kpara(
            mf, comm_h, scfk_blacs_ctxt, "SCF", 19319, scf_root_dense_keys_);
        eigvecs_replicated_ = false;
        scf_replicated_keys_.clear();
        eigvecs_kpara_redistributed_ = true;
        eigvecs_kpara_2d_redistributed_ = true;
    }
    // root_keys tracks every not-yet-scattered full source.  Successful keys
    // commit one at a time after world agreement; no whole-MF duplicate map.
    restore_root_dense_keys_to_tiles(false);
}

void Dataset::redistribute_band_eigvecs_kpara_2d()
{
    if (!band_eigvecs_kpara_2d_redistributed_)
    {
        initialize_band_kblacs_wfc_layout();
        redistribute_meanfield_eigvecs_kpara(
            mf_band, comm_h, bandk_blacs_ctxt, "band-path", 19320, band_root_dense_keys_);
        band_eigvecs_replicated_ = false;
        band_replicated_keys_.clear();
        band_eigvecs_kpara_redistributed_ = true;
        band_eigvecs_kpara_2d_redistributed_ = true;
    }
    restore_root_dense_keys_to_tiles(true);
}

void Dataset::finalize_comm_blacs_coul()
{
    if (!comm_blacs_coul_initialized_) return;
    global::profiler.start(__FUNCTION__);
    desc_coul_intra_q.reset_handler();
    blacs_coul_intra_q_h.reset_comm();
    comm_coul_inter_q_h.free_comm();
    comm_coul_intra_q_h.free_comm();
    comm_coul_h.free_comm();
    comm_blacs_coul_initialized_ = false;
    global::profiler.stop(__FUNCTION__);
}

}
