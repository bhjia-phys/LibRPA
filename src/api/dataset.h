#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <unordered_map>

#include "librpa_options.h"
#include "../core/atom.h"
#include "../core/atomic_basis.h"
#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/exx.h"
#include "../core/geometry.h"
#include "../core/gw.h"
#include "../core/symmetry_context.h"
#include "../core/meanfield.h"
#include "../core/pbc.h"
#include "../core/ri.h"
#include "../core/timefreq.h"
#include "../math/matrix_m.h"
#include "../math/symmetry.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"

namespace librpa_int
{

// Physical input changes are recorded locally by setters.  Communication belongs
// to an already collective compute/prepare entry, never an owner-local setter.
enum class InputChange : std::uint32_t
{
    None = 0,
    Spectrum = 1u << 0,
    Occupations = 1u << 1,
    ScfWavefunctions = 1u << 2,
    Band = 1u << 3,
    Ri = 1u << 4,
    Coulomb = 1u << 5,
    AuxProjection = 1u << 6,
    Velocity = 1u << 7,
    ExternalDielectric = 1u << 8,
    Structure = 1u << 9,
    DirectOptical = 1u << 10
};

inline InputChange operator|(InputChange left, InputChange right)
{
    return static_cast<InputChange>(static_cast<std::uint32_t>(left) |
                                    static_cast<std::uint32_t>(right));
}

// Exactly the arguments of a successfully completed TFGrids::generate call.
// A node match alone does not establish equality of transform weights.
struct TFGridGenerationRecipe
{
    int nfreq = 0;
    LibrpaTimeFreqGrid type = LIBRPA_TFGRID_UNSET;
    std::array<double, 6> parameters{};
    bool inverse_sine = false;
    bool operator==(const TFGridGenerationRecipe &other) const
    {
        return nfreq == other.nfreq && type == other.type &&
               parameters == other.parameters && inverse_sine == other.inverse_sine;
    }
};

/*!
 * Core object to hold runtime environment, input and output data
 */
class Dataset
{
private:
    bool input_blacs_matloc_row_major_;
    bool comm_blacs_coul_initialized_;
    bool coul_blacs2ap_redistributed_;
    bool eigvecs_kpara_redistributed_;
    bool eigvecs_kpara_2d_redistributed_;
    bool band_eigvecs_kpara_2d_redistributed_;
    bool band_eigvecs_kpara_redistributed_ = false;
    bool eigvecs_replicated_ = false;
    bool band_eigvecs_replicated_ = false;
    // Small per-key progress facts make streaming layout conversion retryable
    // without retaining a second whole mean field.
    std::set<std::array<int, 3>> scf_root_dense_keys_, band_root_dense_keys_;
    std::set<std::array<int, 3>> scf_replicated_keys_, band_replicated_keys_;
    std::uint32_t pending_input_changes_ = 0;
    using DenseWfcUpdates = std::map<std::array<int, 3>, ComplexMatrix>;
    DenseWfcUpdates pending_scf_dense_updates_;
    DenseWfcUpdates pending_band_dense_updates_;
    std::set<std::pair<bool, int>> pending_coulomb_2d_updates_;
    bool have_compute_options_ = false;
    LibrpaOptions last_compute_options_{};

    void apply_dense_wfc_updates(bool band);
    void apply_coulomb_2d_updates();
    void collect_wfc_to_kroots(bool band);
    void replicate_dense_wfc(bool band);
    void restore_root_dense_keys_to_tiles(bool band);
    void invalidate_input_dependents(std::uint32_t changes,
                                     bool exx_recipe_changed,
                                     bool optical_recipe_changed,
                                     bool response_recipe_changed,
                                     bool target_recipe_changed);
public:
    /* Member variables */
    // Environment control
    // Global MPI communicators and BLACS context handlers
    MpiCommHandler comm_h;
    BlacsCtxtHandler blacs_h;
    KPointBlacsParallelContext scfk_blacs_ctxt;
    KPointBlacsParallelContext bandk_blacs_ctxt;
    // Communicators for Coulomb matrices 2D input and re-distribution
    MpiCommHandler comm_coul_h;
    MpiCommHandler comm_coul_inter_q_h;
    MpiCommHandler comm_coul_intra_q_h;
    BlacsCtxtHandler blacs_coul_intra_q_h;
    ArrayDesc desc_coul_intra_q;
    //! Array descriptor for matrices of wave-function basis (using dataset.blacs_h)
    ArrayDesc desc_wfc;
    //! Permanent capped rectangular compute layout, N_ao x N_states, on scfk_blacs_ctxt.blacs_h.
    ArrayDesc desc_wfc_kb;
    //! Dense-root compatibility layout used only at API/input redistribution boundaries.
    ArrayDesc desc_wfc_kb_full;
    //! Band-path counterparts using bandk_blacs_ctxt.blacs_h.
    ArrayDesc desc_band_wfc_kb;
    ArrayDesc desc_band_wfc_kb_full;
    //! Array descriptor for matrices of auxiliary basis set size (using blacs_h)
    ArrayDesc desc_abf;
    //! Array descriptor for compressed auxiliary basis set size (using blacs_h)
    ArrayDesc desc_abf_shrink;
    //! Atom pairs on current process for atomic-basis matrix data
    std::vector<atpair_t> atpairs_local;
    //! Distribution of unique atom-pairs on all processes for atomic-basis matrix data
    std::unordered_map<int, std::set<atpair_t>> atpairs_unique_all;
    //! Unit cell vectors on current process
    std::vector<Vector3_Order<int>> Rs_local;

    // Physical system.
    //! Handling object for basis functions for wave function expansion.
    AtomicBasis basis_wfc;
    //! Handling object for auxiliary basis functions for RI
    AtomicBasis basis_aux;
    //! Handling object for shrinked auxiliary basis functions for RI
    AtomicBasis basis_aux_shrink;
    //! Basis ordering and real-spherical-harmonic convention.
    BasisConvention basis_convention;
    SpaceGroupSymOps spg_symops;
    //! Explicit spin-space operation table set by librpa_set_symmetry_spin_operations;
    //! spatial_id indexes spg_symops. Consumed only when spg_spin_ops_explicit is set.
    std::vector<SymmetrySpinOperation> spg_spin_ops;
    //! Whether spg_spin_ops should replace the default grey-group expansion.
    bool spg_spin_ops_explicit = false;
    //! Whether the explicit table is the unitary block of a grey group (LibRPA
    //! appends the antiunitary copies internally). Defaults to true to match the
    //! legacy behavior of librpa_set_symmetry_operations.
    bool spg_grey_group = true;
    //! Symmetry context built from structure and full k-point grid.
    SymmetryContext symmetry_context;
    //! Atomic structure
    Atoms atoms;
    //! Periodic boundary setting
    PeriodicBoundaryData pbc;
    //! Fractional coordiantes of k-points for band calculation
    std::vector<Vector3_Order<double>> kfrac_band_list;
    // Reserved for a future head/wing path with a k grid different from the
    // SCF mean field. The active path currently uses pbc.kfrac_list directly.
    // std::vector<Vector3_Order<double>> kfrac_headwing_list;

    //! Overlap matrix between normal and shrinked ABFs
    std::map<Vector3_Order<double>, ComplexMatrix> sinvS;

    // Input data.
    //! Mean-field starting point
    MeanField mf;
    //! Velocity/momentum matrix input.
    //! Currently consumed by analytic head/wing construction.
    //! File readers may convert producer-specific momentum/velocity files before
    //! storing them here; public API setters store values as provided.
    velocity_matrix_t velocity_matrix;
    //! Mean-field object for band calculation
    MeanField mf_band;
    //! Monotonic id for distinguishing successive band-path input datasets.
    int band_data_id;
    //! Whether band-path input data has been provided through set_band_* APIs.
    bool is_band_data_set;
    //! Whether band-path EXX/GW k-space matrices have been computed for the set band data.
    bool is_band_calc_done;
    //! Time-frequency grids
    TFGrids tfg;
    TFGridGenerationRecipe completed_tfgrid_recipe;
    bool tfgrid_generation_complete = false;
    //! Real-space RI coefficient tensors (local RI)
    Cs_LRI cs_data;
    //! Real-space RI coefficient tensors (local RI) of shrinked auxiliary basis
    Cs_LRI cs_data_shrink;
    // atom-pair distribution of Coulomb matrices
    atpair_k_cplx_mat_t vq;
    atpair_k_cplx_mat_t vq_cut;
    // Local index boundaries and 2D distribution of Coulomb matrices
    // Indices are just for data parsing and blacs context initialization.
    // Further handling should rely on comm_coul_*/blacs_coul_* communicators.
    // Lower bounds are included, and upper bounds are excluded.
    // Bare and cut Coulombs are enforced to use the same layout.
    int vq_lbrow = -1, vq_ubrow = -1;
    int vq_lbcol = -1, vq_ubcol = -1;
    std::map<Vector3_Order<double>, Matz> vq_block_loc;
    std::map<Vector3_Order<double>, Matz> vq_cut_block_loc;
    // Macroscopic dielectric functions at imaginary frequencies
    // Only used for head correction of GW calculation
    std::vector<double> epsmacs_imagfreq;
    std::vector<double> omegas_imagfreq;

    // Independently loaded optical data outlive an invalidated derived context.
    std::vector<double> external_epsmacs_imagfreq;
    std::vector<double> external_omegas_imagfreq;
    std::shared_ptr<const DirectFullBzOpticalInput> direct_full_bz_optical_input;
    std::shared_ptr<const MeanField> optical_meanfield_input;
    OpticalMeanFieldLayout optical_meanfield_layout;
    // Explicit physical readers/native preparation clear this only after a
    // compatible selected input has replaced a structurally stale one.
    bool optical_input_frame_stale = false;
    double strict_2d_auxiliary_monopole_norm_squared = 0.0;

    std::unique_ptr<diele_func> p_headwing;

    // Output data, held by computation objects
    // All computation data objects should be contained here as pointers
    // and are created only when requested.
    std::unique_ptr<Exx> p_exx;
    std::unique_ptr<Chi0> p_chi0;
    std::unique_ptr<G0W0> p_g0w0;

    /* Constructors and destructors */
    Dataset(MPI_Comm comm, const bool input_blacs_matloc_row_major = true);
    ~Dataset() { free(); }
    void free();

    void initialize_comm_blacs_coul();
    void redistribute_coulomb_blacs2ap();
    void finalize_comm_blacs_coul();
    // Collective reset of incompatible loaded Coulomb input.  Destroy borrowed
    // numerical consumers first, then reload V after basis/q-layout changes.
    void reset_coulomb_input_layout();

    // OFF-mode consumers require every dense k coefficient on every rank.
    void collect_eigvecs_kroots();
    void replicate_eigvecs();
    void replicate_band_eigvecs();
    bool eigvecs_replicated() const
    { return eigvecs_replicated_ && scf_root_dense_keys_.empty() && scf_replicated_keys_.empty(); }
    bool band_eigvecs_replicated() const
    { return band_eigvecs_replicated_ && band_root_dense_keys_.empty() && band_replicated_keys_.empty(); }

    void redistribute_eigvecs_kpara();
    void redistribute_band_eigvecs_kpara();
    void redistribute_eigvecs_kpara_2d();
    void redistribute_band_eigvecs_kpara_2d();
    void initialize_band_kblacs_wfc_layout();
    void mark_eigvecs_kpara_2d_ready()
    {
        eigvecs_replicated_ = false;
        scf_root_dense_keys_.clear();
        scf_replicated_keys_.clear();
        eigvecs_kpara_redistributed_ = true;
        eigvecs_kpara_2d_redistributed_ = true;
    }
    void mark_band_eigvecs_kpara_2d_ready()
    {
        band_eigvecs_replicated_ = false;
        band_root_dense_keys_.clear();
        band_replicated_keys_.clear();
        band_eigvecs_kpara_redistributed_ = true;
        band_eigvecs_kpara_2d_redistributed_ = true;
    }
    bool eigvecs_kpara_2d_ready() const
    { return eigvecs_kpara_2d_redistributed_ && scf_root_dense_keys_.empty(); }
    bool band_eigvecs_kpara_2d_ready() const
    {
        return band_eigvecs_kpara_2d_redistributed_ && band_root_dense_keys_.empty();
    }
    void reset_scf_eigvecs_kpara_state();
    void reset_band_eigvecs_kpara_state();
    bool eigvecs_kpara_ready() const
    { return eigvecs_kpara_redistributed_ && scf_root_dense_keys_.empty() && scf_replicated_keys_.empty(); }
    bool band_eigvecs_kpara_ready() const
    { return band_eigvecs_kpara_redistributed_ && band_root_dense_keys_.empty() && band_replicated_keys_.empty(); }

    void note_input_change(InputChange change) noexcept;
    bool has_pending_optical_input_changes() const noexcept;
    void synchronize_input_changes(const LibrpaOptions *opts = nullptr);
    // The supplied complete API matrix is staged if mf already contains tiles.
    ComplexMatrix &dense_wfc_input(bool band, int spin, int component, int k);
    void note_coulomb_2d_update(bool cut, int iq);
    // Preserve physical direct input and normalization before dropping a result.
    void retain_optical_input();

    void invalidate_compute_objects();

    /* Disable copy */
    Dataset(const Dataset &) = delete;
    Dataset &operator=(const Dataset &) = delete;
};

typedef std::shared_ptr<Dataset> dataset_ptr_t;

}
