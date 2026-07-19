#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <librpa_enums.h>

#include "../../src/api/compute_helper.h"
#include "../../src/api/dataset_helper.h"
#include "../../src/api/instance_manager.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/input_elsi.h"
#include "../../src/qsgw/convergence.h"
#include "../../src/qsgw/correlation_potential.h"
#include "../../src/qsgw/distributed_matrix.h"
#include "../../src/qsgw/effective_hamiltonian.h"
#include "../../src/qsgw/fixed_basis.h"
#include "../../src/qsgw/hamiltonian_mixing.h"
#include "../../src/qsgw/hartree_route.h"
#include "../../src/qsgw/hartree_workflow.h"
#include "../../src/qsgw/headwing_update.h"
#include "../../src/qsgw/input_contract.h"
#include "../../src/qsgw/iteration_trace.h"
#include "../../src/qsgw/occupation.h"
#include "../../src/qsgw/projection_target.h"
#include "../../src/qsgw/sha256.h"
#include "../../src/qsgw/vxc_io.h"
#include "../../src/utils/constants.h"
#include "../../src/utils/profiler.h"
#include "../driver.h"
#include "../read_data.h"
#include "../reader_coulomb.h"
#include "../task.h"

namespace
{

using librpa_int::Matz;
using librpa_int::MeanField;
using librpa_int::Vector3_Order;
using librpa_int::cplxdb;
using librpa_int::qsgw::ScopedReferenceEigenvectors;
using librpa_int::qsgw::SpinKMatrixMap;
using librpa_int::qsgw::VelocityMatrix;
using SigmaMatrixMap = librpa_int::qsgw::SpinKFrequencyMatrixMap;

template <typename Function>
void collective_root_stage(
    const librpa_int::MpiCommHandler& communicator,
    const std::string& label,
    Function&& function)
{
    std::exception_ptr root_error;
    int failed = 0;
    if (communicator.is_root())
    {
        try
        {
            function();
        }
        catch (const std::exception& error)
        {
            failed = 1;
            root_error = std::current_exception();
            librpa_int::global::lib_printf(
                LIBRPA_VERBOSE_CRITICAL, "%s failed: %s\n",
                label.c_str(), error.what());
        }
        catch (...)
        {
            failed = 1;
            root_error = std::current_exception();
        }
    }
    communicator.bcast(&failed, 1, 0);
    if (failed)
    {
        if (communicator.is_root()) std::rethrow_exception(root_error);
        throw LIBRPA_RUNTIME_ERROR(label + " failed on root");
    }
}

std::string resolve_input_path(const std::string& base,
                               const std::string& path)
{
    return librpa_int::is_absolute_path(path)
               ? path
               : librpa_int::join_path(base, path);
}

bool role_contains_path(
    const librpa_int::qsgw::QsgwInputContract& contract,
    const std::string& role,
    const std::string& expected)
{
    const std::filesystem::path normalized_expected =
        std::filesystem::path(expected).lexically_normal();
    for (const auto& file : contract.files(role))
    {
        if (std::filesystem::path(file.file).lexically_normal() ==
            normalized_expected)
        {
            return true;
        }
    }
    return false;
}

std::filesystem::path resolved_absolute_path(
    const std::string& base,
    const std::string& path)
{
    const std::filesystem::path value(path);
    const std::filesystem::path resolved = value.is_absolute()
        ? value
        : std::filesystem::path(base) / value;
    return std::filesystem::absolute(resolved).lexically_normal();
}

struct HeadwingKPathInfo
{
    int n_basis = 0;
    int n_states = 0;
    int n_spins = 0;
    std::vector<Vector3_Order<double>> kpoints;
};

struct IndependentHeadwingState
{
    MeanField reference;
    MeanField live;
    VelocityMatrix reference_velocity;
    VelocityMatrix live_velocity;
    std::vector<Vector3_Order<double>> kpoints;
    std::vector<double> weights;
    librpa_int::qsgw::OccupationResult initial_occupations;
    SpinKMatrixMap reference_hamiltonian;
    std::unique_ptr<librpa_int::KPointBlacsParallelContext> kblacs;
    librpa_int::Dataset* owner = nullptr;

    ~IndependentHeadwingState()
    {
        if (owner != nullptr) owner->p_headwing.reset();
    }
};

HeadwingKPathInfo read_independent_headwing_kpoints(
    const std::filesystem::path& path,
    const bool use_spinor_wfc)
{
    librpa_int::require_readable_file(path.string());
    std::ifstream input(path);
    HeadwingKPathInfo result;
    int n_kpoints = 0;
    input >> result.n_basis >> result.n_states >> result.n_spins >> n_kpoints;
    if (!input || result.n_basis <= 0 || result.n_states <= 0 ||
        result.n_spins <= 0 || n_kpoints <= 0)
    {
        throw std::invalid_argument(
            "Invalid QSGW independent head-wing k_path_info header in " +
            path.string());
    }
    if (use_spinor_wfc)
    {
        if (result.n_basis % 2 != 0)
        {
            throw std::invalid_argument(
                "QSGW independent head-wing spinor basis size must be even");
        }
        result.n_basis /= 2;
    }

    result.kpoints.reserve(static_cast<std::size_t>(n_kpoints));
    for (int kpoint = 0; kpoint < n_kpoints; ++kpoint)
    {
        Vector3_Order<double> coordinate;
        input >> coordinate.x >> coordinate.y >> coordinate.z;
        if (!input || !std::isfinite(coordinate.x) ||
            !std::isfinite(coordinate.y) || !std::isfinite(coordinate.z))
        {
            throw std::invalid_argument(
                "Invalid QSGW independent head-wing k point in " +
                path.string());
        }
        result.kpoints.push_back(coordinate);
    }
    return result;
}

void validate_independent_wfc_reader_set(
    const librpa_int::qsgw::IndependentHeadwingPaths& paths)
{
    std::vector<std::filesystem::path> actual;
    const std::string& prefix = driver::driver_params.prefix_eigvecs_scf;
    for (const auto& entry : std::filesystem::directory_iterator(
             paths.directory))
    {
        if (!entry.is_regular_file()) continue;
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0)
        {
            actual.push_back(
                std::filesystem::absolute(entry.path()).lexically_normal());
        }
    }
    std::vector<std::filesystem::path> expected = paths.wavefunctions;
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    if (actual != expected)
    {
        throw std::invalid_argument(
            "QSGW independent head-wing WFC files selected by the reader do not exactly match the input contract");
    }
}

void validate_same_grid_velocity_binding(
    const librpa_int::qsgw::QsgwInputContract& contract,
    const std::string& contract_base,
    const int n_kpoints)
{
    std::vector<std::filesystem::path> expected =
        librpa_int::qsgw::resolve_same_grid_velocity_paths(
            contract.producer(), driver::driver_params.input_dir, n_kpoints);

    std::vector<std::filesystem::path> declared;
    for (const auto& file : contract.files("velocity_mf0"))
    {
        declared.push_back(resolved_absolute_path(contract_base, file.file));
    }
    std::sort(expected.begin(), expected.end());
    std::sort(declared.begin(), declared.end());
    if (declared != expected)
    {
        throw std::invalid_argument(
            "QSGW velocity_mf0 contract files do not exactly match the same-grid head/wing files read by the driver");
    }
}

bool starts_with(const std::string& text, const std::string& prefix)
{
    return text.rfind(prefix, 0) == 0;
}

const librpa_int::qsgw::HartreeReaderRoute& hartree_reader_route()
{
    static const librpa_int::qsgw::HartreeReaderRoute route =
        librpa_int::qsgw::select_hartree_reader_route(
            driver::opts.use_shrink_abfs == LIBRPA_SWITCH_ON,
            driver::driver_params.prefix_lri_coeff,
            driver::driver_params.prefix_lri_coeff_shrink);
    return route;
}

std::vector<std::filesystem::path> discover_hartree_ri_reader_files(
    const librpa_int::qsgw::HartreeReaderRoute& route)
{
    const std::string& input_dir = driver::driver_params.input_dir;
    const std::string& other_prefix = route.use_shrink_basis
        ? driver::driver_params.prefix_lri_coeff
        : driver::driver_params.prefix_lri_coeff_shrink;
    const std::vector<std::string> discovered =
        librpa_int::discover_files_with_prefix(input_dir, route.ri_prefix);
    std::vector<std::filesystem::path> result;
    for (const std::string& path : discovered)
    {
        const std::string filename =
            std::filesystem::path(path).filename().string();
        if (starts_with(other_prefix, route.ri_prefix) &&
            starts_with(filename, other_prefix))
        {
            continue;
        }
        const std::filesystem::path resolved =
            std::filesystem::absolute(path).lexically_normal();
        librpa_int::require_readable_file(resolved.string());
        result.push_back(resolved);
    }
    if (result.empty())
        throw std::invalid_argument(
            "QSGW Hartree RI reader file set is empty for prefix " +
            route.ri_prefix);
    return result;
}

std::vector<std::filesystem::path> discover_hartree_aux_basis_sources(
    const librpa_int::qsgw::HartreeReaderRoute& route,
    const std::vector<std::filesystem::path>& ri_files)
{
    const auto candidate = [](const std::string& filename) {
        return resolved_absolute_path(
            driver::driver_params.input_dir, filename);
    };
    const auto exists = [](const std::filesystem::path& path) {
        return librpa_int::path_exists(path.string().c_str());
    };
    const auto explicit_source = [&](const std::filesystem::path& path) {
        librpa_int::require_readable_file(path.string());
        return std::vector<std::filesystem::path>{path};
    };

    if (route.use_shrink_basis)
    {
        for (const std::string& filename : {
                 driver::driver_params.fn_basis_aux_shrink,
                 std::string("basis_out_shrink"),
                 std::string("basis_out.shrink_backup")})
        {
            const std::filesystem::path path = candidate(filename);
            if (exists(path)) return explicit_source(path);
        }
        return ri_files;
    }

    const std::filesystem::path wavefunction_basis =
        candidate(driver::driver_params.fn_basis_wfc);
    const std::filesystem::path auxiliary_basis =
        candidate(driver::driver_params.fn_basis_aux);
    if (exists(wavefunction_basis) && exists(auxiliary_basis))
        return explicit_source(auxiliary_basis);

    const std::filesystem::path combined_basis =
        candidate(driver::driver_params.fn_basis);
    if (exists(combined_basis)) return explicit_source(combined_basis);
    return ri_files;
}

void validate_hartree_reader_binding(
    const librpa_int::qsgw::QsgwInputContract& contract,
    const std::string& contract_base)
{
    const std::string& input_dir = driver::driver_params.input_dir;
    const auto& route = hartree_reader_route();
    const std::vector<std::filesystem::path> ri_files =
        discover_hartree_ri_reader_files(route);
    const std::string& coulomb_prefix =
        driver::driver_params.qsgw_hartree_coulomb == "full"
            ? driver::driver_params.prefix_coul_full
            : driver::driver_params.prefix_coul_cut;
    if (coulomb_prefix.empty())
    {
        throw std::invalid_argument(
            "QSGW Hartree requires a nonempty Coulomb prefix");
    }
    const std::vector<std::string> discovered_coulomb =
        librpa_int::discover_files_with_prefix(input_dir, coulomb_prefix);
    std::vector<std::filesystem::path> coulomb_files;
    for (const std::string& path : discovered_coulomb)
    {
        coulomb_files.push_back(
            std::filesystem::absolute(path).lexically_normal());
    }

    librpa_int::qsgw::validate_hartree_input_binding(
        contract, contract_base, ri_files, coulomb_files,
        discover_hartree_aux_basis_sources(route, ri_files));
}

void validate_stage_one_contract(
    const librpa_int::qsgw::QsgwInputContract& contract,
    const librpa_int::Dataset& dataset,
    const std::string& contract_base,
    const librpa_int::qsgw::HeadwingGridMode headwing_grid,
    const bool update_hartree,
    const bool compute_band)
{
    using namespace librpa_int::qsgw;
    validate_qsgw_execution_modes(
        contract, headwing_grid, update_hartree, compute_band);
    if (contract.n_spins() != dataset.mf.get_n_spins() ||
        contract.n_bands() != dataset.mf.get_n_bands() ||
        contract.n_aos() != dataset.mf.get_n_aos() ||
        contract.n_scf_kpoints() != dataset.mf.get_n_kpoints() ||
        contract.n_scf_kpoints() !=
            static_cast<int>(dataset.pbc.kfrac_list.size()))
    {
        throw std::invalid_argument(
            "QSGW input-contract dimensions do not match the loaded mf0 dataset");
    }
    validate_projection_target(
        dataset.mf, dataset.pbc.kfrac_list,
        contract.n_spins(), dataset.mf.get_n_spinor(), contract.n_aos(),
        "grid");
    if (!role_contains_path(contract, "mf0_eigenvalues",
                            driver::driver_params.fn_eigocc_scf) ||
        !role_contains_path(contract, "scf_kpoints",
                            driver::driver_params.fn_bz_sampling))
    {
        throw std::invalid_argument(
            "QSGW input contract is not bound to the eigenvalue or k-point file used by the driver");
    }
    const bool producer_matches_constants =
        (contract.producer() == QsgwProducer::FhiAims &&
         driver::driver_params.constants_choice == "aims") ||
        (contract.producer() == QsgwProducer::Abacus &&
         driver::driver_params.constants_choice == "internal");
    if (!producer_matches_constants)
    {
        throw std::invalid_argument(
            "QSGW input-contract producer does not match constants_choice");
    }
    if (headwing_grid == HeadwingGridMode::ScfGrid)
    {
        validate_same_grid_velocity_binding(
            contract, contract_base, dataset.mf.get_n_kpoints());
    }
    else if (headwing_grid == HeadwingGridMode::IndependentFullGrid)
    {
        if (contract.producer() != QsgwProducer::Abacus)
        {
            throw std::invalid_argument(
                "QSGW independent full-grid head/wing currently requires ABACUS PyATB input");
        }
        validate_independent_headwing_binding(
            contract, contract_base, driver::driver_params.input_dir);
    }
    if (update_hartree)
    {
        validate_hartree_reader_binding(contract, contract_base);
    }
    if (compute_band)
    {
        if (contract.n_band_kpoints() != dataset.mf_band.get_n_kpoints() ||
            contract.n_band_kpoints() !=
                static_cast<int>(dataset.kfrac_band_list.size()))
        {
            throw std::invalid_argument(
                "QSGW band input-contract dimensions do not match the loaded band reference");
        }
        validate_projection_target(
            dataset.mf_band, dataset.kfrac_band_list,
            dataset.mf.get_n_spins(), dataset.mf.get_n_spinor(),
            dataset.mf.get_n_aos(), "band");
        validate_band_reference_binding(
            contract, contract_base, driver::driver_params.input_dir,
            driver::driver_params.fn_band_kpath_info);
    }
}

std::unique_ptr<IndependentHeadwingState>
load_independent_headwing_state(
    librpa_int::Dataset& dataset,
    const librpa_int::qsgw::IndependentHeadwingPaths& paths,
    const int expected_n_kpoints,
    const MeanField& source_reference,
    const double electron_count)
{
    using namespace librpa_int;
    using namespace librpa_int::qsgw;

    if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
    {
        throw std::invalid_argument(
            "QSGW independent head-wing operator Fourier currently requires use_kpara_scf_eigvec = false");
    }
    validate_independent_wfc_reader_set(paths);
    const HeadwingKPathInfo info = read_independent_headwing_kpoints(
        paths.kpoints, driver::driver_params.use_spinor_wfc);
    if (static_cast<int>(info.kpoints.size()) != expected_n_kpoints)
    {
        throw std::invalid_argument(
            "QSGW independent head-wing k-point count does not match the input contract");
    }

    auto state = std::make_unique<IndependentHeadwingState>();
    state->kpoints = info.kpoints;
    std::vector<int> identity_map(
        static_cast<std::size_t>(expected_n_kpoints));
    for (int kpoint = 0; kpoint < expected_n_kpoints; ++kpoint)
        identity_map[static_cast<std::size_t>(kpoint)] = kpoint;

    read_scf_occ_eigenvalues(
        paths.eigenvalues.string(), state->live,
        driver::driver_params.use_spinor_wfc, identity_map,
        expected_n_kpoints);
    const int read_status = read_eigenvector(
        path_as_directory(paths.directory.string()), state->live,
        driver::driver_params.use_spinor_wfc, identity_map, nullptr,
        LegacyTextWfcOrder::SpinBasisBand);
    if (read_status != 0)
    {
        throw std::runtime_error(
            "Failed to read QSGW independent head-wing eigenvectors from " +
            paths.directory.string());
    }
    read_velocity(
        paths.velocity.string(), state->live, state->live_velocity);

    const int complete_dimension =
        state->live.get_n_aos() * state->live.get_n_spinor();
    if (info.n_basis != state->live.get_n_aos() ||
        info.n_states != state->live.get_n_bands() ||
        info.n_spins != state->live.get_n_spins() ||
        state->live.get_n_bands() != complete_dimension ||
        source_reference.get_n_spins() != state->live.get_n_spins() ||
        source_reference.get_n_bands() != state->live.get_n_bands() ||
        source_reference.get_n_aos() != state->live.get_n_aos() ||
        source_reference.get_n_spinor() != state->live.get_n_spinor())
    {
        throw std::invalid_argument(
            "QSGW independent head-wing data do not form a complete source-compatible AO basis");
    }
    for (int spin = 0; spin < state->live.get_n_spins(); ++spin)
    {
        for (int spinor = 0; spinor < state->live.get_n_spinor(); ++spinor)
        {
            for (int kpoint = 0;
                 kpoint < state->live.get_n_kpoints(); ++kpoint)
            {
                const ComplexMatrix* wavefunction =
                    state->live.find_wfc(spin, spinor, kpoint);
                if (wavefunction == nullptr ||
                    wavefunction->nr != state->live.get_n_bands() ||
                    wavefunction->nc != state->live.get_n_aos())
                {
                    throw std::invalid_argument(
                        "QSGW independent head-wing wavefunction map is incomplete");
                }
            }
        }
    }

    state->reference = state->live;
    state->reference_velocity = state->live_velocity;
    state->weights.assign(
        static_cast<std::size_t>(expected_n_kpoints),
        1.0 / static_cast<double>(expected_n_kpoints));
    state->initial_occupations = analyze_qsgw_occupations(
        state->reference, state->weights, electron_count);
    state->reference_hamiltonian =
        build_reference_hamiltonian(state->reference);

    KPointBlacsProcessShape process_shape(
        KPointBlacsProcessShape::AUTO,
        KPointBlacsProcessShape::AUTO, true);
    state->kblacs = std::make_unique<KPointBlacsParallelContext>(
        process_shape, dataset.comm_h.comm, expected_n_kpoints);

    std::vector<double> frequency_weights;
    driver::h.get_imaginary_frequency_grids(
        driver::opts, dataset.omegas_imagfreq, frequency_weights);
    const auto& frequencies = dataset.tfg.get_freq_nodes();
    const auto& auxiliary_basis =
        driver::get_bool(driver::opts.use_shrink_abfs)
            ? dataset.basis_aux_shrink
            : dataset.basis_aux;
    if (!auxiliary_basis.initialized())
    {
        throw std::invalid_argument(
            "QSGW independent head-wing auxiliary basis is not initialized");
    }

    state->owner = &dataset;
    dataset.p_headwing = std::make_unique<diele_func>(
        state->live, state->live_velocity, state->kpoints,
        dataset.basis_wfc, auxiliary_basis, frequencies,
        info.n_basis, info.n_states, info.n_spins,
        auxiliary_basis.nb_total, dataset.pbc, dataset.comm_h,
        dataset.blacs_h, state->kblacs.get());
    dataset.p_headwing->use_2d_dielectric =
        driver::get_bool(driver::opts.use_2d_dielectric);
    dataset.p_headwing->use_soc = state->live.get_n_spinor() > 1;
    dataset.p_headwing->debug =
        librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG);
    return state;
}

SpinKMatrixMap load_vxc_manifest_root(
    const std::string& manifest_path,
    const MeanField& reference,
    const std::vector<Vector3_Order<double>>& kpoints,
    const librpa_int::qsgw::VxcDatasetKind dataset_kind)
{
    using namespace librpa_int;
    using namespace librpa_int::qsgw;

    require_readable_file(manifest_path);
    std::ifstream manifest_stream(manifest_path);
    const VxcManifest manifest =
        VxcManifest::parse(manifest_stream, manifest_path);
    if (manifest.producer() == "abacus" && reference.get_n_spinor() != 1)
    {
        throw std::invalid_argument(
            "ABACUS QSGW Vxc currently requires n_spinor=1");
    }
    const int expected_dimension =
        manifest.basis() == VxcBasis::Nao
            ? reference.get_n_aos() * reference.get_n_spinor()
            : reference.get_n_bands();
    manifest.validate(dataset_kind, reference.get_n_spins(),
                      kpoints, expected_dimension, expected_dimension,
                      1.0e-8);
    const std::string manifest_directory = parent_path(manifest_path);
    manifest.validate_file_hashes(manifest_directory);

    SpinKMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            const VxcManifestEntry& entry = manifest.at(spin, kpoint);
            const std::string matrix_path =
                resolve_input_path(manifest_directory, entry.file);
            require_readable_file(matrix_path);
            Matz input;
            if (manifest.producer() == "abacus")
            {
                std::ifstream stream(matrix_path);
                input = read_abacus_vxc_ha(stream, matrix_path);
            }
            else
            {
                input = load_matrix_cplx(matrix_path, MAJOR::COL);
            }
            result[spin][kpoint] = prepare_vxc_in_fixed_state_basis(
                input, manifest.basis(), reference, spin, kpoint);
        }
    }
    return result;
}

SpinKMatrixMap copy_exx_root(const librpa_int::Exx& exchange,
                             const MeanField& reference)
{
    SpinKMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            const auto spin_it = exchange.exx_KS.find(spin);
            if (spin_it == exchange.exx_KS.end())
                throw std::runtime_error("QSGW EXX spin channel is missing");
            const auto k_it = spin_it->second.find(kpoint);
            if (k_it == spin_it->second.end() ||
                k_it->second.nr() != reference.get_n_bands() ||
                k_it->second.nc() != reference.get_n_bands())
            {
                throw std::runtime_error(
                    "QSGW EXX fixed-basis matrix is missing or malformed");
            }
            result[spin][kpoint] = k_it->second.copy();
        }
    }
    return result;
}

SigmaMatrixMap collect_sigma_root(
    const librpa_int::G0W0& self_energy,
    const MeanField& reference,
    const std::vector<double>& frequencies,
    const librpa_int::MpiCommHandler& communicator)
{
    using librpa_int::qsgw::collect_blacs_matrix_root;
    SigmaMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            for (const double frequency : frequencies)
            {
                const Matz* local = nullptr;
                const auto spin_it = self_energy.sigc_is_ik_f_KS.find(spin);
                if (spin_it != self_energy.sigc_is_ik_f_KS.end())
                {
                    const auto k_it = spin_it->second.find(kpoint);
                    if (k_it != spin_it->second.end())
                    {
                        const auto f_it = k_it->second.find(frequency);
                        if (f_it != k_it->second.end()) local = &f_it->second;
                    }
                }
                int valid = local != nullptr &&
                                    local->major() == librpa_int::MAJOR::COL &&
                                    local->nr() ==
                                        self_energy.desc_sigc_is_ik_f_KS.m_loc() &&
                                    local->nc() ==
                                        self_energy.desc_sigc_is_ik_f_KS.n_loc()
                                ? 1
                                : 0;
                int all_valid = 0;
                communicator.allreduce(&valid, &all_valid, 1, MPI_MIN);
                if (!all_valid)
                    throw std::runtime_error(
                        "QSGW distributed fixed-basis Sigma is incomplete");
                Matz full = collect_blacs_matrix_root(
                    *local, self_energy.desc_sigc_is_ik_f_KS);
                if (communicator.is_root())
                {
                    if (full.nr() != reference.get_n_bands() ||
                        full.nc() != reference.get_n_bands())
                    {
                        throw std::runtime_error(
                            "QSGW collected Sigma has an invalid shape");
                    }
                    result[spin][kpoint][frequency] = std::move(full);
                }
            }
        }
    }
    return result;
}

SpinKMatrixMap build_correlation_map(
    const MeanField& live,
    const SigmaMatrixMap& sigma,
    const std::vector<double>& frequencies,
    const LibrpaOptions& options)
{
    using namespace librpa_int::qsgw;
    CorrelationPotentialSettings settings;
    settings.mode = CorrelationPotentialMode::ModeB;
    settings.n_params_anacon = options.n_params_anacon;
    settings.n_params_anacon_resample =
        options.n_params_anacon_resample < 0
            ? options.n_params_anacon
            : options.n_params_anacon_resample;

    SpinKMatrixMap result;
    for (int spin = 0; spin < live.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < live.get_n_kpoints(); ++kpoint)
        {
            result[spin][kpoint] = build_qsgw_correlation_potential(
                live, frequencies, sigma.at(spin).at(kpoint),
                spin, kpoint, settings);
        }
    }
    return result;
}

SigmaMatrixMap copy_head_tensor(
    const librpa_int::diele_func& headwing,
    const std::vector<double>& frequencies)
{
    const auto& head = headwing.get_head_matrices();
    if (head.size() != frequencies.size())
    {
        throw std::invalid_argument(
            "QSGW head tensor and frequency grids have different sizes");
    }

    SigmaMatrixMap result;
    for (std::size_t ifrequency = 0; ifrequency < head.size(); ++ifrequency)
    {
        Matz tensor(3, 3, librpa_int::MAJOR::ROW);
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                tensor(row, column) = head[ifrequency](row, column);
            }
        }
        result[0][0][frequencies[ifrequency]] = std::move(tensor);
    }
    return result;
}

void refresh_headwing(
    librpa_int::Dataset& dataset,
    const librpa::Options& options,
    const MeanField& live)
{
    if (!dataset.p_headwing)
    {
        throw std::invalid_argument(
            "QSGW head/wing object is not initialized");
    }
    dataset.p_headwing->get_meanfield_df() = live;
    dataset.p_headwing->init(options.sqrt_coulomb_threshold, dataset.vq);
    dataset.p_headwing->cal_head();
    dataset.epsmacs_imagfreq = dataset.p_headwing->get_head_vec();
    dataset.omegas_imagfreq = dataset.tfg.get_freq_nodes();
    if (options.option_dielect_func == 3)
    {
        const auto& headwing_cs =
            options.use_shrink_abfs == LIBRPA_SWITCH_ON
                ? dataset.cs_data_shrink
                : dataset.cs_data;
        dataset.p_headwing->cal_wing(
            headwing_cs, options.sqrt_coulomb_threshold, dataset.vq);
    }
}

librpa_int::qsgw::HartreeStaticData load_hartree_static_input_root(
    librpa_int::Dataset& dataset)
{
    using namespace driver;
    using namespace librpa_int;
    using namespace librpa_int::qsgw;

    const HartreeReaderRoute& route = hartree_reader_route();
    const bool use_shrink_basis = route.use_shrink_basis;
    const bool use_cut_coulomb =
        driver_params.qsgw_hartree_coulomb == "truncated";
    const bool reuse_local_coefficients = dataset.comm_h.nprocs == 1;
    const auto saved_routing = opts.parallel_routing;
    Cs_LRI& selected_coefficients = use_shrink_basis
        ? dataset.cs_data_shrink
        : dataset.cs_data;
    const AtomicBasis& selected_auxiliary_basis = use_shrink_basis
        ? dataset.basis_aux_shrink
        : dataset.basis_aux;
    Cs_LRI local_coefficients;
    Cs_LRI saved_coefficients;
    if (reuse_local_coefficients)
    {
        local_coefficients = materialize_hartree_coefficients(
            selected_coefficients, dataset.basis_wfc,
            selected_auxiliary_basis);
    }
    else
    {
        saved_coefficients = std::move(selected_coefficients);
    }

    atpair_k_cplx_mat_t& selected_coulomb =
        use_cut_coulomb ? dataset.vq_cut : dataset.vq;
    atpair_k_cplx_mat_t saved_coulomb = std::move(selected_coulomb);
    std::vector<Vector3_Order<double>> saved_reader_klist =
        dataset.pbc.klist;

    const auto restore_distributed_input = [&] {
        if (!reuse_local_coefficients)
            selected_coefficients = std::move(saved_coefficients);
        selected_coulomb = std::move(saved_coulomb);
        dataset.pbc.klist = std::move(saved_reader_klist);
        opts.parallel_routing = saved_routing;
    };

    HartreeStaticData result;
    try
    {
        if (!reuse_local_coefficients)
        {
            selected_coefficients = Cs_LRI{};
            opts.parallel_routing = LIBRPA_ROUTING_RTAU;
            const auto all_atom_pairs = generate_atom_pair_from_nat(
                n_atoms, false);
            read_Cs(
                driver_params.input_dir, driver_params.cs_threshold,
                all_atom_pairs, route.ri_prefix,
                driver_params.version_lri_reader);
        }

        if (static_cast<int>(dataset.pbc.klist_full.size()) !=
            dataset.pbc.get_n_cells_bvk())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW Hartree full Coulomb staging requires the complete BZ k-list");
        }
        dataset.pbc.klist = dataset.pbc.klist_full;
        read_Vq_full(
            driver_params.input_dir,
            use_cut_coulomb ? driver_params.prefix_coul_cut
                            : driver_params.prefix_coul_full,
            use_cut_coulomb, driver_params.version_coul_reader,
            use_shrink_basis);

        const HartreeKNormalization normalization =
            driver_params.qsgw_hartree_normalization ==
                    "weighted_occupations"
                ? HartreeKNormalization::weighted_occupations
                : HartreeKNormalization::legacy_extra_inverse_nk;
        const auto bvk_remap = api::build_band_bvk_remap(
            dataset.atoms, dataset.pbc, opts.option_bvk_remap);
        result = build_hartree_static_data(
            reuse_local_coefficients ? local_coefficients
                                     : selected_coefficients,
            selected_coulomb, dataset.basis_wfc,
            selected_auxiliary_basis,
            dataset.pbc, bvk_remap, normalization);
    }
    catch (...)
    {
        restore_distributed_input();
        throw;
    }
    restore_distributed_input();
    return result;
}

void write_contract_header(std::ostream& output,
                           const std::string& contract_path,
                           const std::string& contract_sha256,
                           const librpa_int::qsgw::HeadwingGridMode headwing_grid,
                           const bool update_hartree,
                           const bool compute_band)
{
    using librpa_int::qsgw::HeadwingGridMode;
    const bool compute_headwing =
        headwing_grid != HeadwingGridMode::Disabled;
    output << "# qsgw_contract_version 5\n"
           << "# fixed_basis immutable_mf0\n"
           << "# live_update eigenvalues_wfc\n"
           << "# velocity "
           << (headwing_grid == HeadwingGridMode::ScfGrid
                   ? "fixed_basis_rotation"
                   : headwing_grid == HeadwingGridMode::IndependentFullGrid
                         ? "live_ao_fourier_rotation"
                         : "disabled_stage1")
           << "\n"
           << "# headwing "
           << (headwing_grid == HeadwingGridMode::ScfGrid
                   ? "scf_grid_analytic_live"
                   : headwing_grid == HeadwingGridMode::IndependentFullGrid
                         ? "independent_full_grid_analytic_live"
                         : "disabled_stage1")
           << "\n"
           << "# symmetry unsupported_full_bz_only\n"
           << "# hartree "
           << (update_hartree ? "delta_density" : "disabled_stage1")
           << "\n";
    if (update_hartree)
    {
        output << "# hartree_coulomb "
               << driver::driver_params.qsgw_hartree_coulomb << "\n"
               << "# hartree_normalization "
               << driver::driver_params.qsgw_hartree_normalization << "\n";
    }
    output << "# band "
           << (compute_band
                   ? "fixed_reference_operator_fourier_live"
                   : "disabled_stage1")
           << "\n"
           << "# qsgw_input_contract " << contract_path << "\n"
           << "# qsgw_input_contract_sha256 " << contract_sha256 << "\n"
           << "# qsgw_mixer " << driver::driver_params.qsgw_mixer << "\n"
           << "# qsgw_mixing_beta " << std::setprecision(17)
           << driver::driver_params.qsgw_mixing_beta << "\n";
}

void run_qsgw_stage_one(const bool compute_band)
{
    using namespace driver;
    using namespace librpa_int;
    using namespace librpa_int::global;
    using namespace librpa_int::qsgw;

    profiler.start("qsgw", "QSGW fixed-basis self-consistent calculation");
    const auto dataset = api::get_dataset_instance(h);
    if (opts.parallel_routing != LIBRPA_ROUTING_LIBRI &&
        opts.parallel_routing != LIBRPA_ROUTING_AUTO)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW stage one requires parallel_routing=libri or auto");
    }
    if (compute_band)
    {
        if (driver::get_bool(opts.use_kpara_scf_eigvec))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW band fixed-reference preflight does not yet support k-distributed band wavefunctions");
        }
        const std::string band_kpath_path = resolve_input_path(
            driver_params.input_dir, driver_params.fn_band_kpath_info);
        read_band_kpath_info(band_kpath_path);
        dataset->comm_h.barrier();
        read_band_meanfield_data(driver_params.input_dir);
        dataset->comm_h.barrier();
    }
    read_Vq_row(driver_params.input_dir, driver_params.prefix_coul_cut,
                opts.vq_threshold, local_atpair, true,
                driver_params.version_coul_reader,
                driver::get_bool(opts.use_shrink_abfs));

    const bool compute_headwing =
        driver::get_bool(opts.replace_w_head) &&
        (opts.option_dielect_func == 3 || opts.option_dielect_func == 4);
    const HeadwingGridMode headwing_grid = !compute_headwing
        ? HeadwingGridMode::Disabled
        : driver_params.use_pyatb
              ? HeadwingGridMode::IndependentFullGrid
              : HeadwingGridMode::ScfGrid;
    const IterationChannel headwing_channel =
        headwing_grid == HeadwingGridMode::IndependentFullGrid
            ? IterationChannel::Headwing
            : IterationChannel::Grid;

    const std::string contract_path = resolve_input_path(
        driver_params.input_dir, driver_params.qsgw_input_contract);
    std::optional<QsgwInputContract> input_contract;
    std::string contract_sha256;
    int contract_producer = -1;
    int contract_headwing_kpoints = 0;
    collective_root_stage(dataset->comm_h, "QSGW input preflight", [&] {
        require_readable_file(contract_path);
        std::ifstream stream(contract_path);
        input_contract = QsgwInputContract::parse(stream, contract_path);
        const std::string base = parent_path(contract_path);
        input_contract->validate_file_hashes(base);
        validate_stage_one_contract(
            *input_contract, *dataset, base, headwing_grid,
            driver_params.qsgw_update_hartree, compute_band);
        contract_producer = static_cast<int>(input_contract->producer());
        contract_headwing_kpoints =
            input_contract->n_headwing_kpoints();
        contract_sha256 = sha256_file(contract_path);
    });
    dataset->comm_h.bcast(&contract_producer, 1, 0);
    dataset->comm_h.bcast(&contract_headwing_kpoints, 1, 0);
    if (contract_producer != static_cast<int>(QsgwProducer::Abacus) &&
        contract_producer != static_cast<int>(QsgwProducer::FhiAims))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW input producer broadcast is invalid");
    }

    const MeanField reference = dataset->mf;
    const double electron_count = physical_electron_count(
        reference, dataset->pbc.weight_k);
    const OccupationResult initial_occupations = analyze_qsgw_occupations(
        reference, dataset->pbc.weight_k, electron_count);
    const SpinKMatrixMap reference_hamiltonian =
        build_reference_hamiltonian(reference);

    std::unique_ptr<IndependentHeadwingState> independent_headwing;
    if (compute_headwing)
    {
        if (headwing_grid == HeadwingGridMode::ScfGrid)
        {
            read_headwing_input(
                driver_params.input_dir, opts.option_dielect_func == 3);
            if (contract_producer == static_cast<int>(QsgwProducer::FhiAims))
            {
                prepare_fhi_aims_interband_velocity(
                    dataset->velocity_matrix, reference);
            }
            align_distributed_velocity_to_reference_wfc(
                dataset->p_headwing->get_meanfield_df(), reference,
                dataset->velocity_matrix, dataset->comm_h);
            refresh_headwing(*dataset, opts, dataset->mf);
        }
        else
        {
            const IndependentHeadwingPaths paths =
                resolve_independent_headwing_paths(
                    driver_params.input_dir,
                    contract_headwing_kpoints);
            independent_headwing = load_independent_headwing_state(
                *dataset, paths, contract_headwing_kpoints,
                reference, electron_count);
            refresh_headwing(
                *dataset, opts, independent_headwing->live);
        }
    }

    const VelocityMatrix reference_velocity =
        headwing_grid == HeadwingGridMode::ScfGrid
            ? dataset->velocity_matrix
            : VelocityMatrix{};

    std::optional<MeanField> band_reference;
    SpinKMatrixMap band_reference_hamiltonian;
    if (compute_band)
    {
        dataset->mf_band.get_efermi() = reference.get_efermi();
        band_reference = dataset->mf_band;
        band_reference_hamiltonian =
            build_reference_hamiltonian(*band_reference);
    }

    SpinKMatrixMap dft_vxc;
    SpinKMatrixMap dft_vxc_band;
    collective_root_stage(dataset->comm_h, "QSGW Vxc preflight", [&] {
        const auto& records = input_contract->files("vxc_scf_manifest");
        if (records.size() != 1)
            throw std::invalid_argument(
                "QSGW contract requires exactly one SCF Vxc manifest");
        const std::string path = resolve_input_path(
            parent_path(contract_path), records.front().file);
        dft_vxc = load_vxc_manifest_root(
            path, reference, dataset->pbc.kfrac_list,
            VxcDatasetKind::ScfGrid);
        if (compute_band)
        {
            const auto& band_records =
                input_contract->files("vxc_band_manifest");
            if (band_records.size() != 1)
                throw std::invalid_argument(
                    "QSGW contract requires exactly one band Vxc manifest");
            const std::string band_path = resolve_input_path(
                parent_path(contract_path), band_records.front().file);
            dft_vxc_band = load_vxc_manifest_root(
                band_path, *band_reference, dataset->kfrac_band_list,
                VxcDatasetKind::BandPath);
        }
    });

    std::optional<HartreeStaticData> hartree_static;
    if (driver_params.qsgw_update_hartree)
    {
        collective_root_stage(dataset->comm_h, "QSGW Hartree setup", [&] {
            hartree_static = load_hartree_static_input_root(*dataset);
        });
    }

    std::optional<SpinKHamiltonianMixer> mixer;
    if (driver_params.qsgw_mixer == "linear")
    {
        MixingOptions options;
        options.mode = MixingMode::Linear;
        options.beta = driver_params.qsgw_mixing_beta;
        mixer.emplace(options);
        if (dataset->comm_h.is_root())
        {
            if (compute_band)
                mixer->initialize(reference_hamiltonian,
                                  band_reference_hamiltonian);
            else
                mixer->initialize(reference_hamiltonian);
        }
    }
    SpinKMatrixMap current_hamiltonian = reference_hamiltonian;
    SpinKMatrixMap current_band_hamiltonian =
        band_reference_hamiltonian;

    std::ofstream trace;
    std::ofstream eigenvalue_trace;
    std::ofstream matrix_trace;
    collective_root_stage(dataset->comm_h, "QSGW trace setup", [&] {
        trace.open(join_path(opts.output_dir, "qsgw_iterations.dat"));
        eigenvalue_trace.open(
            join_path(opts.output_dir, "qsgw_eigenvalues.dat"));
        if (driver_params.qsgw_write_iteration_matrices)
            matrix_trace.open(join_path(opts.output_dir, "qsgw_matrices.dat"));
        if (!trace || !eigenvalue_trace ||
            (driver_params.qsgw_write_iteration_matrices && !matrix_trace))
            throw std::runtime_error("Cannot open QSGW trace output");
        write_contract_header(
            trace, contract_path, contract_sha256, headwing_grid,
            driver_params.qsgw_update_hartree, compute_band);
        write_contract_header(eigenvalue_trace, contract_path,
                               contract_sha256, headwing_grid,
                               driver_params.qsgw_update_hartree,
                               compute_band);
        write_iteration_summary_header(trace);
        write_eigenvalue_trace_header(eigenvalue_trace);
        if (driver_params.qsgw_write_iteration_matrices)
        {
            write_contract_header(matrix_trace, contract_path,
                                   contract_sha256, headwing_grid,
                                   driver_params.qsgw_update_hartree,
                                   compute_band);
            write_matrix_trace_header(matrix_trace);
            write_matrix_component_trace(
                matrix_trace, 0, IterationChannel::Grid, "h0",
                reference_hamiltonian);
            write_matrix_component_trace(
                matrix_trace, 0, IterationChannel::Grid, "vxc_dft",
                dft_vxc);
            write_wavefunction_trace(
                matrix_trace, 0, IterationChannel::Grid, reference);
            write_occupation_trace(
                matrix_trace, 0, IterationChannel::Grid, dataset->mf);
            if (compute_band)
            {
                write_matrix_component_trace(
                    matrix_trace, 0, IterationChannel::Band, "h0",
                    band_reference_hamiltonian);
                write_matrix_component_trace(
                    matrix_trace, 0, IterationChannel::Band, "vxc_dft",
                    dft_vxc_band);
                write_wavefunction_trace(
                    matrix_trace, 0, IterationChannel::Band,
                    *band_reference);
                write_occupation_trace(
                    matrix_trace, 0, IterationChannel::Band,
                    dataset->mf_band);
            }
            if (compute_headwing)
            {
                if (independent_headwing)
                {
                    write_matrix_component_trace(
                        matrix_trace, 0, IterationChannel::Headwing,
                        "h0", independent_headwing->reference_hamiltonian);
                    write_wavefunction_trace(
                        matrix_trace, 0, IterationChannel::Headwing,
                        independent_headwing->reference);
                    write_occupation_trace(
                        matrix_trace, 0, IterationChannel::Headwing,
                        independent_headwing->live);
                }
                write_velocity_trace(
                    matrix_trace, 0, headwing_channel,
                    independent_headwing
                        ? independent_headwing->reference_velocity
                        : reference_velocity);
                write_frequency_matrix_component_trace(
                    matrix_trace, 0, headwing_channel,
                    "head_tensor", copy_head_tensor(
                        *dataset->p_headwing,
                        dataset->tfg.get_freq_nodes()));
            }
        }
        IterationSummary summary;
        summary.iteration = 0;
        summary.fermi_energy_ev = reference.get_efermi() * HA2EV;
        summary.gap_ev = initial_occupations.gap * HA2EV;
        summary.electron_count = initial_occupations.electron_count;
        summary.has_mixing_decision = false;
        write_iteration_summary(trace, summary);
        write_eigenvalue_trace(eigenvalue_trace, 0,
                               IterationChannel::Grid, reference,
                               dataset->pbc.kfrac_list);
        if (compute_band)
        {
            write_eigenvalue_trace(
                eigenvalue_trace, 0, IterationChannel::Band,
                *band_reference, dataset->kfrac_band_list);
        }
        if (independent_headwing)
        {
            write_eigenvalue_trace(
                eigenvalue_trace, 0, IterationChannel::Headwing,
                independent_headwing->live,
                independent_headwing->kpoints);
        }
    });

    bool converged = false;
    int completed_iterations = 0;
    for (int iteration = 1;
         iteration <= driver_params.qsgw_max_iter; ++iteration)
    {
        completed_iterations = iteration;
        const EigenvalueSnapshot previous = eigenvalue_snapshot(dataset->mf);
        dataset->invalidate_compute_objects();
        SigmaMatrixMap head_tensor;
        std::optional<IndependentHeadwingUpdateResult>
            independent_headwing_update;
        if (compute_headwing)
        {
            refresh_headwing(
                *dataset, opts,
                independent_headwing
                    ? independent_headwing->live
                    : dataset->mf);
            collective_root_stage(
                dataset->comm_h, "QSGW head-tensor trace", [&] {
                    head_tensor = copy_head_tensor(
                        *dataset->p_headwing,
                        dataset->tfg.get_freq_nodes());
                });
        }
        h.build_g0w0_sigma(opts);
        if (!dataset->p_exx || !dataset->p_g0w0)
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW failed to build live EXX/Sigma real-space objects");

        // Upstream uses this field to gate in-memory full-matrix retention as
        // well as optional file output. QSGW consumes the matrices directly;
        // the user-facing output option remains independent and may stay off.
        dataset->p_g0w0->output_sigc_ks_mat_kf = true;
        {
            ScopedReferenceEigenvectors fixed_basis_projection(
                dataset->mf, reference);
            dataset->p_exx->build_KS_kgrid_blacs(
                dataset->blacs_h,
                opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
            dataset->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(
                dataset->blacs_h,
                opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
        }
        const std::vector<double> frequencies =
            dataset->tfg.get_freq_nodes();
        const SigmaMatrixMap sigma = collect_sigma_root(
            *dataset->p_g0w0, reference, frequencies, dataset->comm_h);

        SpinKMatrixMap exchange;
        collective_root_stage(dataset->comm_h, "QSGW EXX collection", [&] {
            exchange = copy_exx_root(*dataset->p_exx, reference);
        });

        SigmaMatrixMap sigma_band;
        SpinKMatrixMap exchange_band;
        if (compute_band)
        {
            dataset->p_exx->reset_kspace();
            dataset->p_g0w0->reset_kspace();
            const auto bvk_remap = api::build_band_bvk_remap(
                dataset->atoms, dataset->pbc, opts.option_bvk_remap);
            dataset->p_exx->build_KS_band_blacs(
                band_reference->get_eigenvectors(),
                dataset->kfrac_band_list, bvk_remap, dataset->blacs_h,
                opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
            dataset->p_g0w0->build_sigc_matrix_KS_band_blacs(
                band_reference->get_eigenvectors(),
                dataset->kfrac_band_list, bvk_remap, dataset->blacs_h,
                opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON,
                nullptr);
            sigma_band = collect_sigma_root(
                *dataset->p_g0w0, *band_reference, frequencies,
                dataset->comm_h);
            collective_root_stage(
                dataset->comm_h, "QSGW band EXX collection", [&] {
                    exchange_band = copy_exx_root(
                        *dataset->p_exx, *band_reference);
                });
            dataset->is_band_calc_done = true;
        }

        SpinKMatrixMap mixed_hamiltonian;
        SpinKMatrixMap mixed_band_hamiltonian;
        double residual_l2 = 0.0;
        double residual_max = 0.0;
        std::optional<MixingDecision> mixing_decision;
        std::string matrix_rows;
        collective_root_stage(dataset->comm_h, "QSGW Hamiltonian update", [&] {
            const SpinKMatrixMap correlation = build_correlation_map(
                dataset->mf, sigma, frequencies, opts);
            std::optional<PeriodicOperatorRMap> hartree_r;
            std::optional<SpinKMatrixMap> hartree;
            std::optional<SpinKMatrixMap> hartree_band;
            if (driver_params.qsgw_update_hartree)
            {
                if (!hartree_static)
                {
                    throw std::runtime_error(
                        "QSGW Hartree static input is unavailable on root");
                }
                hartree_r = build_hartree_delta_periodic_operator(
                    *hartree_static, dataset->mf, reference,
                    dataset->pbc.kfrac_list);
                hartree = project_periodic_operator_to_fixed_basis(
                    *hartree_r, reference, dataset->pbc.kfrac_list);
                if (compute_band)
                {
                    hartree_band = project_periodic_operator_to_fixed_basis(
                        *hartree_r, *band_reference,
                        dataset->kfrac_band_list);
                }
            }
            const SpinKMatrixMap raw = assemble_effective_hamiltonian(
                reference_hamiltonian, dft_vxc, exchange, correlation,
                hartree ? &*hartree : nullptr);
            SpinKMatrixMap correlation_band;
            SpinKMatrixMap raw_band;
            if (compute_band)
            {
                correlation_band = build_correlation_map(
                    dataset->mf_band, sigma_band, frequencies, opts);
                raw_band = assemble_effective_hamiltonian(
                    band_reference_hamiltonian, dft_vxc_band,
                    exchange_band, correlation_band,
                    hartree_band ? &*hartree_band : nullptr);
            }
            const auto residual = measure_spin_k_hamiltonian_residual(
                raw, current_hamiltonian);
            residual_l2 = residual.l2;
            residual_max = residual.maximum;
            if (compute_band)
            {
                const auto band_residual =
                    measure_spin_k_hamiltonian_residual(
                        raw_band, current_band_hamiltonian);
                residual_l2 = std::hypot(
                    residual_l2, band_residual.l2);
                residual_max = std::max(
                    residual_max, band_residual.maximum);
            }
            if (mixer)
            {
                SpinKHamiltonianMixResult result = compute_band
                    ? mixer->mix(raw, raw_band)
                    : mixer->mix(raw);
                mixed_hamiltonian = std::move(result.grid);
                if (compute_band)
                {
                    if (!result.band)
                        throw std::runtime_error(
                            "QSGW synchronized mixer did not return a band Hamiltonian");
                    mixed_band_hamiltonian = std::move(*result.band);
                }
                residual_l2 = result.residual_l2;
                residual_max = result.residual_max;
                mixing_decision = std::move(result.decision);
            }
            else
            {
                mixed_hamiltonian = raw;
                if (compute_band) mixed_band_hamiltonian = raw_band;
            }
            current_hamiltonian = mixed_hamiltonian;
            if (compute_band)
                current_band_hamiltonian = mixed_band_hamiltonian;

            if (driver_params.qsgw_write_iteration_matrices)
            {
                std::ostringstream rows;
                write_frequency_matrix_component_trace(
                    rows, iteration, IterationChannel::Grid,
                    "sigma_c_iw", sigma);
                if (compute_headwing)
                {
                    write_frequency_matrix_component_trace(
                        rows, iteration, headwing_channel,
                        "head_tensor", head_tensor);
                }
                write_matrix_component_trace(
                    rows, iteration, IterationChannel::Grid, "exx",
                    exchange);
                write_matrix_component_trace(
                    rows, iteration, IterationChannel::Grid, "vc",
                    correlation);
                if (hartree)
                {
                    write_matrix_component_trace(
                        rows, iteration, IterationChannel::Grid,
                        "delta_vh", *hartree);
                }
                write_matrix_component_trace(
                    rows, iteration, IterationChannel::Grid, "raw_h", raw);
                write_matrix_component_trace(
                    rows, iteration, IterationChannel::Grid, "mixed_h",
                    mixed_hamiltonian);
                if (compute_band)
                {
                    write_frequency_matrix_component_trace(
                        rows, iteration, IterationChannel::Band,
                        "sigma_c_iw", sigma_band);
                    write_matrix_component_trace(
                        rows, iteration, IterationChannel::Band, "exx",
                        exchange_band);
                    write_matrix_component_trace(
                        rows, iteration, IterationChannel::Band, "vc",
                        correlation_band);
                    if (hartree_band)
                    {
                        write_matrix_component_trace(
                            rows, iteration, IterationChannel::Band,
                            "delta_vh", *hartree_band);
                    }
                    write_matrix_component_trace(
                        rows, iteration, IterationChannel::Band, "raw_h",
                        raw_band);
                    write_matrix_component_trace(
                        rows, iteration, IterationChannel::Band, "mixed_h",
                        mixed_band_hamiltonian);
                }
                matrix_rows = rows.str();
            }
        });

        broadcast_spin_k_matrix_map(
            mixed_hamiltonian, 0, dataset->comm_h);
        if (compute_band)
        {
            broadcast_spin_k_matrix_map(
                mixed_band_hamiltonian, 0, dataset->comm_h);
        }
        const FixedBasisDiagonalizationResult diagonalization =
            diagonalize_in_reference_basis(
                dataset->mf, reference, mixed_hamiltonian,
                headwing_grid == HeadwingGridMode::ScfGrid
                    ? &reference_velocity
                    : nullptr,
                headwing_grid == HeadwingGridMode::ScfGrid
                    ? &dataset->velocity_matrix
                    : nullptr);
        std::optional<FixedBasisDiagonalizationResult> band_diagonalization;
        if (compute_band)
        {
            band_diagonalization = diagonalize_in_reference_basis(
                dataset->mf_band, *band_reference,
                mixed_band_hamiltonian);
        }
        const OccupationResult occupations = update_qsgw_occupations(
            dataset->mf, reference, dataset->pbc.weight_k, electron_count);
        if (independent_headwing)
        {
            independent_headwing_update =
                update_independent_headwing_state(
                    mixed_hamiltonian, reference,
                    dataset->pbc.kfrac_list, dataset->pbc.Rlist,
                    independent_headwing->live,
                    independent_headwing->reference,
                    independent_headwing->kpoints,
                    independent_headwing->reference_velocity,
                    independent_headwing->live_velocity,
                    independent_headwing->weights, electron_count);
        }
        if (compute_band)
            dataset->mf_band.get_efermi() = occupations.chemical_potential;
        const double maximum_change_ev =
            max_eigenvalue_change(dataset->mf, previous) * HA2EV;

        int converged_flag = 0;
        collective_root_stage(dataset->comm_h, "QSGW reporting", [&] {
            converged_flag = qsgw_iteration_converged(
                                 iteration, driver_params.qsgw_min_iter,
                                 maximum_change_ev,
                                 driver_params.qsgw_convergence_tolerance_ev)
                                 ? 1
                                 : 0;
            IterationSummary summary;
            summary.iteration = iteration;
            summary.maximum_eigenvalue_change_ev = maximum_change_ev;
            summary.residual_l2_ha = residual_l2;
            summary.residual_max_ha = residual_max;
            summary.fermi_energy_ev =
                occupations.chemical_potential * HA2EV;
            summary.gap_ev = occupations.gap * HA2EV;
            summary.electron_count = occupations.electron_count;
            summary.converged = converged_flag != 0;
            summary.has_mixing_decision = mixing_decision.has_value();
            if (mixing_decision)
            {
                summary.requested_mode = mixing_decision->requested_mode;
                summary.applied_mode = mixing_decision->applied_mode;
                summary.beta = mixing_decision->beta;
                summary.fell_back = mixing_decision->fell_back;
                summary.reciprocal_condition =
                    mixing_decision->reciprocal_condition;
                summary.coefficients = mixing_decision->coefficients;
                summary.fallback_reason =
                    mixing_decision->fallback_reason;
            }
            write_iteration_summary(trace, summary);
            write_eigenvalue_trace(
                eigenvalue_trace, iteration, IterationChannel::Grid,
                dataset->mf, dataset->pbc.kfrac_list);
            if (compute_band)
            {
                write_eigenvalue_trace(
                    eigenvalue_trace, iteration, IterationChannel::Band,
                    dataset->mf_band, dataset->kfrac_band_list);
            }
            if (independent_headwing)
            {
                write_eigenvalue_trace(
                    eigenvalue_trace, iteration,
                    IterationChannel::Headwing,
                    independent_headwing->live,
                    independent_headwing->kpoints);
            }
            if (driver_params.qsgw_write_iteration_matrices)
            {
                matrix_trace << matrix_rows;
                write_matrix_component_trace(
                    matrix_trace, iteration, IterationChannel::Grid,
                    "rotation_u", diagonalization.unitary);
                write_wavefunction_trace(
                    matrix_trace, iteration, IterationChannel::Grid,
                    dataset->mf);
                write_occupation_trace(
                    matrix_trace, iteration, IterationChannel::Grid,
                    dataset->mf);
                if (compute_band)
                {
                    write_matrix_component_trace(
                        matrix_trace, iteration, IterationChannel::Band,
                        "rotation_u", band_diagonalization->unitary);
                    write_wavefunction_trace(
                        matrix_trace, iteration, IterationChannel::Band,
                        dataset->mf_band);
                    write_occupation_trace(
                        matrix_trace, iteration, IterationChannel::Band,
                        dataset->mf_band);
                }
                if (compute_headwing)
                {
                    write_velocity_trace(
                        matrix_trace, iteration, headwing_channel,
                        independent_headwing
                            ? independent_headwing->live_velocity
                            : dataset->velocity_matrix);
                }
                if (independent_headwing)
                {
                    if (!independent_headwing_update)
                    {
                        throw std::runtime_error(
                            "QSGW independent head-wing update result is missing");
                    }
                    write_matrix_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing, "projected_h",
                        independent_headwing_update->projected_hamiltonian);
                    write_matrix_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing, "rotation_u",
                        independent_headwing_update->unitary);
                    write_wavefunction_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        independent_headwing->live);
                    write_occupation_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        independent_headwing->live);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "basis_inverse_residual",
                        independent_headwing_update
                            ->maximum_basis_inverse_residual);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "basis_condition_estimate",
                        independent_headwing_update
                            ->maximum_basis_condition_estimate);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "fourier_orthogonality_residual",
                        independent_headwing_update
                            ->maximum_fourier_orthogonality_residual);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "source_roundtrip_relative_error",
                        independent_headwing_update
                            ->maximum_source_roundtrip_relative_error);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "target_hermiticity_error",
                        independent_headwing_update
                            ->maximum_target_hermiticity_error);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "target_relative_hermiticity_error",
                        independent_headwing_update
                            ->maximum_target_relative_hermiticity_error);
                    write_scalar_component_trace(
                        matrix_trace, iteration,
                        IterationChannel::Headwing,
                        "repaired_target_hermiticity_error",
                        independent_headwing_update
                            ->maximum_repaired_target_hermiticity_error);
                }
            }
            trace.flush();
            eigenvalue_trace.flush();
            if (driver_params.qsgw_write_iteration_matrices)
                matrix_trace.flush();
            lib_printf(
                "QSGW iteration %d: max_delta=% .8e eV gap=% .8f eV mixer=%s%s\n",
                iteration, maximum_change_ev, occupations.gap * HA2EV,
                driver_params.qsgw_mixer.c_str(),
                converged_flag ? ", converged" : "");
        });
        dataset->comm_h.bcast(&converged_flag, 1, 0);
        converged = converged_flag != 0;
        if (converged) break;
    }

    if (dataset->comm_h.is_root())
    {
        if (!converged)
            lib_printf(LIBRPA_VERBOSE_WARN,
                       "QSGW reached qsgw_max_iter=%d without convergence\n",
                       driver_params.qsgw_max_iter);
        lib_printf("QSGW completed iterations: %d\n", completed_iterations);
    }
    profiler.stop("qsgw");
}

} // namespace

void driver::task_qsgw()
{
    run_qsgw_stage_one(false);
}

void driver::task_qsgw_band()
{
    run_qsgw_stage_one(true);
}
