import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
QSGW_DRIVER = REPO_ROOT / "driver" / "tasks" / "qsgw.cpp"
TESTSUITE = REPO_ROOT / "regression_tests" / "testsuite.xml"
QSGW_HEAD_ONLY_CASE = (
    REPO_ROOT
    / "regression_tests"
    / "testcases"
    / "qsgw_aims_Si_k333_headonly_libri"
)
QSGW_HEAD_ONLY_REF = (
    REPO_ROOT
    / "regression_tests"
    / "refs"
    / "qsgw_aims_Si_k333_headonly_libri"
)


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reduced_grid_symmetry_context_is_prepared_before_contract_validation() -> None:
    source = QSGW_DRIVER.read_text()
    preparation = function_body(
        source, "void prepare_stage_one_symmetry_context("
    )
    assert "librpa_int::initialize_symmetry_context(dataset, true);" in preparation
    assert "dataset.pbc.kfrac_list_full.size()" in preparation
    assert "opts.use_symmetry_exx" in preparation
    assert "opts.use_symmetry_gw" in preparation
    assert "opts.use_symmetry_rpa" in preparation

    runner = function_body(source, "void run_qsgw_stage_one(")
    prepare_call = runner.index("prepare_stage_one_symmetry_context(*dataset);")
    validate_call = runner.index("validate_stage_one_contract(")
    assert prepare_call < validate_call


def test_qsgw_head_only_accepts_valid_symmetry_reduced_scf_input() -> None:
    source = QSGW_DRIVER.read_text()
    assert "QSGW iterative head-only currently requires a full-BZ SCF grid" not in source

    refresh = function_body(source, "void refresh_qsgw_head_only(")
    assert "dataset.p_headwing.reset();" in refresh
    assert "initialize_ds_headwing(dataset, options, false);" in refresh

    header = function_body(source, "void write_contract_header(")
    assert '<< "# head "' in header
    assert '<< "# wing disabled_stage1\\n"' in header
    assert '<< "# headwing "' not in header


def test_qsgw_band_rebuilds_static_operators_in_the_legacy_fixed_band_basis() -> None:
    source = QSGW_DRIVER.read_text()
    assert '#include "../../src/qsgw/operator_fourier.h"' in source

    runner = function_body(source, "void run_qsgw_stage_one(")
    normalized = " ".join(runner.split())
    assert "dataset->p_exx->reset_kspace();" in runner
    assert "dataset->p_g0w0->reset_kspace();" in runner
    assert "api::build_band_bvk_remap(" in runner
    assert "dataset->p_exx->build_KS_band_blacs(" in runner
    assert "dataset->p_g0w0->build_sigc_matrix_KS_band_blacs(" in runner
    assert "band_reference->get_eigenvectors()" in runner
    assert '"fixed_reference_rotation_live"' in source
    assert '"fixed_reference_operator_fourier_live"' not in source
    assert (
        "collect_sigma_root( *dataset->p_g0w0, *band_reference, "
        "frequencies, dataset->comm_h)" in normalized
    )
    assert (
        "build_correlation_map( dataset->mf_band, sigma_band, "
        "frequencies, opts)" in normalized
    )
    assert "project_grid_operator_to_band( exchange," not in normalized
    assert "project_grid_operator_to_band( correlation," not in normalized
    assert "project_grid_operator_to_band( *hartree," not in normalized
    assert (
        "project_periodic_operator_to_fixed_basis( *hartree_r, "
        "*band_reference, dataset->kfrac_band_list)" in normalized
    )
    assert (
        "band_reference_hamiltonian, dft_vxc_band, exchange_band, "
        "correlation_band" in normalized
    )


def test_qsgw_band_applies_the_same_hamiltonian_cut_to_grid_and_band() -> None:
    source = QSGW_DRIVER.read_text()
    assert '#include "../../src/qsgw/hamiltonian_cut.h"' in source
    runner = function_body(source, "void run_qsgw_stage_one(")
    normalized = " ".join(runner.split())

    assert "HamiltonianCutOptions cut_options;" in runner
    assert "cut_options.unoccupied_keep" in runner
    assert "cut_options.mode" in runner
    assert "cut_options.shift_ha" in runner
    assert (
        "current_hamiltonian = compute_band ? apply_hamiltonian_cut( "
        "reference_hamiltonian, reference_hamiltonian, dataset->mf, "
        "cut_options) : reference_hamiltonian" in normalized
    )
    assert (
        "current_band_hamiltonian = compute_band ? apply_hamiltonian_cut( "
        "band_reference_hamiltonian, band_reference_hamiltonian, "
        "dataset->mf_band, cut_options) : band_reference_hamiltonian"
        in normalized
    )
    assert (
        "apply_hamiltonian_cut( raw_uncut, reference_hamiltonian, "
        "dataset->mf, cut_options)" in normalized
    )
    assert (
        "apply_hamiltonian_cut( raw_band_uncut, band_reference_hamiltonian, "
        "dataset->mf_band, cut_options)" in normalized
    )
    assert (
        "apply_hamiltonian_cut( mixed_hamiltonian, reference_hamiltonian, "
        "dataset->mf, cut_options)" in normalized
    )
    assert (
        "apply_hamiltonian_cut( mixed_band_hamiltonian, "
        "band_reference_hamiltonian, dataset->mf_band, cut_options)"
        in normalized
    )
    assert (
        "mixer->initialize(mixed_hamiltonian, mixed_band_hamiltonian)"
        in normalized
    )


def test_qsgw_band_does_not_feed_postprocessing_residual_into_grid_convergence() -> None:
    source = QSGW_DRIVER.read_text()
    runner = function_body(source, "void run_qsgw_stage_one(")

    assert "std::hypot(residual_l2, band_residual.l2)" not in runner
    assert "residual_max, band_residual.maximum" not in runner


def test_qsgw_scopes_internal_sigma_matrix_retention_without_enabling_output() -> None:
    source = QSGW_DRIVER.read_text()
    runner = function_body(source, "void run_qsgw_stage_one(")

    assert "ScopedSigmaMatrixRetention retain_sigma_matrices(" in runner
    assert "output_sigc_ks_mat_kf = true;" not in runner
    assert "write_sigc_matrices_KS_binary" not in runner

    scope_start = source.index("class ScopedSigmaMatrixRetention")
    scope_end = source.index("};", scope_start)
    scope = source[scope_start : scope_end + 2]
    assert "original_(flag_)" in scope
    assert "flag_ = true;" in scope
    assert "~ScopedSigmaMatrixRetention()" in scope
    assert "flag_ = original_;" in scope


def test_qsgw_rejects_k_distributed_wavefunctions_before_dataset_use() -> None:
    source = QSGW_DRIVER.read_text()
    runner = function_body(source, "void run_qsgw_stage_one(")

    rejection = runner.index("if (driver::get_bool(opts.use_kpara_scf_eigvec))")
    dataset_use = runner.index("api::get_dataset_instance(h)")
    assert rejection < dataset_use
    assert "requires replicated SCF wavefunctions" in runner


def test_qsgw_band_exports_the_mixed_grid_hamiltonian_through_ao_real_space() -> None:
    source = QSGW_DRIVER.read_text()
    assert '#include "../../src/qsgw/abacus_csr.h"' in source
    runner = function_body(source, "void run_qsgw_stage_one(")
    normalized = " ".join(runner.split())

    assert "qsgw_export_hamiltonian_for_pyatb" in runner
    assert "project_grid_operator_to_band( mixed_hamiltonian," in normalized
    assert "real_space_ao" in runner
    assert "write_abacus_hamiltonian_csr(" in runner
    assert "_nao_qsgw_iter_" in runner


def test_qsgw_band_writes_legacy_compatible_iteration_tables() -> None:
    source = QSGW_DRIVER.read_text()
    assert '#include "../../src/qsgw/band_output.h"' in source
    runner = function_body(source, "void run_qsgw_stage_one(")

    assert "write_qsgw_band_spin_tables(" in runner
    assert '"KS_band_spin_"' in runner
    assert '"EXX_band_spin_"' in runner
    assert '"QSGW_band_spin_"' in runner


def test_symmetry_hartree_binds_reduced_grid_restoration_data() -> None:
    source = QSGW_DRIVER.read_text()
    runner = function_body(source, "void run_qsgw_stage_one(")
    normalized = " ".join(runner.split())

    assert "std::optional<HartreeSymmetryData> hartree_symmetry;" in runner
    assert "dataset->basis_wfc.build_species_basis_layouts(" in runner
    assert "dataset->basis_wfc.get_atom_nb_map();" in runner
    assert "symmetry_species_layouts_match_atom_counts(" in runner
    assert (
        "dataset->pbc.kfrac_list, hartree_symmetry ? "
        "&*hartree_symmetry : nullptr)" in normalized
    )


def test_qsgw_manual_evidence_is_not_advertised_as_formal_regression() -> None:
    source = TESTSUITE.read_text()
    assert "22/22" not in source
    assert "test_qsgw_headwing_update" not in source

    root = ET.fromstring(source)
    groups = [
        group for group in root.findall("group")
        if group.get("prefix") == "QSGW"
    ]
    assert len(groups) == 1
    assert groups[0].get("name") == "QSGW staged regression gates"

    manual_cases = [
        testcase for testcase in groups[0].findall("testcase")
        if "_manual_" in (testcase.get("directory") or "")
    ]
    assert manual_cases
    for testcase in manual_cases:
        labels = testcase.find("labels")
        assert labels is not None
        assert labels.get("disable") == "reference dataset not committed"
        assert "not a formal regression" in (testcase.get("name") or "")


def test_qsgw_head_only_regression_uses_fixed_legacy_sigcrf_oracle() -> None:
    source = TESTSUITE.read_text()
    root = ET.fromstring(source)
    cases = [
        testcase
        for testcase in root.findall(".//testcase")
        if testcase.get("directory") == "qsgw_aims_Si_k333_headonly_libri"
    ]
    assert len(cases) == 1
    labels = cases[0].find("labels")
    assert labels is not None
    assert labels.get("refhash") == "e08f4a130df7661e9ac355b9be45fb2bf9c3ed01"
    note = labels.get("note") or ""
    assert "reads identical SigcRF files produced by the legacy implementation" in note
    assert "legacy band output" in note
    assert "full-compute Sigma are explicitly not oracles" in note

    settings = (
        QSGW_HEAD_ONLY_CASE / "librpa" / "librpa.in"
    ).read_text()
    assert "task = qsgw" in settings
    assert "restart_from_dir = ../dataset/sigcrf/" in settings
    assert "read_sigc_mat_rf = true" in settings
    assert "qsgw_mixer = none" in settings
    assert "qsgw_min_iter = 1" in settings
    assert "qsgw_max_iter = 1" in settings
    assert "qsgw_band0_cut_mode = 0" in settings
    assert "qsgw_write_iteration_matrices = true" in settings
    assert "qsgw_update_hartree = false" in settings
    assert "replace_w_head = true" in settings
    assert "option_dielect_func = 4" in settings
    assert "qsgw_iterative_headwing" not in settings


def test_qsgw_head_only_dataset_contains_fixed_legacy_sigcrf() -> None:
    archive = QSGW_HEAD_ONLY_CASE / "dataset.tar.gz"
    assert archive.is_file()
    assert sha256_file(archive) == (
        "bc882cd32d703d6e840815c7e87ca33dfa2fc044fc1e8e4eadee7a5187e75ad3"
    )

    with tarfile.open(archive, "r:gz") as dataset:
        names = set(dataset.getnames())
    expected_sigcrf = {
        f"dataset/sigcrf/SigcRF_ispin_00_s_00_iomega_{index:03d}_myid_00000.dat"
        for index in range(6)
    }
    assert expected_sigcrf <= names
    assert "dataset/oracle/legacy-current-fixed-sigcrf-comparison.json" in names
    assert "dataset/oracle/ADAPTER_GATE_PROVENANCE.txt" in names
    assert "dataset/REGRESSION_DATASET_PROVENANCE.txt" in names


def test_qsgw_head_only_reference_is_bound_to_fixed_sigcrf_gate() -> None:
    provenance = json.loads(
        (QSGW_HEAD_ONLY_REF / "REFERENCE_PROVENANCE.json").read_text()
    )
    assert provenance["legacy_commit"] == (
        "e08f4a130df7661e9ac355b9be45fb2bf9c3ed01"
    )
    assert provenance["oracle_scope"] == (
        "qsgw_adapter_from_identical_legacy_sigcrf"
    )
    assert provenance["adapter_gate"]["passed"] is True
    assert provenance["adapter_gate"]["report_sha256"] == (
        "3bef75e6ede886c62a0a7a51f0f6c9c747d811396f8f67d952f0eb204b96825d"
    )
    assert provenance["contract"]["head"] == "on"
    assert provenance["contract"]["wing"] == "off"
    assert provenance["contract"]["hartree"] == "off"

    for relative_path, expected in provenance["references"].items():
        assert sha256_file(QSGW_HEAD_ONLY_REF / relative_path) == expected
