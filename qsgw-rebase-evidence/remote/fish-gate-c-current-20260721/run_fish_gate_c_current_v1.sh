#!/usr/bin/env bash
set -euo pipefail

: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must identify the clean candidate}"
: "${CANDIDATE_GATE0_ROOT:?CANDIDATE_GATE0_ROOT must identify an accepted Gate 0}"
: "${CANDIDATE_GATE0_PROVENANCE_SHA256:?Gate 0 provenance hash is required}"
: "${CANDIDATE_EXE_SHA256:?Candidate executable hash is required}"
: "${HARTREE_BUNDLE_ROOT:?HARTREE_BUNDLE_ROOT must identify the immutable bundle}"
: "${HARTREE_BUNDLE_PROVENANCE_SHA256:?Hartree bundle provenance hash is required}"
: "${HARTREE_BUNDLE_OUTPUT_SHA256:?Hartree bundle output-manifest hash is required}"
: "${HARTREE_DATASET_MANIFEST_SHA256:?Hartree dataset manifest hash is required}"
: "${GRID_BASE_CONTRACT_SHA256:?Grid Hartree-off contract hash is required}"
: "${GRID_HARTREE_FULL_CONTRACT_SHA256:?Grid Hartree contract hash is required}"
: "${BAND_HARTREE_FULL_CONTRACT_SHA256:?Band Hartree contract hash is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"

require_sha() {
  local value=$1
  local length=$2
  local label=$3
  [[ "$value" =~ ^[0-9a-f]+$ ]]
  test "${#value}" -eq "$length" || {
    echo "$label has the wrong length" >&2
    exit 2
  }
}

require_sha "$CANDIDATE_COMMIT" 40 CANDIDATE_COMMIT
for specification in \
  "CANDIDATE_GATE0_PROVENANCE_SHA256:$CANDIDATE_GATE0_PROVENANCE_SHA256" \
  "CANDIDATE_EXE_SHA256:$CANDIDATE_EXE_SHA256" \
  "HARTREE_BUNDLE_PROVENANCE_SHA256:$HARTREE_BUNDLE_PROVENANCE_SHA256" \
  "HARTREE_BUNDLE_OUTPUT_SHA256:$HARTREE_BUNDLE_OUTPUT_SHA256" \
  "HARTREE_DATASET_MANIFEST_SHA256:$HARTREE_DATASET_MANIFEST_SHA256" \
  "GRID_BASE_CONTRACT_SHA256:$GRID_BASE_CONTRACT_SHA256" \
  "GRID_HARTREE_FULL_CONTRACT_SHA256:$GRID_HARTREE_FULL_CONTRACT_SHA256" \
  "BAND_HARTREE_FULL_CONTRACT_SHA256:$BAND_HARTREE_FULL_CONTRACT_SHA256" \
  "RUNNER_SHA256:$RUNNER_SHA256"; do
  require_sha "${specification#*:}" 64 "${specification%%:*}"
done
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-c-${RUN_TAG}
dataset=$HARTREE_BUNDLE_ROOT/dataset
dataset_input_dir=$dataset/
mpi_ranks=1
omp_threads=32
python_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
python=$python_env/bin/python

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

provenance_value() {
  local key=$1
  local file=$2
  awk -F= -v key="$key" '
    $1 == key {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count != 1) exit 2
      print value
    }
  ' "$file"
}

test ! -e "$run_root"
test -x "$python"
test -e "$CANDIDATE_GATE0_ROOT/GREEN_CONFIRMED"
test ! -e "$CANDIDATE_GATE0_ROOT/FAILED"
test "$(sha256sum "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$CANDIDATE_GATE0_PROVENANCE_SHA256"
(
  cd "$CANDIDATE_GATE0_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(provenance_value candidate_commit "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_COMMIT"
candidate_source=$(provenance_value candidate_source "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_build=$(provenance_value candidate_build "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
test "$(provenance_value candidate_executable_sha256 "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_EXE_SHA256"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$CANDIDATE_EXE_SHA256"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$candidate_source" status --porcelain)"

gate_c_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-c-current-20260721
gate_d_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721
historical_source=$candidate_source/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c
runner_source=$gate_c_source/run_fish_gate_c_current_v1.sh
dump_validator_source=$gate_c_source/validate_qsgw_hartree_dump_v2.py
dump_validator_test_source=$gate_c_source/test_validate_qsgw_hartree_dump_v2.py
contraction_validator_source=$gate_c_source/validate_qsgw_hartree_contraction_v1.py
contraction_validator_test_source=$gate_c_source/test_validate_qsgw_hartree_contraction_v1.py
trace_validator_source=$gate_c_source/validate_qsgw_hartree_trace_v6.py
trace_validator_test_source=$gate_c_source/test_validate_qsgw_hartree_trace_v6.py
grid_comparator_source=$gate_c_source/compare_qsgw_grid_channels_v1.py
grid_comparator_test_source=$gate_c_source/test_compare_qsgw_grid_channels_v1.py
null_comparator_source=$gate_c_source/compare_qsgw_hartree_null_delta_v1.py
null_comparator_test_source=$gate_c_source/test_compare_qsgw_hartree_null_delta_v1.py
band_validator_source=$gate_d_source/validate_qsgw_band_v6.py
band_validator_test_source=$gate_d_source/test_validate_qsgw_band_v6.py
recompute_source=$historical_source/recompute_qsgw_hartree_delta_v1.py
contract_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py
for path in "$runner_source" "$dump_validator_source" \
  "$dump_validator_test_source" "$contraction_validator_source" \
  "$contraction_validator_test_source" "$trace_validator_source" \
  "$trace_validator_test_source" "$grid_comparator_source" \
  "$grid_comparator_test_source" "$null_comparator_source" \
  "$null_comparator_test_source" "$band_validator_source" \
  "$band_validator_test_source" "$recompute_source" \
  "$contract_parser_source"; do
  test -f "$path"
done
test "$(sha256sum "$runner_source" | awk '{print $1}')" = "$RUNNER_SHA256"

test -e "$HARTREE_BUNDLE_ROOT/COMPLETE"
test ! -e "$HARTREE_BUNDLE_ROOT/FAILED"
test -z "$(find "$HARTREE_BUNDLE_ROOT" -type l -print -quit)"
test "$(sha256sum "$HARTREE_BUNDLE_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$HARTREE_BUNDLE_PROVENANCE_SHA256"
test "$(sha256sum "$HARTREE_BUNDLE_ROOT/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$HARTREE_BUNDLE_OUTPUT_SHA256"
test "$(sha256sum "$HARTREE_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$HARTREE_DATASET_MANIFEST_SHA256"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$GRID_BASE_CONTRACT_SHA256"
test "$(sha256sum "$dataset/qsgw_input.hartree-full.contract" | awk '{print $1}')" = \
  "$GRID_HARTREE_FULL_CONTRACT_SHA256"
test "$(sha256sum "$dataset/qsgw_band_input.hartree-full.contract" | awk '{print $1}')" = \
  "$BAND_HARTREE_FULL_CONTRACT_SHA256"
(
  cd "$HARTREE_BUNDLE_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
for expected in \
  'gate=pinned_abacus_si_k444_symmetry_band_hartree_bundle_v1' \
  'acceptance_scope=contract_extension_only_no_qsgw_numerical_acceptance' \
  'producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a' \
  'grid=4x4x4' \
  'scf_kpoints=8' \
  'full_bz_kpoints=64' \
  'band_kpoints=143' \
  'n_bands=44' \
  'n_basis=44' \
  'symmetry=on_grid' \
  'headwing=off_fail_fast' \
  'hartree_contracts=full_and_truncated' \
  'hartree_normalizations_runtime_selectable=weighted_occupations,legacy_extra_inverse_nk' \
  'band_update=operator_fourier' \
  "dataset_manifest_sha256=$HARTREE_DATASET_MANIFEST_SHA256" \
  "qsgw_grid_hartree_full_contract_sha256=$GRID_HARTREE_FULL_CONTRACT_SHA256" \
  "qsgw_band_hartree_full_contract_sha256=$BAND_HARTREE_FULL_CONTRACT_SHA256"; do
  grep -Fqx "$expected" "$HARTREE_BUNDLE_ROOT/PROVENANCE.txt"
done
grep -Fqx 'hartree_update delta_density' \
  "$dataset/qsgw_input.hartree-full.contract"
grep -Fqx 'hartree_update delta_density' \
  "$dataset/qsgw_band_input.hartree-full.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.hartree-full.contract"
grep -Fqx 'band_update operator_fourier' \
  "$dataset/qsgw_band_input.hartree-full.contract"

mkdir -p "$run_root/tools" "$run_root/unit-dump" "$run_root/unit-trace" \
  "$run_root/unit-contraction" \
  "$run_root/unit-grid-compare" "$run_root/unit-null-compare" \
  "$run_root/unit-band"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$HARTREE_BUNDLE_ROOT/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$HARTREE_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" \
  "$run_root/bundle-DATASET_SHA256SUMS.txt"
cp "$dump_validator_source" "$run_root/tools/"
cp "$contraction_validator_source" "$run_root/tools/"
cp "$trace_validator_source" "$run_root/tools/"
cp "$grid_comparator_source" "$run_root/tools/"
cp "$null_comparator_source" "$run_root/tools/"
cp "$band_validator_source" "$run_root/tools/"
cp "$recompute_source" "$run_root/tools/"
cp "$contract_parser_source" "$run_root/tools/"

PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_HARTREE_OBSERVER_TEST_TMP="$run_root/unit-dump" \
  "$python" -B "$dump_validator_test_source" \
  >"$run_root/dump-validator-unit-test.stdout" \
  2>"$run_root/dump-validator-unit-test.stderr"
PYTHONPATH="$run_root/tools" \
LIBRPA_QSGW_HARTREE_CONTRACTION_TEST_TMP="$run_root/unit-contraction" \
  "$python" -B "$contraction_validator_test_source" \
  >"$run_root/contraction-validator-unit-test.stdout" \
  2>"$run_root/contraction-validator-unit-test.stderr"
PYTHONPATH="$run_root/tools:$gate_c_source" \
LIBRPA_QSGW_HARTREE_TRACE_TEST_TMP="$run_root/unit-trace" \
  "$python" -B "$trace_validator_test_source" \
  >"$run_root/trace-validator-unit-test.stdout" \
  2>"$run_root/trace-validator-unit-test.stderr"
PYTHONPATH="$run_root/tools:$gate_c_source" \
LIBRPA_QSGW_GRID_COMPARE_TEST_TMP="$run_root/unit-grid-compare" \
  "$python" -B "$grid_comparator_test_source" \
  >"$run_root/grid-comparator-unit-test.stdout" \
  2>"$run_root/grid-comparator-unit-test.stderr"
PYTHONPATH="$run_root/tools:$gate_c_source" \
LIBRPA_QSGW_HARTREE_NULL_TEST_TMP="$run_root/unit-null-compare" \
  "$python" -B "$null_comparator_test_source" \
  >"$run_root/null-comparator-unit-test.stdout" \
  2>"$run_root/null-comparator-unit-test.stderr"
PYTHONPATH="$run_root/tools" LIBRPA_QSGW_TEST_TMP="$run_root/unit-band" \
  "$python" -B "$band_validator_test_source" \
  >"$run_root/band-validator-unit-test.stdout" \
  2>"$run_root/band-validator-unit-test.stderr"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

write_librpa_input() {
  local mode_root=$1
  local task=$2
  local contract=$3
  local iterations=$4
  local update_hartree=$5
  local export_pyatb=false
  if [[ "$task" == qsgw_band ]]; then
    export_pyatb=true
  fi
  cat >"$mode_root/librpa.in" <<EOF
task = $task
input_dir = $dataset_input_dir
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = false
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = $contract
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
qsgw_min_iter = $iterations
qsgw_max_iter = $iterations
qsgw_write_iteration_matrices = true
qsgw_update_hartree = $update_hartree
qsgw_export_hamiltonian_for_pyatb = $export_pyatb
EOF
  if [[ "$update_hartree" == true ]]; then
    cat >>"$mode_root/librpa.in" <<'EOF'
qsgw_hartree_coulomb = full
qsgw_hartree_normalization = weighted_occupations
EOF
  fi
  if [[ "$task" == qsgw_band ]]; then
    cat >>"$mode_root/librpa.in" <<'EOF'
qsgw_band0_unoccupied_keep = 10
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
EOF
  fi
}

run_case() {
  local name=$1
  local task=$2
  local contract=$3
  local contract_sha=$4
  local mode_root=$run_root/$name
  local dump_root=$mode_root/hartree-dump
  mkdir -p "$mode_root" "$dump_root"
  write_librpa_input "$mode_root" "$task" "$contract" 2 true
  cat >"$mode_root/PARAMETERS.txt" <<EOF
task=$task
iterations=0:2
mixer=linear
mixing_beta=0.2
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=delta_density
hartree_coulomb=full
hartree_normalization=weighted_occupations
contract=$contract
contract_sha256=$contract_sha
dataset_manifest_sha256=$HARTREE_DATASET_MANIFEST_SHA256
EOF
  (
    cd "$mode_root"
    export LIBRPA_QSGW_HARTREE_DUMP_DIR="$dump_root"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$mode_root/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$mode_root/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$mode_root/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$mode_root/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$mode_root/$trace"
    grep -Fqx '# headwing disabled_stage1' "$mode_root/$trace"
    grep -Fqx '# hartree delta_density' "$mode_root/$trace"
    grep -Fqx '# hartree_coulomb full' "$mode_root/$trace"
    grep -Fqx '# hartree_normalization weighted_occupations' "$mode_root/$trace"
    grep -Fqx '# qsgw_mixer linear' "$mode_root/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $contract_sha" "$mode_root/$trace"
  done
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$mode_root/qsgw_iterations.dat")" = 2
  test "$(find "$dump_root" -mindepth 1 -maxdepth 1 -type d -name 'call_*' | wc -l)" -eq 2
  test -s "$dump_root/call_001/full_kpoints.txt"
  test -s "$dump_root/call_001/translations.txt"
  test -s "$dump_root/call_001/bvk_remap.txt"
  test -s "$dump_root/call_002/full_kpoints.txt"
  test -s "$dump_root/call_002/translations.txt"
  test -s "$dump_root/call_002/bvk_remap.txt"
  grep -Fq 'libRPA finished successfully' "$mode_root/librpa.stdout"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_hartree_dump_v2.py" \
    "$dump_root" "$mode_root/qsgw_matrices.dat" \
    "$mode_root/qsgw_eigenvalues.dat" "$mode_root/hartree-dump-validation.json" \
    --expected-iterations 2 \
    >"$mode_root/hartree-dump-validation.stdout" \
    2>"$mode_root/hartree-dump-validation.stderr"
  grep -Fq '"passed": true' "$mode_root/hartree-dump-validation.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_hartree_contraction_v1.py" \
    --input-dir "$dataset" \
    --dump-call "$dump_root/call_002" \
    --expected-normalization weighted_occupations \
    --coulomb-prefix coulomb_mat_ \
    --output "$mode_root/hartree-contraction-validation.json" \
    >"$mode_root/hartree-contraction-validation.stdout" \
    2>"$mode_root/hartree-contraction-validation.stderr"
  grep -Fq '"passed": true' \
    "$mode_root/hartree-contraction-validation.json"
}

run_hartree_off_control() {
  local mode_root=$run_root/grid-hartree-off-control
  mkdir -p "$mode_root"
  write_librpa_input "$mode_root" qsgw qsgw_input.contract 1 false
  cat >"$mode_root/PARAMETERS.txt" <<EOF
task=qsgw
iterations=0:1
mixer=linear
mixing_beta=0.2
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=disabled_stage1
contract=qsgw_input.contract
contract_sha256=$GRID_BASE_CONTRACT_SHA256
dataset_manifest_sha256=$HARTREE_DATASET_MANIFEST_SHA256
EOF
  (
    cd "$mode_root"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$mode_root/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$mode_root/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$mode_root/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$mode_root/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$mode_root/$trace"
    grep -Fqx '# headwing disabled_stage1' "$mode_root/$trace"
    grep -Fqx '# hartree disabled_stage1' "$mode_root/$trace"
    ! grep -Fq '# hartree_coulomb ' "$mode_root/$trace"
    ! grep -Fq '# hartree_normalization ' "$mode_root/$trace"
    grep -Fqx '# qsgw_mixer linear' "$mode_root/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $GRID_BASE_CONTRACT_SHA256" \
      "$mode_root/$trace"
  done
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$mode_root/qsgw_iterations.dat")" = 1
  grep -Fq 'libRPA finished successfully' "$mode_root/librpa.stdout"
}

run_hartree_off_control
run_case grid qsgw qsgw_input.hartree-full.contract \
  "$GRID_HARTREE_FULL_CONTRACT_SHA256"
PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_hartree_trace_v6.py" \
  "$run_root/grid/qsgw_matrices.dat" "$run_root/grid/qsgw_iterations.dat" \
  "$run_root/grid/hartree-trace-validation.json" --expected-iterations 2 \
  >"$run_root/grid/hartree-trace-validation.stdout" \
  2>"$run_root/grid/hartree-trace-validation.stderr"
grep -Fq '"passed": true' "$run_root/grid/hartree-trace-validation.json"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/compare_qsgw_hartree_null_delta_v1.py" \
  "$run_root/grid" "$run_root/grid-hartree-off-control" \
  "$run_root/hartree-null-delta-comparison.json" \
  >"$run_root/hartree-null-delta-comparison.stdout" \
  2>"$run_root/hartree-null-delta-comparison.stderr"
grep -Fq '"passed": true' "$run_root/hartree-null-delta-comparison.json"

run_case band qsgw_band qsgw_band_input.hartree-full.contract \
  "$BAND_HARTREE_FULL_CONTRACT_SHA256"
PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/validate_qsgw_band_v6.py" \
  "$run_root/band/qsgw_matrices.dat" \
  "$run_root/band/qsgw_eigenvalues.dat" \
  "$run_root/band/qsgw_iterations.dat" "$run_root/band" \
  "$dataset/bz_sampling_out" "$run_root/band/band-validation.json" \
  --expected-iterations 2 --expected-cut-mode 0 \
  --expected-unoccupied-keep 10 --expected-shift-ha 20.0 \
  >"$run_root/band/band-validation.stdout" \
  2>"$run_root/band/band-validation.stderr"
grep -Fq '"passed": true' "$run_root/band/band-validation.json"

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/compare_qsgw_grid_channels_v1.py" \
  "$run_root/grid" "$run_root/band" "$run_root/grid-channel-comparison.json" \
  >"$run_root/grid-channel-comparison.stdout" \
  2>"$run_root/grid-channel-comparison.stderr"
grep -Fq '"passed": true' "$run_root/grid-channel-comparison.json"

cat >"$run_root/ACCEPTANCE.txt" <<EOF
accepted=true
scope=current_qsgw_hartree_structural_and_metamorphic_gate
legacy_same_dataset_acceptance=false
tasks=qsgw,qsgw_band
iterations=0:2
hartree_off_control_iterations=0:1
iteration1_null_delta_side_effect_guard=true
independent_hartree_contraction_tasks=qsgw,qsgw_band
symmetry=on
headwing=off
hartree_coulomb=full
hartree_normalization=weighted_occupations
mixer=linear
mixing_beta=0.2
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_frobenius_tolerance=1e-8
hermiticity_tolerance_ha=1e-10
electron_count_tolerance=1e-10
iteration1_zero_tolerance_ha=1e-10
iteration2_nonzero_tolerance_ha=1e-12
grid_vs_band_eigenvalue_tolerance_ha=1e-6
grid_vs_band_gap_tolerance_ev=1e-5
EOF

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate_c_current_v1
acceptance=true
acceptance_scope=current_hartree_structural_and_metamorphic_not_legacy_numerical
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
candidate_commit=$CANDIDATE_COMMIT
candidate_gate0_root=$CANDIDATE_GATE0_ROOT
candidate_gate0_provenance_sha256=$CANDIDATE_GATE0_PROVENANCE_SHA256
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
hartree_bundle_root=$HARTREE_BUNDLE_ROOT
hartree_bundle_provenance_sha256=$HARTREE_BUNDLE_PROVENANCE_SHA256
hartree_bundle_output_manifest_sha256=$HARTREE_BUNDLE_OUTPUT_SHA256
dataset_manifest_sha256=$HARTREE_DATASET_MANIFEST_SHA256
grid_base_contract_sha256=$GRID_BASE_CONTRACT_SHA256
grid_hartree_full_contract_sha256=$GRID_HARTREE_FULL_CONTRACT_SHA256
band_hartree_full_contract_sha256=$BAND_HARTREE_FULL_CONTRACT_SHA256
producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
tasks=qsgw,qsgw_band
iterations=0:2
hartree_off_control_iterations=0:1
iteration1_null_delta_side_effect_guard=true
independent_hartree_contraction_tasks=qsgw,qsgw_band
symmetry=on
headwing=off
hartree=delta_density
hartree_coulomb=full
hartree_normalization=weighted_occupations
mixer=linear
mixing_beta=0.2
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
trap - ERR
echo "FISH_GATE_C_CURRENT_V1=PASS"
echo "run_root=$run_root"
