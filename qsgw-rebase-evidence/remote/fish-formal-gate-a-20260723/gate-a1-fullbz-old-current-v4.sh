#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "ERROR line=%s status=%s command=%s\n" \
  "$LINENO" "$status" "$BASH_COMMAND" >&2' ERR

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify this committed runner}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must be a clean checkout at RUNNER_COMMIT}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${GATE0_PROVENANCE_SHA256:?GATE0_PROVENANCE_SHA256 must bind accepted Gate 0}"
: "${GATE1_PROVENANCE_SHA256:?GATE1_PROVENANCE_SHA256 must bind accepted Gate 1}"
: "${GATE2_PROVENANCE_SHA256:?GATE2_PROVENANCE_SHA256 must bind accepted Gate 2}"
: "${CANDIDATE_EXE_SHA256:?CANDIDATE_EXE_SHA256 must bind the Gate 0 executable}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"
test "${#RUNNER_COMMIT}" = 40
test "${#RUNNER_SHA256}" = 64
test "${#GATE0_PROVENANCE_SHA256}" = 64
test "${#GATE1_PROVENANCE_SHA256}" = 64
test "${#GATE2_PROVENANCE_SHA256}" = 64
test "${#CANDIDATE_EXE_SHA256}" = 64
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

root=/tmp/librpa-qsgw-formal-gate-a-20260723
pair_root=/home/bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-20260723-v2
common_root=$pair_root/fullbz
source_input_dir=$common_root/dataset
old_runtime=$root/legacy
old_build=$old_runtime/old/build
old_exe=$old_build/chi0_main.exe
candidate_gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260723-4f9ab0cf-v1
candidate_gate1=/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-20260723-36d74369-v1
candidate_gate2=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260723-dd7a75f2-v1
candidate_source=/tmp/librpa-qsgw-gate0-20260723-4f9ab0cf-v1/candidate
candidate_build=/tmp/librpa-qsgw-gate0-20260723-4f9ab0cf-v1/build-candidate
candidate_exe=$candidate_build/chi0_main.exe
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python
run_root=/home/bhj/ai-runs/librpa-qsgw-formal-gate-a-$RUN_TAG
overlay_root=$run_root/common-input-overlay
input_dir=$overlay_root/dataset
tool_dir=$run_root/tools
mpi_ranks=4
omp_threads=12

expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_old_build_provenance_sha=8dfa0742e5618be14c27a96764ddf240af2ed8fc19662bac4d506bd44e77c96b
expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff
expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e
expected_gate1_output_manifest_sha=74a3453c46df34e7b0ec8430260cd47182e6d3d01a409f4a313dc976948a127f
expected_gate2_output_manifest_sha=48dc9063d56de58b21cd77243ea56e127cfe79013b66728f6a9913ff36ac3dee
expected_pair_output_manifest_sha=159d433d29938f65c637a88b9f4fa3e7974c191267c79a0cd343168bc4b2ddb8
expected_pair_validation_sha=bd6a8c65732518136d5679ceaf29b6454277f832aaf2ac56bce6860375a487da
expected_common_dataset_manifest_sha=ba401b751eb465b957fad2270878022c13dfb922257376b9b2a9faef793873f4
expected_common_output_manifest_sha=9f96b50c098c572d603d649137e411737bfaa6b1ff7e48200cd25e94da3a6a3f
expected_common_provenance_sha=bd5eb751832cfe0919393417e8d93d86fd8a5121cd6d33c152dff9db6927ec8a
expected_common_contract_sha=5c0477af4bc6e4b23c2be5f286c979e44f039ef6b65295c3b577a77ecc220c9e
expected_overlay_dataset_manifest_sha=407b47fa040ff7534d04d08f73ae99ec7d33f673cab8b25a90bae1c160e22d98
expected_overlay_report_sha=0b897345389eb55f45abc392a471bcf68522024d6702660a5c13a0d380b3416d
expected_overlay_stru_sha=2e5c4dff35da70e4fe189b150a8b2528af44ed6b4cc0b6b106e169c4b83d86a4
expected_overlay_builder_sha=bfa451f47201a4bcc04efc91ed1cebf45257b493c9644a7635786a714b4f2a0f
expected_overlay_builder_test_sha=6d0377511b3da42d9924e2ee43fd60d060fcb46eba8c5373eb733b45d6c4a67e
expected_adapter_sha=c63523fa95bbfb3f83e60183cc39e67b057589f7dbd6896d5c96a486aa9016b3
expected_adapter_test_sha=e9f4db574b8a724d4e4beedf4d593a1a27c38c87c6b7f070a1849d9401520f87
expected_base_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_closure_adapter_sha=900f42f1917e4c2a80962e78b80a7226411981fc21923509983e55fa9afbdbfa
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
expected_current_parser_sha=f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb
runner_relative=qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/gate-a1-fullbz-old-current-v4.sh

run_succeeded=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

test ! -e "$run_root"
test -d "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(git -C "$RUNNER_SOURCE" show "$RUNNER_COMMIT:$runner_relative" | \
  sha256sum | awk '{print $1}')" = "$RUNNER_SHA256"
test -e "$pair_root/PAIR_COMPLETE"
test ! -e "$pair_root/FAILED"
test -e "$common_root/COMPLETE"
test -e "$candidate_gate0/GREEN_CONFIRMED"
test ! -e "$candidate_gate0/FAILED"
test -x "$old_exe"
test -x "$candidate_exe"
test -x "$python"
mkdir -p "$tool_dir"
cp "$0" "$run_root/gate-a1-fullbz-old-current-v4.sh"
test "$(sha256sum "$run_root/gate-a1-fullbz-old-current-v4.sh" | awk '{print $1}')" = \
  "$RUNNER_SHA256"

printf 'preflight=legacy_provenance\n'
test "$(sha256sum "$old_exe" | awk '{print $1}')" = \
  "$expected_old_exe_sha"
test "$(sha256sum "$old_runtime/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_old_build_provenance_sha"
grep -Fqx "old_oracle_commit=$expected_old_commit" \
  "$old_runtime/build-provenance.txt"

printf 'preflight=candidate_gate0_provenance\n'
test "$(sha256sum "$candidate_gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$GATE0_PROVENANCE_SHA256"
grep -Fqx 'gate=fish_gate0_current_v2' "$candidate_gate0/PROVENANCE.txt"
grep -Fqx 'acceptance=true' "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "upstream_commit=$expected_upstream_commit" "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_candidate_commit" "$candidate_gate0/PROVENANCE.txt"
grep -Fqx 'protected_diff=empty' "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "candidate_executable=$candidate_exe" "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "candidate_executable_sha256=$CANDIDATE_EXE_SHA256" \
  "$candidate_gate0/PROVENANCE.txt"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$CANDIDATE_EXE_SHA256"
(
  cd "$candidate_gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$expected_candidate_commit"
test -z "$(git -C "$candidate_source" status --porcelain)"

printf 'preflight=candidate_gate1_provenance\n'
test -e "$candidate_gate1/GREEN_CONFIRMED"
test ! -e "$candidate_gate1/FAILED"
test "$(sha256sum "$candidate_gate1/PROVENANCE.txt" | awk '{print $1}')" = \
  "$GATE1_PROVENANCE_SHA256"
grep -Fqx 'gate=fish_gate1_current_g0w0_ab_recovery_v2' "$candidate_gate1/PROVENANCE.txt"
grep -Fqx 'acceptance=true' "$candidate_gate1/PROVENANCE.txt"
grep -Fqx "upstream_commit=$expected_upstream_commit" "$candidate_gate1/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_candidate_commit" "$candidate_gate1/PROVENANCE.txt"
test "$(sha256sum "$candidate_gate1/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate1_output_manifest_sha"
(
  cd "$candidate_gate1"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

printf 'preflight=candidate_gate2_provenance\n'
test -e "$candidate_gate2/GREEN_CONFIRMED"
test ! -e "$candidate_gate2/FAILED"
test "$(sha256sum "$candidate_gate2/PROVENANCE.txt" | awk '{print $1}')" = \
  "$GATE2_PROVENANCE_SHA256"
test "$(sha256sum "$candidate_gate2/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate2_output_manifest_sha"
grep -Fqx 'gate=fish_gate2_current_qsgw_first_self_energy_v2' \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'acceptance=true' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "upstream_commit=$expected_upstream_commit" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_candidate_commit" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "candidate_executable_sha256=$CANDIDATE_EXE_SHA256" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "gate0_provenance_sha256=$GATE0_PROVENANCE_SHA256" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "gate1_accepted_provenance_sha256=$GATE1_PROVENANCE_SHA256" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx "gate1_accepted_output_manifest_sha256=$expected_gate1_output_manifest_sha" \
  "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'sigc_block_count=48' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'symmetry=exx_on_gw_on_rpa_on' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'headwing=off' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'band=off' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'mixing=none' "$candidate_gate2/PROVENANCE.txt"
grep -Fqx 'iterations=0:1' "$candidate_gate2/PROVENANCE.txt"
(
  cd "$candidate_gate2"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

printf 'preflight=common_input_bundle\n'
test "$(sha256sum "$pair_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_pair_output_manifest_sha"
test "$(sha256sum "$pair_root/PAIR_VALIDATION.json" | awk '{print $1}')" = \
  "$expected_pair_validation_sha"
grep -Fq '"status": "PASS"' "$pair_root/PAIR_VALIDATION.json"
test -z "$(find "$pair_root" -type l -print -quit)"
test -z "$(find "$pair_root" -perm /222 -print -quit)"
(
  cd "$pair_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(sha256sum "$common_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_dataset_manifest_sha"
test "$(sha256sum "$common_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_output_manifest_sha"
test "$(sha256sum "$common_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_common_provenance_sha"
test "$(sha256sum "$source_input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_common_contract_sha"
(
  cd "$common_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
sed 's#  \./#  '"$common_root"'/#g' \
  "$common_root/OUTPUT_SHA256SUMS.txt" \
  >"$run_root/common-input-OUTPUT_SHA256SUMS.relocated.txt"
sha256sum --check --quiet "$run_root/common-input-OUTPUT_SHA256SUMS.relocated.txt"
grep -Fqx 'n_scf_kpoints 64' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'n_band_kpoints 0' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$source_input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$source_input_dir/qsgw_input.contract"

printf 'preflight=observer_tools\n'
adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_legacy_v4_current_v6.py
adapter_test_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_compare_qsgw_legacy_v4_current_v6.py
base_comparator_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
closure_adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_component_traces_v6_adapter.py
observer_root=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/observer-tools-v1
closure_source=$observer_root/validate_qsgw_trace_closure.py
fixed_source=$observer_root/validate_qsgw_fixed_basis.py
initial_source=$observer_root/validate_qsgw_initial_state.py
closure_test_source=$observer_root/test_validate_qsgw_trace_closure-v3-38de02fa.py
initial_test_source=$observer_root/test_validate_qsgw_initial_state-v1.py
current_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py
overlay_builder_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/build_legacy_fullbz_stru_overlay_v1.py
overlay_builder_test_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/test_build_legacy_fullbz_stru_overlay_v1.py

while read -r path expected
do
  actual=$(sha256sum "$path" | awk '{print $1}')
  printf 'observer_sha256=%s actual=%s expected=%s\n' \
    "$path" "$actual" "$expected"
  test "$actual" = "$expected"
done <<EOF
$adapter_source $expected_adapter_sha
$adapter_test_source $expected_adapter_test_sha
$base_comparator_source $expected_base_comparator_sha
$closure_adapter_source $expected_closure_adapter_sha
$closure_source $expected_closure_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$closure_test_source $expected_closure_test_sha
$initial_test_source $expected_initial_test_sha
$current_parser_source $expected_current_parser_sha
$overlay_builder_source $expected_overlay_builder_sha
$overlay_builder_test_source $expected_overlay_builder_test_sha
EOF

cp "$adapter_source" "$tool_dir/compare_qsgw_legacy_v4_current_v6.py"
cp "$base_comparator_source" "$tool_dir/base_comparator.py"
cp "$base_comparator_source" "$tool_dir/compare_qsgw_component_traces_v4.py"
cp "$closure_adapter_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$current_parser_source" "$tool_dir/cmp_qsgw_v6.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"
cp "$overlay_builder_source" "$tool_dir/build_legacy_fullbz_stru_overlay_v1.py"
cp "$overlay_builder_test_source" "$tool_dir/test_build_legacy_fullbz_stru_overlay_v1.py"

cp "$old_runtime/build-provenance.txt" "$run_root/old-build-provenance.txt"
cp "$candidate_gate0/PROVENANCE.txt" "$run_root/candidate-gate0-PROVENANCE.txt"
cp "$candidate_gate0/OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate0-OUTPUT_SHA256SUMS.txt"
cp "$candidate_gate1/PROVENANCE.txt" "$run_root/candidate-gate1-PROVENANCE.txt"
cp "$candidate_gate1/OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate1-OUTPUT_SHA256SUMS.txt"
cp "$candidate_gate2/PROVENANCE.txt" "$run_root/candidate-gate2-PROVENANCE.txt"
cp "$candidate_gate2/OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate2-OUTPUT_SHA256SUMS.txt"
cp "$candidate_gate2/qsgw-iter1-invariants.json" \
  "$run_root/candidate-gate2-qsgw-iter1-invariants.json"
cp "$candidate_gate2/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  "$run_root/candidate-gate2-qsgw-iter1-vs-upstream-g0w0-sigc.json"
cp "$common_root/DATASET_SHA256SUMS.txt" "$run_root/DATASET_SHA256SUMS.txt"
cp "$common_root/OUTPUT_SHA256SUMS.txt" "$run_root/common-input-OUTPUT_SHA256SUMS.txt"
cp "$common_root/PROVENANCE.txt" "$run_root/common-input-PROVENANCE.txt"
cp "$source_input_dir/qsgw_input.contract" "$run_root/qsgw_input.contract"

printf 'preflight=observer_unit_tests\n'
PYTHONPATH="$tool_dir" "$python" -B "$adapter_test_source" \
  >"$run_root/adapter-unit-test.stdout" \
  2>"$run_root/adapter-unit-test.stderr"
(
  cd "$tool_dir"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_trace_closure.py \
    >"$run_root/closure-unit-test.stdout" \
    2>"$run_root/closure-unit-test.stderr"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_initial_state.py \
    >"$run_root/initial-unit-test.stdout" \
    2>"$run_root/initial-unit-test.stderr"
  PYTHONPATH="$tool_dir" "$python" -B test_build_legacy_fullbz_stru_overlay_v1.py \
    >"$run_root/overlay-builder-unit-test.stdout" \
    2>"$run_root/overlay-builder-unit-test.stderr"
)

printf 'preflight=legacy_fullbz_same_input_overlay\n'
"$python" -B "$tool_dir/build_legacy_fullbz_stru_overlay_v1.py" \
  "$source_input_dir" "$common_root/DATASET_SHA256SUMS.txt" "$input_dir" \
  >"$run_root/overlay-builder.stdout" \
  2>"$run_root/overlay-builder.stderr"
test "$(sha256sum "$overlay_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_overlay_dataset_manifest_sha"
test "$(sha256sum "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json" | awk '{print $1}')" = \
  "$expected_overlay_report_sha"
test "$(sha256sum "$input_dir/stru_out" | awk '{print $1}')" = \
  "$expected_overlay_stru_sha"
grep -Fq '"status": "PASS"' "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json"
grep -Fq '"n_kpoints": 64' "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json"
grep -Fq '"mapping": "identity"' "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json"
grep -Fq '"unchanged_files_hardlinked": 161' \
  "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json"
test "$(wc -l <"$input_dir/stru_out")" = 138
test "$(sed -n '10p' "$input_dir/stru_out")" = '4 4 4'
test "$(sed -n '75p' "$input_dir/stru_out")" = 1
test "$(sed -n '138p' "$input_dir/stru_out")" = 64
test "$input_dir/band_out" -ef "$source_input_dir/band_out"
test ! "$input_dir/stru_out" -ef "$source_input_dir/stru_out"
test -z "$(find "$overlay_root" -type l -print -quit)"
(
  cd "$overlay_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
(
  cd "$common_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
find "$overlay_root" -type f -exec chmod a-w {} +
find "$overlay_root" -depth -type d -exec chmod a-w {} +
test -z "$(find "$overlay_root" -perm /222 -print -quit)"

set +eu
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
oneapi_rc=$?
set -eu
test "$oneapi_rc" -eq 0
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/CONTROLLED_VARIABLES.txt" <<EOF
gate=A1
oracle=legacy_symmetry_off_full_bz
candidate=current_symmetry_off_full_bz
source_dataset=$source_input_dir
source_dataset_manifest_sha256=$expected_common_dataset_manifest_sha
dataset=$input_dir
dataset_manifest_sha256=$expected_overlay_dataset_manifest_sha
dataset_overlay_contract=legacy_fullbz_stru_overlay_v1
dataset_overlay_report_sha256=$expected_overlay_report_sha
dataset_overlay_stru_sha256=$expected_overlay_stru_sha
same_input_for_legacy_and_candidate=true
pair_root=$pair_root
pair_output_manifest_sha256=$expected_pair_output_manifest_sha
pair_validation_sha256=$expected_pair_validation_sha
input_contract_sha256=$expected_common_contract_sha
scf_kpoints=64
headwing=off
hartree=off
band=off
h_qsgw_cut=disabled_non_band
use_shrink_abfs=false
use_fullcoul_exx=false
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
deterministic_reduction=1
accepted_precondition_gate2=first_self_energy_vs_upstream_g0w0
mode_1=legacy_linear_beta_1_direct_vs_current_none_miniter2
mode_2=legacy_linear_beta_0.2_vs_current_linear_beta_0.2_miniter5
EOF

printf 'numerics=begin\n'
run_mode() {
  local mode_name=$1
  local target_iter=$2
  local legacy_beta=$3
  local candidate_mode=$4
  local candidate_beta=$5
  local mode_root=$run_root/$mode_name
  local legacy_run=$mode_root/legacy
  local candidate_run=$mode_root/current
  mkdir -p "$legacy_run" "$candidate_run"

  cat >"$legacy_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

  cat >"$candidate_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
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
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = $candidate_mode
qsgw_mixing_beta = $candidate_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

  (
    cd "$legacy_run"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE=0
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=$legacy_beta
    export LD_LIBRARY_PATH="$old_build/qsgw:$old_build/src:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    mpirun -np "$mpi_ranks" "$old_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )

  test -s "$legacy_run/qsgw_oracle_matrices.dat"
  test -s "$legacy_run/homo_lumo_vs_iterations.dat"
  grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# qsgw_mixer linear' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx "# qsgw_min_iter $target_iter" "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx "# qsgw_max_iter $target_iter" "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# qsgw_update_hartree 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_gw 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_exx 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# replace_w_head 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# option_dielect_func 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_shrink_abfs 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_exx 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_eps 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_wc 0' "$legacy_run/qsgw_oracle_matrices.dat"
  test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
    "$legacy_run/qsgw_oracle_matrices.dat")" = "$target_iter"

  (
    cd "$candidate_run"
    unset QSGW_ORACLE_TRACE
    unset QSGW_ORACLE_UPDATE_HARTREE
    unset QSGW_HROUND_SCALE
    unset LIBRPA_QSGW_MIXING_BETA
    export LD_LIBRARY_PATH="$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )

  test -s "$candidate_run/qsgw_matrices.dat"
  test -s "$candidate_run/qsgw_eigenvalues.dat"
  test -s "$candidate_run/qsgw_iterations.dat"
  grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat
  do
    grep -Fqx '# qsgw_contract_version 6' "$candidate_run/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$candidate_run/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$candidate_run/$trace"
    grep -Fqx '# symmetry exx_off_gw_off_rpa_off' "$candidate_run/$trace"
    grep -Fqx '# headwing disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# hartree disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# band disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# h_qsgw_cut disabled_non_band' "$candidate_run/$trace"
    grep -Fqx "# qsgw_mixer $candidate_mode" "$candidate_run/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $expected_common_contract_sha" \
      "$candidate_run/$trace"
  done
  test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
    "$candidate_run/qsgw_matrices.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/qsgw_eigenvalues.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/qsgw_iterations.dat")" = "$target_iter"

  "$python" -B "$tool_dir/compare_qsgw_legacy_v4_current_v6.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_eigenvalues.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$mode_root/legacy-current-comparison.json" \
    --base-comparator "$tool_dir/base_comparator.py" \
    --current-contract-parser "$tool_dir/cmp_qsgw_v6.py" \
    --iterations "0:$target_iter" \
    --expected-mode "$candidate_mode" \
    --expected-legacy-beta "$legacy_beta" \
    --expected-current-beta "$candidate_beta" \
    --expected-legacy-symmetry off \
    --expected-current-symmetry off \
    --expected-legacy-use-shrink-abfs 0 \
    --frequency-tolerance 1e-10 \
    --matrix-max-abs-tolerance-ha 1e-8 \
    --matrix-relative-tolerance 1e-8 \
    --eigenvalue-tolerance-ha 1e-6 \
    --gap-tolerance-ev 1e-5 \
    --degeneracy-tolerance-ha 1e-8 \
    --state-tolerance 1e-10 \
    --normalized-current-matrix "$mode_root/current-v5-self-matrices.dat" \
    --normalized-current-eigenvalues "$mode_root/current-v5-self-eigenvalues.dat" \
    --normalized-current-iterations "$mode_root/current-v5-self-iterations.dat" \
    >"$mode_root/legacy-current-comparison.stdout" \
    2>"$mode_root/legacy-current-comparison.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-current-comparison.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_trace_closure.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$mode_root/legacy-closure.json" \
    --iterations "0:$target_iter" --channel 0 --legacy-contract \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/legacy-closure.stdout" \
    2>"$mode_root/legacy-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-closure.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_trace_closure.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-closure.json" \
    --iterations "0:$target_iter" --channel 0 \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/current-closure.stdout" \
    2>"$mode_root/current-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/current-closure.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_fixed_basis.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-v5-self-eigenvalues.dat" \
    "$input_dir/band_out" "$mode_root/current-fixed-basis.json" \
    --iterations "0:$target_iter" --channel 0 \
    --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
    >"$mode_root/current-fixed-basis.stdout" \
    2>"$mode_root/current-fixed-basis.stderr"
  grep -Fq '"passed": true' "$mode_root/current-fixed-basis.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_initial_state.py" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$input_dir/band_out" "$mode_root/current-initial-state.json" \
    --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
    >"$mode_root/current-initial-state.stdout" \
    2>"$mode_root/current-initial-state.stderr"
  grep -Fq '"passed": true' "$mode_root/current-initial-state.json"

  awk -v tolerance=1e-10 '
    NF && $1 !~ /^#/ {
      value = $7 + 0.0
      if (!seen) { reference = value; max_abs = 0.0; seen = 1 }
      delta = value - reference
      if (delta < 0.0) delta = -delta
      if (delta > max_abs) max_abs = delta
      if (delta > tolerance) failed = 1
      count += 1
    }
    END {
      printf "reference_electron_count=%.17g\n", reference
      printf "max_abs_delta=%.17g\n", max_abs
      printf "tolerance=%.17g\n", tolerance
      printf "iteration_count=%d\n", count
      if (!seen || failed) exit 1
    }
  ' "$candidate_run/qsgw_iterations.dat" \
    >"$mode_root/current-electron-count.txt"

  cat >"$mode_root/ACCEPTANCE.txt" <<EOF
accepted=true
gate=A1
mode=$mode_name
iterations=0:$target_iter
legacy_symmetry=off_full_bz
current_symmetry=off_full_bz
legacy_current_matrix_relative_tolerance=1e-8
legacy_current_eigenvalue_tolerance_ha=1e-6
legacy_current_gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_tolerance=1e-10
electron_count_tolerance=1e-10
EOF
}

run_mode no-mix-miniter2 2 1 none 0.2
run_mode linear-beta-0.2-miniter5 5 0.2 linear 0.2

printf 'postflight=input_integrity\n'
(
  cd "$common_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
(
  cd "$overlay_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
test "$(sha256sum "$overlay_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_overlay_dataset_manifest_sha"
test "$(sha256sum "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json" | awk '{print $1}')" = \
  "$expected_overlay_report_sha"
test "$(sha256sum "$input_dir/stru_out" | awk '{print $1}')" = \
  "$expected_overlay_stru_sha"
test -z "$(find "$overlay_root" -perm /222 -print -quit)"
(
  cd "$candidate_gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$candidate_gate1"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$candidate_gate2"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
sha256sum --check --quiet "$run_root/common-input-OUTPUT_SHA256SUMS.relocated.txt"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=A1_legacy_fullbz_vs_current_fullbz
acceptance=true
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
legacy_commit=$expected_old_commit
legacy_executable=$old_exe
legacy_executable_sha256=$expected_old_exe_sha
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
candidate_gate0=$candidate_gate0
candidate_gate0_provenance_sha256=$GATE0_PROVENANCE_SHA256
candidate_gate1=$candidate_gate1
candidate_gate1_provenance_sha256=$GATE1_PROVENANCE_SHA256
candidate_gate1_output_manifest_sha256=$expected_gate1_output_manifest_sha
candidate_gate2=$candidate_gate2
candidate_gate2_provenance_sha256=$GATE2_PROVENANCE_SHA256
candidate_gate2_output_manifest_sha256=$expected_gate2_output_manifest_sha
source_dataset=$source_input_dir
source_dataset_manifest_sha256=$expected_common_dataset_manifest_sha
dataset=$input_dir
dataset_manifest_sha256=$expected_overlay_dataset_manifest_sha
dataset_overlay_contract=legacy_fullbz_stru_overlay_v1
dataset_overlay_report_sha256=$expected_overlay_report_sha
dataset_overlay_stru_sha256=$expected_overlay_stru_sha
same_input_for_legacy_and_candidate=true
pair_root=$pair_root
pair_output_manifest_sha256=$expected_pair_output_manifest_sha
pair_validation_sha256=$expected_pair_validation_sha
input_contract_sha256=$expected_common_contract_sha
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=off
band=off
h_qsgw_cut=disabled_non_band
host=$(hostname -f 2>/dev/null || hostname)
execution_surface=fish_direct
mpi_ranks=$mpi_ranks
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cat >"$run_root/A1_ACCEPTANCE.txt" <<EOF
accepted=true
oracle=legacy_symmetry_off_full_bz
candidate=current_symmetry_off_full_bz
no_mix_miniter2=passed
linear_beta_0.2_miniter5=passed
iteration_zero_through_final=compared
matrix_state_and_invariants=passed
EOF

(
  cd "$run_root"
  find no-mix-miniter2 linear-beta-0.2-miniter5 -type f -print0 | \
    sort -z | xargs -0 sha256sum >MODE_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet MODE_OUTPUT_SHA256SUMS.txt
)
sha256sum \
  "$run_root/PROVENANCE.txt" \
  "$run_root/CONTROLLED_VARIABLES.txt" \
  "$run_root/A1_ACCEPTANCE.txt" \
  "$run_root/MODE_OUTPUT_SHA256SUMS.txt" \
  "$run_root/old-build-provenance.txt" \
  "$run_root/candidate-gate0-PROVENANCE.txt" \
  "$run_root/candidate-gate0-OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate1-PROVENANCE.txt" \
  "$run_root/candidate-gate1-OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate2-PROVENANCE.txt" \
  "$run_root/candidate-gate2-OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate2-qsgw-iter1-invariants.json" \
  "$run_root/candidate-gate2-qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  "$run_root/DATASET_SHA256SUMS.txt" \
  "$run_root/common-input-OUTPUT_SHA256SUMS.txt" \
  "$run_root/common-input-OUTPUT_SHA256SUMS.relocated.txt" \
  "$run_root/common-input-PROVENANCE.txt" \
  "$run_root/qsgw_input.contract" \
  "$run_root/gate-a1-fullbz-old-current-v4.sh" \
  "$overlay_root/DATASET_SHA256SUMS.txt" \
  "$overlay_root/LEGACY_FULLBZ_STRU_OVERLAY.json" \
  "$tool_dir/build_legacy_fullbz_stru_overlay_v1.py" \
  "$tool_dir/test_build_legacy_fullbz_stru_overlay_v1.py" \
  "$run_root/adapter-unit-test.stdout" \
  "$run_root/adapter-unit-test.stderr" \
  "$run_root/closure-unit-test.stdout" \
  "$run_root/closure-unit-test.stderr" \
  "$run_root/initial-unit-test.stdout" \
  "$run_root/initial-unit-test.stderr" \
  "$run_root/overlay-builder-unit-test.stdout" \
  "$run_root/overlay-builder-unit-test.stderr" \
  "$run_root/overlay-builder.stdout" \
  "$run_root/overlay-builder.stderr" \
  "$run_root/oneapi-setvars.log" \
  >"$run_root/SHA256SUMS.txt"
(
  cd "$run_root"
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
touch "$run_root/GREEN_CONFIRMED"
run_succeeded=1
trap - EXIT
printf 'GREEN_CONFIRMED %s\n' "$run_root"
