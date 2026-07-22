#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-shrink-off-iter1-${RUN_TAG}
gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260722-66bfe1cf-v1
expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274

legacy_gate=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-symmetry-oracle-build-20260720-v10
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-symmetry-oracle-20260720-v10/build
legacy_exe=$legacy_build/chi0_main.exe
legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
legacy_exe_sha=ee198669c8e57d5e2d923f2284062572dbaa06d3d6652f29830afa898a7dd225
legacy_gate_provenance_sha=35efa46f326697220f2b67697ab05e8433e7477c0fbb6989817ac66bd45476fc
legacy_gate_manifest_sha=4d913cd39b76bc81e117814e1ac5fb7251800d75bdca5fa1442ed821a38a6a48
legacy_harness_patch_sha=bc56a7a25bf85eebd7418711164e99b332ee3f5d4d4a152bfcc1015bdf6dccf3

bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
bundle_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
source_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179

gate2_accept=/home/bhj/ai-runs/librpa-qsgw-gate2-current-postcheck-20260722-2ad6b353-v1
gate2_source=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1
expected_gate2_provenance_sha=79d50e4f5ea14b624ff3aebc2c43802c1ee01e3b97445401467048b57e8f0526
expected_gate2_manifest_sha=1023c711b76da6161cf24b98d766f90517d215fb7407f47b45abcc0d204abb27
expected_gate2_source_manifest_sha=964d7e1f53c214aaa388458d75e64808683c07b9b968391d7bc449cb681d8c81
input_overlay=$gate2_source/input-overlay
legacy_input_view=$run_root/input-views/legacy
candidate_input_view=$run_root/input-views/candidate
legacy_input_dir=$legacy_input_view/
candidate_input_dir=$candidate_input_view/
source_stru_sha=5d943ee64376bc4e3315cc7ae779a1a91785b42e515290d37dacd78147947b5a
overlay_contract_sha=7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7
overlay_stru_sha=e756fd9551bfa9df748473880259ba019de904867c1aaff126b1b3a9c51a8873
vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd

symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
observer_bundle=$symmetry_dir/observer-tools-v1
base_comparator_source=$symmetry_dir/compare_qsgw_component_traces-v4-c3daf072.py
closure_source=$observer_bundle/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$observer_bundle/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$observer_bundle/validate_qsgw_fixed_basis.py
initial_source=$observer_bundle/validate_qsgw_initial_state-v1.py
initial_test_source=$observer_bundle/test_validate_qsgw_initial_state-v1.py
base_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
adapter_sha=279fa6221f4384c9caf8e6999f1fdc6238f7b7c9cace6c49d73ecce3ece5f457
adapter_test_sha=c87292f3efbe42a617d34e097294eb58d635d1181219db455f3908ed99170e1a
current_contract_parser_sha=f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb
input_view_validator_sha=93a9fcf9bffb40e43daeafd0709b38c8bae37c4789bd5810e872ae227f6860cc
input_view_test_sha=df045826f24cb591a88bfd87ad8bc7665cd17f944eca79ef7a574b40f4c9815f

mpi_ranks=1
omp_threads=32
pytest_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
python=$pytest_env/bin/python

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
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"

test -e "$gate0/GREEN_CONFIRMED"
test ! -e "$gate0/FAILED"
test "$(sha256sum "$gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate0_provenance_sha"
test "$(sha256sum "$gate0/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate0_manifest_sha"
(
  cd "$gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
candidate_source=$(provenance_value candidate_source "$gate0/PROVENANCE.txt")
candidate_build=$(provenance_value candidate_build "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test "$(provenance_value candidate_executable_sha256 "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_exe_sha"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$expected_candidate_exe_sha"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$expected_candidate_commit"
test -z "$(git -C "$candidate_source" status --porcelain)"

test -e "$legacy_gate/GREEN_CONFIRMED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "$(sha256sum "$legacy_gate/PROVENANCE.txt" | awk '{print $1}')" = \
  "$legacy_gate_provenance_sha"
test "$(sha256sum "$legacy_gate/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$legacy_gate_manifest_sha"
(
  cd "$legacy_gate"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fqx 'acceptance=true_oracle_harness_build' "$legacy_gate/PROVENANCE.txt"
grep -Fqx "combined_oracle_patch_sha256=$legacy_harness_patch_sha" \
  "$legacy_gate/PROVENANCE.txt"

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$bundle_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$source_contract_sha"
test "$(sha256sum "$dataset/stru_out" | awk '{print $1}')" = "$source_stru_sha"
test "$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$vxc_manifest_sha"
test "${legacy_input_dir: -1}" = "/"
test "${candidate_input_dir: -1}" = "/"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test -e "$gate2_accept/GREEN_CONFIRMED"
test ! -e "$gate2_accept/FAILED"
test "$(sha256sum "$gate2_accept/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate2_provenance_sha"
test "$(sha256sum "$gate2_accept/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate2_manifest_sha"
test "$(sha256sum "$gate2_accept/SOURCE_RUN_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate2_source_manifest_sha"
(
  cd "$gate2_accept"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$gate2_source"
  sha256sum --check --quiet "$gate2_accept/SOURCE_RUN_SHA256SUMS.txt"
)
test "$(sha256sum "$input_overlay/qsgw_input.contract" | awk '{print $1}')" = \
  "$overlay_contract_sha"
test "$(sha256sum "$input_overlay/stru_out" | awk '{print $1}')" = \
  "$overlay_stru_sha"
test "$(sha256sum "$input_overlay/vxc_out" | awk '{print $1}')" = \
  "$vxc_sha"
test "$(find "$input_overlay" -maxdepth 1 -type l | wc -l)" -eq 54
test "$(find "$input_overlay" -maxdepth 1 -type f | wc -l)" -eq 3
(
  cd "$input_overlay"
  sha256sum --check --quiet ../input-overlay.sha256
)

adapter_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_legacy_v4_current_v6.py
adapter_test_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_compare_qsgw_legacy_v4_current_v6.py
input_view_validator_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/validate_gate_a_input_views_v1.py
input_view_test_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_validate_gate_a_input_views_v1.py
current_contract_parser_source=$RUNNER_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py
while read -r path expected; do
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$base_comparator_source $base_comparator_sha
$closure_source $closure_sha
$closure_test_source $closure_test_sha
$fixed_source $fixed_sha
$initial_source $initial_sha
$initial_test_source $initial_test_sha
$adapter_source $adapter_sha
$adapter_test_source $adapter_test_sha
$input_view_validator_source $input_view_validator_sha
$input_view_test_source $input_view_test_sha
$current_contract_parser_source $current_contract_parser_sha
EOF

test -x "$python"
"$python" -c 'import numpy; print(numpy.__version__)' >/dev/null

mkdir -p "$run_root/tools"
cp "$0" "$run_root/run_fish_gate_a_shrink_off_iter1_v1.sh"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$gate0/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$gate0/OUTPUT_SHA256SUMS.txt" "$run_root/gate0-OUTPUT_SHA256SUMS.txt"
cp "$gate2_accept/PROVENANCE.txt" "$run_root/gate2-accepted-PROVENANCE.txt"
cp "$gate2_accept/OUTPUT_SHA256SUMS.txt" "$run_root/gate2-accepted-OUTPUT_SHA256SUMS.txt"
cp "$gate2_accept/SOURCE_RUN_SHA256SUMS.txt" "$run_root/gate2-SOURCE_RUN_SHA256SUMS.txt"
cp "$legacy_gate/PROVENANCE.txt" "$run_root/legacy-gate-PROVENANCE.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$base_comparator_source" "$run_root/tools/base_comparator.py"
cp "$base_comparator_source" "$run_root/tools/compare_qsgw_component_traces.py"
cp "$closure_source" "$run_root/tools/validate_qsgw_trace_closure.py"
cp "$closure_source" "$run_root/tools/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" "$run_root/tools/test_validate_qsgw_trace_closure.py"
cp "$fixed_source" "$run_root/tools/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$run_root/tools/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$run_root/tools/test_validate_qsgw_initial_state.py"
cp "$adapter_source" "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py"
cp "$adapter_test_source" "$run_root/tools/test_compare_qsgw_legacy_v4_current_v6.py"
cp "$input_view_validator_source" "$run_root/tools/validate_gate_a_input_views_v1.py"
cp "$input_view_test_source" "$run_root/tools/test_validate_gate_a_input_views_v1.py"
cp "$current_contract_parser_source" "$run_root/tools/cmp_qsgw_v6.py"

(
  cd "$run_root/tools"
  "$python" -B "$adapter_test_source" \
    >"$run_root/adapter-unit-test.stdout" \
    2>"$run_root/adapter-unit-test.stderr"
  PYTHONPATH="$run_root/tools" "$python" -B test_validate_qsgw_trace_closure.py \
    >"$run_root/closure-unit-test.stdout" \
    2>"$run_root/closure-unit-test.stderr"
  PYTHONPATH="$run_root/tools" "$python" -B test_validate_qsgw_initial_state.py \
    >"$run_root/initial-unit-test.stdout" \
    2>"$run_root/initial-unit-test.stderr"
  PYTHONPATH="$run_root/tools" "$python" -B test_validate_gate_a_input_views_v1.py \
    >"$run_root/input-view-unit-test.stdout" \
    2>"$run_root/input-view-unit-test.stderr"
)

mkdir -p "$legacy_input_view" "$candidate_input_view"
while IFS= read -r -d '' input; do
  name=$(basename "$input")
  common_target=$(readlink -f "$input")
  legacy_target=$common_target
  case "$name" in
    stru_out) legacy_target=$dataset/stru_out ;;
    qsgw_input.contract) legacy_target=$dataset/qsgw_input.contract ;;
  esac
  ln -s "$legacy_target" "$legacy_input_view/$name"
  ln -s "$common_target" "$candidate_input_view/$name"
done < <(find "$input_overlay" -mindepth 1 -maxdepth 1 -print0)

test "$(find "$legacy_input_view" -maxdepth 1 -type l | wc -l)" -eq 57
test "$(find "$candidate_input_view" -maxdepth 1 -type l | wc -l)" -eq 57
test ! -e "$legacy_input_view/s1k1_nao.txt"
test ! -e "$candidate_input_view/s1k1_nao.txt"
"$python" -B "$run_root/tools/validate_gate_a_input_views_v1.py" \
  "$legacy_input_view" "$candidate_input_view" \
  "$run_root/input-view-validation.json" \
  >"$run_root/input-view-validation.stdout" \
  2>"$run_root/input-view-validation.stderr"
grep -Fq '"passed": true' "$run_root/input-view-validation.json"
test "$(sha256sum "$legacy_input_view/stru_out" | awk '{print $1}')" = \
  "$source_stru_sha"
test "$(sha256sum "$candidate_input_view/stru_out" | awk '{print $1}')" = \
  "$overlay_stru_sha"
test "$(sha256sum "$legacy_input_view/qsgw_input.contract" | awk '{print $1}')" = \
  "$source_contract_sha"
test "$(sha256sum "$candidate_input_view/qsgw_input.contract" | awk '{print $1}')" = \
  "$overlay_contract_sha"
while IFS= read -r -d '' input; do
  name=$(basename "$input")
  case "$name" in
    stru_out|qsgw_input.contract) continue ;;
  esac
  test "$(readlink -f "$legacy_input_view/$name")" = \
    "$(readlink -f "$candidate_input_view/$name")"
done < <(find "$candidate_input_view" -mindepth 1 -maxdepth 1 -print0)

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
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/CONTROLLED_PAIR.txt" <<EOF
pair=legacy_compatibility_harness_vs_candidate_shrink_off_iter1
changed_factor=use_shrink_abfs_false_on_both_implementations
mode=direct_beta_1_vs_none_iter1
candidate_configured_beta=0.2
legacy_semantic_mapping=direct_beta_1_vs_candidate_none
dataset_sha256=$dataset_manifest_sha
frozen_physical_input_source=$input_overlay
legacy_input_view=$legacy_input_view
candidate_input_view=$candidate_input_view
reader_metadata_differences=qsgw_input.contract,stru_out
shared_legacy_reader_aliases=none_identity_overlap_fallback
common_physical_file_sha256_identical=true
legacy_input_contract_sha256=$source_contract_sha
candidate_input_contract_sha256=$overlay_contract_sha
legacy_stru_out_sha256=$source_stru_sha
candidate_stru_out_sha256=$overlay_stru_sha
vxc_out_sha256=$vxc_sha
runner_commit=$RUNNER_COMMIT
candidate_commit=$expected_candidate_commit
candidate_executable_sha256=$expected_candidate_exe_sha
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
symmetry=on
headwing=off
hartree=off
band=off
EOF

run_mode() {
  local mode_name=$1
  local target_iter=$2
  local legacy_beta=$3
  local candidate_mode=$4
  local candidate_beta=$5
  local mode_root=$run_root/$mode_name
  local legacy_run=$mode_root/legacy
  local candidate_run=$mode_root/candidate

  mkdir -p "$legacy_run" "$candidate_run"
  cat >"$legacy_run/librpa.in" <<EOF
task = qsgw
input_dir = $legacy_input_dir
output_dir = .
fn_vxc_scf = vxc_out
constants_choice = internal
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
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

  cat >"$candidate_run/librpa.in" <<EOF
task = qsgw
input_dir = $candidate_input_dir
output_dir = .
fn_vxc_scf = vxc_out
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
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = $candidate_mode
qsgw_mixing_beta = $candidate_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

  cat >"$mode_root/PARAMETERS.txt" <<EOF
mode=$mode_name
iterations=0:$target_iter
legacy_role=compatibility_harness_not_raw_source
legacy_effective_beta=$legacy_beta
candidate_mixer=$candidate_mode
candidate_configured_beta=$candidate_beta
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
dataset=$dataset
dataset_manifest_sha256=$dataset_manifest_sha
frozen_physical_input_source=$input_overlay
legacy_input_view=$legacy_input_view
candidate_input_view=$candidate_input_view
reader_metadata_differences=qsgw_input.contract,stru_out
shared_legacy_reader_aliases=none_identity_overlap_fallback
common_physical_file_sha256_identical=true
legacy_input_contract_sha256=$source_contract_sha
candidate_input_contract_sha256=$overlay_contract_sha
legacy_stru_out_sha256=$source_stru_sha
candidate_stru_out_sha256=$overlay_stru_sha
vxc_out_sha256=$vxc_sha
EOF

  (
    cd "$legacy_run"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE=0
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=$legacy_beta
    export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$legacy_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  test -s "$legacy_run/qsgw_oracle_matrices.dat"
  test -s "$legacy_run/homo_lumo_vs_iterations.dat"
  grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_gw 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_exx 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_shrink_abfs 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fq 'libRPA finished successfully' "$legacy_run/librpa.stdout"
  test "$(grep -c '^S matrix file not found:' "$legacy_run/librpa.stderr")" -eq 8
  test "$(grep -c '^HF file not found:' "$legacy_run/librpa.stderr")" -eq 8
  test -z "$(grep -Ev '^(S matrix file not found:|HF file not found:)' \
    "$legacy_run/librpa.stderr")"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$legacy_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

  (
    cd "$candidate_run"
    unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
    unset LIBRPA_QSGW_MIXING_BETA
    export LD_LIBRARY_PATH="$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$candidate_run/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$candidate_run/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$candidate_run/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$candidate_run/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$candidate_run/$trace"
    grep -Fqx '# headwing disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# hartree disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# band disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# h_qsgw_cut disabled_non_band' "$candidate_run/$trace"
    grep -Fqx "# qsgw_mixer $candidate_mode" "$candidate_run/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $overlay_contract_sha" "$candidate_run/$trace"
  done
  test -s "$candidate_run/homo_lumo_vs_iterations.dat"
  grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

  "$python" -B "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_eigenvalues.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$mode_root/legacy-current-comparison.json" \
    --base-comparator "$run_root/tools/base_comparator.py" \
    --current-contract-parser "$run_root/tools/cmp_qsgw_v6.py" \
    --iterations "0:$target_iter" \
    --expected-mode "$candidate_mode" \
    --expected-legacy-beta "$legacy_beta" \
    --expected-current-beta "$candidate_beta" \
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

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_trace_closure.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$mode_root/legacy-closure.json" \
    --iterations "0:$target_iter" --channel 0 --legacy-contract \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/legacy-closure.stdout" \
    2>"$mode_root/legacy-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-closure.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_trace_closure.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-closure.json" \
    --iterations "0:$target_iter" --channel 0 \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/current-closure.stdout" \
    2>"$mode_root/current-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/current-closure.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_fixed_basis.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-v5-self-eigenvalues.dat" \
    "$candidate_input_view/band_out" "$mode_root/current-fixed-basis.json" \
    --iterations "0:$target_iter" --channel 0 \
    --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
    >"$mode_root/current-fixed-basis.stdout" \
    2>"$mode_root/current-fixed-basis.stderr"
  grep -Fq '"passed": true' "$mode_root/current-fixed-basis.json"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_initial_state.py" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$candidate_input_view/band_out" "$mode_root/current-initial-state.json" \
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
mode=$mode_name
iterations=0:$target_iter
legacy_current_matrix_relative_tolerance=1e-8
legacy_current_eigenvalue_tolerance_ha=1e-6
legacy_current_gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_tolerance=1e-10
electron_count_tolerance=1e-10
EOF
}

run_mode shrink-off-no-mix-iter1 1 1 none 0.2

(
  cd "$gate2_source"
  sha256sum --check --quiet "$gate2_accept/SOURCE_RUN_SHA256SUMS.txt"
)

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate_a_shrink_off_iter1_v1
acceptance=true
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
legacy_role=compatibility_harness_not_raw_source
legacy_commit=$legacy_commit
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
legacy_harness_patch_sha256=$legacy_harness_patch_sha
candidate_commit=$expected_candidate_commit
candidate_gate0_root=$gate0
candidate_gate0_provenance_sha256=$expected_gate0_provenance_sha
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
dataset=$dataset
dataset_manifest_sha256=$dataset_manifest_sha
source_contract_sha256=$source_contract_sha
source_stru_out_sha256=$source_stru_sha
vxc_manifest_sha256=$vxc_manifest_sha
frozen_physical_input_source=$input_overlay
legacy_input_view=$legacy_input_view
candidate_input_view=$candidate_input_view
reader_metadata_differences=qsgw_input.contract,stru_out
shared_legacy_reader_aliases=none_identity_overlap_fallback
common_physical_file_sha256_identical=true
legacy_input_contract_sha256=$source_contract_sha
candidate_input_contract_sha256=$overlay_contract_sha
legacy_stru_out_sha256=$source_stru_sha
candidate_stru_out_sha256=$overlay_stru_sha
shared_vxc_out_sha256=$vxc_sha
gate2_accepted_root=$gate2_accept
gate2_accepted_provenance_sha256=$expected_gate2_provenance_sha
gate2_source_manifest_sha256=$expected_gate2_source_manifest_sha
symmetry=on
headwing=off
hartree=off
band=off
use_shrink_abfs=false_both_implementations
mode=none_iter1
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
printf 'FISH_GATE_A_SHRINK_OFF_ITER1_V1=PASS\n'
cat "$run_root/shrink-off-no-mix-iter1/legacy/homo_lumo_vs_iterations.dat"
cat "$run_root/shrink-off-no-mix-iter1/candidate/homo_lumo_vs_iterations.dat"
run_succeeded=1
trap - EXIT
