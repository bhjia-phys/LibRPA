#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "ERROR line=%s status=%s command=%s\n" \
  "$LINENO" "$status" "$BASH_COMMAND" >&2' ERR

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify this committed runner}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must be a clean checkout at RUNNER_COMMIT}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${GATE0_PROVENANCE_SHA256:?GATE0_PROVENANCE_SHA256 must bind accepted Gate 0}"
: "${CANDIDATE_EXE_SHA256:?CANDIDATE_EXE_SHA256 must bind the Gate 0 executable}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"
test "${#RUNNER_COMMIT}" = 40
test "${#RUNNER_SHA256}" = 64
test "${#GATE0_PROVENANCE_SHA256}" = 64
test "${#CANDIDATE_EXE_SHA256}" = 64
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

bundle=/home/bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-physical-bundle-20260723-v2
input_dir=$bundle/dataset
candidate_gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260723-4f9ab0cf-v1
candidate_source=/tmp/librpa-qsgw-gate0-20260723-4f9ab0cf-v1/candidate
candidate_build=/tmp/librpa-qsgw-gate0-20260723-4f9ab0cf-v1/build-candidate
candidate_exe=$candidate_build/chi0_main.exe
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a23-current-symmetry-side-$RUN_TAG
tool_dir=$run_root/tools
mpi_ranks=4
omp_threads=12

expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff
expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e
expected_bundle_provenance_sha=9d2c7e63572e26ee2bb59d72c145da2c6ab9babf2c6e2c3bf352de8be5abab8d
expected_bundle_output_sha=a2fe0a055c7da236c5a17ad2a9c5769d7e2c549c08d40a260a2166b8d225963b
expected_dataset_manifest_sha=f8f97072567fbb819c406c0d7776d0209486c245b4e0140bdff7da34508b6c79
expected_contract_sha=70f69e02fb749229448b6139bc01f944789fbc21c23d10e10b7a7b56d952f340
expected_vxc_manifest_sha=545e595eab871410b4e06e46c6d4394d82f6d46fa46eb08e572a0546763a3ed4
expected_base_comparator_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_closure_adapter_sha=900f42f1917e4c2a80962e78b80a7226411981fc21923509983e55fa9afbdbfa
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
expected_current_parser_sha=f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb
runner_relative=qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/gate-a23-current-symmetry-side-v3.sh
normalizer_relative=qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/normalize_qsgw_v6_self_traces_v1.py
normalizer_test_relative=qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723/test_normalize_qsgw_v6_self_traces_v1.py

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
test -x "$candidate_exe"
test -x "$python"
mkdir -p "$tool_dir"
cp "$0" "$run_root/gate-a23-current-symmetry-side-v3.sh"
test "$(sha256sum "$run_root/gate-a23-current-symmetry-side-v3.sh" | awk '{print $1}')" = \
  "$RUNNER_SHA256"

printf 'preflight=candidate_gate0\n'
test -e "$candidate_gate0/GREEN_CONFIRMED"
test ! -e "$candidate_gate0/FAILED"
test "$(sha256sum "$candidate_gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$GATE0_PROVENANCE_SHA256"
grep -Fqx 'acceptance=true' "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "upstream_commit=$expected_upstream_commit" "$candidate_gate0/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_candidate_commit" "$candidate_gate0/PROVENANCE.txt"
grep -Fqx 'protected_diff=empty' "$candidate_gate0/PROVENANCE.txt"
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

printf 'preflight=pinned_symmetry_bundle\n'
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
test "$(sha256sum "$bundle/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_bundle_provenance_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_contract_sha"
test "$(sha256sum "$input_dir/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$expected_vxc_manifest_sha"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
grep -Fqx 'grid=4x4x4' "$bundle/PROVENANCE.txt"
grep -Fqx 'scf_kpoints=8' "$bundle/PROVENANCE.txt"
grep -Fqx 'full_bz_kpoints=64' "$bundle/PROVENANCE.txt"
grep -Fqx 'use_shrink_abfs=false' "$bundle/PROVENANCE.txt"
grep -Fqx 'symmetry=on' "$bundle/PROVENANCE.txt"
grep -Fqx 'physical_lattice_source=matching_abacus_input_STRU' \
  "$bundle/PROVENANCE.txt"
grep -Fqx 'shared_gw_source_changes=none' "$bundle/PROVENANCE.txt"
grep -Fqx 'headwing=off' "$bundle/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$bundle/PROVENANCE.txt"
grep -Fqx 'band_update=off' "$bundle/PROVENANCE.txt"
grep -Fqx 'n_scf_kpoints 8' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_band_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"

printf 'preflight=observer_tools\n'
normalizer_source=$RUNNER_SOURCE/$normalizer_relative
normalizer_test_source=$RUNNER_SOURCE/$normalizer_test_relative
normalizer_sha=$(git -C "$RUNNER_SOURCE" show "$RUNNER_COMMIT:$normalizer_relative" | \
  sha256sum | awk '{print $1}')
normalizer_test_sha=$(git -C "$RUNNER_SOURCE" show "$RUNNER_COMMIT:$normalizer_test_relative" | \
  sha256sum | awk '{print $1}')
test "$(sha256sum "$normalizer_source" | awk '{print $1}')" = "$normalizer_sha"
test "$(sha256sum "$normalizer_test_source" | awk '{print $1}')" = \
  "$normalizer_test_sha"
closure_adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_component_traces_v6_adapter.py
base_comparator_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
observer_root=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/observer-tools-v1
closure_source=$observer_root/validate_qsgw_trace_closure.py
fixed_source=$observer_root/validate_qsgw_fixed_basis.py
initial_source=$observer_root/validate_qsgw_initial_state.py
closure_test_source=$observer_root/test_validate_qsgw_trace_closure-v3-38de02fa.py
initial_test_source=$observer_root/test_validate_qsgw_initial_state-v1.py
current_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py

while read -r path expected
do
  actual=$(sha256sum "$path" | awk '{print $1}')
  printf 'observer_sha256=%s actual=%s expected=%s\n' \
    "$path" "$actual" "$expected"
  test "$actual" = "$expected"
done <<EOF
$base_comparator_source $expected_base_comparator_sha
$closure_adapter_source $expected_closure_adapter_sha
$closure_source $expected_closure_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$closure_test_source $expected_closure_test_sha
$initial_test_source $expected_initial_test_sha
$current_parser_source $expected_current_parser_sha
EOF

cp "$normalizer_source" "$tool_dir/normalize_qsgw_v6_self_traces_v1.py"
cp "$normalizer_test_source" "$tool_dir/test_normalize_qsgw_v6_self_traces_v1.py"
cp "$base_comparator_source" "$tool_dir/compare_qsgw_component_traces_v4.py"
cp "$closure_adapter_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$current_parser_source" "$tool_dir/cmp_qsgw_v6.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"
cp "$candidate_gate0/PROVENANCE.txt" "$run_root/candidate-gate0-PROVENANCE.txt"
cp "$candidate_gate0/OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-gate0-OUTPUT_SHA256SUMS.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/input-bundle-PROVENANCE.txt"
cp "$bundle/OUTPUT_SHA256SUMS.txt" "$run_root/input-bundle-OUTPUT_SHA256SUMS.txt"
cp "$bundle/DATASET_SHA256SUMS.txt" "$run_root/DATASET_SHA256SUMS.txt"
cp "$input_dir/qsgw_input.contract" "$run_root/qsgw_input.contract"
cp "$input_dir/qsgw_vxc_scf.manifest" "$run_root/qsgw_vxc_scf.manifest"

(
  cd "$tool_dir"
  "$python" -B test_normalize_qsgw_v6_self_traces_v1.py \
    >"$run_root/normalizer-unit-test.stdout" \
    2>"$run_root/normalizer-unit-test.stderr"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_trace_closure.py \
    >"$run_root/closure-unit-test.stdout" \
    2>"$run_root/closure-unit-test.stderr"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_initial_state.py \
    >"$run_root/initial-unit-test.stdout" \
    2>"$run_root/initial-unit-test.stderr"
)

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
gate=current_symmetry_side
acceptance=false_pending_full_bz_control
candidate=current_symmetry_on_ibz
dataset=$input_dir
dataset_manifest_sha256=$expected_dataset_manifest_sha
input_contract_sha256=$expected_contract_sha
vxc_manifest_sha256=$expected_vxc_manifest_sha
scf_kpoints=8
restored_full_bz_kpoints=64
headwing=off
hartree=off
band=off
use_shrink_abfs=false
use_fullcoul_exx=false
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
deterministic_reduction=1
mode_1=none_miniter2
mode_2=linear_beta_0.2_miniter5
EOF

run_mode() {
  local mode_name=$1
  local target_iter=$2
  local mixer=$3
  local mode_root=$run_root/$mode_name
  local current_run=$mode_root/current
  mkdir -p "$current_run"

  cat >"$current_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
fn_vxc_scf = vxc_out.dat
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
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = $mixer
qsgw_mixing_beta = 0.2
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

  (
    cd "$current_run"
    export LD_LIBRARY_PATH="$candidate_build/qsgw:$candidate_build/src:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )

  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat
  do
    test -s "$current_run/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$current_run/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$current_run/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$current_run/$trace"
    grep -Fqx '# velocity disabled_stage1' "$current_run/$trace"
    grep -Fqx '# headwing disabled_stage1' "$current_run/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$current_run/$trace"
    grep -Fqx '# hartree disabled_stage1' "$current_run/$trace"
    grep -Fqx '# band disabled_stage1' "$current_run/$trace"
    grep -Fqx '# h_qsgw_cut disabled_non_band' "$current_run/$trace"
    grep -Fqx "# qsgw_mixer $mixer" "$current_run/$trace"
    grep -Fqx '# qsgw_mixing_beta 0.2' "$current_run/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $expected_contract_sha" \
      "$current_run/$trace"
  done
  test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
    "$current_run/qsgw_matrices.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$current_run/qsgw_eigenvalues.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$current_run/qsgw_iterations.dat")" = "$target_iter"

  "$python" -B "$tool_dir/normalize_qsgw_v6_self_traces_v1.py" \
    "$current_run/qsgw_matrices.dat" \
    "$current_run/qsgw_eigenvalues.dat" \
    "$current_run/qsgw_iterations.dat" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-v5-self-eigenvalues.dat" \
    "$mode_root/current-v5-self-iterations.dat" \
    "$mode_root/current-v6-to-v5-normalization.json" \
    >"$mode_root/current-v6-to-v5-normalization.stdout" \
    2>"$mode_root/current-v6-to-v5-normalization.stderr"
  grep -Fq '"passed": true' "$mode_root/current-v6-to-v5-normalization.json"
  grep -Fq '"numeric_rows_unchanged": true' \
    "$mode_root/current-v6-to-v5-normalization.json"

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
    "$current_run/qsgw_matrices.dat" \
    "$current_run/qsgw_iterations.dat" \
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
  ' "$current_run/qsgw_iterations.dat" \
    >"$mode_root/current-electron-count.txt"

  cat >"$mode_root/SELF_VALIDATION.txt" <<EOF
passed=true
acceptance=false_pending_full_bz_control
mode=$mode_name
iterations=0:$target_iter
symmetry=on_ibz_live
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_tolerance=1e-10
electron_count_tolerance=1e-10
EOF
}

run_mode no-mix-miniter2 2 none
run_mode linear-beta-0.2-miniter5 5 linear

(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
(
  cd "$candidate_gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=current_symmetry_side_v3
acceptance=false_pending_full_bz_control
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
normalizer_sha256=$normalizer_sha
normalizer_test_sha256=$normalizer_test_sha
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
gate0_provenance_sha256=$GATE0_PROVENANCE_SHA256
input_bundle=$bundle
input_bundle_provenance_sha256=$expected_bundle_provenance_sha
input_bundle_output_manifest_sha256=$expected_bundle_output_sha
dataset_manifest_sha256=$expected_dataset_manifest_sha
qsgw_input_contract_sha256=$expected_contract_sha
qsgw_vxc_manifest_sha256=$expected_vxc_manifest_sha
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mixing_modes=none_miniter2,linear_beta_0.2_miniter5
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
deterministic_reduction=1
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name SYMMETRY_SIDE_COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/SYMMETRY_SIDE_COMPLETE"
find "$run_root" -type f -exec chmod a-w {} +
find "$run_root" -depth -type d -exec chmod a-w {} +
run_succeeded=1
trap - EXIT

test -e "$run_root/SYMMETRY_SIDE_COMPLETE"
test ! -e "$run_root/FAILED"
test -z "$(find "$run_root" -type l -print -quit)"
test -z "$(find "$run_root" -perm /222 -print -quit)"
echo CURRENT_QSGW_SYMMETRY_SIDE_V3=PASS_PENDING_FULL_BZ_CONTROL
