#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run immutable}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a-legacy-no-overlap-iter1-$RUN_TAG
legacy_gate=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-symmetry-oracle-build-20260720-v10
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-symmetry-oracle-20260720-v10/build
legacy_exe=$legacy_build/chi0_main.exe
legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
legacy_exe_sha=ee198669c8e57d5e2d923f2284062572dbaa06d3d6652f29830afa898a7dd225
legacy_gate_provenance_sha=35efa46f326697220f2b67697ab05e8433e7477c0fbb6989817ac66bd45476fc

bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
source_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
source_stru_sha=5d943ee64376bc4e3315cc7ae779a1a91785b42e515290d37dacd78147947b5a

gate2_source=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1
gate2_current=$gate2_source/candidate-qsgw
gate2_postcheck=/home/bhj/ai-runs/librpa-qsgw-gate2-current-postcheck-20260722-2ad6b353-v1
input_overlay=$gate2_source/input-overlay
expected_gate2_provenance_sha=79d50e4f5ea14b624ff3aebc2c43802c1ee01e3b97445401467048b57e8f0526
expected_current_matrix_sha=85a8e04bd29780630a6720832898e06e18953810c2b183f87e5be4a9af93af37
expected_current_eigenvalue_sha=1ee2b4e9afe21987a95bfc3315e9945b0dd9dfa69d8f41e1734f4e4607e009f8
expected_current_iteration_sha=e7ae3485578958d1ead4bead993e27b508bbe32f7ff7ab688f1b82d5d7837ef9
vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd

tool_root=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
base_comparator=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
current_parser=$RUNNER_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py
python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
mpi_ranks=1
omp_threads=32

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
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test -e "$legacy_gate/GREEN_CONFIRMED"
test ! -e "$legacy_gate/FAILED"
test "$(sha256sum "$legacy_gate/PROVENANCE.txt" | awk '{print $1}')" = "$legacy_gate_provenance_sha"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = "$dataset_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = "$source_contract_sha"
test "$(sha256sum "$dataset/stru_out" | awk '{print $1}')" = "$source_stru_sha"
test -e "$gate2_postcheck/GREEN_CONFIRMED"
test ! -e "$gate2_postcheck/FAILED"
test "$(sha256sum "$gate2_postcheck/PROVENANCE.txt" | awk '{print $1}')" = "$expected_gate2_provenance_sha"
test "$(sha256sum "$gate2_current/qsgw_matrices.dat" | awk '{print $1}')" = "$expected_current_matrix_sha"
test "$(sha256sum "$gate2_current/qsgw_eigenvalues.dat" | awk '{print $1}')" = "$expected_current_eigenvalue_sha"
test "$(sha256sum "$gate2_current/qsgw_iterations.dat" | awk '{print $1}')" = "$expected_current_iteration_sha"
test "$(sha256sum "$input_overlay/vxc_out" | awk '{print $1}')" = "$vxc_sha"
test -f "$tool_root/compare_qsgw_legacy_v4_current_v6.py"
test -f "$tool_root/test_compare_qsgw_legacy_v4_current_v6.py"
test -f "$tool_root/summarize_qsgw_trace_components_v1.py"
test -f "$base_comparator"
test -f "$current_parser"
test -x "$python"

mkdir -p "$run_root/tools" "$run_root/input-view" "$run_root/legacy"
cp "$0" "$run_root/"
cp "$legacy_gate/PROVENANCE.txt" "$run_root/legacy-build-PROVENANCE.txt"
cp "$gate2_postcheck/PROVENANCE.txt" "$run_root/gate2-PROVENANCE.txt"
cp "$tool_root/compare_qsgw_legacy_v4_current_v6.py" "$run_root/tools/"
cp "$tool_root/test_compare_qsgw_legacy_v4_current_v6.py" "$run_root/tools/"
cp "$tool_root/summarize_qsgw_trace_components_v1.py" "$run_root/tools/"
cp "$base_comparator" "$run_root/tools/base_comparator.py"
cp "$current_parser" "$run_root/tools/cmp_qsgw.py"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

while IFS= read -r -d '' input; do
  name=$(basename "$input")
  target=$(readlink -f "$input")
  case "$name" in
    stru_out) target=$dataset/stru_out ;;
    qsgw_input.contract) target=$dataset/qsgw_input.contract ;;
  esac
  ln -s "$target" "$run_root/input-view/$name"
done < <(find "$input_overlay" -mindepth 1 -maxdepth 1 -print0)

test "$(find "$run_root/input-view" -maxdepth 1 -type l | wc -l)" -eq 57
test ! -e "$run_root/input-view/s1k1_nao.txt"
test -e "$run_root/input-view/sks1k1_nao.txt"
test "$(sha256sum "$run_root/input-view/stru_out" | awk '{print $1}')" = "$source_stru_sha"
test "$(sha256sum "$run_root/input-view/qsgw_input.contract" | awk '{print $1}')" = "$source_contract_sha"
test "$(sha256sum "$run_root/input-view/vxc_out" | awk '{print $1}')" = "$vxc_sha"

cat >"$run_root/legacy/librpa.in" <<EOF
task = qsgw
input_dir = $run_root/input-view/
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
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
qsgw_iterative_headwing = false
max_iter = 1
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" 2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

(
  cd "$run_root/legacy"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=1
  export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 21600 mpirun -np "$mpi_ranks" "$legacy_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)

test -s "$run_root/legacy/qsgw_oracle_matrices.dat"
test -s "$run_root/legacy/homo_lumo_vs_iterations.dat"
grep -Fq 'libRPA finished successfully' "$run_root/legacy/librpa.stdout"
test "$(grep -c '^S matrix file not found:' "$run_root/legacy/librpa.stderr")" -eq 8
test "$(grep -c '^HF file not found:' "$run_root/legacy/librpa.stderr")" -eq 8
test -z "$(grep -Ev '^(S matrix file not found:|HF file not found:)' "$run_root/legacy/librpa.stderr")"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' "$run_root/legacy/homo_lumo_vs_iterations.dat")" = 1

"$python" -B "$run_root/tools/summarize_qsgw_trace_components_v1.py" \
  "$run_root/legacy/qsgw_oracle_matrices.dat" \
  "$run_root/legacy-component-summary.json" --iterations 0:1 \
  >"$run_root/legacy-component-summary.stdout" \
  2>"$run_root/legacy-component-summary.stderr"

"$python" -B "$run_root/tools/compare_qsgw_legacy_v4_current_v6.py" \
  "$run_root/legacy/qsgw_oracle_matrices.dat" \
  "$gate2_current/qsgw_matrices.dat" \
  "$gate2_current/qsgw_eigenvalues.dat" \
  "$gate2_current/qsgw_iterations.dat" \
  "$run_root/legacy-current-comparison.json" \
  --base-comparator "$run_root/tools/base_comparator.py" \
  --current-contract-parser "$run_root/tools/cmp_qsgw.py" \
  --iterations 0:1 --expected-mode none \
  --expected-legacy-beta 1 --expected-current-beta 0.2 \
  --frequency-tolerance 1e-10 --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 --degeneracy-tolerance-ha 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/legacy-current-comparison.stdout" \
  2>"$run_root/legacy-current-comparison.stderr"
grep -Fq '"passed": true' "$run_root/legacy-current-comparison.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a_legacy_no_overlap_iter1_v1
acceptance=true_iter0_iter1_component_parity
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
legacy_commit=$legacy_commit
legacy_executable_sha256=$legacy_exe_sha
current_gate2_matrix_sha256=$expected_current_matrix_sha
dataset_manifest_sha256=$dataset_manifest_sha
legacy_input_contract_sha256=$source_contract_sha
legacy_stru_out_sha256=$source_stru_sha
shared_vxc_out_sha256=$vxc_sha
legacy_overlap_policy=historical_identity_fallback
expected_s_matrix_warning_count=8
symmetry=on_ibz_8_to_full_bz_64
headwing=off
hartree=off
band=off
iterations=0:1
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
run_succeeded=1
trap - EXIT
cat "$run_root/PROVENANCE.txt"
cat "$run_root/legacy/homo_lumo_vs_iterations.dat"
