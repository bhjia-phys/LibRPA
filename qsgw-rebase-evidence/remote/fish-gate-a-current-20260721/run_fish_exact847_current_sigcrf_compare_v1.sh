#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the clean runner checkout}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must identify the clean runner checkout}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable run directory}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_ID" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID contains unsafe characters" >&2; exit 2 ;;
esac

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
legacy_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-shrinkchi-off-miniter2-20260722-v3
legacy_work=$legacy_run/legacy
legacy_sigcrf=$legacy_work/librpa.d
source_candidate_run=$base/librpa-qsgw-gate-a1-exact847-candidate-one-update-20260722-69c33c2f-v3
input_overlay=$source_candidate_run/input-overlay-state-basis
observer_run=$base/librpa-qsgw-gate-a1-exact847-component-observer-20260722-17720999-v1
component_dir=$observer_run/legacy/librpa.d/qsgw_legacy_components/iter_00001
gate0=$base/librpa-qsgw-gate0-20260722-66bfe1cf-v1
candidate=$run_root/candidate
tools_dir=$run_root/tools
python=$base/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_legacy_failed_sha=acea8cb23bf445e18da415193e099532413036a0ce9c28393835b38fa1cc5ddf
expected_legacy_provenance_sha=ba3db4dd770806337c85e4f4c446eb02ad959473f69f99d02d81e80c1dc5df0c
expected_legacy_input_sha=75c895f04642497b6578e9ec061cb29c7a1af8d6ae680eaddd8d0e7115b20b85
expected_legacy_history_sha=8157cd3154c5515a8bbe3ede38731cf40c0bb952d47879004f8934d5d0addeb7
expected_component_tree_sha=e1065cedb703c2eba9ee32aca3303815e0c539553cebabb1c3b333eb9275f4e3

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
    $1 == key { count += 1; value = substr($0, length(key) + 2) }
    END { if (count != 1) exit 2; print value }
  ' "$file"
}

test ! -e "$run_root"
test -e "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test -x "$python"

test -e "$legacy_run/FAILED"
test ! -e "$legacy_run/COMPLETE"
test "$(sha256sum "$legacy_run/FAILED" | awk '{print $1}')" = \
  "$expected_legacy_failed_sha"
test "$(sha256sum "$legacy_run/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_legacy_provenance_sha"
test "$(sha256sum "$legacy_work/librpa.in" | awk '{print $1}')" = \
  "$expected_legacy_input_sha"
test "$(sha256sum "$legacy_work/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  "$expected_legacy_history_sha"
test "$(find "$legacy_sigcrf" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 16
grep -Fqx 'use_shrink_chi = f' "$legacy_work/librpa.in"
grep -Fqx 'output_gw_sigc_mat_rf = t' "$legacy_work/librpa.in"

test -e "$observer_run/FAILED"
test -f "$component_dir/metadata.txt"
test "$(find "$component_dir" -maxdepth 1 -type f -name '*.bin' | wc -l)" -eq 168
component_tree_sha=$(
  cd "$component_dir"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum \
    | awk '{print $1}'
)
test "$component_tree_sha" = "$expected_component_tree_sha"

test -e "$gate0/GREEN_CONFIRMED"
test ! -e "$gate0/FAILED"
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
candidate_build=$(provenance_value candidate_build "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test -f "$input_overlay/qsgw_input.contract"
grep -Fqx 'basis state' "$input_overlay/qsgw_vxc_scf.manifest"
grep -Fqx 'gauge mf0_state' "$input_overlay/qsgw_vxc_scf.manifest"

mkdir -p "$candidate" "$tools_dir"
cp "$0" "$run_root/"
for tool in \
  compare_exact847_sigcrf_v1.py \
  compare_exact847_component_dump_v1.py \
  diagnose_exact847_component_parity_v1.py \
  compare_legacy_h0_candidate_trace_v2.py; do
  cp "$current_dir/$tool" "$tools_dir/$tool"
done
cp "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" "$tools_dir/"
cp "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" "$tools_dir/"

(
  cd "$legacy_sigcrf"
  find . -maxdepth 1 -type f -name 'SigcRF*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$run_root/legacy-sigcrf-sha256.txt"
)

cat >"$candidate/librpa.in" <<EOF
task = qsgw
input_dir = $input_overlay/
output_dir = ./
constants_choice = internal
nfreq = 16
tfgrid_type = minimax
n_params_anacon = 16
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
gf_R_threshold = 1e-12
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = false
output_gw_sigc_mat_rf = true
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = true
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LD_LIBRARY_PATH="$candidate_build/src:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_current_sigcrf_compare_v1
acceptance=false_diagnostic_only
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
legacy_sigcrf_source_run=$legacy_run
legacy_source_status=failed_after_iteration1_checkpoint
legacy_sigcrf_iteration=1
legacy_component_observer_run=$observer_run
legacy_component_tree_sha256=$component_tree_sha
source_candidate_overlay=$input_overlay
read_sigc_mat_rf=false
output_gw_sigc_mat_rf=true
symmetry=on
headwing=off
hartree=off
iteration=1
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$candidate"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 10800 mpirun -np 1 "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)

grep -Fq 'QSGW iteration 1:' "$candidate/librpa.stdout"
grep -Fq 'QSGW completed iterations: 1' "$candidate/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$candidate/librpa.stdout"
test -s "$candidate/qsgw_matrices.dat"
test "$(find "$candidate" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 16

"$python" -B "$tools_dir/compare_exact847_sigcrf_v1.py" \
  "$legacy_sigcrf" "$candidate" \
  "$run_root/exact847-current-sigcrf-comparison.json" \
  >"$run_root/sigcrf-comparison.stdout" \
  2>"$run_root/sigcrf-comparison.stderr"
test ! -s "$run_root/sigcrf-comparison.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/exact847-current-sigcrf-comparison.json"

PYTHONPATH="$tools_dir" "$python" -B \
  "$tools_dir/compare_exact847_component_dump_v1.py" \
  "$component_dir" "$candidate/qsgw_matrices.dat" \
  "$run_root/exact847-current-component-comparison.json" \
  --iteration 1 --n-frequencies 16 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  >"$run_root/component-comparison.stdout" \
  2>"$run_root/component-comparison.stderr"
test ! -s "$run_root/component-comparison.stderr"
grep -Fq '"diagnostic_complete": true' \
  "$run_root/exact847-current-component-comparison.json"

(
  cd "$candidate"
  find . -maxdepth 1 -type f -name 'SigcRF*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$run_root/current-sigcrf-sha256.txt"
)

cat >>"$run_root/PROVENANCE.txt" <<EOF
candidate_trace_sha256=$(sha256sum "$candidate/qsgw_matrices.dat" | awk '{print $1}')
sigcrf_comparison_sha256=$(sha256sum "$run_root/exact847-current-sigcrf-comparison.json" | awk '{print $1}')
component_comparison_sha256=$(sha256sum "$run_root/exact847-current-component-comparison.json" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name DIAGNOSTIC_COMPLETE ! -name FAILED -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/DIAGNOSTIC_COMPLETE"
cat "$run_root/exact847-current-sigcrf-comparison.json"
run_succeeded=1
trap - EXIT
