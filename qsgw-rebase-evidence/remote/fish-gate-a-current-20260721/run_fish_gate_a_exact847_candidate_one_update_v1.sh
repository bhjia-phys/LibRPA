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
legacy_checkpoint=$legacy_work/librpa.d/qsgw_checkpoints
legacy_bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
legacy_dataset=$legacy_bundle/dataset
gate0=$base/librpa-qsgw-gate0-20260722-66bfe1cf-v1
gate2=$base/librpa-qsgw-gate2-current-20260722-aca53374-v1
input_overlay=$gate2/input-overlay
candidate=$run_root/candidate
tools_dir=$run_root/tools
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_failed_sha=acea8cb23bf445e18da415193e099532413036a0ce9c28393835b38fa1cc5ddf
expected_legacy_provenance_sha=ba3db4dd770806337c85e4f4c446eb02ad959473f69f99d02d81e80c1dc5df0c
expected_legacy_input_sha=75c895f04642497b6578e9ec061cb29c7a1af8d6ae680eaddd8d0e7115b20b85
expected_legacy_history_sha=8157cd3154c5515a8bbe3ede38731cf40c0bb952d47879004f8934d5d0addeb7
expected_latest_sha=4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865
expected_checkpoint_meta_sha=5e7ff5f8a44a77e57d860f57f72e0623d5ef503077f841267ffa42c6460ac7ee
expected_overlay_contract_sha=7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7
expected_checkpoint_matrix_shas=(
  0e28d870136a05e67f012b22b31ddac6af31e4ee6ffef3cddc2dc65eabc92a03
  2c35d317cdff6a3f189d1da8177616fa7cca09731d71cb2d74a0fc06f1cf3cca
  0836c9dd0e82dc2235c3d2134ed678f7483d3534c2455f83547171454539603a
  525a185516d4d9965b00137de39465d90b38ab01893bfc45c834966e8ffb3cf2
  9c6c58dc7d707f87676783f4356998c54c1a893ede6adb21defc3a4aa2822fe8
  f6ab1d381cba4a0236c4a8ec4ff7b2a8b5ac5ee833128d6f49ae5cac63629804
  acf93703141588102afa9b18b5e31bd29e726743030034e89ac120e50070911a
  70b9efe5e82eb3cb44cf0e830225e43367291c496eeea9dcf83b2f205930f37b
)

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

# Bind only the usable iteration-1 legacy checkpoint. The source run remains a
# failed miniter2 diagnostic and is never promoted to a green legacy run.
test -e "$legacy_run/FAILED"
test ! -e "$legacy_run/COMPLETE"
test "$(sha256sum "$legacy_run/FAILED" | awk '{print $1}')" = "$expected_failed_sha"
test "$(sha256sum "$legacy_run/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_legacy_provenance_sha"
test "$(sha256sum "$legacy_work/librpa.in" | awk '{print $1}')" = \
  "$expected_legacy_input_sha"
test "$(sha256sum "$legacy_work/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  "$expected_legacy_history_sha"
test "$(sha256sum "$legacy_checkpoint/latest_iteration.txt" | awk '{print $1}')" = \
  "$expected_latest_sha"
test "$(cat "$legacy_checkpoint/latest_iteration.txt")" = 1
test "$(sha256sum "$legacy_checkpoint/iter_00001/checkpoint.meta" | awk '{print $1}')" = \
  "$expected_checkpoint_meta_sha"
grep -Fqx '1 4.08114 4.08114 4.08114' \
  "$legacy_work/homo_lumo_vs_iterations.dat"
for ik in $(seq 1 8); do
  printf -v name 'H0_GW_spin_01_k_%06d.bin' "$ik"
  test "$(sha256sum "$legacy_checkpoint/iter_00001/$name" | awk '{print $1}')" = \
    "${expected_checkpoint_matrix_shas[$((ik - 1))]}"
done

test -e "$gate0/GREEN_CONFIRMED"
test ! -e "$gate0/FAILED"
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
candidate_source=$(provenance_value candidate_source "$gate0/PROVENANCE.txt")
candidate_build=$(provenance_value candidate_build "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$expected_candidate_commit"
test -z "$(git -C "$candidate_source" status --porcelain)"
test "$(sha256sum "$input_overlay/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_overlay_contract_sha"

mkdir -p "$candidate" "$tools_dir"
"$python" - "$legacy_dataset" "$input_overlay" \
  "$run_root/shared-reader-input-audit.json" <<'PY'
import hashlib
import json
import pathlib
import sys

legacy = pathlib.Path(sys.argv[1])
candidate = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])

def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

excluded = {"qsgw_input.contract", "stru_out"}
rows = []
for entry in sorted(candidate.iterdir(), key=lambda path: path.name):
    if entry.name in excluded:
        continue
    old = legacy / entry.name
    if not old.is_file() or digest(old) != digest(entry):
        raise AssertionError(f"legacy/candidate reader mismatch: {entry.name}")
    rows.append({"file": entry.name, "sha256": digest(entry)})
if not rows:
    raise AssertionError("no common reader inputs were checked")
output.write_text(json.dumps({
    "excluded_metadata_files": sorted(excluded),
    "matched_reader_file_count": len(rows),
    "matched_reader_files": rows,
    "passed": True,
}, indent=2, sort_keys=True) + "\n")
PY

symmetry_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720
current_dir=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721
for tool in \
  "$symmetry_dir/compare_legacy_band0_native_outputs_v1.py" \
  "$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py" \
  "$current_dir/compare_legacy_h0_candidate_trace_v2.py" \
  "$symmetry_dir/observer-tools-v1/validate_qsgw_trace_closure-v3-4a5de94e.py" \
  "$symmetry_dir/observer-tools-v1/validate_qsgw_fixed_basis.py" \
  "$symmetry_dir/observer-tools-v1/validate_qsgw_initial_state-v1.py"; do
  test -f "$tool"
  cp "$tool" "$tools_dir/$(basename "$tool")"
done

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

cat >"$run_root/PARAMETER_MAPPING.txt" <<'EOF'
comparison=exact847_iteration1_checkpoint_to_current_qsgw_one_update
legacy_status=failed_after_iteration1_checkpoint
iterations=0:1
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_direct,candidate_none_direct
headwing=off
hartree=off
band=off_for_comparison
h_qsgw_cut=off_mode0
nfreq=16
n_params_anacon=16
use_shrink_abfs=true
use_shrink_chi=false
use_fullcoul_exx=true
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=1
omp_threads=32
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
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$candidate_build/src:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_candidate_one_update_v1
acceptance=pending_one_update_parity
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
legacy_run=$legacy_run
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_checkpoint_iteration=1
legacy_run_is_green=false
target_iteration=1
LIBRI_DETERMINISTIC_REDUCTION_requested=1
LIBRI_DETERMINISTIC_REDUCTION_binary_support=unverified
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$candidate"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 10800 mpirun -np 1 "$candidate_exe" >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)

grep -Fq 'QSGW iteration 1:' "$candidate/librpa.stdout"
grep -Fq 'QSGW completed iterations: 1' "$candidate/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$candidate/librpa.stdout"
for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
  test -s "$candidate/$trace"
  grep -Fqx '# qsgw_contract_version 6' "$candidate/$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$candidate/$trace"
  grep -Fqx '# qsgw_mixer none' "$candidate/$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $expected_overlay_contract_sha" \
    "$candidate/$trace"
done

"$python" -B "$tools_dir/validate_qsgw_trace_closure-v3-4a5de94e.py" \
  "$candidate/qsgw_matrices.dat" "$run_root/candidate-closure.json" \
  --iterations 0:1 --channel 0
"$python" -B "$tools_dir/validate_qsgw_fixed_basis.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_eigenvalues.dat" \
  "$input_overlay/band_out" "$run_root/candidate-fixed-basis.json" \
  --iterations 0:1 --channel 0
"$python" -B "$tools_dir/validate_qsgw_initial_state-v1.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_iterations.dat" \
  "$input_overlay/band_out" "$run_root/candidate-initial-state.json"

"$python" -B "$tools_dir/compare_legacy_h0_candidate_trace_v2.py" \
  "$legacy_checkpoint" "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" \
  "$run_root/legacy-candidate-comparison.json" \
  --iterations 1 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  --occupied-bands 4 \
  >"$run_root/comparator.stdout" 2>"$run_root/comparator.stderr"

for report in \
  candidate-closure.json candidate-fixed-basis.json \
  candidate-initial-state.json legacy-candidate-comparison.json; do
  grep -Fq '"passed": true' "$run_root/$report"
done
grep -Fq '"parity_passed": true' \
  "$run_root/legacy-candidate-comparison.json"
grep -Fq '"candidate_invariants_passed": true' \
  "$run_root/legacy-candidate-comparison.json"

printf 'acceptance=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$run_root/ONE_UPDATE_PARITY_GREEN"
(
  cd "$run_root"
  find . -type f \
    ! -name OUTPUT_SHA256SUMS.txt \
    ! -name COMPLETE \
    ! -name FAILED \
    -print0 | sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
run_succeeded=1
trap - EXIT
