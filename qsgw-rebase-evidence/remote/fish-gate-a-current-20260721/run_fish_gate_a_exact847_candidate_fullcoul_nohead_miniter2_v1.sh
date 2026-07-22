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
legacy_run=$base/librpa-qsgw-gate-a1-exact847-fullcoul-nohead-miniter2-20260722-6a85c7fc-v1
legacy_work=$legacy_run/legacy
legacy_checkpoints=$legacy_work/librpa.d/qsgw_checkpoints
legacy_bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
legacy_dataset=$legacy_bundle/dataset

gate0=$base/librpa-qsgw-gate0-20260722-66bfe1cf-v1
gate2_accept=$base/librpa-qsgw-gate2-current-postcheck-20260722-2ad6b353-v1
gate2_source=$base/librpa-qsgw-gate2-current-20260722-aca53374-v1
input_overlay=$gate2_source/input-overlay
candidate=$run_root/candidate
tools_dir=$run_root/tools

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274
expected_gate2_provenance_sha=79d50e4f5ea14b624ff3aebc2c43802c1ee01e3b97445401467048b57e8f0526
expected_gate2_manifest_sha=1023c711b76da6161cf24b98d766f90517d215fb7407f47b45abcc0d204abb27
expected_gate2_source_manifest_sha=964d7e1f53c214aaa388458d75e64808683c07b9b968391d7bc449cb681d8c81
expected_overlay_contract_sha=7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7
expected_overlay_stru_sha=e756fd9551bfa9df748473880259ba019de904867c1aaff126b1b3a9c51a8873
expected_vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd
expected_legacy_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_legacy_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43

mpi_ranks=1
omp_threads=32
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python

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

test -e "$legacy_run/COMPLETE"
test -e "$legacy_run/LEGACY_MINITER2_GREEN"
test -e "$legacy_run/RUN_GREEN"
test ! -e "$legacy_run/FAILED"
test -s "$legacy_run/iteration-audit.json"
grep -Fq '"passed": true' "$legacy_run/iteration-audit.json"
test -d "$legacy_checkpoints/iter_00001"
test -d "$legacy_checkpoints/iter_00002"
(
  cd "$legacy_run"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test -e "$legacy_bundle/COMPLETE"
test ! -e "$legacy_bundle/FAILED"
test "$(sha256sum "$legacy_bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_legacy_bundle_dataset_sha"
test "$(sha256sum "$legacy_bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_legacy_bundle_output_sha"
(
  cd "$legacy_bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

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
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$expected_candidate_commit"
test -z "$(git -C "$candidate_source" status --porcelain)"

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
  "$expected_overlay_contract_sha"
test "$(sha256sum "$input_overlay/stru_out" | awk '{print $1}')" = \
  "$expected_overlay_stru_sha"
test "$(sha256sum "$input_overlay/vxc_out" | awk '{print $1}')" = \
  "$expected_vxc_sha"
(
  cd "$input_overlay"
  sha256sum --check --quiet ../input-overlay.sha256
)

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
    legacy_path = legacy / entry.name
    if not legacy_path.is_file():
        raise AssertionError(f"legacy dataset is missing {entry.name}")
    legacy_sha = digest(legacy_path)
    candidate_sha = digest(entry)
    if legacy_sha != candidate_sha:
        raise AssertionError(f"reader input differs: {entry.name}")
    rows.append({"file": entry.name, "sha256": legacy_sha})
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
native_v1=$symmetry_dir/compare_legacy_band0_native_outputs_v1.py
checkpoint_v1=$symmetry_dir/compare_legacy_h0_candidate_trace_v1.py
checkpoint_test_v1=$symmetry_dir/test_compare_legacy_h0_candidate_trace_v1.py
checkpoint_v2=$current_dir/compare_legacy_h0_candidate_trace_v2.py
checkpoint_test_v2=$current_dir/test_compare_legacy_h0_candidate_trace_v2.py
closure=$symmetry_dir/observer-tools-v1/validate_qsgw_trace_closure-v3-4a5de94e.py
fixed=$symmetry_dir/observer-tools-v1/validate_qsgw_fixed_basis.py
initial=$symmetry_dir/observer-tools-v1/validate_qsgw_initial_state-v1.py
for tool in "$native_v1" "$checkpoint_v1" "$checkpoint_test_v1" \
  "$checkpoint_v2" "$checkpoint_test_v2" "$closure" "$fixed" "$initial"; do
  test -f "$tool"
  cp "$tool" "$tools_dir/$(basename "$tool")"
done
(
  cd "$tools_dir"
  "$python" -B -m unittest test_compare_legacy_h0_candidate_trace_v1.py
  "$python" -B -m unittest test_compare_legacy_h0_candidate_trace_v2.py
) >"$run_root/comparator-tests.stdout" 2>"$run_root/comparator-tests.stderr"

cat >"$candidate/librpa.in" <<EOF
task = qsgw
input_dir = $input_overlay/
output_dir = ./
constants_choice = internal
nfreq = 16
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
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
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=exact847_qsgw_band0_grid_to_current_qsgw_grid
iterations=0:2
legacy_task=qsgw_band0
candidate_task=qsgw
legacy_patch=scheme_a_occupation_only
same_physical_reader_files=true
reader_metadata_differences=qsgw_input.contract,stru_out
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_direct,candidate_none_direct
headwing=off
hartree=off
band=candidate_off_legacy_not_compared
h_qsgw_cut=off_mode0
nfreq=16
n_params_anacon=all_16_points
use_shrink_abfs=true
use_shrink_chi=false
use_fullcoul_exx=true
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
EOF

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
export LD_LIBRARY_PATH="$candidate_build/src:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_candidate_fullcoul_nohead_miniter2_v1
acceptance=pending_candidate_runtime_and_parity
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
legacy_run=$legacy_run
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_role=exact847_plus_scheme_a_occupation_only
candidate_input_overlay=$input_overlay
candidate_input_contract_sha256=$expected_overlay_contract_sha
same_physical_reader_files=true
target_iteration=2
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_direct_candidate_none
headwing=off
hartree=off
band=candidate_off
h_qsgw_cut=off
nfreq=16
use_fullcoul_exx=true
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$candidate/librpa.in" "$input_overlay/qsgw_input.contract" \
  >>"$run_root/PROVENANCE.txt"

(
  cd "$candidate"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)
grep -Fq 'QSGW iteration 1:' "$candidate/librpa.stdout"
grep -Fq 'QSGW iteration 2:' "$candidate/librpa.stdout"
grep -Fq 'QSGW completed iterations: 2' "$candidate/librpa.stdout"
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
  --iterations 0:2 --channel 0
"$python" -B "$tools_dir/validate_qsgw_fixed_basis.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_eigenvalues.dat" \
  "$input_overlay/band_out" "$run_root/candidate-fixed-basis.json" \
  --iterations 0:2 --channel 0
"$python" -B "$tools_dir/validate_qsgw_initial_state-v1.py" \
  "$candidate/qsgw_matrices.dat" "$candidate/qsgw_iterations.dat" \
  "$input_overlay/band_out" "$run_root/candidate-initial-state.json"
grep -Fq '"passed": true' "$run_root/candidate-closure.json"
grep -Fq '"passed": true' "$run_root/candidate-fixed-basis.json"
grep -Fq '"passed": true' "$run_root/candidate-initial-state.json"

"$python" -B "$tools_dir/compare_legacy_h0_candidate_trace_v2.py" \
  "$legacy_checkpoints" "$candidate/qsgw_matrices.dat" \
  "$candidate/qsgw_eigenvalues.dat" \
  "$run_root/legacy-candidate-comparison.json" \
  --iterations 1:2 --n-spins 1 --n-kpoints 8 --n-bands 44 \
  --occupied-bands 4 \
  >"$run_root/comparator.stdout" 2>"$run_root/comparator.stderr"
grep -Fq '"parity_passed": true' "$run_root/legacy-candidate-comparison.json"
grep -Fq '"candidate_invariants_passed": true' \
  "$run_root/legacy-candidate-comparison.json"
grep -Fq '"passed": true' "$run_root/legacy-candidate-comparison.json"

printf 'acceptance=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/PARITY_GREEN"
touch "$run_root/CANDIDATE_INVARIANTS_GREEN"
touch "$run_root/RUN_GREEN"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
run_succeeded=1

echo GATE_A1_EXACT847_CANDIDATE_FULLCOUL_NOHEAD_MINITER2_V1=PASS
cat "$run_root/legacy-candidate-comparison.json"
