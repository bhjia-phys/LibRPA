#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
run_root=$base/librpa-qsgw-gate-a1-candidate-symmetry-nomix-miniter2-capability-20260720-v1
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
mixing_audit=$base/librpa-qsgw-gate-a1-mixing-frontier-audit-20260720-v2
candidate_build=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13/build
candidate_exe=$candidate_build/chi0_main.exe
candidate=$run_root/candidate

expected_dataset_manifest_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43
expected_candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -e "$mixing_audit/COMPLETE"
test -e "$mixing_audit/DIAGNOSTIC_GREEN"
test ! -e "$mixing_audit/FAILED"
test -x "$candidate_exe"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"

mkdir -p "$candidate"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

cat >"$candidate/librpa.in" <<EOF
task = qsgw
input_dir = $dataset/
output_dir = ./
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
use_shrink_abfs = true
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
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
gate_kind=expected_capability_failure_diagnostic
input_bundle=$bundle
same_physical_bundle_as_legacy=true
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=none_direct_update
headwing=off
hartree=off
h_qsgw_cut=off
nfreq=6
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
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_candidate_symmetry_nomix_miniter2_capability_v1
acceptance=expected_failure_pending
claim=diagnostic_only_not_iterative_qsgw_acceptance
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_dataset_manifest_sha
mixing_frontier_audit=$mixing_audit
mixing=none_direct_update
symmetry=on_ibz_8_to_full_bz_64
headwing=off
hartree=off
target_iterations=2
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$candidate/librpa.in" "$dataset/qsgw_input.contract" \
  >>"$run_root/PROVENANCE.txt"

set +e
(
  cd "$candidate"
  export LD_LIBRARY_PATH="$candidate_build/src:$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np 1 "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  rc=$?
  printf 'completed_utc=%s\nexit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>runtime.txt
  exit "$rc"
)
candidate_rc=$?
set -e
printf '%s\n' "$candidate_rc" >"$run_root/candidate-exit-code.txt"

test "$candidate_rc" -ne 0
test -s "$candidate/qsgw_iterations.dat"
test -s "$candidate/qsgw_eigenvalues.dat"
test -s "$candidate/qsgw_matrices.dat"
grep -Fqx '# qsgw_contract_version 5' "$candidate/qsgw_iterations.dat"
grep -Fqx '# qsgw_mixer none' "$candidate/qsgw_iterations.dat"
grep -Fq 'QSGW iteration 1:' "$candidate/librpa.stdout"
test "$(grep -Ec '^2[[:space:]]' "$candidate/qsgw_iterations.dat")" -eq 0

python3 - "$candidate/qsgw_iterations.dat" \
  "$candidate/librpa.stdout" "$candidate/librpa.stderr" \
  "$run_root/capability-classification.json" <<'PY'
import json
import math
import pathlib
import sys

trace_path = pathlib.Path(sys.argv[1])
stdout_path = pathlib.Path(sys.argv[2])
stderr_path = pathlib.Path(sys.argv[3])
output_path = pathlib.Path(sys.argv[4])
rows = []
for raw in trace_path.read_text().splitlines():
    fields = raw.split()
    if not fields or raw.lstrip().startswith("#"):
        continue
    assert len(fields) >= 17, raw
    rows.append(
        {
            "iteration": int(fields[0]),
            "efermi_ev": float(fields[4]),
            "gap_ev": float(fields[5]),
            "electron_count": float(fields[6]),
        }
    )
assert [row["iteration"] for row in rows] == [0, 1], rows
iteration_one = rows[1]
assert abs(iteration_one["electron_count"] - 8.0) <= 1.0e-10
assert abs(iteration_one["gap_ev"]) <= 1.0e-12
stdout = stdout_path.read_text(errors="replace")
stderr = stderr_path.read_text(errors="replace")
assert "Minimax energy window: emin" in stdout
assert stderr.strip()
result = {
    "passed": True,
    "claim": "expected_capability_failure_reproduced",
    "iterative_qsgw_accepted": False,
    "iterations_written": rows,
    "iteration_one_charge_conserving_metal": True,
    "exception_contains_map_at": "map::at" in stderr,
    "stderr_tail": stderr.splitlines()[-20:],
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

printf 'acceptance=true_expected_capability_failure\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/EXPECTED_FAILURE_REPRODUCED"
touch "$run_root/DIAGNOSTIC_GREEN"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
find "$run_root" -type d -exec chmod 0555 {} +
find "$run_root" -type f -exec chmod 0444 {} +

echo GATE_A1_CANDIDATE_SYMMETRY_NOMIX_MINITER2_CAPABILITY_V1=PASS
cat "$run_root/capability-classification.json"
