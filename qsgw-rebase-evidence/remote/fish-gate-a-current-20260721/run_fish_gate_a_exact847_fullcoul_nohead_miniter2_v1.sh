#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_ID:?RUN_ID must name a fresh immutable run directory}"

base=/home/bhj/ai-runs
run_root=$base/$RUN_ID
build_evidence=$base/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-build-20260720-v12
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-20260720-v12
legacy_build=$legacy_source/build
legacy_exe=$legacy_build/chi0_main.exe
work=$run_root/legacy

expected_build_provenance_sha=01c9cd0d9cb8a5e1ac95a4706144994d9141bd3d0349408281ef3313352f3fba
expected_build_output_sha=95b97146491167a5a68e60c0fafc528718e31ea12321315668387cd34e1048f2
expected_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_bundle_output_sha=25590340b95edd0aaa91e3c8b34e332545afd9143ba96807a391290a7e0e9a43
expected_grid_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_legacy_exe_sha=6edd4e9847ce6536815eee17627bfd35be5b8443283d52818726ab2e5a74af50
expected_modified_task_sha=34e5c93fe12259f4838469b0e19b2c2316d4b6871ba9d6c9da5b3c057a01ab34
expected_corrected_fermi_cpp_sha=e4dcf3cb0998f312eaeab2e530c1cb306c0b9790608034784dad0ca3ea1437f3
target_iter=2
mpi_ranks=1
omp_threads=32

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
test -e "$build_evidence/GREEN_CONFIRMED"
test ! -e "$build_evidence/FAILED"
test "$(sha256sum "$build_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_build_provenance_sha"
test "$(sha256sum "$build_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_build_output_sha"
(
  cd "$build_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_dataset_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_output_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_grid_contract_sha"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = \
  "$expected_legacy_exe_sha"
test "$(sha256sum "$legacy_source/driver/task_qsgw_band_0.cpp" | awk '{print $1}')" = \
  "$expected_modified_task_sha"
test "$(sha256sum "$legacy_source/qsgw/fermi_energy_occupation.cpp" | awk '{print $1}')" = \
  "$expected_corrected_fermi_cpp_sha"

mkdir -p "$work/librpa.d"
for entry in "$dataset"/*; do
  ln -s "$entry" "$work/$(basename "$entry")"
done
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

cat >"$work/librpa.in" <<'EOF'
task = qsgw_band0
nfreq = 16
n_params_anacon = 16
option_dielect_func = 0
replace_w_head = f
use_scalapack_gw_wc = t
use_scalapack_ecrpa = t
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
use_fullcoul_eps = t
use_fullcoul_wc = f
use_pyatb = f
output_energy_qp = t
output_gw_sigc_mat_rf = t
libri_chi0_threshold_C = 1e-4
libri_chi0_threshold_G = 1e-5
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
libri_g0w0_threshold_C = 1e-5
libri_g0w0_threshold_G = 1e-5
libri_g0w0_threshold_Wc = 1e-6
max_iter = 2
qsgw_checkpoint_every = 1
qsgw_export_hamiltonian_for_pyatb = f
qsgw_band0_unoccupied_keep = 44
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
qsgw_band0_update_hartree = f
output_dir = librpa.d/
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
diagnostic=restore_historical_full_coulomb_exx_and_frequency_grid
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_patch=scheme_a_occupation_only
iterations=0:$target_iter
input_bundle=$bundle
crystal_symmetry=on_ibz_8_to_full_bz_64
legacy_task=qsgw_band0
legacy_mixing=commented_out_direct_update
headwing=off
hartree=off
h_qsgw_cut=off_mode0
nfreq=16
n_params_anacon=all_16_points
use_shrink_abfs=true
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
export LIBRPA_WCFQ_DUMP=1
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_exact847_scheme_a_legacy_fullcoul_nohead_miniter2_v1
acceptance=pending_legacy_runtime
legacy_role=corrected_multi_iteration_oracle_harness_not_raw_historical_binary
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_build_evidence=$build_evidence
legacy_build_provenance_sha256=$expected_build_provenance_sha
legacy_build_output_manifest_sha256=$expected_build_output_sha
legacy_executable=$legacy_exe
legacy_executable_sha256=$expected_legacy_exe_sha
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_bundle_dataset_sha
grid_contract_sha256=$expected_grid_contract_sha
target_iteration=$target_iter
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=direct_update_none
headwing=off
hartree=off
h_qsgw_cut=off_mode0
band=enabled_by_legacy_qsgw_band0_but_not_a_gate_observer
nfreq=16
use_fullcoul_exx=true
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$work/librpa.in" "$dataset/qsgw_input.contract" \
  >>"$run_root/PROVENANCE.txt"

(
  cd "$work"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 21600 mpirun -np "$mpi_ranks" "$legacy_exe" 16 1e-12 \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)

grep -Fq 'Task work begins: qsgw_band0' "$work/librpa.stdout"
grep -Fq 'QSGW band0: max_iterations = 2' "$work/librpa.stdout"
grep -Fq 'QSGW band0: H0 cut mode 0' "$work/librpa.stdout"
grep -Fq 'Iteration 1: HOMO =' "$work/librpa.stdout"
grep -Fq 'Iteration 2: HOMO =' "$work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$work/librpa.stdout"
test -s "$work/homo_lumo_vs_iterations.dat"

python3 - "$work/homo_lumo_vs_iterations.dat" "$run_root/iteration-audit.json" <<'PY'
import json
import math
import pathlib
import sys

history = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
rows = []
for raw in history.read_text().splitlines():
    fields = raw.split()
    if not fields:
        continue
    assert len(fields) == 4, raw
    row = {
        "iteration": int(fields[0]),
        "homo_ev": float(fields[1]),
        "lumo_ev": float(fields[2]),
        "efermi_ev": float(fields[3]),
    }
    assert all(math.isfinite(row[key]) for key in ("homo_ev", "lumo_ev", "efermi_ev")), row
    assert -100.0 < row["homo_ev"] < 100.0, row
    assert -100.0 < row["lumo_ev"] < 100.0, row
    assert row["lumo_ev"] >= row["homo_ev"] - 1.0e-8, row
    rows.append(row)
assert [row["iteration"] for row in rows] == [0, 1, 2], rows
assert all(row["lumo_ev"] - row["homo_ev"] > 0.0 for row in rows), rows
output.write_text(json.dumps({
    "expected_iterations": [0, 1, 2],
    "rows": rows,
    "finite_physical_frontier": True,
    "positive_gap_each_iteration": True,
    "passed": True,
}, indent=2, sort_keys=True) + "\n")
PY

for iteration in 1 2; do
  checkpoint="$work/librpa.d/qsgw_checkpoints/iter_$(printf '%05d' "$iteration")"
  test -f "$checkpoint/meta.txt"
  test "$(find "$checkpoint" -maxdepth 1 \
    -name 'H0_GW_spin_01_k_*.bin' -type f | wc -l)" -eq 8
done

printf 'acceptance=true_legacy_runtime_only_candidate_not_started\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
touch "$run_root/LEGACY_MINITER2_GREEN"
touch "$run_root/RUN_GREEN"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"

echo GATE_A1_EXACT847_SCHEME_A_LEGACY_FULLCOUL_NOHEAD_MINITER2_V1=PASS
cat "$run_root/iteration-audit.json"
