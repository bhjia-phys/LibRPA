#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
run_root=$base/librpa-qsgw-gate-a0-exact847-exx-band-isolation-20260720-v1
qsgw_run=$base/librpa-qsgw-gate-a0-legacy-exact847-band0-iter1-20260720-v1
oracle_bundle=$base/librpa-qsgw-gate-a-historical-band0-oracle-20260720-v1-2380156
oracle=$oracle_bundle/oracle
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8/build
legacy_exe=$legacy_build/chi0_main.exe
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1
analyzer=analyze_legacy_band0_text_diff_v1.py
analyzer_test=test_analyze_legacy_band0_text_diff_v1.py

expected_legacy_exe_sha=481ec33b3118747eb33ff3c252ab23fe23f7202c3cee7ee7147ac60b2e5cedaa
expected_bundle_dataset_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_analyzer_sha=d50d699ccf54dce8b17288f0a5e30bbfe51cd68173c6412d4fe2013abe5f6a41
expected_analyzer_test_sha=448778bca260be5bb364ad2ba079453232bbf52dc77974bf132b01ed86ad7c8f
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
test -e "$qsgw_run/RUN_GREEN"
test ! -e "$qsgw_run/FAILED"
test -e "$oracle_bundle/COMPLETE"
test ! -e "$oracle_bundle/FAILED"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -x "$legacy_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = \
  "$expected_legacy_exe_sha"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_dataset_sha"
test "$(sha256sum "$staging/$analyzer" | awk '{print $1}')" = \
  "$expected_analyzer_sha"
test "$(sha256sum "$staging/$analyzer_test" | awk '{print $1}')" = \
  "$expected_analyzer_test_sha"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$qsgw_run"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$oracle_bundle/oracle"
  sha256sum --check --quiet "$oracle_bundle/ORACLE_SHA256SUMS.txt"
)

mkdir -p "$run_root/work" "$run_root/tools"
for entry in "$dataset"/*; do
  ln -s "$entry" "$run_root/work/$(basename "$entry")"
done
install -m 0444 "$staging/$analyzer" "$run_root/tools/$analyzer"
install -m 0444 "$staging/$analyzer_test" \
  "$run_root/tools/$analyzer_test"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

cat >"$run_root/work/librpa.in" <<'EOF'
task = exx_band
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
output_dir = ./
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
export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_exact847_standalone_exx_band_isolation_v1
acceptance_scope=diagnostic_only
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
legacy_executable=$legacy_exe
legacy_executable_sha256=$expected_legacy_exe_sha
input_bundle=$bundle
input_dataset_manifest_sha256=$expected_bundle_dataset_sha
task=exx_band
symmetry=on
use_shrink_abfs=true
use_fullcoul_exx=true
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root/tools"
  python3 -m unittest "$analyzer_test"
) >"$run_root/analyzer-tests.stdout" \
  2>"$run_root/analyzer-tests.stderr"

(
  cd "$run_root/work"
  timeout 3600 mpirun -np "$mpi_ranks" "$legacy_exe" 16 1e-12 \
    >librpa.stdout 2>librpa.stderr
)
grep -Fq 'Task work begins: exx_band' "$run_root/work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$run_root/work/librpa.stdout"
test -s "$run_root/work/EXX_band_spin_1.dat"
test -s "$run_root/work/KS_band_spin_1.dat"

if cmp -s "$run_root/work/EXX_band_spin_1.dat" \
  "$qsgw_run/work/EXX_band_spin_1_1.dat"; then
  standalone_matches_reproduced_qsgw=true
  touch "$run_root/STANDALONE_EQUALS_REPRODUCED_QSGW_EXX"
else
  standalone_matches_reproduced_qsgw=false
fi

python3 "$run_root/tools/$analyzer" \
  "$oracle/EXX_band_spin_1_1.dat" \
  "$run_root/work/EXX_band_spin_1.dat" \
  "$oracle/QSGW_band_spin_1_1.dat" \
  "$qsgw_run/work/QSGW_band_spin_1_1.dat" \
  "$run_root/band-isolation-analysis.json" \
  >"$run_root/analyzer.stdout" 2>"$run_root/analyzer.stderr"

python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); e=r["exx"]; q=r["qsgw"]; print("standalone_exx_max_abs_diff_ev="+repr(e["energy_max_abs_diff_ev"])); print("standalone_exx_max_location="+json.dumps(e["energy_max_location"],sort_keys=True)); print("qsgw_gap_reference_ev="+repr(q["reference_edges"]["gap_ev"])); print("qsgw_gap_reproduced_ev="+repr(q["observed_edges"]["gap_ev"])); print("qsgw_gap_abs_diff_ev="+repr(q["gap_abs_diff_ev"]))' \
  "$run_root/band-isolation-analysis.json" \
  >"$run_root/diagnostic-summary.txt"

cat >>"$run_root/PROVENANCE.txt" <<EOF
standalone_matches_reproduced_qsgw_exx=$standalone_matches_reproduced_qsgw
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
touch "$run_root/RUN_COMPLETE"
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

echo GATE_A0_EXACT847_EXX_BAND_ISOLATION_V1=COMPLETE
echo standalone_matches_reproduced_qsgw_exx=$standalone_matches_reproduced_qsgw
cat "$run_root/diagnostic-summary.txt"
