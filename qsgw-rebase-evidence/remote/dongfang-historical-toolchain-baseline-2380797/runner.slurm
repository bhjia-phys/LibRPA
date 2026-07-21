#!/bin/bash
#SBATCH -J q847-band0
#SBATCH -p 48cp3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH -o /data/home/df_iopcas_bhj/q847-band0-%j.out
#SBATCH -e /data/home/df_iopcas_bhj/q847-band0-%j.err

set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

home=/data/home/df_iopcas_bhj
runtime_parent=$home/ai-runs/librpa-qsgw-gate-a0-exact847-historical-toolchain-exx-20260720-v1
run_root=$home/ai-runs/librpa-qsgw-gate-a0-exact847-historical-toolchain-band0-iter1-20260720-v1
scratch_root=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/librpa-qsgw-gate-a0-exact847-historical-toolchain-band0-iter1-20260720-v1-${SLURM_JOB_ID}
work=$scratch_root/work
runtime=$scratch_root/runtime
historical=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa
bundle_manifest=$home/librpa-qsgw-gate-a-symmetry-band0-DATASET_SHA256SUMS-20260720.txt
source_verifier=$home/verify_historical_band0_source_against_bundle_v1.py
source_verifier_test=$home/test_verify_historical_band0_source_against_bundle_v1.py
comparator_v1=$home/compare_legacy_band0_native_outputs_v1.py
comparator_v1_test=$home/test_compare_legacy_band0_native_outputs_v1.py
comparator_v2=$home/compare_legacy_band0_native_outputs_v2.py
comparator_v2_test=$home/test_compare_legacy_band0_native_outputs_v2.py

expected_runtime_exe_sha=33fbed74ff744e3e6e85fcb825603d1fb478440b3500f3fb6d058a68726f14a1
expected_runtime_qsgw_sha=fa515da33ef666bd96feff18af840fecfd9260f742261275a47df6c6108efc32
expected_runtime_rpa_sha=088de821a4cdce32595dde44b5893fecff6c43a4dea1644ac727fccc06a0cf95
expected_bundle_manifest_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_source_verifier_sha=25238d020dad3056840ce0290836580679ea27180777e80326dd50566eb44266
expected_source_verifier_test_sha=8d731a434e2044540c0ae070310a9e06dd36ea29e869d2f7a54c99a33b71ba71
expected_comparator_v1_sha=350e6589b74a27a588d44ba7f1b53f42bcd46cfebd4c9a5c7658debdb520e88d
expected_comparator_v1_test_sha=402060c6b4c9b2eeefb6652670d3825ae835b86ae9f30622bb6f876a2a5c7293
expected_comparator_v2_sha=69bb86ecf7675e644fbd500a49e0f44c92950aba478e7f764bcea30c28f7afab
expected_comparator_v2_test_sha=4cd649cdb65dae707f698780e860acea5454be86c2c4f7419ee8c6c10804e551
expected_librpa_input_sha=ab6e1f986b13adcd68169c38d55b12b5cde018a5cc8557f78addd7d390e1d4e4
expected_oracle_exx_sha=a95d0a6b6a68483a483e499dc3060d127dfe12c50b04edfb9a43dad4795ce4b8
expected_oracle_qsgw_sha=835d4e35a346cd5cc1639fa1696ca3aa9283e877c2ff10c8ee309ed162a66059
expected_initial='Initial HOMO = 6.28506 eV, LUMO = 6.97617 eV, Fermi Energy = 6.48709 eV'
stage=preflight

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\nstage=%s\nslurm_job_id=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "${stage:-unknown}" \
      "${SLURM_JOB_ID:-unknown}" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
test ! -e "$scratch_root"
test -e "$runtime_parent/COMPLETE"
test ! -e "$runtime_parent/FAILED"
test -d "$historical"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
(
  cd "$runtime_parent"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(sha256sum "$runtime_parent/runtime/chi0_main.exe" | awk '{print $1}')" = "$expected_runtime_exe_sha"
test "$(sha256sum "$runtime_parent/runtime/libqsgw.so.0.3.0" | awk '{print $1}')" = "$expected_runtime_qsgw_sha"
test "$(sha256sum "$runtime_parent/runtime/librpa.so.0.3.0" | awk '{print $1}')" = "$expected_runtime_rpa_sha"
test "$(sha256sum "$bundle_manifest" | awk '{print $1}')" = "$expected_bundle_manifest_sha"
test "$(sha256sum "$source_verifier" | awk '{print $1}')" = "$expected_source_verifier_sha"
test "$(sha256sum "$source_verifier_test" | awk '{print $1}')" = "$expected_source_verifier_test_sha"
test "$(sha256sum "$comparator_v1" | awk '{print $1}')" = "$expected_comparator_v1_sha"
test "$(sha256sum "$comparator_v1_test" | awk '{print $1}')" = "$expected_comparator_v1_test_sha"
test "$(sha256sum "$comparator_v2" | awk '{print $1}')" = "$expected_comparator_v2_sha"
test "$(sha256sum "$comparator_v2_test" | awk '{print $1}')" = "$expected_comparator_v2_test_sha"
test "$(sha256sum "$historical/librpa.in" | awk '{print $1}')" = "$expected_librpa_input_sha"
test "$(sha256sum "$historical/EXX_band_spin_1_1.dat" | awk '{print $1}')" = "$expected_oracle_exx_sha"
test "$(sha256sum "$historical/QSGW_band_spin_1_1.dat" | awk '{print $1}')" = "$expected_oracle_qsgw_sha"

mkdir -p "$run_root" "$scratch_root" "$work/librpa.d" "$runtime" \
  "$run_root/reproduced/librpa.d/qsgw_checkpoints/iter_00001" \
  "$run_root/tools"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$0" "$run_root/runner.slurm"
cp "$source_verifier" "$source_verifier_test" "$comparator_v1" \
  "$comparator_v1_test" "$comparator_v2" "$comparator_v2_test" \
  "$run_root/tools/"

stage=tool_tests
(
  cd "$home"
  python3 -m unittest "$(basename "$source_verifier_test")" \
    "$(basename "$comparator_v1_test")" "$(basename "$comparator_v2_test")"
) >"$run_root/tool-tests.stdout" 2>"$run_root/tool-tests.stderr"

stage=input_manifest
python3 "$source_verifier" "$bundle_manifest" "$historical" \
  "$run_root/historical-source-vs-bundle.json" --minimum-existing 1000 \
  >"$run_root/historical-source-vs-bundle.stdout" \
  2>"$run_root/historical-source-vs-bundle.stderr"

stage=runtime_prepare
cp "$runtime_parent/runtime/chi0_main.exe" "$runtime/chi0_main.exe"
cp "$runtime_parent/runtime/libqsgw.so.0.3.0" "$runtime/libqsgw.so.0.3.0"
cp "$runtime_parent/runtime/librpa.so.0.3.0" "$runtime/librpa.so.0.3.0"
chmod 0555 "$runtime/chi0_main.exe"
chmod 0444 "$runtime/libqsgw.so.0.3.0" "$runtime/librpa.so.0.3.0"
test "$(sha256sum "$runtime/chi0_main.exe" | awk '{print $1}')" = "$expected_runtime_exe_sha"

stage=work_prepare
while read -r _ relative; do
  relative=${relative#dataset/}
  source_file=$historical/$relative
  if [[ -f "$source_file" ]]; then
    mkdir -p "$work/$(dirname "$relative")"
    ln -s "$source_file" "$work/$relative"
  fi
done <"$bundle_manifest"
cp "$historical/librpa.in" "$work/librpa.in"
test "$(sha256sum "$work/librpa.in" | awk '{print $1}')" = "$expected_librpa_input_sha"

stage=librpa
set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" 2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LIBRPA_WCFQ_DUMP=1
export LD_LIBRARY_PATH="$runtime:${LD_LIBRARY_PATH:-}"
ldd "$runtime/chi0_main.exe" >"$run_root/runtime-ldd.txt"
grep -Fq "$runtime/libqsgw.so.0.3.0" "$run_root/runtime-ldd.txt"
grep -Fq "$runtime/librpa.so.0.3.0" "$run_root/runtime-ldd.txt"
(
  cd "$work"
  timeout 14400 mpirun -np 1 "$runtime/chi0_main.exe" 16 1e-12 \
    >librpa.stdout 2>librpa.stderr
)
grep -Fq 'Initialization finished' "$work/librpa.stdout"
grep -Fq 'Task work begins: qsgw_band0' "$work/librpa.stdout"
grep -Fq 'QSGW band0: max_iterations = 1' "$work/librpa.stdout"
grep -Fqx "$expected_initial" "$work/librpa.stdout"
grep -Fq 'Iteration 1: HOMO =' "$work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$work/librpa.stdout"
test "$(wc -l <"$work/librpa.stderr")" -eq 8
for kpoint in $(seq 1 8); do
  grep -Fqx "HF file not found: hf_exchange_spin_01_kpt_$(printf '%06d' "$kpoint").csc" \
    "$work/librpa.stderr"
done
test -s "$work/QSGW_band_spin_1_1.dat"
test -s "$work/KS_band_spin_1_1.dat"
test -s "$work/EXX_band_spin_1_1.dat"
test -s "$work/homo_lumo_vs_iterations.dat"
test -s "$work/hrs1_nao_qsgw_iter_0001.csr"
test "$(find "$work/librpa.d/qsgw_checkpoints/iter_00001" \
  -maxdepth 1 -name 'H0_GW_spin_01_k_*.bin' -type f | wc -l)" -eq 8
test "$(find "$work/librpa.d" -maxdepth 1 \
  -name 'SigcRF_ispin_00_s_00_iomega_*_myid_00000.dat' -type f | wc -l)" -eq 16

stage=archive_native_outputs
cp "$work/librpa.in" "$work/KS_band_spin_1_1.dat" \
  "$work/EXX_band_spin_1_1.dat" "$work/QSGW_band_spin_1_1.dat" \
  "$work/homo_lumo_vs_iterations.dat" \
  "$work/hrs1_nao_qsgw_iter_0001.csr" \
  "$run_root/reproduced/"
cp "$work"/librpa.d/SigcRF_*.dat "$run_root/reproduced/librpa.d/"
cp "$work"/librpa.d/qsgw_checkpoints/iter_00001/H0_GW_spin_01_k_*.bin \
  "$run_root/reproduced/librpa.d/qsgw_checkpoints/iter_00001/"
cp "$work/librpa.stdout" "$work/librpa.stderr" "$run_root/"
grep -F 'Iteration 1: HOMO =' "$work/librpa.stdout" \
  >"$run_root/iteration1-summary.txt"

stage=compare
set +e
(
  cd "$home"
  python3 "$comparator_v2" "$historical" "$run_root/reproduced" \
    "$run_root/native-comparison-v2.json"
) >"$run_root/native-comparison-v2.stdout" \
  2>"$run_root/native-comparison-v2.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" >"$run_root/native-comparison-v2.exit-code"
if [[ "$comparison_rc" -eq 0 ]]; then
  touch "$run_root/HISTORICAL_REPRODUCTION_GREEN"
else
  touch "$run_root/HISTORICAL_REPRODUCTION_FAILED"
fi
python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); h=r.get("h0",{}); s=r.get("sigcrf",{}); t=r.get("text_outputs",{}); print("historical_reproduction_passed="+repr(r.get("historical_reproduction_passed"))); print("goal_thresholds_passed="+repr(r.get("goal_thresholds_passed"))); print("h0_max_abs_ha="+repr(h.get("max_abs_ha"))); print("h0_max_relative_frobenius="+repr(h.get("max_relative_frobenius"))); print("oracle_hermiticity_max_abs_ha="+repr(h.get("oracle_hermiticity_max_abs_ha"))); print("sigcrf_max_abs_ha="+repr(s.get("max_abs_ha"))); print("sigcrf_max_relative_frobenius="+repr(s.get("max_relative_frobenius"))); print("text_max_abs_ev="+repr(t.get("max_abs_ev")))' \
  "$run_root/native-comparison-v2.json" >"$run_root/diagnostic-summary.txt"

stage=freeze
input_map_summary=$(python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); print(str(r["matching_entries"])+"/"+str(r["existing_entries"]))' \
  "$run_root/historical-source-vs-bundle.json")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_exact847_historical_toolchain_qsgw_band0_iter1_v1
acceptance_scope=historical_native_reproduction
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
source_patch=none
runtime_parent=$runtime_parent
runtime_executable_sha256=$expected_runtime_exe_sha
runtime_libqsgw_sha256=$expected_runtime_qsgw_sha
runtime_librpa_sha256=$expected_runtime_rpa_sha
historical_input_root=$historical
librpa_input_sha256=$expected_librpa_input_sha
frozen_bundle_manifest_sha256=$expected_bundle_manifest_sha
historical_existing_input_matches=$input_map_summary
task=qsgw_band0
max_iter=1
nfreq=16
symmetry=on_ibz_8_to_full_bz_64
headwing=historical_option3_pyatb
hartree=off
band=historical_201_point_reference
h_qsgw_cut=mode2_keep10_shift20Ha
LIBRPA_WCFQ_DUMP=1
command_arguments=16_1e-12
mpi_ranks=1
omp_threads=${SLURM_CPUS_PER_TASK:-32}
native_comparison_exit_code=$comparison_rc
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=${SLURM_NODELIST:-unknown}
runner_sha256=$RUNNER_SHA256
scratch_root=$scratch_root
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
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

echo GATE_A0_EXACT847_HISTORICAL_TOOLCHAIN_BAND0_ITER1_V1=COMPLETE
cat "$run_root/iteration1-summary.txt"
cat "$run_root/diagnostic-summary.txt"
