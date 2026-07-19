#!/usr/bin/env bash
# Gate 7B: band-channel SigC external-consistency check. The pure-upstream
# (95c4c080) g0w0_band executable computes the band-path correlation
# self-energy dumps on the frozen normalized Si k444 dataset; the new
# implementation's frozen Gate 7A trace (job 2358253) must reproduce the
# same band sigma_c(iw) blocks at iteration 1 within 1e-10 Ha / 1e-10
# relative Frobenius (Gate-2-equivalent for the band channel).
#SBATCH --job-name=qsgw-g7b-g0w0-band
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7b-g0w0-band-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate7b-g0w0-band-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-band-v3-2358467
input_dir=$input_root/dataset
gate7a_run=$root/runs/gate7a-qsgw-band-ad9fd6ed-miniter2-v3-2358253
upstream_build=$root/build-upstream-95c4c080-v1
upstream_exe=$upstream_build/chi0_main.exe
trace_comparator=$root/jobs/compare_qsgw_trace_g0w0_sigc.py
closure_v5=$root/jobs/validate_qsgw_trace_closure-v5-1cf1fcd6.py
summarize=$root/jobs/summarize_librpa_sigc_matrix_dumps-a2bf9dce.py
comparator_module=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate7b-g0w0-band-vs-7a-iter1-$SLURM_JOB_ID
upstream_run=$run_root/upstream
tool_dir=$run_root/tools

expected_upstream_commit=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_upstream_exe_sha=af93da0fccd7c9346734234f0c2ef890f3a156938fd3048294de93173434e268
expected_input_manifest_sha=20558326c86aded8cb902c03dad84c0aade8a83e374c785ae23fcd30ec24de6f
expected_band_contract_sha=94c1832de2c73fc3d8f44c93bf206a39f80b9e2233d40398af075b19dd40cdd3
expected_7a_trace_sha=202fee60b1be7bf6d336cf6e0fa8378e75076b18e028019b12584e5ad02743a7
expected_trace_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_closure_v5_sha=1cf1fcd6a409d4b7b84df2d03e36d2109d1fd37ad8afd9082d8353b1c4bbd7b6
expected_summarize_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_comparator_module_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

assert_passed() {
  "$python" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$1"
}

test ! -e "$run_root"
mkdir -p "$upstream_run" "$tool_dir"
test -e "$input_root/COMPLETE"
test -e "$upstream_build/COMPLETE"
test -e "$gate7a_run/FAILED"
test ! -e "$gate7a_run/COMPLETE"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$upstream_exe $expected_upstream_exe_sha
$input_dir/DATASET_SHA256SUMS.txt $expected_input_manifest_sha
$input_dir/qsgw_input.band.contract $expected_band_contract_sha
$gate7a_run/current/qsgw_matrices.dat $expected_7a_trace_sha
$trace_comparator $expected_trace_comparator_sha
$closure_v5 $expected_closure_v5_sha
$summarize $expected_summarize_sha
$comparator_module $expected_comparator_module_sha
EOF
grep -Fqx "source_commit=$expected_upstream_commit" \
  "$upstream_build/build-provenance.txt"
(
  cd "$input_dir"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

cp "$0" "$run_root/gate7b-g0w0-band-vs-7a-iter1-v5.slurm"
cp "$upstream_build/build-provenance.txt" "$run_root/upstream-build-provenance.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.band.contract" "$run_root/"
cp "$trace_comparator" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py"
cp "$closure_v5" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$summarize" "$tool_dir/summarize_librpa_sigc_matrix_dumps.py"
cp "$comparator_module" "$tool_dir/compare_qsgw_component_traces.py"

cat >"$upstream_run/librpa.in" <<EOF
task = g0w0_band
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
output_gw_sigc_ks_mat_kf = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
output_energy_qp = false
EOF

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate7b-g0w0-band-vs-7a-iter1
final_acceptance=false_pending_comparison
result=PENDING
upstream_commit=$expected_upstream_commit
upstream_executable_sha256=$expected_upstream_exe_sha
gate7a_job=2358253
gate7a_trace_sha256=$expected_7a_trace_sha
band_input_dataset_manifest_sha256=$expected_input_manifest_sha
band_contract_sha256=$expected_band_contract_sha
comparison=iteration1_band_channel_sigma_c_iw_vs_upstream_g0w0_band
max_abs_tolerance_ha=1e-10
relative_frobenius_tolerance=1e-10
mpi_ranks=${SLURM_NTASKS:-unknown}
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$upstream_run"
  unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA LIBRPA_QSGW_HARTREE_DUMP_DIR
  export LD_LIBRARY_PATH="$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$upstream_exe" >librpa.stdout 2>librpa.stderr
)

n_dumps=$(find "$upstream_run" -maxdepth 1 -type f \
  -name 'Sigc_fk_mn_band_0_ispin_*' | wc -l)
test "$n_dumps" -gt 0
printf 'upstream_band_sigc_dump_count=%s\n' "$n_dumps" \
  >>"$run_root/PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$gate7a_run/current/qsgw_matrices.dat" \
  "$upstream_run" \
  "$run_root/qsgw-band-iter1-vs-upstream-g0w0-band-sigc.json" \
  --iteration 1 \
  --channel 1 \
  --source band_0 \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.band.contract" \
  >"$run_root/sigc-comparator.stdout" \
  2>"$run_root/sigc-comparator.stderr"
assert_passed "$run_root/qsgw-band-iter1-vs-upstream-g0w0-band-sigc.json"

printf 'final_acceptance=true\nresult=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED ! -path './upstream/*' -print0 | sort -z | \
    xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE7B_G0W0_BAND_VS_7A_ITER1=PASS
