#!/usr/bin/env bash
#SBATCH --job-name=qsgw-reader-head-ab
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/run-reader-head-ablation-2e446af2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/run-reader-head-ablation-2e446af2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
build=$root/build-diagnostic-reader-head-2e446af2-v5
exe=$build/chi0_main.exe
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
reference_root=$root/runs/gate4-head-only-limit-live-2e446af2-2346965
reference_g0w0=$reference_root/g0w0-head-only
run_root=$root/runs/diagnostic-reader-head-ablation-2e446af2-$SLURM_JOB_ID
qsgw_run=$run_root/qsgw-head-only
evidence=$root/evidence/diagnostic-reader-head-ablation-2e446af2-$SLURM_JOB_ID
tool_dir=$evidence/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
patch_file=$root/jobs/diagnostic-gate4-preserve-reader-head-2e446af2.patch
comparator_source=$root/jobs/compare_qsgw_trace_g0w0_sigc.py
reader_source=$root/jobs/summarize_librpa_sigc_matrix_dumps.py

expected_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418
expected_patch_sha=ac06b195fa4985b4059a80dcd3ebae17813450ef4cf465f6f4044f64e7aa50e7
expected_exe_sha=e654b032b2b0b23254dfdd3bedbd483b927e3553a0497b2ceeb4f728c605b8d3
expected_build_manifest_sha=9c9b1fb353759cb46502b14c13a250cedf91de38219f9867b96e3774c2a794b0
expected_build_provenance_sha=a656dcfac74e44500634161cf41c38703de57c6136c2d7cd1345b61bfdfd703b
expected_input_dataset_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_input_output_manifest_sha=b8c26ba28f64666c9f1efa3a840bc1edd1ed15cb84a6ef911990a0c1982de4dc
expected_input_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_head_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_reference_comparison_sha=ad8bf5864f56823d1f62e795f89493c0573755b25d5f9bd832d4944cdf71fab9
expected_reference_g0w0_manifest_sha=a2bc8e8837c96cfd17f71a96382292f9da5cc72b3712bbfdef2f2411d00952ba

test "${SLURM_NTASKS:-0}" -eq 4
test "${SLURM_CPUS_PER_TASK:-0}" -eq 12
test -e "$build/COMPLETE"
test -x "$exe"
test -e "$input_root/COMPLETE"
test -d "$reference_g0w0"
test ! -e "$run_root"
test ! -e "$evidence"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = "$expected_patch_sha"
test "$(sha256sum "$exe" | awk '{print $1}')" = "$expected_exe_sha"
test "$(sha256sum "$build/SHA256SUMS.txt" | awk '{print $1}')" = "$expected_build_manifest_sha"
test "$(sha256sum "$build/build-provenance.txt" | awk '{print $1}')" = "$expected_build_provenance_sha"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_input_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_input_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = "$expected_input_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.head-only.contract" | awk '{print $1}')" = "$expected_head_contract_sha"
test "$(sha256sum "$comparator_source" | awk '{print $1}')" = "$expected_comparator_sha"
test "$(sha256sum "$reader_source" | awk '{print $1}')" = "$expected_reader_sha"
test "$(sha256sum "$reference_root/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = "$expected_reference_comparison_sha"
grep -Fq "\"g0w0_sigc_manifest_sha256\": \"$expected_reference_g0w0_manifest_sha\"" \
  "$reference_root/head-only-iter1-vs-g0w0-sigc.json"
grep -Fq '"passed": false' "$reference_root/head-only-iter1-vs-g0w0-sigc.json"
test "$(find "$reference_g0w0" -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
grep -Fq 'libRPA finished successfully' "$reference_g0w0/librpa.stdout"
(
  cd "$build"
  sha256sum --check --quiet SHA256SUMS.txt
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_READER_HEAD_ABLATION_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$qsgw_run" "$tool_dir"
cp "$0" "$evidence/run-diagnostic-reader-head-ablation-2e446af2-v1.slurm"
cp "$patch_file" "$evidence/diagnostic-gate4-preserve-reader-head-2e446af2.patch"
cp "$build/SHA256SUMS.txt" "$evidence/build-SHA256SUMS.txt"
cp "$build/build-provenance.txt" "$evidence/build-provenance.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$evidence/"
cp "$input_root/OUTPUT_SHA256SUMS.txt" "$evidence/input-OUTPUT_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$evidence/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.head-only.contract" "$evidence/"
cp "$reference_root/head-only-iter1-vs-g0w0-sigc.json" \
  "$evidence/reference-unmodified-comparison.json"
cp "$comparator_source" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py"
cp "$reader_source" "$tool_dir/summarize_librpa_sigc_matrix_dumps.py"

cat >"$qsgw_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
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
replace_w_head = true
option_dielect_func = 4
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.head-only.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF
cp "$qsgw_run/librpa.in" "$evidence/librpa.in"

set +u
source /data/app/intel/oneapi-2024.2/compiler/2024.2/env/vars.sh \
  >"$evidence/compiler-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/mpi/2021.13/env/vars.sh \
  >"$evidence/mpi-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/mkl/2024.2/env/vars.sh \
  >"$evidence/mkl-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/tbb/2021.13/env/vars.sh \
  >"$evidence/tbb-vars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

{
  printf 'diagnostic=preserve_reader_head_for_scf_grid_iteration_one\n'
  printf 'acceptance_gate=false\n'
  printf 'base_source_commit=%s\n' "$expected_commit"
  printf 'diagnostic_patch_sha256=%s\n' "$expected_patch_sha"
  printf 'diagnostic_executable_sha256=%s\n' "$expected_exe_sha"
  printf 'reference_g0w0_job=2346965\n'
  printf 'reference_g0w0_manifest_sha256=%s\n' "$expected_reference_g0w0_manifest_sha"
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mixing=none\n'
  printf 'headwing=head_only_scf_grid\n'
  printf 'qsgw_iterations=0:1\n'
  printf 'mpi_ranks=%s\n' "$SLURM_NTASKS"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$evidence/PROVENANCE.txt"

(
  cd "$qsgw_run"
  mpirun -np "$SLURM_NTASKS" "$exe" \
    >librpa.stdout 2>librpa.stderr
)
test -f "$qsgw_run/qsgw_matrices.dat"
test -f "$qsgw_run/qsgw_iterations.dat"
test -f "$qsgw_run/qsgw_eigenvalues.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' "$qsgw_run/qsgw_iterations.dat")" = 1
grep -Fq 'libRPA finished successfully' "$qsgw_run/librpa.stdout"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$qsgw_run/qsgw_matrices.dat" \
  "$reference_g0w0" \
  "$evidence/ablation-vs-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.head-only.contract" \
  >"$evidence/comparator.stdout" \
  2>"$evidence/comparator.stderr"
grep -Fq "\"g0w0_sigc_manifest_sha256\": \"$expected_reference_g0w0_manifest_sha\"" \
  "$evidence/ablation-vs-g0w0-sigc.json"

if grep -Fq '"passed": true' "$evidence/ablation-vs-g0w0-sigc.json"; then
  hypothesis_supported=true
  touch "$evidence/HYPOTHESIS_SUPPORTED"
else
  hypothesis_supported=false
  touch "$evidence/HYPOTHESIS_NOT_SUPPORTED"
fi

cp "$qsgw_run/librpa.stdout" "$evidence/qsgw-librpa.stdout"
cp "$qsgw_run/librpa.stderr" "$evidence/qsgw-librpa.stderr"
cp "$qsgw_run/qsgw_iterations.dat" "$evidence/"
cp "$qsgw_run/qsgw_eigenvalues.dat" "$evidence/"
(
  cd "$qsgw_run"
  find . -type f -print0 | sort -z | xargs -0 sha256sum \
    >"$evidence/RUN_RAW_SHA256SUMS.txt"
)
(
  cd "$reference_g0w0"
  find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >"$evidence/REFERENCE_G0W0_SHA256SUMS.txt"
)

{
  printf 'hypothesis_supported=%s\n' "$hypothesis_supported"
  printf 'qsgw_matrix_trace_sha256=%s\n' "$(sha256sum "$qsgw_run/qsgw_matrices.dat" | awk '{print $1}')"
  printf 'comparison_sha256=%s\n' "$(sha256sum "$evidence/ablation-vs-g0w0-sigc.json" | awk '{print $1}')"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$evidence/PROVENANCE.txt"

(
  cd "$evidence"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$evidence/DIAGNOSTIC_COMPLETE"
printf 'QSGW_READER_HEAD_ABLATION=PASS hypothesis_supported=%s\n' "$hypothesis_supported"
