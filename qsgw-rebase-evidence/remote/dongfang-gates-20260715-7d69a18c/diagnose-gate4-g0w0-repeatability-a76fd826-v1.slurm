#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g0w0-repeat
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/g0w0-repeat-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/g0w0-repeat-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_run=$root/runs/gate4-head-only-limit-live-a76fd826-2345990
source_g0w0=$source_run/g0w0-head-only
source_qsgw=$source_run/head-only
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-a76fd826-v3
candidate_exe=$candidate_build/chi0_main.exe
target_root=$root/diagnostics/gate4-g0w0-repeatability-a76fd826-$SLURM_JOB_ID
repeat_run=$target_root/g0w0-repeat
tool_dir=$target_root/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

expected_candidate_exe_sha=8b508963f90027b5483fb2020e5feb1b5a72495dab4bef92f0e9b9be988d9b02
expected_candidate_build_manifest_sha=b7598f2b01531ffab9017db2ac39da2413a9a31ee76b52572ca1348bd4ac4d12
expected_candidate_build_provenance_sha=7ed01a1cb203cb28734d026a16c57bc1c6367a78a6b9a96e456c233df25040a8
expected_input_dataset_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_input_output_manifest_sha=b8c26ba28f64666c9f1efa3a840bc1edd1ed15cb84a6ef911990a0c1982de4dc
expected_input_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_input_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_gate4_script_sha=c4d5d9b6091e06b72c0fabb854df5278f3a1a8735269997c372f216e65e8edd2
expected_source_input_sha=41c115083cb2776e0b772571d33becf3968de6a55b39792034099b3830c030c5
expected_source_comparison_sha=fcbafa525de4515145560ee793947bd3e11d733d57edd732c82b43610f76872c
expected_source_trace_sha=f4fe1bdea67f86e9b141a02bcb2bee040d8ad093ff58ef4d5cbde34e7ae33c7c
expected_source_dump_manifest_sha=0489ef13d60ab7db841de38a84bc0eafe7a56d710cda80cf1f5d2098e000a2bc
expected_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf

test ! -e "$target_root"
test "${SLURM_NTASKS:-0}" -eq 4
test "${SLURM_CPUS_PER_TASK:-0}" -eq 12
test -x "$candidate_exe"
test -e "$candidate_build/COMPLETE"
test -e "$input_root/COMPLETE"
test ! -e "$source_run/COMPLETE"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_input_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.head-only.contract" | awk '{print $1}')" = \
  "$expected_input_contract_sha"
test "$(sha256sum "$source_run/gate4-head-only-limit-live-a76fd826-v3.slurm" | awk '{print $1}')" = \
  "$expected_gate4_script_sha"
test "$(sha256sum "$source_g0w0/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$source_run/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = \
  "$expected_source_comparison_sha"
test "$(sha256sum "$source_qsgw/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_source_trace_sha"
test "$(sha256sum "$source_run/tools/compare_qsgw_trace_g0w0_sigc.py" | awk '{print $1}')" = \
  "$expected_comparator_sha"
test "$(sha256sum "$source_run/tools/summarize_librpa_sigc_matrix_dumps.py" | awk '{print $1}')" = \
  "$expected_reader_sha"
test "$(sha256sum "$source_run/tools/validate_qsgw_trace_closure.py" | awk '{print $1}')" = \
  "$expected_closure_sha"
test "$(find "$source_g0w0" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
grep -Fq 'libRPA finished successfully' "$source_g0w0/librpa.stdout"
grep -Fq '"passed": false' \
  "$source_run/head-only-iter1-vs-g0w0-sigc.json"
grep -Fq '"relative_frobenius_difference": 2.174636782761653e-11' \
  "$source_run/head-only-iter1-vs-g0w0-sigc.json"
grep -Fq '"max_abs_difference_ha": 2.479842502490722e-10' \
  "$source_run/head-only-iter1-vs-g0w0-sigc.json"

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4_G0W0_REPEAT_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$repeat_run" "$tool_dir"
cp "$0" "$target_root/diagnose-gate4-g0w0-repeatability-a76fd826-v1.slurm"
cp "$source_g0w0/librpa.in" "$repeat_run/librpa.in"
cp "$source_run/tools/compare_qsgw_trace_g0w0_sigc.py" "$tool_dir/"
cp "$source_run/tools/summarize_librpa_sigc_matrix_dumps.py" "$tool_dir/"
cp "$source_run/tools/validate_qsgw_trace_closure.py" "$tool_dir/"
cp "$source_run/head-only-iter1-vs-g0w0-sigc.json" \
  "$target_root/source-head-only-vs-g0w0.json"
cp "$input_root/DATASET_SHA256SUMS.txt" "$target_root/"
cp "$input_root/OUTPUT_SHA256SUMS.txt" \
  "$target_root/input-OUTPUT_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$target_root/input-PROVENANCE.txt"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$target_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

{
  printf 'analysis=gate4_g0w0_repeatability\n'
  printf 'acceptance_gate=false\n'
  printf 'candidate_commit=a76fd826eed3929185ea47cfcb992075a72f2b77\n'
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'source_gate4_job=2345990\n'
  printf 'source_gate4_script_sha256=%s\n' "$expected_gate4_script_sha"
  printf 'source_g0w0_input_sha256=%s\n' "$expected_source_input_sha"
  printf 'source_g0w0_dump_manifest_sha256=%s\n' \
    "$expected_source_dump_manifest_sha"
  printf 'source_qsgw_trace_sha256=%s\n' "$expected_source_trace_sha"
  printf 'controlled_factor=repeat_run\n'
  printf 'mpi_ranks=%s\n' "$SLURM_NTASKS"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' \
    "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$target_root/PROVENANCE.txt"

(
  cd "$repeat_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
)
grep -Fq 'libRPA finished successfully' "$repeat_run/librpa.stdout"
test "$(find "$repeat_run" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384

build_dump_manifest() {
  local directory=$1
  local output=$2
  (
    cd "$directory"
    find . -maxdepth 1 -type f \
      -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' \
      -printf '%f\n' | LC_ALL=C sort | while IFS= read -r filename; do
        printf '%s %s\n' "$filename" \
          "$(sha256sum "$filename" | awk '{print $1}')"
      done
  ) >"$output"
}

build_dump_manifest "$source_g0w0" \
  "$target_root/source-g0w0-dump-manifest.txt"
build_dump_manifest "$repeat_run" \
  "$target_root/repeat-g0w0-dump-manifest.txt"
test "$(wc -l <"$target_root/source-g0w0-dump-manifest.txt")" -eq 384
test "$(wc -l <"$target_root/repeat-g0w0-dump-manifest.txt")" -eq 384
test "$(sha256sum "$target_root/source-g0w0-dump-manifest.txt" | awk '{print $1}')" = \
  "$expected_source_dump_manifest_sha"
cmp -s "$target_root/source-g0w0-dump-manifest.txt" \
  "$target_root/repeat-g0w0-dump-manifest.txt"

set +e
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$source_qsgw/qsgw_matrices.dat" \
  "$repeat_run" \
  "$target_root/head-only-vs-repeat-g0w0.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.head-only.contract" \
  >"$target_root/head-only-vs-repeat-g0w0.stdout" \
  2>"$target_root/head-only-vs-repeat-g0w0.stderr"
comparison_status=$?
set -e
test "$comparison_status" -eq 1
cmp -s "$source_run/head-only-iter1-vs-g0w0-sigc.json" \
  "$target_root/head-only-vs-repeat-g0w0.json"

cat >>"$target_root/PROVENANCE.txt" <<EOF
repeat_g0w0_dump_manifest_sha256=$expected_source_dump_manifest_sha
g0w0_repeatability=EXACT_384_OF_384
qsgw_vs_repeat_comparison_identical_to_source=PASS
diagnostic_conclusion=residual_is_not_g0w0_run_to_run_noise
product_source_changed=false
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target_root/COMPLETE"
printf 'QSGW_GATE4_G0W0_REPEATABILITY=PASS\n'
