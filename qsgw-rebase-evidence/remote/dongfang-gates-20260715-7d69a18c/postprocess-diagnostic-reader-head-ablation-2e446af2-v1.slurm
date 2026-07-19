#!/usr/bin/env bash
#SBATCH --job-name=qsgw-reader-head-post
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/post-reader-head-ablation-2e446af2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/post-reader-head-ablation-2e446af2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_job=2347529
source_run=$root/runs/diagnostic-reader-head-ablation-2e446af2-$source_job
qsgw_run=$source_run/qsgw-head-only
source_evidence=$root/evidence/diagnostic-reader-head-ablation-2e446af2-$source_job
reference_root=$root/runs/gate4-head-only-limit-live-2e446af2-2346965
reference_g0w0=$reference_root/g0w0-head-only
target=$root/evidence/diagnostic-reader-head-ablation-postprocess-2e446af2-$SLURM_JOB_ID
tool_dir=$target/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
comparator_source=$root/jobs/compare_qsgw_trace_g0w0_sigc.py
reader_source=$root/jobs/summarize_librpa_sigc_matrix_dumps.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
component_source=$root/jobs/compare_qsgw_component_traces-v4-c3daf072.py

expected_run_script_sha=043a91cd9846a56aaa423b189517eefd9c0cd7885ebaffdc61a81a060bac533a
expected_exe_sha=e654b032b2b0b23254dfdd3bedbd483b927e3553a0497b2ceeb4f728c605b8d3
expected_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_component_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_reference_comparison_sha=ad8bf5864f56823d1f62e795f89493c0573755b25d5f9bd832d4944cdf71fab9
expected_reference_g0w0_manifest_sha=a2bc8e8837c96cfd17f71a96382292f9da5cc72b3712bbfdef2f2411d00952ba

test ! -e "$target"
test -f "$qsgw_run/qsgw_matrices.dat"
test -f "$qsgw_run/qsgw_iterations.dat"
test -f "$qsgw_run/qsgw_eigenvalues.dat"
test -f "$source_evidence/PROVENANCE.txt"
test ! -e "$source_evidence/DIAGNOSTIC_COMPLETE"
test "$(sha256sum "$source_evidence/run-diagnostic-reader-head-ablation-2e446af2-v1.slurm" | awk '{print $1}')" = "$expected_run_script_sha"
test "$(sha256sum "$root/build-diagnostic-reader-head-2e446af2-v5/chi0_main.exe" | awk '{print $1}')" = "$expected_exe_sha"
test "$(sha256sum "$comparator_source" | awk '{print $1}')" = "$expected_comparator_sha"
test "$(sha256sum "$reader_source" | awk '{print $1}')" = "$expected_reader_sha"
test "$(sha256sum "$closure_source" | awk '{print $1}')" = "$expected_closure_sha"
test "$(sha256sum "$component_source" | awk '{print $1}')" = "$expected_component_sha"
test "$(sha256sum "$reference_root/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = "$expected_reference_comparison_sha"
grep -Fq "\"g0w0_sigc_manifest_sha256\": \"$expected_reference_g0w0_manifest_sha\"" \
  "$reference_root/head-only-iter1-vs-g0w0-sigc.json"
test "$(find "$reference_g0w0" -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' "$qsgw_run/qsgw_iterations.dat")" = 1
grep -Fq 'libRPA finished successfully' "$qsgw_run/librpa.stdout"

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_READER_HEAD_ABLATION_POST_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$tool_dir"
cp "$0" "$target/postprocess-diagnostic-reader-head-ablation-2e446af2-v1.slurm"
cp "$comparator_source" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py"
cp "$reader_source" "$tool_dir/summarize_librpa_sigc_matrix_dumps.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$component_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$source_evidence/PROVENANCE.txt" "$target/source-PROVENANCE.txt"
cp "$source_evidence/librpa.in" "$target/"
cp "$source_evidence/build-SHA256SUMS.txt" "$target/"
cp "$source_evidence/build-provenance.txt" "$target/"
cp "$source_evidence/DATASET_SHA256SUMS.txt" "$target/"
cp "$source_evidence/input-OUTPUT_SHA256SUMS.txt" "$target/"
cp "$source_evidence/input-PROVENANCE.txt" "$target/"
cp "$source_evidence/qsgw_input.head-only.contract" "$target/"
cp "$source_evidence/reference-unmodified-comparison.json" "$target/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$qsgw_run/qsgw_matrices.dat" \
  "$reference_g0w0" \
  "$target/ablation-vs-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$root/inputs/si-k444-headwing-scf-synced-v2-2345960/dataset/qsgw_input.head-only.contract" \
  >"$target/comparator.stdout" \
  2>"$target/comparator.stderr"
grep -Fq "\"g0w0_sigc_manifest_sha256\": \"$expected_reference_g0w0_manifest_sha\"" \
  "$target/ablation-vs-g0w0-sigc.json"

if grep -Fq '"passed": true' "$target/ablation-vs-g0w0-sigc.json"; then
  hypothesis_supported=true
  touch "$target/HYPOTHESIS_SUPPORTED"
else
  hypothesis_supported=false
  touch "$target/HYPOTHESIS_NOT_SUPPORTED"
fi

cp "$qsgw_run/librpa.stdout" "$target/qsgw-librpa.stdout"
cp "$qsgw_run/librpa.stderr" "$target/qsgw-librpa.stderr"
cp "$qsgw_run/qsgw_iterations.dat" "$target/"
cp "$qsgw_run/qsgw_eigenvalues.dat" "$target/"
(
  cd "$qsgw_run"
  find . -type f -print0 | sort -z | xargs -0 sha256sum \
    >"$target/RUN_RAW_SHA256SUMS.txt"
)
(
  cd "$reference_g0w0"
  find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >"$target/REFERENCE_G0W0_SHA256SUMS.txt"
)
sacct -j "$source_job" \
  --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,Start,End \
  -n -P >"$target/source-job.sacct"

{
  printf 'diagnostic=preserve_reader_head_for_scf_grid_iteration_one\n'
  printf 'acceptance_gate=false\n'
  printf 'source_numerical_job=%s\n' "$source_job"
  printf 'source_numerical_step_state=COMPLETED\n'
  printf 'source_job_state=FAILED_observer_dependency_only\n'
  printf 'diagnostic_executable_sha256=%s\n' "$expected_exe_sha"
  printf 'reference_g0w0_job=2346965\n'
  printf 'reference_g0w0_manifest_sha256=%s\n' "$expected_reference_g0w0_manifest_sha"
  printf 'strict_max_abs_tolerance_ha=1e-10\n'
  printf 'strict_relative_frobenius_tolerance=1e-10\n'
  printf 'hypothesis_supported=%s\n' "$hypothesis_supported"
  printf 'qsgw_matrix_trace_sha256=%s\n' "$(sha256sum "$qsgw_run/qsgw_matrices.dat" | awk '{print $1}')"
  printf 'comparison_sha256=%s\n' "$(sha256sum "$target/ablation-vs-g0w0-sigc.json" | awk '{print $1}')"
  printf 'execution_surface=dongfang_sbatch_compute_node\n'
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$target/PROVENANCE.txt"

(
  cd "$target"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target/POSTPROCESS_COMPLETE"
printf 'QSGW_READER_HEAD_ABLATION_POSTPROCESS=PASS hypothesis_supported=%s\n' "$hypothesis_supported"
