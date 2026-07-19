#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-fail-post
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-failure-post-2e446af2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-failure-post-2e446af2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_job=2346965
source_run=$root/runs/gate4-head-only-limit-live-2e446af2-$source_job
target=$root/evidence/gate4-head-only-failed-2e446af2-$SLURM_JOB_ID
tool_dir=$target/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

expected_gate_script_sha=71f8d17a3efdf5cd251b9bc28cd96890f41283da3e04bf473fc460c837028228
expected_provenance_sha=72d7da92aa1655ae78c34361b75aa069aac93881c4d087716f8808d89760d968
expected_executable_sha=e4892316bd9b61f1203e0265fab9ebd38e8d2501582f89e6a056fa225894b2f1
expected_control_sha=bdf9677692e7b3fedb938a07f3126cb934af712f872565bb18e18052fa96e1ad
expected_control_canonical_sha=06d55e781d9a350151af36fe63692546fcd17e41fb3cce676e70bf82f40c71e6
expected_disabled_trace_sha=b28af06137ba7f641e1101cfdf7b33b889b0bdab589a11972f1a6840d1ad38b1
expected_head_trace_sha=cede567b676e2e88ad0585c221997228ae8f23fafcc0f95a443ecf75789e26ac
expected_disabled_fixed_sha=8567e46436883d28544eee2c33eafc01f0f445fa01d2b5136de8d1c78506d240
expected_head_fixed_sha=56b47a5b65255a00017c172794807e35387f77e25ec4103e7890494cee2b81cb
expected_disabled_closure_sha=0df119023574f75a106b74e54abdff87812640f213dd929c9aaf97e98a4fe61d
expected_head_closure_sha=1e5b002999089c203f870150bbb535f7dc3dbc3755ff3d81f01a56cebf099908
expected_strict_sigc_sha=ad8bf5864f56823d1f62e795f89493c0573755b25d5f9bd832d4944cdf71fab9
expected_component_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_pair_sha=0ec188b1e67a480878940ac3691a4be3888789737271b8cd8049befcd322e79f
expected_pair_test_sha=0d3f12f1db4d80d298487e31dd730336bd7f1f4eb0214135e4441adbee7cc216
expected_head_sha=e65d30e7444d868a71d4979b7899327488d92cb5d7ece2371707aead561a1e8a
expected_head_test_sha=1b9095eb0ba32a2fe76f15620fd98d95449d6049400565547a4590b6c3331f58

test ! -e "$target"
test ! -e "$source_run/COMPLETE"
test "$(sha256sum "$source_run/gate4-head-only-limit-live-2e446af2-v1.slurm" | awk '{print $1}')" = "$expected_gate_script_sha"
test "$(sha256sum "$source_run/PROVENANCE.txt" | awk '{print $1}')" = "$expected_provenance_sha"
test "$(sha256sum "$root/build-candidate-2e446af2-v2/chi0_main.exe" | awk '{print $1}')" = "$expected_executable_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.json" | awk '{print $1}')" = "$expected_control_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.canonical.json" | awk '{print $1}')" = "$expected_control_canonical_sha"
test "$(sha256sum "$source_run/disabled/qsgw_matrices.dat" | awk '{print $1}')" = "$expected_disabled_trace_sha"
test "$(sha256sum "$source_run/head-only/qsgw_matrices.dat" | awk '{print $1}')" = "$expected_head_trace_sha"
test "$(sha256sum "$source_run/disabled-fixed-basis.json" | awk '{print $1}')" = "$expected_disabled_fixed_sha"
test "$(sha256sum "$source_run/head-only-fixed-basis.json" | awk '{print $1}')" = "$expected_head_fixed_sha"
test "$(sha256sum "$source_run/disabled-closure.json" | awk '{print $1}')" = "$expected_disabled_closure_sha"
test "$(sha256sum "$source_run/head-only-closure.json" | awk '{print $1}')" = "$expected_head_closure_sha"
test "$(sha256sum "$source_run/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = "$expected_strict_sigc_sha"
test "$(sha256sum "$source_run/tools/compare_qsgw_component_traces_v3.py" | awk '{print $1}')" = "$expected_component_sha"
test "$(sha256sum "$source_run/tools/compare_qsgw_head_mode_pair.py" | awk '{print $1}')" = "$expected_pair_sha"
test "$(sha256sum "$source_run/tools/test_compare_qsgw_head_mode_pair.py" | awk '{print $1}')" = "$expected_pair_test_sha"
test "$(sha256sum "$source_run/tools/validate_qsgw_headwing_trace.py" | awk '{print $1}')" = "$expected_head_sha"
test "$(sha256sum "$source_run/tools/test_validate_qsgw_headwing_trace.py" | awk '{print $1}')" = "$expected_head_test_sha"
grep -Fq '"passed": true' "$source_run/disabled-fixed-basis.json"
grep -Fq '"passed": true' "$source_run/head-only-fixed-basis.json"
grep -Fq '"passed": true' "$source_run/disabled-closure.json"
grep -Fq '"passed": true' "$source_run/head-only-closure.json"
grep -Fq '"passed": false' "$source_run/head-only-iter1-vs-g0w0-sigc.json"

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4_HEAD_FAILURE_POST_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$tool_dir"
cp "$0" "$target/postprocess-gate4-head-failure-2e446af2-v1.slurm"
cp "$source_run/tools/compare_qsgw_component_traces_v3.py" "$tool_dir/"
cp "$source_run/tools/compare_qsgw_head_mode_pair.py" "$tool_dir/"
cp "$source_run/tools/test_compare_qsgw_head_mode_pair.py" "$tool_dir/"
cp "$source_run/tools/validate_qsgw_headwing_trace.py" "$tool_dir/"
cp "$source_run/tools/test_validate_qsgw_headwing_trace.py" "$tool_dir/"
cp "$source_run/CONTROLLED_PAIR.json" "$target/"
cp "$source_run/CONTROLLED_PAIR.canonical.json" "$target/"
cp "$source_run/PROVENANCE.txt" "$target/source-PROVENANCE.txt"
cp "$source_run/disabled-initial-state.json" "$target/"
cp "$source_run/head-only-initial-state.json" "$target/"
cp "$source_run/disabled-fixed-basis.json" "$target/"
cp "$source_run/head-only-fixed-basis.json" "$target/"
cp "$source_run/disabled-closure.json" "$target/"
cp "$source_run/head-only-closure.json" "$target/"
cp "$source_run/head-only-iter1-vs-g0w0-sigc.json" "$target/"
cp "$source_run/candidate-build-SHA256SUMS.txt" "$target/"
cp "$source_run/candidate-build-provenance.txt" "$target/"
cp "$source_run/DATASET_SHA256SUMS.txt" "$target/"
cp "$source_run/qsgw_input.disabled.contract" "$target/"
cp "$source_run/qsgw_input.head-only.contract" "$target/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_head_mode_pair.py" \
  >"$target/pair-unit-test.stdout" \
  2>"$target/pair-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_headwing_trace.py" \
  >"$target/head-unit-test.stdout" \
  2>"$target/head-unit-test.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_head_mode_pair.py" \
  "$source_run/disabled/qsgw_matrices.dat" \
  "$source_run/head-only/qsgw_matrices.dat" \
  "$source_run/CONTROLLED_PAIR.json" \
  "$target/head-mode-pair-comparison.json" \
  --initial-tolerance 1e-10 \
  --exx-tolerance 1e-10 \
  --head-effect-minimum 1e-12 \
  >"$target/head-mode-pair-comparator.stdout" \
  2>"$target/head-mode-pair-comparator.stderr"
grep -Fq '"passed": true' "$target/head-mode-pair-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_headwing_trace.py" \
  "$source_run/head-only/qsgw_matrices.dat" \
  "$target/head-live-validation.json" \
  --iterations 0:2 \
  --channel 0 \
  --invariant-tolerance 1e-10 \
  --initial-replay-tolerance 1e-10 \
  --head-live-change-minimum 1e-12 \
  >"$target/head-live-validator.stdout" \
  2>"$target/head-live-validator.stderr"
grep -Fq '"passed": true' "$target/head-live-validation.json"

sacct -j "$source_job" \
  --format=JobID,JobName%28,Partition,State,ExitCode,Elapsed,Start,End \
  -n -P >"$target/source-job.sacct"
(
  cd "$source_run"
  find . -type f -print0 | sort -z | xargs -0 sha256sum \
    >"$target/SOURCE_RAW_SHA256SUMS.txt"
)

{
  printf 'analysis=gate4_head_failure_postprocess\n'
  printf 'acceptance_gate=false\n'
  printf 'gate_status=FAIL\n'
  printf 'candidate_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418\n'
  printf 'candidate_executable_sha256=%s\n' "$expected_executable_sha"
  printf 'source_failed_gate4_job=%s\n' "$source_job"
  printf 'source_gate4_script_sha256=%s\n' "$expected_gate_script_sha"
  printf 'source_disabled_trace_sha256=%s\n' "$expected_disabled_trace_sha"
  printf 'source_head_trace_sha256=%s\n' "$expected_head_trace_sha"
  printf 'controlled_pair_validation=PASS\n'
  printf 'head_live_update_validation=PASS\n'
  printf 'fixed_basis_validation=PASS\n'
  printf 'closure_validation=PASS\n'
  printf 'strict_sigc_comparison=FAIL\n'
  printf 'strict_sigc_max_abs_difference_ha=2.475853458928731e-10\n'
  printf 'strict_sigc_max_abs_tolerance_ha=1e-10\n'
  printf 'strict_sigc_relative_frobenius_difference=2.1723851533680614e-11\n'
  printf 'strict_sigc_relative_frobenius_tolerance=1e-10\n'
  printf 'product_source_changed=false\n'
  printf 'execution_surface=dongfang_sbatch_compute_node\n'
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$target/PROVENANCE.txt"

printf '{\n  "gate": "gate4a-head-only-g0w0-limit-live-update",\n  "passed": false,\n  "reason": "strict_sigc_max_abs_difference_exceeds_tolerance",\n  "max_abs_difference_ha": 2.475853458928731e-10,\n  "max_abs_tolerance_ha": 1e-10,\n  "relative_frobenius_difference": 2.1723851533680614e-11,\n  "relative_frobenius_tolerance": 1e-10\n}\n' \
  >"$target/GATE_STATUS.json"

(
  cd "$target"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target/GATE_FAILED"
touch "$target/POSTPROCESS_COMPLETE"
printf 'QSGW_GATE4_HEAD_FAILURE_POSTPROCESS=PASS\n'
