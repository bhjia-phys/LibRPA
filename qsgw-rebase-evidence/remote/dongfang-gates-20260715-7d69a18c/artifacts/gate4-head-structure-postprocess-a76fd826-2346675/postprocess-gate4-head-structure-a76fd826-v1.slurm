#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-post
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-structure-post-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-structure-post-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_run=$root/runs/gate4-head-only-limit-live-a76fd826-2345990
target=$root/evidence/gate4-head-structure-postprocess-a76fd826-$SLURM_JOB_ID
tool_dir=$target/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

expected_gate_script_sha=c4d5d9b6091e06b72c0fabb854df5278f3a1a8735269997c372f216e65e8edd2
expected_control_sha=3b359aaa46d3fd5bb97711baa089e677d9fa0f7b65d0f3f1f2c7ac0f44697d39
expected_control_canonical_sha=5d6877fa4f283c2f0209a8de8cea7bd91558d2c1c9a8031a9e5ed942f1b9b29c
expected_disabled_trace_sha=989b228bb5e7c5fc0662578588206c2c9a6872d867550add5460c85e944523a3
expected_head_trace_sha=f4fe1bdea67f86e9b141a02bcb2bee040d8ad093ff58ef4d5cbde34e7ae33c7c
expected_disabled_fixed_sha=bf359f46642777449ec8e737f797e60109a261c32bdb717c42b4e5902aef7497
expected_head_fixed_sha=dafbe296dc6485144fba198f55d1760a061cc7e34c99f9b1a9637116bc0a3108
expected_disabled_closure_sha=0df119023574f75a106b74e54abdff87812640f213dd929c9aaf97e98a4fe61d
expected_head_closure_sha=1e5b002999089c203f870150bbb535f7dc3dbc3755ff3d81f01a56cebf099908
expected_strict_sigc_sha=fcbafa525de4515145560ee793947bd3e11d733d57edd732c82b43610f76872c
expected_component_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_pair_sha=0ec188b1e67a480878940ac3691a4be3888789737271b8cd8049befcd322e79f
expected_pair_test_sha=0d3f12f1db4d80d298487e31dd730336bd7f1f4eb0214135e4441adbee7cc216
expected_head_sha=e65d30e7444d868a71d4979b7899327488d92cb5d7ece2371707aead561a1e8a
expected_head_test_sha=1b9095eb0ba32a2fe76f15620fd98d95449d6049400565547a4590b6c3331f58

test ! -e "$target"
test ! -e "$source_run/COMPLETE"
test "$(sha256sum "$source_run/gate4-head-only-limit-live-a76fd826-v3.slurm" | awk '{print $1}')" = \
  "$expected_gate_script_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.json" | awk '{print $1}')" = \
  "$expected_control_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.canonical.json" | awk '{print $1}')" = \
  "$expected_control_canonical_sha"
test "$(sha256sum "$source_run/disabled/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_disabled_trace_sha"
test "$(sha256sum "$source_run/head-only/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_head_trace_sha"
test "$(sha256sum "$source_run/disabled-fixed-basis.json" | awk '{print $1}')" = \
  "$expected_disabled_fixed_sha"
test "$(sha256sum "$source_run/head-only-fixed-basis.json" | awk '{print $1}')" = \
  "$expected_head_fixed_sha"
test "$(sha256sum "$source_run/disabled-closure.json" | awk '{print $1}')" = \
  "$expected_disabled_closure_sha"
test "$(sha256sum "$source_run/head-only-closure.json" | awk '{print $1}')" = \
  "$expected_head_closure_sha"
test "$(sha256sum "$source_run/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = \
  "$expected_strict_sigc_sha"
test "$(sha256sum "$source_run/tools/compare_qsgw_component_traces_v3.py" | awk '{print $1}')" = \
  "$expected_component_sha"
test "$(sha256sum "$source_run/tools/compare_qsgw_head_mode_pair.py" | awk '{print $1}')" = \
  "$expected_pair_sha"
test "$(sha256sum "$source_run/tools/test_compare_qsgw_head_mode_pair.py" | awk '{print $1}')" = \
  "$expected_pair_test_sha"
test "$(sha256sum "$source_run/tools/validate_qsgw_headwing_trace.py" | awk '{print $1}')" = \
  "$expected_head_sha"
test "$(sha256sum "$source_run/tools/test_validate_qsgw_headwing_trace.py" | awk '{print $1}')" = \
  "$expected_head_test_sha"
grep -Fq '"passed": true' "$source_run/disabled-fixed-basis.json"
grep -Fq '"passed": true' "$source_run/head-only-fixed-basis.json"
grep -Fq '"passed": true' "$source_run/disabled-closure.json"
grep -Fq '"passed": true' "$source_run/head-only-closure.json"
grep -Fq '"passed": false' "$source_run/head-only-iter1-vs-g0w0-sigc.json"

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4_HEAD_STRUCTURE_POST_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$tool_dir"
cp "$0" "$target/postprocess-gate4-head-structure-a76fd826-v1.slurm"
cp "$source_run/tools/compare_qsgw_component_traces_v3.py" "$tool_dir/"
cp "$source_run/tools/compare_qsgw_head_mode_pair.py" "$tool_dir/"
cp "$source_run/tools/test_compare_qsgw_head_mode_pair.py" "$tool_dir/"
cp "$source_run/tools/validate_qsgw_headwing_trace.py" "$tool_dir/"
cp "$source_run/tools/test_validate_qsgw_headwing_trace.py" "$tool_dir/"
cp "$source_run/CONTROLLED_PAIR.json" "$target/"
cp "$source_run/CONTROLLED_PAIR.canonical.json" "$target/"
cp "$source_run/disabled-fixed-basis.json" "$target/"
cp "$source_run/head-only-fixed-basis.json" "$target/"
cp "$source_run/disabled-closure.json" "$target/"
cp "$source_run/head-only-closure.json" "$target/"
cp "$source_run/head-only-iter1-vs-g0w0-sigc.json" "$target/"

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

cat >"$target/PROVENANCE.txt" <<EOF
analysis=gate4_head_structural_postprocess
acceptance_gate=false
candidate_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
source_failed_gate4_job=2345990
source_gate4_script_sha256=$expected_gate_script_sha
source_disabled_trace_sha256=$expected_disabled_trace_sha
source_head_trace_sha256=$expected_head_trace_sha
controlled_pair_validation=PASS
head_live_update_validation=PASS
fixed_basis_validation=PASS
closure_validation=PASS
strict_sigc_comparison=FAIL
strict_sigc_max_abs_difference_ha=2.479842502490722e-10
strict_sigc_relative_frobenius_difference=2.174636782761653e-11
product_source_changed=false
execution_surface=dongfang_sbatch_compute_node
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target/COMPLETE"
printf 'QSGW_GATE4_HEAD_STRUCTURE_POSTPROCESS=PASS\n'
