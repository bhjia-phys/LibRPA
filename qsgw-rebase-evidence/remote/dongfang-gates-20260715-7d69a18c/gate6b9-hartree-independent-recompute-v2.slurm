#!/usr/bin/env bash
# Gate 6B9 v2: Hartree delta-density independent-recomputation acceptance.
# v2 supersedes v1 (job 2352094, rejected): v1 compared gauge-dependent
# rotation_u/wfc_spinor0 rows element-wise across two builds; eigensolver
# gauge freedom within (near-)degenerate subspaces makes that comparison
# physically naive. In v2:
#   - Acceptance B binds rerun to frozen through gauge-INVARIANT
#     components only (rotation_u/wfc_spinor0 excluded explicitly).
#   - Acceptance B2 validates the rerun trace's rotation/WFC internally
#     via the v4 closure and fixed-basis observers (U unitarity, U^T H U
#     diagonal, C = C0 U), which is the gauge-correct acceptance.
#   - Top-level JSON "passed" is asserted via Python, not substring grep.
# Acceptance (fail-closed):
#   A. observer unit tests (recompute 29, v5-compare 16)
#   B. rerun vs frozen v5 comparison, iterations 0:1, max-abs <= 1e-12 Ha,
#      excluding rotation_u/wfc_spinor0 (documented gauge freedom)
#   B2. v4 closure + fixed-basis on the rerun trace, iterations 0:2
#   C. independent Python recomputation on call_001 (iteration-1 delta_vh
#      vanishes) and call_002 (iteration-2 end-to-end <= 1e-8 Ha)
# Diagnostic (recorded, not gate-blocking):
#   D. full-window v5 comparison at 0:2 without exclusions + occupation
#      flip analysis (QSGW feedback sensitivity near E_F)
#SBATCH --job-name=qsgw-g6b9-vh-recompute
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b9-hartree-recompute-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b9-hartree-recompute-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
rerun=$root/runs/gate6c1-hartree-dump-rerun-ad9fd6ed-miniter2-2352021
frozen_run=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-2350992
input_dir=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832/dataset
recompute=$root/jobs/recompute_qsgw_hartree_delta-v1-bdab2d8f.py
recompute_test=$root/jobs/test_recompute_qsgw_hartree_delta-v1-cb40a80c.py
v5cmp=$root/jobs/compare_qsgw_current_v5_traces-v1-d29960a4.py
v5cmp_test=$root/jobs/test_compare_qsgw_current_v5_traces-v1-40a9a07b.py
closure_v4=$root/jobs/validate_qsgw_trace_closure-v4-860e8bcd.py
comparator_module=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
fixed_basis=$root/jobs/validate_qsgw_fixed_basis.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate6b9-hartree-independent-recompute-v2-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_recompute_sha=bdab2d8f5deef8a6940af99c1236045ac8ea2bc97379cb4c5ca235690e34a3e4
expected_recompute_test_sha=cb40a80c1071d0bd16f0b77397315c6a1f05ff642a1f6f97947a173cf0d07ef5
expected_v5cmp_sha=d29960a40711f4f2f1da180f119e584315770789f22c9087d9c993c378670bb1
expected_v5cmp_test_sha=40a9a07b752447dac080299e38fb3ba9d71d675ebfc2f72a5188332a714a23c1
expected_closure_v4_sha=860e8bcd3c641ebab510d5b41ece95992bb960611855525772a48d58559ca3d5
expected_comparator_module_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_fixed_basis_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_rerun_output_sha=b2bd39fff282e540df0d1a052a97e6683198cdf969f1a2c193713703ed9636ff
expected_rerun_provenance_sha=cb91af27ef4baf0f36beee673a399817acc7d2f237f3a2f08951f4d2be8c93a2
expected_rerun_trace_sha=a028cee91a21f5812258beb1d3ed3955ec3ad045cfa9aee58ef0c27e189e391b
expected_rerun_eigenvalues_sha=d18df450c21336797873ea785d50061bda2b6c06e43bbb83ada01efc1e891e2e
expected_rerun_iterations_sha=ea561e4d053c8e618e71b10bb25ead19d871f5fbc6ff37e449bb19230b733960
expected_frozen_trace_sha=f27fb95c7849fd75539d94b142c4b556cf9b74f4ab11e1b9c69492f9a289527a
expected_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6

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
mkdir -p "$tool_dir"
test -e "$rerun/DUMP_COMPLETE"
test ! -e "$rerun/FAILED"
test -e "$frozen_run/FAILED"
test ! -e "$frozen_run/COMPLETE"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$recompute $expected_recompute_sha
$recompute_test $expected_recompute_test_sha
$v5cmp $expected_v5cmp_sha
$v5cmp_test $expected_v5cmp_test_sha
$closure_v4 $expected_closure_v4_sha
$comparator_module $expected_comparator_module_sha
$fixed_basis $expected_fixed_basis_sha
$rerun/OUTPUT_SHA256SUMS.txt $expected_rerun_output_sha
$rerun/PROVENANCE.txt $expected_rerun_provenance_sha
$rerun/current/qsgw_matrices.dat $expected_rerun_trace_sha
$rerun/current/qsgw_eigenvalues.dat $expected_rerun_eigenvalues_sha
$rerun/current/qsgw_iterations.dat $expected_rerun_iterations_sha
$frozen_run/current/qsgw_matrices.dat $expected_frozen_trace_sha
$input_dir/qsgw_input.hartree.contract $expected_hartree_contract_sha
EOF
(
  cd "$rerun"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

cp "$0" "$run_root/gate6b9-hartree-independent-recompute-v2.slurm"
cp "$recompute" "$tool_dir/recompute_qsgw_hartree_delta_v1.py"
cp "$recompute_test" "$tool_dir/test_recompute_qsgw_hartree_delta_v1.py"
cp "$v5cmp" "$tool_dir/compare_qsgw_current_v5_traces_v1.py"
cp "$v5cmp_test" "$tool_dir/test_compare_qsgw_current_v5_traces_v1.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_v4" "$tool_dir/validate_qsgw_trace_closure_v4.py"
cp "$comparator_module" "$tool_dir/compare_qsgw_component_traces.py"
cp "$fixed_basis" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$rerun/PROVENANCE.txt" "$run_root/rerun-PROVENANCE.txt"
cp "$rerun/OUTPUT_SHA256SUMS.txt" "$run_root/rerun-OUTPUT_SHA256SUMS.txt"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate6b9-hartree-independent-recompute-v2
supersedes=gate6b9-v1-job-2352094-rejected-gauge-naive-comparison
final_acceptance=false_pending_observers
result=PENDING
rerun_job=2352021
rerun_commit=ad9fd6ed65552b5ad2b3e80483aaf56538d15805
frozen_trace_job=2350992
frozen_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
acceptance_b_excluded_components=rotation_u,wfc_spinor0
acceptance_b_exclusion_reason=eigensolver_gauge_freedom_within_degenerate_subspaces
gauge_coverage=closure_v4_and_fixed_basis_on_rerun_trace
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# ---- Acceptance A: observer unit tests ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_recompute_qsgw_hartree_delta_v1.py" \
  >"$run_root/recompute-unit-test.stdout" \
  2>"$run_root/recompute-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_current_v5_traces_v1.py" \
  >"$run_root/v5cmp-unit-test.stdout" \
  2>"$run_root/v5cmp-unit-test.stderr"

# ---- Acceptance B: rerun vs frozen, iterations 0:1, gauge-invariant ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_current_v5_traces_v1.py" \
  "$frozen_run/current/qsgw_matrices.dat" \
  "$rerun/current/qsgw_matrices.dat" \
  "$run_root/rerun-vs-frozen-iterations0-1.json" \
  --iterations 0:1 --channel 0 \
  --matrix-max-abs-tolerance-ha 1e-12 \
  --matrix-relative-tolerance 1e-8 \
  --exclude-components rotation_u wfc_spinor0 \
  >"$run_root/rerun-vs-frozen-iterations0-1.stdout" \
  2>"$run_root/rerun-vs-frozen-iterations0-1.stderr"
assert_passed "$run_root/rerun-vs-frozen-iterations0-1.json"

paste \
  <(awk 'NF && $1 !~ /^#/ && $1 <= 1' "$frozen_run/current/qsgw_eigenvalues.dat") \
  <(awk 'NF && $1 !~ /^#/ && $1 <= 1' "$rerun/current/qsgw_eigenvalues.dat") \
  >"$run_root/eigenvalues-0-1-paste.txt"
awk '{
    for (i = 1; i <= 9; i++) if ($i != $(i + 9)) {
      d = $i - $(i + 9); if (d < 0) d = -d;
      if (i <= 4 && d > 0) exit 2;
      if (d > max) max = d;
    }
  }
  END { printf "eigenvalue_trace_0_1_max_abs_diff=%.17e\n", max + 0.0;
        if (max > 1e-12) exit 1 }' \
  "$run_root/eigenvalues-0-1-paste.txt" \
  >"$run_root/eigenvalues-0-1-diff.txt"

# ---- Acceptance B2: rerun trace closure + fixed basis ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure_v4.py" \
  "$rerun/current/qsgw_matrices.dat" \
  "$run_root/rerun-closure-validation.json" \
  --iterations 0:2 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$run_root/rerun-closure-validator.stdout" \
  2>"$run_root/rerun-closure-validator.stderr"
assert_passed "$run_root/rerun-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$rerun/current/qsgw_matrices.dat" \
  "$rerun/current/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/rerun-fixed-basis-validation.json" \
  --iterations 0:2 --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$run_root/rerun-fixed-basis-validator.stdout" \
  2>"$run_root/rerun-fixed-basis-validator.stderr"
assert_passed "$run_root/rerun-fixed-basis-validation.json"

# ---- Acceptance C: independent recomputation on both dump calls ----
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/recompute_qsgw_hartree_delta_v1.py" \
  --input-dir "$input_dir" \
  --dump-call "$rerun/hartree-dump/call_001" \
  --trace "$rerun/current/qsgw_matrices.dat" \
  --iterations 0:2 \
  --output "$run_root/recompute-call-001.json" \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  >"$run_root/recompute-call-001.stdout" \
  2>"$run_root/recompute-call-001.stderr"
assert_passed "$run_root/recompute-call-001.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/recompute_qsgw_hartree_delta_v1.py" \
  --input-dir "$input_dir" \
  --dump-call "$rerun/hartree-dump/call_002" \
  --trace "$rerun/current/qsgw_matrices.dat" \
  --iterations 0:2 \
  --output "$run_root/recompute-call-002.json" \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  >"$run_root/recompute-call-002.stdout" \
  2>"$run_root/recompute-call-002.stderr"
assert_passed "$run_root/recompute-call-002.json"

# ---- Diagnostic D: full-window comparison + occupation flip analysis ----
set +e
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_current_v5_traces_v1.py" \
  "$frozen_run/current/qsgw_matrices.dat" \
  "$rerun/current/qsgw_matrices.dat" \
  "$run_root/rerun-vs-frozen-iterations0-2.json" \
  --iterations 0:2 --channel 0 \
  --matrix-max-abs-tolerance-ha 1e-12 \
  --matrix-relative-tolerance 1e-8 \
  >"$run_root/rerun-vs-frozen-iterations0-2.stdout" \
  2>"$run_root/rerun-vs-frozen-iterations0-2.stderr"
diagnostic_v5_rc=$?
set -e

awk '$3 == "occupation" && $1 >= 1 {print $1, $4, $5, $8, $10}' \
  "$frozen_run/current/qsgw_matrices.dat" | sort -n -k1,1 -k2,2 -k3,3 -k4,4 \
  >"$run_root/frozen-occupation-rows.txt"
awk '$3 == "occupation" && $1 >= 1 {print $1, $4, $5, $8, $10}' \
  "$rerun/current/qsgw_matrices.dat" | sort -n -k1,1 -k2,2 -k3,3 -k4,4 \
  >"$run_root/rerun-occupation-rows.txt"
paste "$run_root/frozen-occupation-rows.txt" \
  "$run_root/rerun-occupation-rows.txt" | \
awk '{
    if ($1 != $6 || $2 != $7 || $3 != $8 || $4 != $9) { bad++; next }
    d = $5 - $10; if (d < 0) d = -d;
    if (d > 0) { flips++; printf "occupation_diff iter=%d spin=%d k=%d band=%d frozen=%.17e rerun=%.17e\n", $1, $2, $3, $4, $5, $10 }
  }
  END {
    printf "occupation_key_mismatches=%d\n", bad + 0
    printf "occupation_differing_rows=%d\n", flips + 0
  }' \
  >"$run_root/occupation-flip-analysis.txt"

{
  printf 'final_acceptance=true\nresult=PASS\n'
  printf 'diagnostic_v5_iterations0_2_rc=%s\n' "$diagnostic_v5_rc"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE6B9_HARTREE_INDEPENDENT_RECOMPUTE_V2=PASS
