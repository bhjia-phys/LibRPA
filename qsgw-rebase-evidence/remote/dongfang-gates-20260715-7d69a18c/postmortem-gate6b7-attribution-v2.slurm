#!/usr/bin/env bash
# Attribution postmortem for rejected Gate 6B7 (job 2350992).
# The Gate 6B7 postmortem (evidence gate6b7-miniter2-postmortem-v2-2351093)
# showed that the old v40 Hartree-on oracle and the new a25e4e43 candidate
# already diverge at iterations 0:1 in the base G0W0 quantities
# (exx max_abs 23.95 Ha, sigma_c_iw 2.0 Ha, vc 0.396 Ha), while
# h0/vxc_dft/occupation/delta_vh agree exactly.
# This observer compares the two LEGACY v4 traces on the same frozen input:
#   side A: Gate 6A2 old v36 trace (Hartree OFF, job 2348804)
#   side B: Gate 6B6/B7 oracle v40 trace (Hartree ON, job 2348987)
# restricted to iterations 0:1, where the Sigma frequency grids still agree.
# Interpretation (with Gate 6B5 v36==v40 Hartree-off and Gate 6A2 old==new):
#   - components pass  -> old side stable under the Hartree switch; the
#                         divergence is new-side (merge bug in Hartree mode).
#   - components fail  -> the v40 full-reader Hartree env changed the old
#                         base G0W0 quantities; the oracle is not a valid
#                         same-input reference for Gate 6B7 as designed.
# benchmark_acceptance=false: COMPLETE only means the diagnostics executed.
#SBATCH --job-name=qsgw-g6b7-attr-v2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-attribution-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-attribution-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate6a2_run=$root/runs/gate6a2-full-abfs-baseline-miniter2-a25e4e43-2348804
oracle_run=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-2348987
failed_run=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-2350992
prior_postmortem=$root/evidence/gate6b7-miniter2-postmortem-v2-2351093
old_off_trace=$gate6a2_run/old/qsgw_oracle_matrices.dat
old_on_trace=$oracle_run/old/qsgw_oracle_matrices.dat
same_comparator=$root/jobs/compare_qsgw_same_contract_traces-v1-ae9778a5.py
base_comparator=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
evidence=$root/evidence/gate6b7-attribution-v2-$SLURM_JOB_ID

expected_same_comparator_sha=ae9778a561f42e0ca613eaf571549f696eb82f585a8daea0b3fa3f8503b4a9a8
expected_base_comparator_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f

test ! -e "$evidence"
test -e "$gate6a2_run/COMPLETE"
test ! -e "$gate6a2_run/FAILED"
test -e "$oracle_run/COMPLETE"
test ! -e "$oracle_run/FAILED"
test -e "$failed_run/FAILED"
test -e "$prior_postmortem/COMPLETE"
test -s "$old_off_trace"
test -s "$old_on_trace"
test -s "$same_comparator"
test "$(sha256sum "$same_comparator" | awk '{print $1}')" = \
  "$expected_same_comparator_sha"
test "$(sha256sum "$base_comparator" | awk '{print $1}')" = \
  "$expected_base_comparator_sha"
grep -Fqx 'hartree=off' "$gate6a2_run/PROVENANCE.txt"
grep -Fqx 'final_acceptance=true' "$gate6a2_run/PROVENANCE.txt"
grep -Fqx 'old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01' \
  "$gate6a2_run/PROVENANCE.txt"
grep -Fqx \
  'input_dataset_manifest_sha256=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1' \
  "$gate6a2_run/PROVENANCE.txt"
grep -Fqx \
  'input_dataset_manifest_sha256=6e183ba7e91eb03311836bf5d2e6f0bf2c209338679fae1fc9c9cc627423bba1' \
  "$failed_run/PROVENANCE.txt"

mkdir -p "$evidence/tools"
cp "$0" "$evidence/postmortem-gate6b7-attribution-v2.slurm"
cp "$same_comparator" "$evidence/tools/compare_qsgw_same_contract_traces.py"
cp "$base_comparator" "$evidence/tools/compare_qsgw_component_traces_v3.py"

sha256sum \
  "$old_off_trace" \
  "$old_on_trace" \
  "$gate6a2_run/old/librpa.in" \
  "$oracle_run/old/librpa.in" \
  "$same_comparator" \
  "$base_comparator" \
  >"$evidence/SOURCE_SHA256SUMS.txt"

grep '^#' "$old_off_trace" >"$evidence/old-off-contract-header.txt"
grep '^#' "$old_on_trace" >"$evidence/old-on-contract-header.txt"

# delta_vh layout handling: the comparator requires identical row layouts.
# The Hartree-on oracle writes delta_vh rows; verify they are exactly zero
# at iterations 0:1, then drop them from a comparison copy if the
# Hartree-off trace has no such rows. The transform is recorded.
off_dvh_count=$(awk '$1 ~ /^[0-9]+$/ && $1 <= 1 && $3 == "delta_vh" {c++} END {print c+0}' \
  "$old_off_trace")
on_dvh_stats=$(awk '$1 ~ /^[0-9]+$/ && $1 <= 1 && $3 == "delta_vh" {r=$10; im=$11; if (r<0) r=-r; if (im<0) im=-im; if (r>m) m=r; if (im>m) m=im; c++} END {printf "%d %.17e", c+0, m+0.0}' \
  "$old_on_trace")
printf 'old_off_delta_vh_rows_iter0_1=%s\n' "$off_dvh_count" \
  >"$evidence/delta-vh-layout.txt"
printf 'old_on_delta_vh_rows_and_maxabs_iter0_1=%s\n' "$on_dvh_stats" \
  >>"$evidence/delta-vh-layout.txt"
test "$(printf '%s' "$on_dvh_stats" | awk '{print $2}')" = "0.00000000000000000e+00"

comparison_old_on=$old_on_trace
if [[ "$off_dvh_count" -eq 0 ]]; then
  on_dvh_rows=$(printf '%s' "$on_dvh_stats" | awk '{print $1}')
  if [[ "$on_dvh_rows" -gt 0 ]]; then
    awk '$3 != "delta_vh"' "$old_on_trace" \
      >"$evidence/old-on-trace-no-delta-vh.dat"
    printf 'transform=dropped %s identically_zero delta_vh rows from oracle copy\n' \
      "$on_dvh_rows" >>"$evidence/delta-vh-layout.txt"
    comparison_old_on=$evidence/old-on-trace-no-delta-vh.dat
    sha256sum "$comparison_old_on" \
      >"$evidence/TRANSFORMED_TRACE_SHA256SUMS.txt"
  fi
fi

set +e
PYTHONPATH="$evidence/tools" "$python" -B \
  "$evidence/tools/compare_qsgw_same_contract_traces.py" \
  "$old_off_trace" \
  "$comparison_old_on" \
  "$evidence/old-off-vs-old-on-comparison.json" \
  --iterations 0:1 \
  --channel 0 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --state-tolerance 1e-10 \
  >"$evidence/attribution-comparator.stdout" \
  2>"$evidence/attribution-comparator.stderr"
attribution_rc=$?
set -e
test -s "$evidence/old-off-vs-old-on-comparison.json"

"$python" -B - "$evidence/old-off-vs-old-on-comparison.json" \
  >"$evidence/attribution-component-summary.txt" <<'PYEOF'
import json, sys
report = json.load(open(sys.argv[1]))
print("overall_passed=", report.get("passed"))
contract = report.get("contract") or {}
print("contract_passed=", contract.get("passed"))
for item in (contract.get("differences") or [])[:10]:
    print("contract_difference=", item)
components = report.get("components") or {}
for name in sorted(components):
    metric = components[name]
    print(
        "component=%s max_abs=%.6e rel_frobenius=%.6e blocks=%s passed=%s"
        % (
            name,
            metric.get("max_abs_diff_ha", float("nan")),
            metric.get("max_relative_frobenius", float("nan")),
            metric.get("block_count"),
            metric.get("passed"),
        )
    )
state = report.get("state") or {}
print("state_passed=", state.get("passed"))
for key in sorted(state):
    if isinstance(state[key], (int, float)):
        print("state_%s=%.6e" % (key, state[key]))
PYEOF

{
  printf 'analysis=gate6b7_attribution_old_off_vs_old_on_v2\n'
  printf 'benchmark_acceptance=false\n'
  printf 'diagnostic_only=true\n'
  printf 'side_a=gate6a2_old_v36_hartree_off_job_2348804\n'
  printf 'side_b=gate6b6_oracle_v40_hartree_on_job_2348987\n'
  printf 'selection=iterations_0_1\n'
  printf 'attribution_rc=%s\n' "$attribution_rc"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$evidence/PROVENANCE.txt"

(
  cd "$evidence"
  find . -type f ! -name 'SHA256SUMS.txt' -print0 | sort -z | \
    xargs -0 sha256sum >SHA256SUMS.txt
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
echo GATE6B7_ATTRIBUTION_V2=DONE
