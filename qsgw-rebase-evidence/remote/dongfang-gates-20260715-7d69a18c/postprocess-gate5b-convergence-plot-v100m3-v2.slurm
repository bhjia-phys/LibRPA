#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g5b-plot-v2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate5b-convergence-plot-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate5b-convergence-plot-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate5b_root=$root/runs/gate5b-linear-beta02-miniter10-a25e4e43-2348654
plot_source=$root/jobs/plot_qsgw_linear_convergence-v1-49b68d4a.py
plot_test_source=$root/jobs/test_plot_qsgw_linear_convergence-v1-1cd68e59.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
artifact=$root/artifacts/gate5b-linear-convergence-2348654-$SLURM_JOB_ID
tool_dir=$artifact/tools

expected_gate5b_output_sha=822a6ff869f222ac4596585cbc49c2ff2bbfb6bd84fb8c064404223bba725d77
expected_gate5b_provenance_sha=4a51ff4d9a0bad374d62d56daf15915152e5f1ba2b6569a3b4a67429815ed11b
expected_trajectory_sha=34a6bc807637f9a4e23fa67b7c7157ddea8b5d472dc10fe4f6c8ca8e081992cf
expected_plot_sha=49b68d4afa0dce412b77eaa0153e6314101dda434f54bf8186b5a8e5b07a3597
expected_plot_test_sha=1cd68e59d4bf18c69e4025e24cd9590c2fe395f92de884c0274d7b7d8dec8d3b

record_failure() {
  local rc=$?
  if [[ -d ${artifact:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$artifact/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$artifact"
mkdir -p "$tool_dir"
test -e "$gate5b_root/COMPLETE"
while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  test "$observed" = "$expected"
done <<EOF
$gate5b_root/OUTPUT_SHA256SUMS.txt $expected_gate5b_output_sha
$gate5b_root/PROVENANCE.txt $expected_gate5b_provenance_sha
$gate5b_root/linear-residual-trajectory.csv $expected_trajectory_sha
$plot_source $expected_plot_sha
$plot_test_source $expected_plot_test_sha
EOF
grep -Fqx 'final_acceptance=true' "$gate5b_root/PROVENANCE.txt"

cp "$0" "$artifact/postprocess-gate5b-convergence-plot-v100m3-v2.slurm"
cp "$gate5b_root/linear-residual-trajectory.csv" "$artifact/"
cp "$gate5b_root/PROVENANCE.txt" "$artifact/gate5b-PROVENANCE.txt"
cp "$gate5b_root/OUTPUT_SHA256SUMS.txt" \
  "$artifact/gate5b-OUTPUT_SHA256SUMS.txt"
cp "$plot_source" "$tool_dir/plot_qsgw_linear_convergence_v1.py"
cp "$plot_test_source" "$tool_dir/test_plot_qsgw_linear_convergence_v1.py"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_plot_qsgw_linear_convergence_v1.py" \
  >"$artifact/plot-unit-test.stdout" \
  2>"$artifact/plot-unit-test.stderr"
"$python" -B "$tool_dir/plot_qsgw_linear_convergence_v1.py" \
  "$artifact/linear-residual-trajectory.csv" \
  "$artifact/qsgw-linear-beta02-miniter10-convergence.svg" \
  "$artifact/qsgw-linear-beta02-miniter10-convergence.json" \
  >"$artifact/plot-observer.stdout" \
  2>"$artifact/plot-observer.stderr"
grep -Fq '"passed": true' \
  "$artifact/qsgw-linear-beta02-miniter10-convergence.json"
grep -Fq '<svg xmlns=' \
  "$artifact/qsgw-linear-beta02-miniter10-convergence.svg"

cat >"$artifact/PROVENANCE.txt" <<EOF
artifact=gate5b-linear-beta02-miniter10-convergence
source_job=2348654
source_output_manifest_sha256=$expected_gate5b_output_sha
source_provenance_sha256=$expected_gate5b_provenance_sha
source_trajectory_sha256=$expected_trajectory_sha
plot_observer_sha256=$expected_plot_sha
plot_test_sha256=$expected_plot_test_sha
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
result=PASS
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
(
  cd "$artifact"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$artifact/COMPLETE"
echo GATE5B_CONVERGENCE_PLOT=PASS
