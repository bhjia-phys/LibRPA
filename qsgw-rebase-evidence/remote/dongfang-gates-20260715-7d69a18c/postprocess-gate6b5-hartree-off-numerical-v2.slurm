#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g6b5-numcmp2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b5-numerical-postprocess-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b5-numerical-postprocess-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_run=$root/runs/gate6b5-legacy-full-reader-hartree-off-ab-mpi4-omp12-miniter2-2348914
artifact=$root/artifacts/gate6b5-hartree-off-numerical-postprocess-2348914-$SLURM_JOB_ID
tools=$artifact/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
base_source=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
base_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v6-hartree-576f860d.py
same_source=$root/jobs/compare_qsgw_same_contract_traces-v1-ae9778a5.py
same_test_source=$root/jobs/test_compare_qsgw_same_contract_traces-v1-fb20c06d.py

expected_base_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_base_test_sha=576f860db9a00e1653f4503e6845716b24d35ca425fa392eb96703faa1833065
expected_same_sha=ae9778a561f42e0ca613eaf571549f696eb82f585a8daea0b3fa3f8503b4a9a8
expected_same_test_sha=fb20c06d3b95206da5a3e575de9e26dda939885d700b93bdc1f3c978d8321f4c

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
test -f "$source_run/FAILED"
test ! -e "$source_run/COMPLETE"
test -s "$source_run/old/qsgw_oracle_matrices.dat"
test -s "$source_run/adapter/qsgw_oracle_matrices.dat"
mkdir -p "$tools"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$base_source $expected_base_sha
$base_test_source $expected_base_test_sha
$same_source $expected_same_sha
$same_test_source $expected_same_test_sha
EOF

cp "$base_source" "$tools/compare_qsgw_component_traces_v3.py"
cp "$base_test_source" "$tools/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$same_source" "$tools/compare_qsgw_same_contract_traces_v1.py"
cp "$same_test_source" "$tools/test_compare_qsgw_same_contract_traces_v1.py"
cp "$0" "$artifact/postprocess-gate6b5-hartree-off-numerical-v2.slurm"
cp "$source_run/FAILED" "$artifact/source-run-FAILED.txt"

sha256sum \
  "$source_run/old/qsgw_oracle_matrices.dat" \
  "$source_run/adapter/qsgw_oracle_matrices.dat" \
  "$source_run/old/homo_lumo_vs_iterations.dat" \
  "$source_run/adapter/homo_lumo_vs_iterations.dat" \
  >"$artifact/INPUT_SHA256SUMS.txt"

PYTHONPATH="$tools" "$python" -B \
  "$tools/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$artifact/base-comparator-unit-test.stdout" \
  2>"$artifact/base-comparator-unit-test.stderr"
PYTHONPATH="$tools" "$python" -B \
  "$tools/test_compare_qsgw_same_contract_traces_v1.py" \
  >"$artifact/same-contract-unit-test.stdout" \
  2>"$artifact/same-contract-unit-test.stderr"

PYTHONPATH="$tools" "$python" -B \
  "$tools/compare_qsgw_same_contract_traces_v1.py" \
  "$source_run/old/qsgw_oracle_matrices.dat" \
  "$source_run/adapter/qsgw_oracle_matrices.dat" \
  "$artifact/comparison.json" \
  --iterations 0:2 --channel 0 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance-ha 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --state-tolerance 1e-10 \
  >"$artifact/comparator.stdout" \
  2>"$artifact/comparator.stderr"
grep -Fq '"passed": true' "$artifact/comparison.json"

cat >"$artifact/PROVENANCE.txt" <<EOF
observer=gate6b5_hartree_off_numerical_postprocess
source_run=$source_run
source_job=2348914
source_result=FAILED_bitwise_cmp_only
adapter_isolation=numerical_thresholds
benchmark_acceptance=false
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_frobenius_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
state_tolerance=1e-10
mpi_ranks=4
omp_threads=12
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
result=PASS
EOF

(
  cd "$artifact"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$artifact/COMPLETE"
echo GATE6B5_HARTREE_OFF_NUMERICAL_POSTPROCESS=PASS
