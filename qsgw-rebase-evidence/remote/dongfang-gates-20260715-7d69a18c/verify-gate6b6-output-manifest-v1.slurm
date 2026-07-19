#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g6b6-verify
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/verify-gate6b6-output-manifest-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/verify-gate6b6-output-manifest-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_job=2348987
source_run=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-$source_job
artifact=$root/artifacts/gate6b6-output-manifest-recheck-$source_job-$SLURM_JOB_ID

expected_source_output_manifest_sha=11758525d80a1025eac9dbe35a6794d52444960899bff90482a2f344c327e78f
expected_source_provenance_sha=35be3d726dffb010c92158ba0847aeed4b06b460bec9d27a5dac8045c66c8ddf
expected_source_closure_sha=ae15401792cfbf028bb1964bcf8299a481c0a819f92b9b77604341d6eee5fbd9
expected_source_script_sha=ed6068b4e40122bbf1dae494340872f751467ee0a26e5976e3d14d2a33bd7ade

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
mkdir -p "$artifact"
test -e "$source_run/COMPLETE"
test ! -e "$source_run/FAILED"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$source_run/OUTPUT_SHA256SUMS.txt $expected_source_output_manifest_sha
$source_run/PROVENANCE.txt $expected_source_provenance_sha
$source_run/old-closure-validation.json $expected_source_closure_sha
$source_run/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-v2.slurm $expected_source_script_sha
EOF

(
  cd "$source_run"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fq '"passed": true' "$source_run/old-closure-validation.json"

cp "$0" "$artifact/verify-gate6b6-output-manifest-v1.slurm"
cp "$source_run/OUTPUT_SHA256SUMS.txt" "$artifact/source-OUTPUT_SHA256SUMS.txt"
cp "$source_run/PROVENANCE.txt" "$artifact/source-PROVENANCE.txt"
cp "$source_run/old-closure-validation.json" "$artifact/old-closure-validation.json"
cat >"$artifact/PROVENANCE.txt" <<EOF
observer=verify_gate6b6_output_manifest
source_job=$source_job
source_run=$source_run
source_complete=true
source_failed=false
source_output_manifest_sha256=$expected_source_output_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
source_closure_sha256=$expected_source_closure_sha
source_script_sha256=$expected_source_script_sha
source_output_manifest_recheck=PASS
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
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
echo GATE6B6_OUTPUT_MANIFEST_RECHECK=PASS
