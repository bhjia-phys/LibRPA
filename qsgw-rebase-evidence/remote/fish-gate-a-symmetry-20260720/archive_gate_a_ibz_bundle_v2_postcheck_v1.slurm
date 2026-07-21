#!/usr/bin/env bash
#SBATCH --job-name=qsgw-a0-ibz-archive
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-freeze-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-freeze-submit-20260720-v1/slurm-%j.err

set -euo pipefail

bundle=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526
archive=$bundle.tar.gz
evidence=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-archive-postcheck-20260720-v1-$SLURM_JOB_ID

record_failure() {
  local rc=$?
  if [[ -d ${evidence:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$evidence/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test ! -e "$archive"
test ! -e "$archive.sha256"
test ! -e "$evidence"
mkdir -p "$evidence"

(
  cd "$bundle"
  sha256sum --check --quiet SOURCE_DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(find "$bundle/dataset" -maxdepth 1 -type f | wc -l)" -eq 48
test "$(find "$bundle" -type l | wc -l)" -eq 0
test "$(find "$bundle" -perm /222 | wc -l)" -eq 0
grep -Fqx 'scf_kpoints=8' "$bundle/PROVENANCE.txt"
grep -Fqx 'full_bz_kpoints=64' "$bundle/PROVENANCE.txt"
grep -Fqx 'headwing=off' "$bundle/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$bundle/PROVENANCE.txt"
grep -Fqx 'band=off' "$bundle/PROVENANCE.txt"

tar -C "$(dirname "$bundle")" -czf "$archive" "$(basename "$bundle")"
sha256sum "$archive" >"$archive.sha256"
tar -tzf "$archive" >"$evidence/archive-members.txt"
test "$(grep -Ec '/dataset/[^/]+$' "$evidence/archive-members.txt")" -eq 48

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_ibz_bundle_archive_postcheck
acceptance=true
source_bundle=$bundle
source_bundle_complete=true
source_bundle_read_only=true
source_bundle_symlinks=0
source_dataset_files=48
failed_freeze_job=2378526
failed_freeze_job_scope=archive_transport_only_tar_zstd_unavailable
archive=$archive
archive_sha256=$(awk '{print $1}' "$archive.sha256")
archive_format=tar_gzip
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$evidence/archive-members.txt" "$evidence/PROVENANCE.txt" \
  >"$evidence/OUTPUT_SHA256SUMS.txt"
sha256sum --check --quiet "$evidence/OUTPUT_SHA256SUMS.txt"
touch "$evidence/COMPLETE"

echo GATE_A0_IBZ_ARCHIVE_POSTCHECK=PASS
cat "$evidence/PROVENANCE.txt"
cat "$archive.sha256"
exit 0
