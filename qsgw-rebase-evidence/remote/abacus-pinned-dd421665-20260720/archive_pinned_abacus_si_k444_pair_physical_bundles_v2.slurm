#!/usr/bin/env bash
#SBATCH --job-name=si444-pair-archive
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-archive-submit-20260723-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-archive-submit-20260723-v1/slurm-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?this archive task must run under Slurm}"
: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the committed archive runner}"

base=/data/home/df_iopcas_bhj/ai-runs
bundle=$base/abacus-pinned-dd421665-si-k444-pair-physical-bundles-20260723-v2
stage_dir=$base/abacus-pinned-dd421665-si-k444-pair-physical-bundles-archive-submit-20260723-v1
root=$base/abacus-pinned-dd421665-si-k444-pair-physical-bundles-archive-20260723-v1
bundle_name=$(basename "$bundle")
archive=$root/$bundle_name.tar.zst
runner_name=archive_pinned_abacus_si_k444_pair_physical_bundles_v2.slurm

expected_bundle_output_manifest_sha=159d433d29938f65c637a88b9f4fa3e7974c191267c79a0cd343168bc4b2ddb8
expected_pair_validation_sha=bd6a8c65732518136d5679ceaf29b6454277f832aaf2ac56bce6860375a487da
expected_symmetry_dataset_manifest_sha=c0aea2f49b4844cd6804c99aa09ceba891229c3c8a4b0830ee056aacd8b6b05a
expected_fullbz_dataset_manifest_sha=ba401b751eb465b957fad2270878022c13dfb922257376b9b2a9faef793873f4

run_succeeded=0
record_exit()
{
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\nslurm_job_id=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$SLURM_JOB_ID" >"$root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

sha_of()
{
  sha256sum "$1" | awk '{print $1}'
}

test ! -e "$root"
test -d "$stage_dir"
test -f "$stage_dir/$runner_name"
test -f "$stage_dir/STAGED_SHA256SUMS.txt"
(
  cd "$stage_dir"
  sha256sum --check --quiet STAGED_SHA256SUMS.txt
)

test -e "$bundle/PAIR_COMPLETE"
test ! -e "$bundle/FAILED"
test -e "$bundle/symmetry/COMPLETE"
test -e "$bundle/fullbz/COMPLETE"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
test "$(sha_of "$bundle/OUTPUT_SHA256SUMS.txt")" = \
  "$expected_bundle_output_manifest_sha"
test "$(sha_of "$bundle/PAIR_VALIDATION.json")" = \
  "$expected_pair_validation_sha"
test "$(sha_of "$bundle/symmetry/DATASET_SHA256SUMS.txt")" = \
  "$expected_symmetry_dataset_manifest_sha"
test "$(sha_of "$bundle/fullbz/DATASET_SHA256SUMS.txt")" = \
  "$expected_fullbz_dataset_manifest_sha"
grep -Fq '"status": "PASS"' "$bundle/PAIR_VALIDATION.json"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd symmetry
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd ../fullbz
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$root"
cp "$stage_dir/$runner_name" "$root/$runner_name"
cp "$stage_dir/STAGED_SHA256SUMS.txt" "$root/STAGED_SHA256SUMS.txt"
tar -C "$(dirname "$bundle")" -cf - "$bundle_name" | \
  zstd -T"${SLURM_CPUS_PER_TASK:-8}" -3 -o "$archive"
(
  cd "$root"
  sha256sum "$(basename "$archive")" >ARCHIVE_SHA256SUMS.txt
  sha256sum --check --quiet ARCHIVE_SHA256SUMS.txt
)
zstd -dc "$archive" | tar -tf - >"$root/archive-members.txt"
python3 -B - "$root/archive-members.txt" "$bundle_name" <<'PY'
import pathlib
import sys

members = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
prefix = sys.argv[2] + "/"
assert members
assert all(member == sys.argv[2] or member.startswith(prefix) for member in members)
assert all(not member.startswith("/") and ".." not in pathlib.PurePosixPath(member).parts for member in members)
required = {
    f"{prefix}PAIR_COMPLETE",
    f"{prefix}PAIR_VALIDATION.json",
    f"{prefix}OUTPUT_SHA256SUMS.txt",
    f"{prefix}symmetry/COMPLETE",
    f"{prefix}fullbz/COMPLETE",
}
assert required.issubset(set(members))
PY
zstd -dc "$archive" | \
  tar -xOf - "$bundle_name/OUTPUT_SHA256SUMS.txt" | \
  sha256sum >"$root/source_output_manifest_from_archive.sha256"
test "$(awk '{print $1}' "$root/source_output_manifest_from_archive.sha256")" = \
  "$expected_bundle_output_manifest_sha"

cat >"$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_pair_physical_bundles_archive_v1
acceptance_scope=transport_only
runner_commit=$RUNNER_COMMIT
source_bundle=$bundle
source_bundle_output_manifest_sha256=$expected_bundle_output_manifest_sha
source_pair_validation_sha256=$expected_pair_validation_sha
source_bundle_complete=true
source_bundle_read_only=true
source_bundle_symlinks=0
archive=$archive
archive_sha256=$(awk '{print $1}' "$root/ARCHIVE_SHA256SUMS.txt")
archive_format=tar_zstd
zstd_level=3
zstd_threads=${SLURM_CPUS_PER_TASK:-8}
slurm_job_id=$SLURM_JOB_ID
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
(
  cd "$root"
  find . -type f ! -path './OUTPUT_SHA256SUMS.txt' ! -name ARCHIVE_COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/ARCHIVE_COMPLETE"
find "$root" -type f -exec chmod a-w {} +
find "$root" -depth -type d -exec chmod a-w {} +
run_succeeded=1
trap - EXIT

test -e "$root/ARCHIVE_COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -type l -print -quit)"
test -z "$(find "$root" -perm /222 -print -quit)"
echo PINNED_ABACUS_SI_K444_PAIR_PHYSICAL_BUNDLES_ARCHIVE_V1=PASS
cat "$root/PROVENANCE.txt"
