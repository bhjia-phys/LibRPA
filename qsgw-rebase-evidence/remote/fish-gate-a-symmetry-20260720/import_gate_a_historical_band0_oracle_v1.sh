#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
archive=$base/librpa-qsgw-gate-a-historical-band0-oracle-20260720-v1-2380156.tar.gz
oracle=$base/librpa-qsgw-gate-a-historical-band0-oracle-20260720-v1-2380156
evidence=$base/librpa-qsgw-gate-a-historical-band0-oracle-import-20260720-v1

expected_archive_sha=5f150a46f01607075ec27c9c181d045c528c07f7d22e099f98f01e8d61b3ee79
expected_output_sha=8370446a8239754ff1c839e800e4a072fe2fe3699a7a5c8f83262183028b5986
expected_oracle_sha=1b9646b744d82de506fdfb0487644427e47ec0430929944430a9b9c586aba800
expected_provenance_sha=104ef00112c60ccbdb4d647ad6e7bd395851d46dc0a7db641544daad0c02d658

test -f "$archive"
test ! -e "$oracle"
test ! -e "$evidence"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_archive_sha"
mkdir -p "$evidence"
printf '%s\n' "$RUNNER_SHA256" >"$evidence/runner-sha256.txt"
tar -tzf "$archive" >"$evidence/archive-members.txt"
tar -C "$base" -xzf "$archive"

test -e "$oracle/COMPLETE"
test ! -e "$oracle/FAILED"
test "$(find "$oracle/oracle" -type f | wc -l)" -eq 36
test -z "$(find "$oracle" -type l -print -quit)"
test -z "$(find "$oracle" -perm /222 -print -quit)"
test "$(sha256sum "$oracle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_output_sha"
test "$(sha256sum "$oracle/ORACLE_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_oracle_sha"
test "$(sha256sum "$oracle/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_provenance_sha"
(
  cd "$oracle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$oracle/oracle"
  sha256sum --check --quiet "$oracle/ORACLE_SHA256SUMS.txt"
)

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_qsgw_band0_oracle_import_v1
acceptance=true
source_host=dongfang
destination_host=$(hostname -f 2>/dev/null || hostname)
archive=$archive
archive_sha256=$expected_archive_sha
oracle=$oracle
native_output_file_count=36
oracle_manifest_sha256=$expected_oracle_sha
source_provenance_sha256=$expected_provenance_sha
runner_sha256=$RUNNER_SHA256
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$evidence/archive-members.txt" "$evidence/PROVENANCE.txt" \
  "$evidence/runner-sha256.txt" >"$evidence/OUTPUT_SHA256SUMS.txt"
sha256sum --check --quiet "$evidence/OUTPUT_SHA256SUMS.txt"
touch "$evidence/COMPLETE"

echo GATE_A_HISTORICAL_BAND0_ORACLE_IMPORT_V1=PASS
cat "$evidence/PROVENANCE.txt"
