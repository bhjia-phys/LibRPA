#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
archive=$base/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526.tar.gz
bundle=$base/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526
evidence=$base/librpa-qsgw-gate-a-bundle-import-20260720-v1
expected_archive_sha=2f6b4ed5e1a88178058bce46c6dfe3444e8edb2a387716960f5a2d9012c9389a

test -f "$archive"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_archive_sha"
test ! -e "$bundle"
test ! -e "$evidence"
mkdir -p "$evidence"
printf '%s\n' "$RUNNER_SHA256" >"$evidence/runner-sha256.txt"
tar -tzf "$archive" >"$evidence/archive-members.txt"
test "$(grep -Ec '/dataset/[^/]+$' "$evidence/archive-members.txt")" -eq 48
tar -C "$base" -xzf "$archive"

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(find "$bundle/dataset" -maxdepth 1 -type f | wc -l)" -eq 48
test "$(find "$bundle" -type l | wc -l)" -eq 0
(
  cd "$bundle"
  sha256sum --check --quiet SOURCE_DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fqx 'scf_kpoints=8' "$bundle/PROVENANCE.txt"
grep -Fqx 'full_bz_kpoints=64' "$bundle/PROVENANCE.txt"
grep -Fqx 'kstar_multiplicities=1,8,4,6,24,12,3,6' "$bundle/PROVENANCE.txt"

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=gate_a0_ibz_bundle_import_on_fish
acceptance=true
source_host=dongfang
destination_host=$(hostname -f 2>/dev/null || hostname)
archive=$archive
archive_sha256=$expected_archive_sha
bundle=$bundle
dataset=$bundle/dataset
dataset_files=48
dataset_symlinks=0
dataset_manifest_sha256=$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')
qsgw_input_contract_sha256=$(sha256sum "$bundle/dataset/qsgw_input.contract" | awk '{print $1}')
qsgw_vxc_scf_manifest_sha256=$(sha256sum "$bundle/dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$evidence/archive-members.txt" "$evidence/PROVENANCE.txt" \
  "$evidence/runner-sha256.txt" >"$evidence/OUTPUT_SHA256SUMS.txt"
sha256sum --check --quiet "$evidence/OUTPUT_SHA256SUMS.txt"
touch "$evidence/COMPLETE"
echo GATE_A0_IBZ_BUNDLE_IMPORT=PASS
cat "$evidence/PROVENANCE.txt"
exit 0
