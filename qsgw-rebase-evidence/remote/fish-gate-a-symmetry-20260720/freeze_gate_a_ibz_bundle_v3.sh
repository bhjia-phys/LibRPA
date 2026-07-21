#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

parent=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v2-2378526
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
staging=/tmp/librpa-qsgw-gate-a-sks-v3-20260720
producer=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1

expected_parent_output_sha=9b013776b7fb57a57ac1d407a4d57c2d91f02a09ee128e9789e25caa39a06e9b
expected_parent_dataset_sha=fd75c8adae425c03df1bf8e73f8b21dc97302f3fc1a4d8a65e81640c5acdc540
expected_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179

declare -A expected_sks_sha=(
  [sks1k1_nao.txt]=1d6ef5e400589211114f04c643a2cc5a2d353e5945cd70a870517d40081514ad
  [sks1k2_nao.txt]=d53c7d4b68f46e61992f686815e6aeb7d4a5d22b3741c5701e2f5dc3f87d535f
  [sks1k3_nao.txt]=4d8a2a472e4d2d22636df05b972203b30c73316119a0b4ad84da28438e47160b
  [sks1k4_nao.txt]=f119307b5ea2565f3c0af605f730e92042082d6920cbf4472148e173517f1f8a
  [sks1k5_nao.txt]=ea610bed6c67498ab5988e35e53d8c47ad856e6d4b226721b0bb389ba8003e1d
  [sks1k6_nao.txt]=e23ebeb20c065296ab39c395cb55b62ef609e202c7f0ad8ba4169ba06aae22e5
  [sks1k7_nao.txt]=5d8f659fc91b297c77b19bb6f7e3611f1ea7575663df7ea01171b88b2d30737f
  [sks1k8_nao.txt]=dbc1cac6272bcecba02ba996f8b650ba098c223c5daf231722e96663d0136b64
)

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${bundle:-/nonexistent} ]]; then
    chmod u+w "$bundle" 2>/dev/null || true
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$bundle/FAILED"
    chmod a-w "$bundle/FAILED" "$bundle" 2>/dev/null || true
  fi
  exit "$rc"
}
trap record_failure ERR

test -d "$parent/dataset"
test -e "$parent/COMPLETE"
test ! -e "$parent/FAILED"
test -d "$staging"
test ! -e "$bundle"
test "$(sha256sum "$parent/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_parent_output_sha"
test "$(sha256sum "$parent/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_parent_dataset_sha"
test "$(sha256sum "$parent/dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_contract_sha"
test "$(sha256sum "$parent/dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$expected_vxc_manifest_sha"
test "$(find "$parent/dataset" -type f | wc -l)" -eq 48
test -z "$(find "$parent" -type l -print -quit)"
test -z "$(find "$parent" -perm /222 -print -quit)"
(
  cd "$parent"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test "$(find "$staging" -maxdepth 1 -type f -name 'sks1k*_nao.txt' | wc -l)" -eq 8
test "$(find "$staging" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 8
test -z "$(find "$staging" -mindepth 1 -maxdepth 1 ! -type f -print -quit)"
for file in "${!expected_sks_sha[@]}"; do
  test -f "$staging/$file"
  test "$(sha256sum "$staging/$file" | awk '{print $1}')" = \
    "${expected_sks_sha[$file]}"
done

cp -a "$parent" "$bundle"
chmod -R u+w "$bundle"
rm -f "$bundle/COMPLETE" "$bundle/FAILED" "$bundle/OUTPUT_SHA256SUMS.txt"
cp "$parent/PROVENANCE.txt" "$bundle/provenance/parent-v2-PROVENANCE.txt"
cp "$parent/DATASET_SHA256SUMS.txt" \
  "$bundle/provenance/parent-v2-DATASET_SHA256SUMS.txt"
cp "$parent/OUTPUT_SHA256SUMS.txt" \
  "$bundle/provenance/parent-v2-OUTPUT_SHA256SUMS.txt"
printf '%s\n' "$RUNNER_SHA256" >"$bundle/provenance/freeze-v3-runner-sha256.txt"

for index in {1..8}; do
  install -m 0644 "$staging/sks1k${index}_nao.txt" \
    "$bundle/dataset/sks1k${index}_nao.txt"
done

(
  cd "$bundle"
  find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

dataset_manifest_sha=$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')
contract_sha=$(sha256sum "$bundle/dataset/qsgw_input.contract" | awk '{print $1}')
vxc_manifest_sha=$(sha256sum "$bundle/dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
test "$contract_sha" = "$expected_contract_sha"
test "$vxc_manifest_sha" = "$expected_vxc_manifest_sha"
test "$(find "$bundle/dataset" -type f | wc -l)" -eq 56
test -z "$(find "$bundle" -type l -print -quit)"

cat >"$bundle/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_ibz_bundle_freeze_v3
acceptance_scope=historical_legacy_candidate_parity_only
parent_bundle=$parent
parent_bundle_output_manifest_sha256=$expected_parent_output_sha
parent_dataset_manifest_sha256=$expected_parent_dataset_sha
copy_mode=full_physical_cp_a_no_symlinks
augmentation=legacy_overlap_matrices_sks1k1_through_sks1k8
augmentation_source_host=dongfang
augmentation_source_root=$producer
augmentation_file_count=8
dataset_file_count=56
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_input_contract_sha256=$contract_sha
qsgw_vxc_scf_manifest_sha256=$vxc_manifest_sha
producer=abacus_historical_dirty_checkout_not_pinned
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
kstar_multiplicities=1,8,4,6,24,12,3,6
use_shrink_abfs=true
headwing=off
hartree=off
band=off
runner_sha256=$RUNNER_SHA256
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

for index in {1..8}; do
  file="sks1k${index}_nao.txt"
  printf '%s  %s\n' "${expected_sks_sha[$file]}" "dataset/$file"
done >"$bundle/provenance/sks-source-SHA256SUMS.txt"
(
  cd "$bundle"
  sha256sum --check --quiet provenance/sks-source-SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$bundle/COMPLETE"
find "$bundle" -type f -exec chmod a-w {} +
find "$bundle" -depth -type d -exec chmod a-w {} +

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -z "$(find "$bundle" -type l -print -quit)"
test -z "$(find "$bundle" -perm /222 -print -quit)"
test "$(find "$bundle/dataset" -type f | wc -l)" -eq 56
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

echo GATE_A_IBZ_BUNDLE_V3=PASS
cat "$bundle/PROVENANCE.txt"
sha256sum "$bundle/DATASET_SHA256SUMS.txt" \
  "$bundle/OUTPUT_SHA256SUMS.txt" \
  "$bundle/dataset/qsgw_input.contract" \
  "$bundle/dataset/qsgw_vxc_scf.manifest"
