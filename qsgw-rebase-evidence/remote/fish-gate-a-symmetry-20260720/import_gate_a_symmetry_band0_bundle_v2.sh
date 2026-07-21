#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
failed=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v1
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v2
dataset=$bundle/dataset
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1

expected_failed_runner=57964ad8337a8f614dfc9673592e6ecf6234eeaaf06b013585c2d1337eb1eb17
expected_parent_dataset_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
expected_asset_dataset_sha=bcdd94d197babc504b355374c778693fadcfd92009b79153ea0f32f6a469668c
expected_symrot_abf_sha=8e9ffde672debf4fcf3b6efedc08c56c70bb69a5a66f3777bad6e0bda9325ae8
expected_grid_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_band_generator_sha=93802dbdd2ead01d2cdda57406645736197c319d8c71b506aebfa7bb6b8578ca
expected_band_test_sha=10dbad783c41db9fb6acb257879e7b00ece111f139fb0836a9cbc322cbb02e8f

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${bundle:-/nonexistent} ]]; then
    chmod -R u+w "$bundle" 2>/dev/null || true
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$bundle/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test -d "$failed/dataset"
test -e "$failed/FAILED"
test ! -e "$failed/COMPLETE"
test "$(cat "$failed/FAILED")" = $'failed_utc=2026-07-20T15:14:47Z\nexit_code=1'
test "$(cat "$failed/provenance/import-runner-sha256.txt")" = \
  "$expected_failed_runner"
test "$(find "$failed/dataset" -type f | wc -l)" -eq 1330
test ! -e "$failed/dataset/symrot_abf_k.txt"
test ! -e "$failed/dataset/qsgw_input.disabled.contract"
test ! -e "$failed/dataset/qsgw_input.band.contract"
test ! -e "$failed/dataset/qsgw_vxc_band.v2.manifest"
test "$(sha256sum "$failed/provenance/parent-v3-DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_parent_dataset_sha"
test "$(sha256sum "$failed/provenance/band-assets-DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_asset_dataset_sha"
test "$(sha256sum "$failed/dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_grid_contract_sha"
test "$(sha256sum "$staging/symrot_abf_k.txt" | awk '{print $1}')" = \
  "$expected_symrot_abf_sha"
test "$(sha256sum "$staging/prepare_qsgw_band_contract_v1.py" | awk '{print $1}')" = \
  "$expected_band_generator_sha"
test "$(sha256sum "$staging/test_prepare_qsgw_band_contract_v1.py" | awk '{print $1}')" = \
  "$expected_band_test_sha"
test ! -e "$bundle"
test -z "$(find "$failed" -type l -print -quit)"
(
  cd "$failed"
  sha256sum --check --quiet provenance/parent-v3-DATASET_SHA256SUMS.txt
  sha256sum --check --quiet provenance/band-assets-DATASET_SHA256SUMS.txt
)

TMPDIR=/tmp python3 -B "$staging/test_prepare_qsgw_band_contract_v1.py" \
  >"$staging/band-generator-unit-test-v2.stdout" \
  2>"$staging/band-generator-unit-test-v2.stderr"

cp -a --reflink=auto "$failed" "$bundle"
chmod -R u+w "$bundle"
rm -f "$bundle/FAILED" "$bundle/COMPLETE" "$bundle/OUTPUT_SHA256SUMS.txt"
cp "$failed/FAILED" "$bundle/provenance/failed-v1-FAILED.txt"
printf '%s\n' "$RUNNER_SHA256" >"$bundle/provenance/import-v2-runner-sha256.txt"
cp "$staging/band-generator-unit-test-v2.stdout" "$bundle/provenance/"
cp "$staging/band-generator-unit-test-v2.stderr" "$bundle/provenance/"

install -m 0644 "$staging/symrot_abf_k.txt" "$dataset/symrot_abf_k.txt"
cp "$dataset/qsgw_input.contract" "$dataset/qsgw_input.disabled.contract"
python3 -B "$bundle/provenance/prepare_qsgw_band_contract_v1.py" \
  --dataset "$dataset" \
  >"$bundle/provenance/band-contract-generator.stdout" \
  2>"$bundle/provenance/band-contract-generator.stderr"
test ! -s "$bundle/provenance/band-contract-generator.stderr"

test "$(find "$dataset" -type f | wc -l)" -eq 1334
test "$(find "$dataset" -maxdepth 1 -name 'band_KS_eigenvalue_k_*.txt' -type f | wc -l)" -eq 201
test "$(find "$dataset" -maxdepth 1 -name 'band_KS_eigenvector_k_*.txt' -type f | wc -l)" -eq 201
test "$(find "$dataset" -maxdepth 1 -name 'band_vxck*_nao.txt' -type f | wc -l)" -eq 201
test "$(find "$dataset/vxc_band" -maxdepth 1 -name 'vxck*s1_nao.txt' -type f | wc -l)" -eq 201
test "$(find "$dataset/pyatb_librpa_df" -type f | wc -l)" -eq 67
test -z "$(find "$bundle" -type l -print -quit)"
cmp "$dataset/qsgw_input.contract" "$dataset/qsgw_input.disabled.contract"
grep -Fqx 'n_band_kpoints 201' "$dataset/qsgw_input.band.contract"
grep -Fqx 'band_update operator_fourier' "$dataset/qsgw_input.band.contract"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.band.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.band.contract"
test "$(grep -Ec '^band_mf0_eigenvalues [0-9a-f]{64} band_KS_eigenvalue_k_[0-9]{5}\.txt$' "$dataset/qsgw_input.band.contract")" -eq 201
test "$(grep -Ec '^band_mf0_wavefunctions [0-9a-f]{64} band_KS_eigenvector_k_[0-9]{5}\.txt$' "$dataset/qsgw_input.band.contract")" -eq 201
test "$(grep -Ec '^1 [0-9]+ ' "$dataset/qsgw_vxc_band.v2.manifest")" -eq 201

(
  cd "$bundle"
  find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

dataset_manifest_sha=$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')
grid_contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
band_contract_sha=$(sha256sum "$dataset/qsgw_input.band.contract" | awk '{print $1}')
band_vxc_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_band.v2.manifest" | awk '{print $1}')
cat >"$bundle/PROVENANCE.txt" <<EOF
gate=gate_a0_symmetry_band0_combined_bundle_v2
acceptance_scope=legacy_candidate_same_physical_input
recovery_source=$failed
recovery_source_state=failed_permission_order_after_verified_parent_and_asset_copy
recovery_source_runner_sha256=$expected_failed_runner
recovery_source_parent_dataset_manifest_sha256=$expected_parent_dataset_sha
recovery_source_asset_dataset_manifest_sha256=$expected_asset_dataset_sha
recovery_action=copy_verified_bytes_then_restore_directory_write_before_augmentation
copy_mode=physical_copy_no_symlinks_reflink_allowed
symrot_abf_sha256=$expected_symrot_abf_sha
dataset_file_count=1334
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
band_kpoints=201
pyatb_files=67
use_shrink_abfs=true
symmetry=on_ibz_8_to_full_bz_64
headwing_contract=off
hartree_contract=off
band_contract=operator_fourier
dataset_manifest_sha256=$dataset_manifest_sha
grid_contract_sha256=$grid_contract_sha
band_contract_sha256=$band_contract_sha
band_vxc_manifest_sha256=$band_vxc_manifest_sha
band_generator_sha256=$expected_band_generator_sha
band_generator_test_sha256=$expected_band_test_sha
runner_sha256=$RUNNER_SHA256
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$bundle"
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
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

echo GATE_A_SYMMETRY_BAND0_COMBINED_BUNDLE_V2=PASS
cat "$bundle/PROVENANCE.txt"
sha256sum "$bundle/DATASET_SHA256SUMS.txt" \
  "$bundle/OUTPUT_SHA256SUMS.txt" \
  "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.band.contract" \
  "$dataset/qsgw_vxc_band.v2.manifest"
