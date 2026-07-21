#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

base=/home/bhj/ai-runs
parent=$base/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
asset_archive=$base/librpa-qsgw-gate-a-band0-assets-20260720-v1-2379975.tar.gz
asset=$base/librpa-qsgw-gate-a-band0-assets-20260720-v1-2379975
staging=/tmp/librpa-qsgw-gate-a-band0-import-20260720-v1
bundle=$base/librpa-qsgw-gate-a-symmetry-band0-bundle-20260720-v1
dataset=$bundle/dataset

expected_parent_dataset_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
expected_parent_output_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
expected_parent_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_parent_vxc_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179
expected_asset_archive_sha=5867063d7731811c2716b2991d3afe7c090f7f45c88a551f893442a2568150f5
expected_asset_dataset_sha=bcdd94d197babc504b355374c778693fadcfd92009b79153ea0f32f6a469668c
expected_asset_output_sha=2e5637c21da99e6a368a273fc7923e29a9953e8b9a940a6434b402db9b178375
expected_asset_provenance_sha=0c0b13717ae797d63a0a3e88eb3b5d2551c3faf3f96c80dc34d1caa91fc75b0e
expected_asset_source_manifest_sha=69ecbea258c9845e97540aed5f4b52425fef8922e9dcc8cd0fca14c534e2f824
expected_symrot_abf_sha=8e9ffde672debf4fcf3b6efedc08c56c70bb69a5a66f3777bad6e0bda9325ae8
expected_grid_generator_sha=a7fce0fc0c4dbab5f5c259f54eecd8d76023b40e9d39105b72c63c6e3889b019
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

test -d "$parent/dataset"
test -e "$parent/COMPLETE"
test ! -e "$parent/FAILED"
test -f "$asset_archive"
test -d "$staging"
test ! -e "$asset"
test ! -e "$bundle"
test "$(sha256sum "$parent/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_parent_dataset_sha"
test "$(sha256sum "$parent/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_parent_output_sha"
test "$(sha256sum "$parent/dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_parent_contract_sha"
test "$(sha256sum "$parent/dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$expected_parent_vxc_sha"
test "$(sha256sum "$asset_archive" | awk '{print $1}')" = \
  "$expected_asset_archive_sha"
test "$(sha256sum "$staging/symrot_abf_k.txt" | awk '{print $1}')" = \
  "$expected_symrot_abf_sha"
test "$(sha256sum "$staging/prepare_abacus_qsgw_ibz_contract_v1.py" | awk '{print $1}')" = \
  "$expected_grid_generator_sha"
test "$(sha256sum "$staging/prepare_qsgw_band_contract_v1.py" | awk '{print $1}')" = \
  "$expected_band_generator_sha"
test "$(sha256sum "$staging/test_prepare_qsgw_band_contract_v1.py" | awk '{print $1}')" = \
  "$expected_band_test_sha"
test -z "$(find "$parent" -type l -print -quit)"
test -z "$(find "$parent" -perm /222 -print -quit)"
(
  cd "$parent"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

TMPDIR=/tmp python3 -B "$staging/test_prepare_qsgw_band_contract_v1.py" \
  >"$staging/band-generator-unit-test.stdout" \
  2>"$staging/band-generator-unit-test.stderr"

tar -C "$base" -xzf "$asset_archive"
test -e "$asset/COMPLETE"
test ! -e "$asset/FAILED"
test "$(sha256sum "$asset/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_asset_dataset_sha"
test "$(sha256sum "$asset/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_asset_output_sha"
test "$(sha256sum "$asset/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_asset_provenance_sha"
test "$(sha256sum "$asset/provenance/SOURCE_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_asset_source_manifest_sha"
test "$(find "$asset/dataset" -type f | wc -l)" -eq 1274
test -z "$(find "$asset" -type l -print -quit)"
test -z "$(find "$asset" -perm /222 -print -quit)"
(
  cd "$asset"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test -z "$(comm -12 \
  <(cd "$parent/dataset" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
  <(cd "$asset/dataset" && find . -type f -printf '%P\n' | LC_ALL=C sort))"
test ! -e "$parent/dataset/symrot_abf_k.txt"
test ! -e "$asset/dataset/symrot_abf_k.txt"

cp -a --reflink=auto "$parent" "$bundle"
chmod -R u+w "$bundle"
rm -f "$bundle/COMPLETE" "$bundle/FAILED" "$bundle/OUTPUT_SHA256SUMS.txt"
mkdir -p "$bundle/provenance"
cp "$parent/PROVENANCE.txt" "$bundle/provenance/parent-v3-PROVENANCE.txt"
cp "$parent/DATASET_SHA256SUMS.txt" \
  "$bundle/provenance/parent-v3-DATASET_SHA256SUMS.txt"
cp "$parent/OUTPUT_SHA256SUMS.txt" \
  "$bundle/provenance/parent-v3-OUTPUT_SHA256SUMS.txt"
cp "$asset/PROVENANCE.txt" "$bundle/provenance/band-assets-PROVENANCE.txt"
cp "$asset/DATASET_SHA256SUMS.txt" \
  "$bundle/provenance/band-assets-DATASET_SHA256SUMS.txt"
cp "$asset/OUTPUT_SHA256SUMS.txt" \
  "$bundle/provenance/band-assets-OUTPUT_SHA256SUMS.txt"
cp "$asset/provenance/SOURCE_SHA256SUMS.txt" \
  "$bundle/provenance/band-assets-SOURCE_SHA256SUMS.txt"
cp "$staging/prepare_abacus_qsgw_ibz_contract_v1.py" "$bundle/provenance/"
cp "$staging/prepare_qsgw_band_contract_v1.py" "$bundle/provenance/"
cp "$staging/test_prepare_qsgw_band_contract_v1.py" "$bundle/provenance/"
cp "$staging/band-generator-unit-test.stdout" "$bundle/provenance/"
cp "$staging/band-generator-unit-test.stderr" "$bundle/provenance/"
printf '%s\n' "$RUNNER_SHA256" >"$bundle/provenance/import-runner-sha256.txt"

cp -a "$asset/dataset/." "$dataset/"
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
gate=gate_a0_symmetry_band0_combined_bundle_v1
acceptance_scope=legacy_candidate_same_physical_input
parent_grid_bundle=$parent
parent_grid_dataset_manifest_sha256=$expected_parent_dataset_sha
parent_grid_output_manifest_sha256=$expected_parent_output_sha
band_asset_bundle=$asset
band_asset_archive_sha256=$expected_asset_archive_sha
band_asset_dataset_manifest_sha256=$expected_asset_dataset_sha
band_asset_output_manifest_sha256=$expected_asset_output_sha
symrot_abf_source_host=dongfang
symrot_abf_source_root=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1
symrot_abf_sha256=$expected_symrot_abf_sha
copy_mode=physical_copy_no_symlinks_reflink_allowed
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
grid_generator_sha256=$expected_grid_generator_sha
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

echo GATE_A_SYMMETRY_BAND0_COMBINED_BUNDLE_V1=PASS
cat "$bundle/PROVENANCE.txt"
sha256sum "$bundle/DATASET_SHA256SUMS.txt" \
  "$bundle/OUTPUT_SHA256SUMS.txt" \
  "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.band.contract" \
  "$dataset/qsgw_vxc_band.v2.manifest"
