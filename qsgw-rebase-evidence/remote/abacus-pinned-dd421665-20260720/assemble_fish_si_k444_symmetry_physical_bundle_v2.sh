#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "ERROR line=%s status=%s command=%s\n" \
  "$LINENO" "$status" "$BASH_COMMAND" >&2' ERR

: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify this committed runner}"
: "${RUNNER_SOURCE:?RUNNER_SOURCE must be a clean checkout at RUNNER_COMMIT}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
test "${#RUNNER_COMMIT}" = 40
test "${#RUNNER_SHA256}" = 64

base=/home/bhj/ai-runs
source_bundle=$base/abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v4
source_dataset=$source_bundle/dataset
root=$base/abacus-pinned-dd421665-si-k444-symmetry-physical-bundle-20260723-v2
dataset=$root/dataset
provenance=$root/provenance
tool_dir=$provenance/tools

expected_source_provenance_sha=88cbcde30bc927495a96d0b779bfb4854cabb014911c08907643ff661d06463a
expected_source_output_manifest_sha=d16cb9df89de56895e9533a9f2253ea8ec0b4f8ea6f7b027b659f5ff8dfb5c46
expected_source_dataset_manifest_sha=9f47e5c85dc9c697e808270a602da473d70654c4a2bfa98af47fb1d495680c7e
expected_source_stru_sha=28943ba2c3b359acbf95f6f8bf1dfc7f61502dc928952058f31e62cab0dfe279
expected_input_stru_sha=d907d2cf15182a5106e4c1d5285c4523837fe3b0848ba7ebe41a1b2dff26d3fb

runner_relative=qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/assemble_fish_si_k444_symmetry_physical_bundle_v2.sh
builder_relative=qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/build_abacus_physical_stru_overlay_v1.py
basis_relative=qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/generate_abacus_basis_metadata_v1.py
contract_relative=qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/prepare_abacus_qsgw_ibz_contract_v3.py

run_succeeded=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

test ! -e "$root"
test -d "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(git -C "$RUNNER_SOURCE" show "$RUNNER_COMMIT:$runner_relative" | \
  sha256sum | awk '{print $1}')" = "$RUNNER_SHA256"

test -e "$source_bundle/COMPLETE"
test ! -e "$source_bundle/FAILED"
test -z "$(find "$source_bundle" -type l -print -quit)"
test -z "$(find "$source_bundle" -perm /222 -print -quit)"
test "$(sha256sum "$source_bundle/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_output_manifest_sha"
test "$(sha256sum "$source_bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_dataset_manifest_sha"
test "$(sha256sum "$source_dataset/stru_out" | awk '{print $1}')" = \
  "$expected_source_stru_sha"
test "$(sha256sum "$source_bundle/provenance/producer-inputs-v2/STRU" | \
  awk '{print $1}')" = "$expected_input_stru_sha"
(
  cd "$source_bundle"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$dataset" "$tool_dir" "$provenance/source-v4"
cp "$0" "$provenance/$(basename "$runner_relative")"
for relative in "$builder_relative" "$basis_relative" "$contract_relative"; do
  source_path=$RUNNER_SOURCE/$relative
  target_path=$tool_dir/$(basename "$relative")
  test -f "$source_path"
  cp "$source_path" "$target_path"
  test "$(sha256sum "$source_path" | awk '{print $1}')" = \
    "$(git -C "$RUNNER_SOURCE" show "$RUNNER_COMMIT:$relative" | \
      sha256sum | awk '{print $1}')"
done

cp "$source_bundle/PROVENANCE.txt" "$provenance/source-v4/PROVENANCE.txt"
cp "$source_bundle/DATASET_SHA256SUMS.txt" \
  "$provenance/source-v4/DATASET_SHA256SUMS.txt"
cp "$source_bundle/OUTPUT_SHA256SUMS.txt" \
  "$provenance/source-v4/OUTPUT_SHA256SUMS.txt"
cp "$source_bundle/provenance/producer-inputs-v2/STRU" "$provenance/STRU"
cp "$source_bundle/provenance/assets/Si_gga_8au_100Ry_3s3p2d.orb" "$provenance/"
cp "$source_bundle/provenance/assets/Si_3s3p2d1f1g_pca1e-6.abfs" "$provenance/"

while IFS= read -r -d '' path; do
  name=$(basename "$path")
  case "$name" in
    stru_out|basis_wfc_out|basis_aux_out|basis_out|basis_metadata_summary.json|\
    qsgw_input.contract|qsgw_vxc_scf.manifest|qsgw_input_contract.summary.json)
      continue
      ;;
  esac
  cp --reflink=auto "$path" "$dataset/$name"
done < <(find "$source_dataset" -maxdepth 1 -type f -print0)

python3 -B "$tool_dir/build_abacus_physical_stru_overlay_v1.py" \
  "$provenance/STRU" "$source_dataset/stru_out" "$dataset/stru_out" \
  "$provenance/PHYSICAL_STRU_OVERLAY.json" \
  >"$provenance/physical-stru-overlay.stdout"
python3 -B - "$provenance/PHYSICAL_STRU_OVERLAY.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["passed"] is True
assert report["symmetry_convention"] == "row"
assert report["symmetry_operation_count"] == 48
assert report["atom_cartesian_max_abs_bohr"] <= 1.0e-10
assert report["reciprocal_closure_max_abs"] <= 1.0e-12
assert report["atom_symmetry_max_abs"] <= 1.0e-10
PY

python3 -B "$tool_dir/generate_abacus_basis_metadata_v1.py" \
  --orb "$provenance/Si_gga_8au_100Ry_3s3p2d.orb" \
  --abfs "$provenance/Si_3s3p2d1f1g_pca1e-6.abfs" \
  --n-atoms 2 --output-dir "$dataset" \
  >"$provenance/basis-generator.stdout"
(
  cd "$tool_dir"
  python3 -B prepare_abacus_qsgw_ibz_contract_v3.py "$dataset" \
    >"$provenance/contract-generator.stdout"
)

test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | \
  wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxck*_nao.txt' | wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxcs*k*_nao.txt' | wc -l)" -eq 0
grep -Fqx 'basis state' "$dataset/qsgw_vxc_scf.manifest"
grep -Fqx 'gauge mf0_state' "$dataset/qsgw_vxc_scf.manifest"
grep -Fqx 'n_scf_kpoints 8' "$dataset/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$dataset/qsgw_input.contract"
grep -Fqx 'n_band_kpoints 0' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
test -z "$(find "$root" -type l -print -quit)"

python3 -B - "$source_dataset" "$dataset" \
  >"$provenance/SOURCE_REUSE_VERIFICATION.txt" <<'PY'
import hashlib
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
derived = {
    "stru_out",
    "basis_wfc_out",
    "basis_aux_out",
    "basis_out",
    "basis_metadata_summary.json",
    "qsgw_input.contract",
    "qsgw_vxc_scf.manifest",
    "qsgw_input_contract.summary.json",
}

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

source_names = {path.name for path in source.iterdir() if path.is_file()} - derived
target_names = {path.name for path in target.iterdir() if path.is_file()} - derived
assert source_names == target_names, (sorted(source_names - target_names), sorted(target_names - source_names))
for name in sorted(source_names):
    assert digest(source / name) == digest(target / name), name
print(f"PASS: {len(source_names)} non-derived dataset files are SHA256-identical")
PY

(
  cd "$root"
  find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

cat >"$root/PROVENANCE.txt" <<EOF
gate=fish_pinned_abacus_si_k444_symmetry_physical_bundle_v2
acceptance_scope=corrected_physical_stru_stage1_input
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
source_bundle=$source_bundle
source_bundle_provenance_sha256=$expected_source_provenance_sha
source_dataset_manifest_sha256=$expected_source_dataset_manifest_sha
source_stru_out_sha256=$expected_source_stru_sha
input_stru_sha256=$expected_input_stru_sha
physical_lattice_source=matching_abacus_input_STRU
physical_lattice_constant_bohr=10.2
symmetry_operation_source=source_bundle_stru_out_fractional_tail
symmetry_operation_count=48
symmetry=on
scf_kpoints=8
full_bz_kpoints=64
grid=4x4x4
use_shrink_abfs=false
headwing=off
hartree=off
band_update=off
shared_gw_source_changes=none
copy_mode=physical_copy_reflink_allowed_no_symlinks
dataset_manifest_sha256=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
qsgw_input_contract_sha256=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
qsgw_vxc_manifest_sha256=$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/COMPLETE"
find "$root" -type f -exec chmod a-w {} +
find "$root" -depth -type d -exec chmod a-w {} +
run_succeeded=1
trap - EXIT

test -e "$root/COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -type l -print -quit)"
test -z "$(find "$root" -perm /222 -print -quit)"
echo FISH_SI_K444_SYMMETRY_PHYSICAL_BUNDLE_V2=PASS
cat "$root/PROVENANCE.txt"
