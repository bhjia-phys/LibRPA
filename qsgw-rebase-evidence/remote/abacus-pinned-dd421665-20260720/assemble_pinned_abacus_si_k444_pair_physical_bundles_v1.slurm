#!/usr/bin/env bash
#SBATCH --job-name=si444-pair-bundles
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-submit-20260723-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-submit-20260723-v1/slurm-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?this assembler must run under Slurm}"
: "${RUNNER_COMMIT:?RUNNER_COMMIT must identify the committed assembler}"

source_pair=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-p1-20260723-v1
source_symmetry=$source_pair/symmetry
source_fullbz=$source_pair/fullbz
source_common=$source_pair/common
stage_dir=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-submit-20260723-v1
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-physical-bundles-20260723-v1
tools=$root/provenance/tools

expected_common_manifest_sha=02e413fb565d02450f7a3709dd346baacd37bc64bd263c5e55a5e007d59a5f50
expected_staging_provenance_sha=db965373f11224fc3e6c475b0a1f235c1da60648a1c089f3b0b57ffa34fc78ec
expected_symmetry_artifact_manifest_sha=8eed09fcedadcd270dce3feafea8d0df05592fc6fb2649500809c366a669e303
expected_symmetry_provenance_sha=22cc8badd7ff6f34511fa1d163d91f54efa3388ffdbe295fb6fcdfdbb15c0d9c
expected_symmetry_validation_sha=e8e534db7c92bb0bded9b7d583e821fefb175ed4740d804b2064bd0d663d18b9
expected_fullbz_artifact_manifest_sha=b96b4344408a50154152b8e2e6bdde67fcb6ec24958dc01237a5361869b840c6
expected_fullbz_provenance_sha=0d966e7c6532ee5f937a5df364827c2bc5face4f88b01da0d1aeb3300761cc75
expected_fullbz_validation_sha=b1d9d56dc29861f0b6fd5c760d3b0a2c31b68ded08bd128788fe503e5224b1be

runner_name=assemble_pinned_abacus_si_k444_pair_physical_bundles_v1.slurm
required_tools=(
  build_abacus_physical_stru_overlay_v1.py
  build_abacus_physical_bz_sampling_overlay_v1.py
  generate_abacus_basis_metadata_v1.py
  prepare_abacus_qsgw_ibz_contract_v3.py
  prepare_abacus_qsgw_fullbz_contract_v1.py
)

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

require_sha()
{
  local path=$1
  local expected=$2
  test -f "$path"
  test "$(sha_of "$path")" = "$expected"
}

test ! -e "$root"
test -d "$stage_dir"
test -f "$stage_dir/STAGED_SHA256SUMS.txt"
(
  cd "$stage_dir"
  sha256sum --check --quiet STAGED_SHA256SUMS.txt
)
test -f "$stage_dir/$runner_name"
for tool in "${required_tools[@]}"; do
  test -f "$stage_dir/$tool"
done

test -e "$source_symmetry/COMPLETE"
test -e "$source_fullbz/COMPLETE"
test ! -e "$source_symmetry/FAILED"
test ! -e "$source_fullbz/FAILED"
test -z "$(find "$source_pair" -type l -print -quit)"
test -z "$(find "$source_common" "$source_symmetry" "$source_fullbz" -perm /222 -print -quit)"
require_sha "$source_pair/COMMON_SHA256SUMS.txt" "$expected_common_manifest_sha"
require_sha "$source_pair/STAGING_PROVENANCE.txt" "$expected_staging_provenance_sha"
require_sha "$source_symmetry/ARTIFACT_SHA256SUMS.txt" "$expected_symmetry_artifact_manifest_sha"
require_sha "$source_symmetry/RUN_PROVENANCE.txt" "$expected_symmetry_provenance_sha"
require_sha "$source_symmetry/OUTPUT_VALIDATION.json" "$expected_symmetry_validation_sha"
require_sha "$source_fullbz/ARTIFACT_SHA256SUMS.txt" "$expected_fullbz_artifact_manifest_sha"
require_sha "$source_fullbz/RUN_PROVENANCE.txt" "$expected_fullbz_provenance_sha"
require_sha "$source_fullbz/OUTPUT_VALIDATION.json" "$expected_fullbz_validation_sha"
(
  cd "$source_pair"
  sha256sum --check --quiet COMMON_SHA256SUMS.txt
)
(
  cd "$source_symmetry"
  sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)
(
  cd "$source_fullbz"
  sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)

grep -Fqx 'producer_mode=symmetry' "$source_symmetry/RUN_PROVENANCE.txt"
grep -Fqx 'producer_mode=fullbz' "$source_fullbz/RUN_PROVENANCE.txt"
grep -Fqx 'all_other_producer_inputs=identical' "$source_symmetry/RUN_PROVENANCE.txt"
grep -Fqx 'all_other_producer_inputs=identical' "$source_fullbz/RUN_PROVENANCE.txt"
for provenance in "$source_symmetry/RUN_PROVENANCE.txt" "$source_fullbz/RUN_PROVENANCE.txt"; do
  grep -Fqx 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a' "$provenance"
  grep -Fqx 'binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17' "$provenance"
  grep -Fqx 'k_grid=4x4x4' "$provenance"
done
python3 -B - "$source_common/producer-inputs-v2/INPUT_scf_symmetry" \
  "$source_common/producer-inputs-v2/INPUT_scf_fullbz" <<'PY'
import pathlib
import sys

changed = {"symmetry", "exx_symmetry_realspace"}

def read(path):
    rows = []
    values = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            key = fields[0]
            if key in changed:
                values[key] = fields[1:]
                rows.append((key, ("<changed>",)))
            else:
                rows.append((key, tuple(fields[1:])))
        else:
            rows.append(("literal", (line,)))
    return rows, values

symmetry_rows, symmetry_values = read(sys.argv[1])
fullbz_rows, fullbz_values = read(sys.argv[2])
assert symmetry_rows == fullbz_rows
assert symmetry_values == {"symmetry": ["1"], "exx_symmetry_realspace": ["1"]}
assert fullbz_values == {"symmetry": ["-1"], "exx_symmetry_realspace": ["0"]}
PY

mkdir -p "$root/provenance/source-pair" "$tools"
cp "$stage_dir/$runner_name" "$root/provenance/$runner_name"
cp "$stage_dir/STAGED_SHA256SUMS.txt" "$root/provenance/STAGED_SHA256SUMS.txt"
for tool in "${required_tools[@]}"; do
  cp "$stage_dir/$tool" "$tools/$tool"
done
cp "$source_pair/COMMON_SHA256SUMS.txt" "$root/provenance/source-pair/COMMON_SHA256SUMS.txt"
cp "$source_pair/STAGING_PROVENANCE.txt" "$root/provenance/source-pair/STAGING_PROVENANCE.txt"
cp "$source_symmetry/RUN_PROVENANCE.txt" "$root/provenance/source-pair/SYMMETRY_RUN_PROVENANCE.txt"
cp "$source_symmetry/OUTPUT_VALIDATION.json" "$root/provenance/source-pair/SYMMETRY_OUTPUT_VALIDATION.json"
cp "$source_symmetry/ARTIFACT_SHA256SUMS.txt" "$root/provenance/source-pair/SYMMETRY_ARTIFACT_SHA256SUMS.txt"
cp "$source_fullbz/RUN_PROVENANCE.txt" "$root/provenance/source-pair/FULLBZ_RUN_PROVENANCE.txt"
cp "$source_fullbz/OUTPUT_VALIDATION.json" "$root/provenance/source-pair/FULLBZ_OUTPUT_VALIDATION.json"
cp "$source_fullbz/ARTIFACT_SHA256SUMS.txt" "$root/provenance/source-pair/FULLBZ_ARTIFACT_SHA256SUMS.txt"

copy_selected()
{
  local source_path=$1
  local target_path=$2
  local source_manifest=$3
  local verification=$4
  local source_sha target_sha

  test -f "$source_path"
  test ! -e "$target_path"
  mkdir -p "$(dirname "$target_path")"
  source_sha=$(sha_of "$source_path")
  printf '%s  %s\n' "$source_sha" "$source_path" >>"$source_manifest"
  cp --reflink=auto "$source_path" "$target_path"
  target_sha=$(sha_of "$target_path")
  test "$target_sha" = "$source_sha"
  printf '%s  %s -> %s\n' "$source_sha" "$source_path" "$target_path" >>"$verification"
}

build_mode()
{
  local mode=$1
  local expected_kpoints contract_generator allow_no_symmetry
  local source_mode source_run member dataset provenance assets
  local source_manifest verification path

  case "$mode" in
    symmetry)
      expected_kpoints=8
      source_mode=$source_symmetry
      contract_generator=prepare_abacus_qsgw_ibz_contract_v3.py
      allow_no_symmetry=0
      ;;
    fullbz)
      expected_kpoints=64
      source_mode=$source_fullbz
      contract_generator=prepare_abacus_qsgw_fullbz_contract_v1.py
      allow_no_symmetry=1
      ;;
    *)
      printf 'invalid mode: %s\n' "$mode" >&2
      return 2
      ;;
  esac

  source_run=$source_mode/run
  member=$root/$mode
  dataset=$member/dataset
  provenance=$member/provenance
  assets=$provenance/assets
  source_manifest=$provenance/SOURCE_SELECTED_SHA256SUMS.txt
  verification=$provenance/SOURCE_COPY_VERIFICATION.txt
  mkdir -p "$dataset" "$assets"
  : >"$source_manifest"
  : >"$verification"

  copy_selected "$source_run/band_out" "$dataset/band_out" "$source_manifest" "$verification"
  for path in "$source_run"/KS_eigenvector_*.dat \
              "$source_run"/Cs_data_*.txt \
              "$source_run"/coulomb_mat_*.txt \
              "$source_run"/coulomb_cut_*.txt; do
    copy_selected "$path" "$dataset/$(basename "$path")" "$source_manifest" "$verification"
  done
  for path in "$source_run"/OUT.ABACUS/vxck*_nao.txt; do
    copy_selected "$path" "$dataset/$(basename "$path")" "$source_manifest" "$verification"
  done
  for path in "$source_common"/assets/Si_gga_8au_100Ry_3s3p2d.orb \
              "$source_common"/assets/Si_3s3p2d1f1g_pca1e-6.abfs; do
    copy_selected "$path" "$assets/$(basename "$path")" "$source_manifest" "$verification"
  done

  if [[ $allow_no_symmetry -eq 1 ]]; then
    python3 -B "$tools/build_abacus_physical_stru_overlay_v1.py" \
      "$source_common/producer-inputs-v2/STRU" "$source_run/stru_out" \
      "$dataset/stru_out" "$provenance/PHYSICAL_STRU_OVERLAY.json" \
      --allow-no-symmetry >"$provenance/physical-stru-overlay.stdout"
  else
    python3 -B "$tools/build_abacus_physical_stru_overlay_v1.py" \
      "$source_common/producer-inputs-v2/STRU" "$source_run/stru_out" \
      "$dataset/stru_out" "$provenance/PHYSICAL_STRU_OVERLAY.json" \
      >"$provenance/physical-stru-overlay.stdout"
  fi
  python3 -B "$tools/build_abacus_physical_bz_sampling_overlay_v1.py" \
    "$dataset/stru_out" "$source_run/bz_sampling_out" "$dataset/bz_sampling_out" \
    "$provenance/PHYSICAL_BZ_SAMPLING_OVERLAY.json" \
    >"$provenance/physical-bz-sampling-overlay.stdout"

  python3 -B - "$provenance/PHYSICAL_STRU_OVERLAY.json" \
    "$provenance/PHYSICAL_BZ_SAMPLING_OVERLAY.json" "$mode" "$expected_kpoints" <<'PY'
import json
import sys

stru = json.load(open(sys.argv[1], encoding="utf-8"))
bz = json.load(open(sys.argv[2], encoding="utf-8"))
mode = sys.argv[3]
expected_kpoints = int(sys.argv[4])
assert stru["passed"] is True
assert stru["symmetry_convention"] == "row"
assert stru["atom_cartesian_max_abs_bohr"] <= 1.0e-10
assert stru["reciprocal_closure_max_abs"] <= 1.0e-12
assert stru["symmetry_operation_count"] == (48 if mode == "symmetry" else 0)
assert bz["passed"] is True
assert bz["grid"] == [4, 4, 4]
assert bz["scf_kpoint_count"] == expected_kpoints
assert bz["ibz_kpoint_count"] == expected_kpoints
assert bz["non_cartesian_tokens_unchanged"] is True
assert bz["fractional_recovery_max_abs"] <= 1.0e-12
PY

  python3 -B "$tools/generate_abacus_basis_metadata_v1.py" \
    --orb "$assets/Si_gga_8au_100Ry_3s3p2d.orb" \
    --abfs "$assets/Si_3s3p2d1f1g_pca1e-6.abfs" \
    --n-atoms 2 --output-dir "$dataset" >"$provenance/basis-generator.stdout"
  (
    cd "$tools"
    python3 -B "$contract_generator" "$dataset" >"$provenance/contract-generator.stdout"
  )

  test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | wc -l)" -eq "$expected_kpoints"
  test "$(find "$dataset" -maxdepth 1 -type f -name 'vxck*_nao.txt' | wc -l)" -eq "$expected_kpoints"
  test "$(find "$dataset" -maxdepth 1 -type f -name 'Cs_data_*.txt' | wc -l)" -eq 8
  test "$(find "$dataset" -maxdepth 1 -type f -name 'coulomb_mat_*.txt' | wc -l)" -eq 8
  test "$(find "$dataset" -maxdepth 1 -type f -name 'coulomb_cut_*.txt' | wc -l)" -eq 8
  grep -Fqx 'basis state' "$dataset/qsgw_vxc_scf.manifest"
  grep -Fqx 'gauge mf0_state' "$dataset/qsgw_vxc_scf.manifest"
  grep -Fqx "n_scf_kpoints $expected_kpoints" "$dataset/qsgw_input.contract"
  grep -Fqx 'n_headwing_kpoints 0' "$dataset/qsgw_input.contract"
  grep -Fqx 'n_band_kpoints 0' "$dataset/qsgw_input.contract"
  grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
  grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
  grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
  test -z "$(find "$member" -type l -print -quit)"

  (
    cd "$member"
    find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
      >DATASET_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
  )

  cat >"$member/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_pair_physical_bundle_v1
producer_mode=$mode
all_other_producer_inputs=identical
source_pair=$source_pair
source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17
runner_commit=$RUNNER_COMMIT
source_artifact_manifest_sha256=$(sha_of "$source_mode/ARTIFACT_SHA256SUMS.txt")
source_validation_sha256=$(sha_of "$source_mode/OUTPUT_VALIDATION.json")
physical_lattice_source=matching_abacus_input_STRU
cartesian_kvector_source=fractional_bz_times_physical_reciprocal_lattice
grid=4x4x4
scf_kpoints=$expected_kpoints
full_bz_kpoints=64
use_shrink_abfs=false
headwing=off
hartree=off
band_update=off
shared_gw_source_changes=none
copy_mode=selected_physical_copy_reflink_allowed_no_symlinks
dataset_manifest_sha256=$(sha_of "$member/DATASET_SHA256SUMS.txt")
qsgw_input_contract_sha256=$(sha_of "$dataset/qsgw_input.contract")
qsgw_vxc_manifest_sha256=$(sha_of "$dataset/qsgw_vxc_scf.manifest")
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
  (
    cd "$member"
    find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE ! -name FAILED \
      -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
    sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  )
  touch "$member/COMPLETE"
}

build_mode symmetry
build_mode fullbz

python3 -B - "$source_symmetry/OUTPUT_VALIDATION.json" \
  "$source_fullbz/OUTPUT_VALIDATION.json" "$root/PAIR_VALIDATION.json" <<'PY'
import json
import pathlib
import sys

symmetry = json.load(open(sys.argv[1], encoding="utf-8"))
fullbz = json.load(open(sys.argv[2], encoding="utf-8"))
output = pathlib.Path(sys.argv[3])
assert symmetry["status"] == "PASS"
assert fullbz["status"] == "PASS"
for key in ("n_spins", "n_bands", "n_basis"):
    assert symmetry["band"][key] == fullbz["band"][key]
assert symmetry["band"]["n_kpoints"] == 8
assert fullbz["band"]["n_kpoints"] == 64
gap_difference_ev = abs(symmetry["band"]["gap_ev"] - fullbz["band"]["gap_ev"])
assert gap_difference_ev <= 1.0e-6
report = {
    "status": "PASS",
    "changed_factor": "abacus_input_symmetry",
    "all_other_producer_inputs": "identical",
    "symmetry_gap_ev": symmetry["band"]["gap_ev"],
    "fullbz_gap_ev": fullbz["band"]["gap_ev"],
    "gap_difference_ev": gap_difference_ev,
    "gap_tolerance_ev": 1.0e-6,
    "symmetry_kpoints": 8,
    "fullbz_kpoints": 64,
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

cat >"$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_pair_physical_bundles_v1
acceptance_scope=matched_symmetry_and_fullbz_stage1_inputs
runner_commit=$RUNNER_COMMIT
source_pair=$source_pair
source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17
changed_factor=abacus_input_symmetry
all_other_producer_inputs=identical
symmetry_bundle=$root/symmetry
fullbz_bundle=$root/fullbz
shared_gw_source_changes=none
slurm_job_id=$SLURM_JOB_ID
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$root"
  find . -type f ! -path './OUTPUT_SHA256SUMS.txt' ! -name PAIR_COMPLETE ! -name FAILED \
    -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/PAIR_COMPLETE"
find "$root" -type f -exec chmod a-w {} +
find "$root" -depth -type d -exec chmod a-w {} +
run_succeeded=1
trap - EXIT

test -e "$root/PAIR_COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -type l -print -quit)"
test -z "$(find "$root" -perm /222 -print -quit)"
echo PINNED_ABACUS_SI_K444_PAIR_PHYSICAL_BUNDLES_V1=PASS
cat "$root/PAIR_VALIDATION.json"
