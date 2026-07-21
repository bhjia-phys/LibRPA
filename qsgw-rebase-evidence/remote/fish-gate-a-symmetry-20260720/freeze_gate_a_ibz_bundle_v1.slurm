#!/usr/bin/env bash
#SBATCH --job-name=qsgw-a0-ibz-freeze
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-freeze-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-freeze-submit-20260720-v1/slurm-%j.err

set -euo pipefail

submit_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-freeze-submit-20260720-v1
source_root=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-newarch-previnput-wcfq-current-20260704-013610/librpa
producer_root=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1
metadata_root=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-sym-newarch-exactpyatb-20260630-032256
generator=$submit_root/prepare_abacus_qsgw_ibz_contract_v1.py
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v1-$SLURM_JOB_ID
dataset=$root/dataset
provenance=$root/provenance
expected_generator_sha=a7fce0fc0c4dbab5f5c259f54eecd8d76023b40e9d39105b72c63c6e3889b019

record_failure() {
  local rc=$?
  if [[ -d ${root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test -d "$source_root"
test -d "$producer_root"
test -d "$metadata_root"
test -f "$generator"
test "$(sha256sum "$generator" | awk '{print $1}')" = "$expected_generator_sha"
test ! -e "$root"
test ! -e "$root.tar.zst"
mkdir -p "$dataset" "$provenance"

copy_one() {
  local name=$1
  test -e "$source_root/$name"
  cp -L --preserve=mode,timestamps "$source_root/$name" "$dataset/$name"
  test -f "$dataset/$name"
  test ! -L "$dataset/$name"
}

for name in \
  stru_out band_out bz_sampling_out basis_wfc_out basis_aux_out \
  basis_aux_shrink_out irreducible_sector.txt symrot_k.txt symrot_R.txt; do
  copy_one "$name"
done

for pattern in \
  'KS_eigenvector_*.dat' 'vxcs1k*_nao.txt' 'Cs_data_*.txt' \
  'Cs_shrinked_data_*.txt' 'shrink_sinvS_*.txt' \
  'coulomb_mat_*.txt' 'coulomb_cut_*.txt'; do
  mapfile -d '' matches < <(
    find "$source_root" -maxdepth 1 \( -type f -o -type l \) \
      -name "$pattern" -printf '%f\0' | sort -z
  )
  expected=4
  case "$pattern" in
    'KS_eigenvector_*.dat'|'vxcs1k*_nao.txt') expected=8 ;;
  esac
  test "${#matches[@]}" -eq "$expected"
  for name in "${matches[@]}"; do
    copy_one "$name"
  done
done

test "$(find "$dataset" -maxdepth 1 -type f | wc -l)" -eq 45
test "$(find "$dataset" -type l | wc -l)" -eq 0
test "$(awk 'NR == 1 {print $1, $2, $3}' "$dataset/bz_sampling_out")" = '4 4 4'
test "$(awk 'NR == 2 {print $1, $2}' "$dataset/bz_sampling_out")" = '8 8'
test "$(head -n 1 "$dataset/band_out")" = '8 1 44 44'

(
  cd "$root"
  find dataset -type f -print0 | sort -z | xargs -0 sha256sum \
    >SOURCE_DATASET_SHA256SUMS.txt
  sha256sum --check --quiet SOURCE_DATASET_SHA256SUMS.txt
)

python3 "$generator" "$dataset" --use-shrink-abfs \
  >"$provenance/contract-generator.stdout" \
  2>"$provenance/contract-generator.stderr"
test ! -s "$provenance/contract-generator.stderr"
grep -Fqx 'n_scf_kpoints 8' "$dataset/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
test "$(grep -Ec '^mf0_wavefunctions [0-9a-f]{64} KS_eigenvector_[0-7]\.dat$' "$dataset/qsgw_input.contract")" -eq 8
test "$(grep -Ec '^reader_static ' "$dataset/qsgw_input.contract")" -eq 24
test "$(grep -Ec '^1 [1-8] ' "$dataset/qsgw_vxc_scf.manifest")" -eq 8
python3 - "$dataset/qsgw_input_contract.summary.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert summary["grid"] == [4, 4, 4]
assert summary["n_scf_kpoints"] == 8
assert summary["n_bands"] == 44
assert summary["n_aos"] == 44
assert summary["kstar_multiplicities"] == [1, 8, 4, 6, 24, 12, 3, 6]
assert summary["kstar_coverage"] == 64
assert summary["reader_static_count"] == 24
assert summary["contract_file_count"] == 35
assert summary["use_shrink_abfs"] is True
PY

cp "$generator" "$provenance/"
for name in \
  prepare_dataset.py prepare_dataset.slurm run_qsgw.slurm \
  dataset_setup_summary.txt input_contract_hashes.txt; do
  if [[ -f $metadata_root/$name ]]; then
    cp "$metadata_root/$name" "$provenance/metadata-$name"
  fi
done
cp "$source_root/librpa.in" "$provenance/stitched-source-librpa.in"

(
  cd "$root"
  find dataset -type f -print0 | sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
test "$(find "$dataset" -maxdepth 1 -type f | wc -l)" -eq 48
test "$(find "$dataset" -type l | wc -l)" -eq 0

source_manifest_sha=$(sha256sum "$root/SOURCE_DATASET_SHA256SUMS.txt" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
vxc_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
cat >"$root/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_ibz_bundle_freeze
acceptance_scope=historical_legacy_candidate_parity_only
producer=abacus_historical_dirty_checkout_not_pinned
source_root=$source_root
producer_root=$producer_root
metadata_root=$metadata_root
copy_mode=selected_physical_cp_L_no_symlinks
source_file_count=45
dataset_file_count=48
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
kstar_multiplicities=1,8,4,6,24,12,3,6
use_shrink_abfs=true
headwing=off
hartree=off
band=off
generator_sha256=$expected_generator_sha
source_dataset_manifest_sha256=$source_manifest_sha
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_input_contract_sha256=$contract_sha
qsgw_vxc_scf_manifest_sha256=$vxc_manifest_sha
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    -print0 | sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/COMPLETE"
chmod -R a-w "$root"
tar -C "$(dirname "$root")" --zstd -cf "$root.tar.zst" "$(basename "$root")"
sha256sum "$root.tar.zst" >"$root.tar.zst.sha256"

echo "GATE_A0_IBZ_BUNDLE_FREEZE=PASS"
echo "bundle_root=$root"
echo "archive=$root.tar.zst"
cat "$root.tar.zst.sha256"
exit 0
