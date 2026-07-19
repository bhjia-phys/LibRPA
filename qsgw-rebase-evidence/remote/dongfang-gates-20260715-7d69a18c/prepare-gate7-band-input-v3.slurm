#!/usr/bin/env bash
# Prepare the Gate-7B upstream reference input (v3): the frozen normalized
# band input (v2) plus the band_vxc_k diagonal files the upstream
# g0w0_band reader requires. Conversion (documented in the converter):
# band_vxck triangular diagonal real part / 2 (Ry -> Ha), verified against
# the May 2026 reference pair. The v2 dataset and the band contract are
# not modified.
#SBATCH --job-name=qsgw-band-input-v3
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/band-input-v3-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/band-input-v3-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/inputs/si-k444-band-v2-2358129
source_dataset=$source_root/dataset
converter=$root/jobs/convert_band_vxck_to_band_vxc_k-v1-bfd43623.py
converter_test=$root/jobs/test_convert_band_vxck_to_band_vxc_k-v1-d035a541.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
target_root=$root/inputs/si-k444-band-v3-$SLURM_JOB_ID
target_dataset=$target_root/dataset

expected_source_provenance_sha=92345ccc043d63501ca496b099cc4887692313b28e11874bfe6572bc69904e21
expected_source_manifest_sha=604fcc0f59999d37df0388649fe8a21ad2b9af1e29bd3861e2068d18e04e1a2b
expected_converter_sha=bfd43623503ab3c360351685345e72cd8b17b114f150cc4adcc1a8769c6e006d
expected_converter_test_sha=d035a541c424c7a2dc0fd93a5dfc44795d66c64407fb37854389d812fbdcacff
expected_n_band_kpoints=201

record_failure() {
  local rc=$?
  if [[ -d ${target_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$target_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$target_root"
test -e "$source_root/COMPLETE"
test "$(sha256sum "$source_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_dataset/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_manifest_sha"
test "$(sha256sum "$converter" | awk '{print $1}')" = "$expected_converter_sha"
test "$(sha256sum "$converter_test" | awk '{print $1}')" = \
  "$expected_converter_test_sha"

mkdir -p "$target_root/tools"
cp "$0" "$target_root/prepare-gate7-band-input-v3.slurm"
cp "$converter" "$target_root/tools/convert_band_vxck_to_band_vxc_k_v1.py"
cp "$converter_test" \
  "$target_root/tools/test_convert_band_vxck_to_band_vxc_k_v1.py"
cp "$source_root/PROVENANCE.txt" "$target_root/source-PROVENANCE.txt"
cp "$source_dataset/DATASET_SHA256SUMS.txt" \
  "$target_root/source-DATASET_SHA256SUMS.txt"

"$python" -B "$target_root/tools/test_convert_band_vxck_to_band_vxc_k_v1.py" \
  >"$target_root/converter-unit-test.stdout" \
  2>"$target_root/converter-unit-test.stderr"

cp -a "$source_dataset" "$target_dataset"
test "$(find "$target_dataset" -type l | wc -l)" -eq 0
test "$(find "$target_dataset" -type f | wc -l)" -eq 1372
test "$(find "$target_dataset" -maxdepth 1 -name 'band_vxc_k_*.txt' | wc -l)" -eq 0

n_converted=0
for kpoint in $(seq -w 00001 "$expected_n_band_kpoints"); do
  source=$target_dataset/band_vxck$((10#$kpoint))_nao.txt
  target=$target_dataset/band_vxc_k_$kpoint.txt
  test -f "$source"
  "$python" -B "$target_root/tools/convert_band_vxck_to_band_vxc_k_v1.py" \
    "$source" "$target" >>"$target_root/converter.stdout"
  test -s "$target"
  n_converted=$((n_converted + 1))
done
test "$n_converted" -eq "$expected_n_band_kpoints"
test "$(find "$target_dataset" -maxdepth 1 -name 'band_vxc_k_*.txt' | wc -l)" -eq \
  "$expected_n_band_kpoints"

# spot check: band_vxc_k_00001 band 1 == band_vxck1 diagonal / 2
diag=$(grep -A1 '^Row 1$' "$target_dataset/band_vxck1_nao.txt" | tail -1 | \
  grep -oE '\(-?[0-9.]+e[+-][0-9]+,' | head -1 | tr -d '(,')
converted=$(awk 'NR==1 {print $3}' "$target_dataset/band_vxc_k_00001.txt")
"$python" -c "
import sys
diag=float('$diag')/2.0
conv=float('$converted')
sys.exit(0 if abs(diag-conv) < 1e-12 else 1)
"

(
  cd "$target_dataset"
  find . -type f ! -name 'DATASET_SHA256SUMS.txt' -print0 | sort -z | \
    xargs -0 sha256sum >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

dataset_manifest_sha=$(sha256sum "$target_dataset/DATASET_SHA256SUMS.txt" \
  | awk '{print $1}')
band_contract_sha=$(sha256sum "$target_dataset/qsgw_input.band.contract" \
  | awk '{print $1}')

cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=prepare_gate7b_upstream_reference_input_v3
conversion=band_vxck_triangular_diagonal_real_over_2_ry_to_ha
conversion_reference=may2026_reference_pair_minus_8.00126971e-01_to_minus_4.0006348548059606e-01
source_input_root=$source_root
source_dataset_manifest_sha256=$expected_source_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
converter_sha256=$expected_converter_sha
n_band_vxc_k_files=$n_converted
band_contract_sha256=$band_contract_sha
dataset_manifest_sha256=$dataset_manifest_sha
headwing=off
hartree=off
band_update=operator_fourier
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target_root"
  find . -maxdepth 1 -type f ! -name 'OUTPUT_SHA256SUMS.txt' \
    ! -name COMPLETE -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target_root/COMPLETE"
echo GATE7_BAND_INPUT_V3=COMPLETE
