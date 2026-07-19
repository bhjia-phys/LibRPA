#!/usr/bin/env bash
# Normalize the frozen Gate-7 band input to per-state occupations (v2).
# Root cause (Gate 7A job 2357440 band initial-state rejection):
# ABACUS band-mode band_KS_eigenvalue files store k-weighted occupations
# (occupation * k_weight, here 2/201), while the shared LibRPA band reader
# (src/api/input.cpp: librpa_set_band_occ_eigval) applies an additional
# 1/n_kpts_band factor expecting band_out-style per-state occupations.
# Upstream code is not modified; the dataset is normalized instead by
# multiplying the weight column by n_band_kpoints (physical content
# unchanged: 2/201 * 201 = 2.0, the band_out convention).
# All other payloads stay byte-identical to the frozen v1 dataset.
#SBATCH --job-name=qsgw-band-input-v2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/band-input-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/band-input-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/inputs/si-k444-band-v1-2356667
source_dataset=$source_root/dataset
normalizer=$root/jobs/normalize_band_eigenvalue_weights-v1-349146361.py
normalizer_test=$root/jobs/test_normalize_band_eigenvalue_weights-v1-1133d23a.py
preparer=$root/jobs/prepare_qsgw_band_contract-v1-adf8b56b.py
preparer_test=$root/jobs/test_prepare_qsgw_band_contract-v1-1e4fab6d.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
target_root=$root/inputs/si-k444-band-v2-$SLURM_JOB_ID
target_dataset=$target_root/dataset

expected_source_provenance_sha=febb2aaf4266d0996aaae3ebf4bd246604dd69814435d4bd072535f268a2e753
expected_source_manifest_sha=b279a3d2eea04e8ea65245277a2fee668e84fa1a1e229fb8d006fa0af04ba79a
expected_normalizer_sha=349146361f2c2693da7a68f591c7c16aa6b7e65ca041760921b77dd6df0cffe9
expected_normalizer_test_sha=1133d23a5759e8117547a7c09c3333899976fa8df2516e96235b3f3715a3fbe4
expected_preparer_sha=adf8b56b57773cc0e2b57b740ea84080ed768a5439717fec1ae75ba2181502f8
expected_preparer_test_sha=1e4fab6dea4f3db0e80adda260a1ea900eed02a88427135799c3ffa97c182780
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
test "$(sha256sum "$normalizer" | awk '{print $1}')" = \
  "$expected_normalizer_sha"
test "$(sha256sum "$normalizer_test" | awk '{print $1}')" = \
  "$expected_normalizer_test_sha"
test "$(sha256sum "$preparer" | awk '{print $1}')" = "$expected_preparer_sha"
test "$(sha256sum "$preparer_test" | awk '{print $1}')" = \
  "$expected_preparer_test_sha"

mkdir -p "$target_root/tools"
cp "$0" "$target_root/normalize-gate7-band-input-v2.slurm"
cp "$normalizer" "$target_root/tools/normalize_band_eigenvalue_weights_v1.py"
cp "$normalizer_test" \
  "$target_root/tools/test_normalize_band_eigenvalue_weights_v1.py"
cp "$preparer" "$target_root/tools/prepare_qsgw_band_contract_v1.py"
cp "$preparer_test" "$target_root/tools/test_prepare_qsgw_band_contract_v1.py"
cp "$source_root/PROVENANCE.txt" "$target_root/source-PROVENANCE.txt"
cp "$source_dataset/DATASET_SHA256SUMS.txt" \
  "$target_root/source-DATASET_SHA256SUMS.txt"

"$python" -B \
  "$target_root/tools/test_normalize_band_eigenvalue_weights_v1.py" \
  >"$target_root/normalizer-unit-test.stdout" \
  2>"$target_root/normalizer-unit-test.stderr"
"$python" -B "$target_root/tools/test_prepare_qsgw_band_contract_v1.py" \
  >"$target_root/preparer-unit-test.stdout" \
  2>"$target_root/preparer-unit-test.stderr"

mkdir -p "$target_dataset"
(
  cd "$source_dataset"
  find . -type f ! -name 'qsgw_input.band.contract' \
    ! -name 'qsgw_vxc_band.v2.manifest' ! -name 'DATASET_SHA256SUMS.txt' \
    -print0 | tar --null -cf - --files-from=- \
) | (
  cd "$target_dataset"
  tar -xf -
)
test "$(find "$target_dataset" -type l | wc -l)" -eq 0
test -f "$target_dataset/qsgw_input.disabled.contract"
test -f "$target_dataset/band_kpath_info"
test ! -e "$target_dataset/qsgw_input.band.contract"
test ! -e "$target_dataset/qsgw_vxc_band.v2.manifest"
test "$(find "$target_dataset" -type f | wc -l)" -eq 1369

n_normalized=0
for kpoint in $(seq -w 00001 "$expected_n_band_kpoints"); do
  file=$target_dataset/band_KS_eigenvalue_k_$kpoint.txt
  test -f "$file"
  sha256sum "$file" >>"$target_root/eigenvalue-before.sha256"
  "$python" -B \
    "$target_root/tools/normalize_band_eigenvalue_weights_v1.py" \
    "$file" "$expected_n_band_kpoints" \
    >>"$target_root/normalizer.stdout"
  n_normalized=$((n_normalized + 1))
done
test "$n_normalized" -eq "$expected_n_band_kpoints"
awk 'NR==1 && NF==5 && $3 >= 1.99 && $3 <= 2.01 {ok=1} END {exit !ok}' \
  "$target_dataset/band_KS_eigenvalue_k_00001.txt"
for kpoint in $(seq -w 00001 "$expected_n_band_kpoints"); do
  sha256sum "$target_dataset/band_KS_eigenvalue_k_$kpoint.txt" \
    >>"$target_root/eigenvalue-after.sha256"
done

"$python" -B "$target_root/tools/prepare_qsgw_band_contract_v1.py" \
  --dataset "$target_dataset" \
  >"$target_root/generator-report.json" \
  2>"$target_root/generator.stderr"

test -s "$target_dataset/qsgw_input.band.contract"
test -s "$target_dataset/qsgw_vxc_band.v2.manifest"
grep -Fqx "n_band_kpoints $expected_n_band_kpoints" \
  "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'band_update operator_fourier' \
  "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'headwing_grid disabled' "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'hartree_update off' "$target_dataset/qsgw_input.band.contract"
test "$(grep -Ec '^band_mf0_eigenvalues ' \
  "$target_dataset/qsgw_input.band.contract")" -eq \
  "$expected_n_band_kpoints"
test "$(grep -Ec '^band_mf0_wavefunctions ' \
  "$target_dataset/qsgw_input.band.contract")" -eq \
  "$expected_n_band_kpoints"
test "$(grep -Ec '^vxc_band_manifest ' \
  "$target_dataset/qsgw_input.band.contract")" -eq 1

(
  cd "$target_dataset"
  find . -type f ! -name 'DATASET_SHA256SUMS.txt' -print0 | sort -z | \
    xargs -0 sha256sum >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

band_contract_sha=$(sha256sum "$target_dataset/qsgw_input.band.contract" \
  | awk '{print $1}')
band_manifest_sha=$(sha256sum "$target_dataset/qsgw_vxc_band.v2.manifest" \
  | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$target_dataset/DATASET_SHA256SUMS.txt" \
  | awk '{print $1}')

cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=normalize_band_input_to_per_state_occupation_v2
normalization_reason=abacus_band_mode_kweighted_occupations_vs_shared_reader_per_state_assumption
reader_reference=src/api/input.cpp:librpa_set_band_occ_eigval_multiplies_by_1_over_n_kpts_band
normalization_transform=band_KS_eigenvalue_weight_column_times_n_band_kpoints
source_input_root=$source_root
source_dataset_manifest_sha256=$expected_source_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
normalizer_sha256=$expected_normalizer_sha
preparer_sha256=$expected_preparer_sha
n_band_kpoints=$expected_n_band_kpoints
normalized_file_count=$n_normalized
band_contract_sha256=$band_contract_sha
band_vxc_manifest_sha256=$band_manifest_sha
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
echo GATE7_BAND_INPUT_V2_NORMALIZE=COMPLETE
