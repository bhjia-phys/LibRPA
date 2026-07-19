#!/usr/bin/env bash
#SBATCH --job-name=qsgw-sync-bandout
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/sync-bandout-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/sync-bandout-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/inputs/si-k444-headwing-scf-v1-2343259
source_dataset=$source_root/dataset
target_root=$root/inputs/si-k444-headwing-scf-synced-v2-$SLURM_JOB_ID
dataset=$target_root/dataset
tool_dir=$target_root/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
comparator_source=$root/jobs/compare_qsgw_band_out-2d452fe9.py
comparator_test_source=$root/jobs/test_compare_qsgw_band_out-9a6b76ab.py

expected_source_dataset_manifest_sha=16b05f98f326c9cf183cd8cc5ea9a5344892aafb77245bb5cadcd2c03d718db6
expected_source_output_manifest_sha=4455e7dd4fc3becd2979cbf843b674421b9d59029660e35e3fe3972eb4c1497f
expected_source_provenance_sha=e4bf804a732ad3f65cb32a06d8c97f86b9601673c71bc7aa3c9c34cf3b5b287c
expected_disabled_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_source_head_contract_sha=8130a0b09f76e40e211b81bcbe60372313164412b0e7239cbc101977ec1f375e
expected_source_variants_sha=ba7d46f884e2218a4e4493e3d7ef7fd50e244152adb544c341a6337d3aaee3d7
expected_source_reader_manifest_sha=24a987b711a725698c8a98c202621dd95fdbd6d42028ec1f328e6e1c539cbc73
expected_main_band_out_sha=f63409ef843dee38d65e3b8d21cc407f9f59eb0a47b9e74d45cdd15ae7bbb56b
expected_pyatb_band_out_sha=1ebf3df9022f68fd041b4582ed6f29eda1f0889b9d1d76b8a3fb124af005b0bd
expected_comparator_sha=2d452fe98a56c8e036d5a181ec9879a8ef6004f6076b6b8cf2534317d11fe9b0
expected_comparator_test_sha=9a6b76abbb6a93a6314d740c20e6535e1e862e3f4b47b135aa22a45ab8b046d4
old_band_line="velocity_mf0 $expected_pyatb_band_out_sha pyatb_librpa_df/band_out"
new_band_line="velocity_mf0 $expected_main_band_out_sha pyatb_librpa_df/band_out"

test ! -e "$target_root"
test -e "$source_root/COMPLETE"
test "$(sha256sum "$source_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_dataset_manifest_sha"
test "$(sha256sum "$source_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_output_manifest_sha"
test "$(sha256sum "$source_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_dataset/qsgw_input.disabled.contract" | awk '{print $1}')" = \
  "$expected_disabled_contract_sha"
test "$(sha256sum "$source_dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_source_head_contract_sha"
test "$(sha256sum "$source_dataset/qsgw_input.head-only.contract" | awk '{print $1}')" = \
  "$expected_source_head_contract_sha"
test "$(sha256sum "$source_root/HEAD_MODE_VARIANTS.tsv" | awk '{print $1}')" = \
  "$expected_source_variants_sha"
test "$(sha256sum "$source_root/HEADWING_READER_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_reader_manifest_sha"
test "$(sha256sum "$source_dataset/band_out" | awk '{print $1}')" = \
  "$expected_main_band_out_sha"
test "$(sha256sum "$source_dataset/pyatb_librpa_df/band_out" | awk '{print $1}')" = \
  "$expected_pyatb_band_out_sha"
test "$(sha256sum "$comparator_source" | awk '{print $1}')" = \
  "$expected_comparator_sha"
test "$(sha256sum "$comparator_test_source" | awk '{print $1}')" = \
  "$expected_comparator_test_sha"
test "$(grep -Fxc "$old_band_line" "$source_dataset/qsgw_input.contract")" -eq 1
test "$(grep -Fxc "$old_band_line" "$source_dataset/qsgw_input.head-only.contract")" -eq 1
! cmp -s "$source_dataset/band_out" "$source_dataset/pyatb_librpa_df/band_out"
(
  cd "$source_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd "$source_dataset"
  sha256sum --check --quiet "$source_root/HEADWING_READER_SHA256SUMS.txt"
)
test "$(find "$source_dataset" -type f | wc -l)" -eq 1369
test "$(find "$source_dataset" -type l | wc -l)" -eq 0

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_SYNC_BAND_OUT_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$dataset" "$tool_dir"
cp -a --reflink=auto "$source_dataset"/. "$dataset"/
cp "$0" "$target_root/normalize-headwing-band-out-a76fd826-v1.slurm"
cp "$comparator_source" "$tool_dir/compare_qsgw_band_out.py"
cp "$comparator_test_source" "$tool_dir/test_compare_qsgw_band_out.py"
cp "$source_root/DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt"
cp "$source_root/OUTPUT_SHA256SUMS.txt" \
  "$target_root/SOURCE_OUTPUT_SHA256SUMS.txt"
cp "$source_root/PROVENANCE.txt" "$target_root/SOURCE_PROVENANCE.txt"
cp "$source_root/HEADWING_READER_SHA256SUMS.txt" \
  "$target_root/SOURCE_HEADWING_READER_SHA256SUMS.txt"
cp "$source_root/HEAD_MODE_VARIANTS.tsv" "$target_root/HEAD_MODE_VARIANTS.tsv"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_band_out.py" \
  >"$target_root/comparator-unit-test.stdout" \
  2>"$target_root/comparator-unit-test.stderr"

set +e
"$python" -B "$tool_dir/compare_qsgw_band_out.py" \
  "$source_dataset/band_out" \
  "$source_dataset/pyatb_librpa_df/band_out" \
  "$target_root/source-band-out-comparison.json" \
  --eigenvalue-tolerance-ha 1e-10 \
  --occupation-tolerance 0 \
  --fermi-tolerance-ha 1e-10 \
  >"$target_root/source-band-out-comparison.stdout" \
  2>"$target_root/source-band-out-comparison.stderr"
source_comparison_status=$?
set -e
test "$source_comparison_status" -eq 1
grep -Fq '"passed": false' "$target_root/source-band-out-comparison.json"
grep -Fq '"eigenvalues_within_tolerance": false' \
  "$target_root/source-band-out-comparison.json"
grep -Fq '"occupations_within_tolerance": true' \
  "$target_root/source-band-out-comparison.json"

cp "$dataset/band_out" "$dataset/pyatb_librpa_df/band_out"
cmp -s "$dataset/band_out" "$dataset/pyatb_librpa_df/band_out"
test "$(sha256sum "$dataset/pyatb_librpa_df/band_out" | awk '{print $1}')" = \
  "$expected_main_band_out_sha"

for contract in qsgw_input.contract qsgw_input.head-only.contract; do
  test "$(grep -Fxc "$old_band_line" "$dataset/$contract")" -eq 1
  sed -i "s|^$old_band_line$|$new_band_line|" "$dataset/$contract"
  test "$(grep -Fxc "$new_band_line" "$dataset/$contract")" -eq 1
  ! grep -Fq "$old_band_line" "$dataset/$contract"
done
cmp -s "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.head-only.contract"
test "$(sha256sum "$dataset/qsgw_input.disabled.contract" | awk '{print $1}')" = \
  "$expected_disabled_contract_sha"
cmp -s "$source_dataset/qsgw_input.disabled.contract" \
  "$dataset/qsgw_input.disabled.contract"

awk '$1 == "velocity_mf0" {print $2 "  " $3}' \
  "$dataset/qsgw_input.contract" \
  >"$target_root/HEADWING_READER_SHA256SUMS.txt"
test "$(wc -l <"$target_root/HEADWING_READER_SHA256SUMS.txt")" -eq 67
test "$(grep -Fxc "$expected_main_band_out_sha  pyatb_librpa_df/band_out" \
  "$target_root/HEADWING_READER_SHA256SUMS.txt")" -eq 1
(
  cd "$dataset"
  sha256sum --check --quiet "$target_root/HEADWING_READER_SHA256SUMS.txt"
)

"$python" -B "$tool_dir/compare_qsgw_band_out.py" \
  "$dataset/band_out" \
  "$dataset/pyatb_librpa_df/band_out" \
  "$target_root/synchronized-band-out-comparison.json" \
  --eigenvalue-tolerance-ha 0 \
  --occupation-tolerance 0 \
  --fermi-tolerance-ha 0 \
  --require-byte-identical \
  >"$target_root/synchronized-band-out-comparison.stdout" \
  2>"$target_root/synchronized-band-out-comparison.stderr"
grep -Fq '"passed": true' \
  "$target_root/synchronized-band-out-comparison.json"
grep -Fq '"byte_identical": true' \
  "$target_root/synchronized-band-out-comparison.json"

(
  cd "$target_root"
  find dataset -type f -print0 | sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
test "$(wc -l <"$target_root/DATASET_SHA256SUMS.txt")" -eq 1369
grep -Ev ' dataset/(qsgw_input\.contract|qsgw_input\.head-only\.contract|pyatb_librpa_df/band_out)$' \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt" \
  >"$target_root/source-without-normalized-files.sha256"
grep -Ev ' dataset/(qsgw_input\.contract|qsgw_input\.head-only\.contract|pyatb_librpa_df/band_out)$' \
  "$target_root/DATASET_SHA256SUMS.txt" \
  >"$target_root/target-without-normalized-files.sha256"
test "$(wc -l <"$target_root/source-without-normalized-files.sha256")" -eq 1366
cmp -s "$target_root/source-without-normalized-files.sha256" \
  "$target_root/target-without-normalized-files.sha256"

head_contract_sha=$(
  sha256sum "$dataset/qsgw_input.head-only.contract" | awk '{print $1}'
)
test "$head_contract_sha" = \
  "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')"
reader_manifest_sha=$(
  sha256sum "$target_root/HEADWING_READER_SHA256SUMS.txt" | awk '{print $1}'
)
dataset_manifest_sha=$(
  sha256sum "$target_root/DATASET_SHA256SUMS.txt" | awk '{print $1}'
)
cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=synchronize_same_grid_headwing_band_out
acceptance_gate=false
candidate_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
source_failed_gate4_job=2345640
source_input_root=$source_root
source_dataset_manifest_sha256=$expected_source_dataset_manifest_sha
source_output_manifest_sha256=$expected_source_output_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
source_disabled_contract_sha256=$expected_disabled_contract_sha
source_head_contract_sha256=$expected_source_head_contract_sha
source_main_band_out_sha256=$expected_main_band_out_sha
source_pyatb_band_out_sha256=$expected_pyatb_band_out_sha
source_band_out_eigenvalue_max_abs_ha=2.1026311207705817e-06
source_band_out_eigenvalue_max_kpoint=64
source_band_out_eigenvalue_max_spin=1
source_band_out_eigenvalue_max_band=26
source_band_out_fermi_max_abs_ha=0.0030039101593575257
source_band_out_occupation_max_abs=0.0
normalization_policy=pyatb_band_out_replaced_with_main_mf0_band_out
normalization_changed_file_count=3
normalization_changed_files=qsgw_input.contract,qsgw_input.head-only.contract,pyatb_librpa_df/band_out
all_other_dataset_payloads_byte_identical=PASS
unchanged_dataset_payload_file_count=1366
target_dataset_file_count=1369
target_dataset_manifest_sha256=$dataset_manifest_sha
target_disabled_contract_sha256=$expected_disabled_contract_sha
target_head_contract_sha256=$head_contract_sha
target_main_band_out_sha256=$expected_main_band_out_sha
target_pyatb_band_out_sha256=$expected_main_band_out_sha
target_band_out_byte_identical=PASS
target_reader_manifest_sha256=$reader_manifest_sha
head_mode_variants_sha256=$expected_source_variants_sha
comparator_sha256=$expected_comparator_sha
comparator_test_sha256=$expected_comparator_test_sha
product_source_changed=false
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

sha256sum \
  "$target_root/DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_OUTPUT_SHA256SUMS.txt" \
  "$target_root/SOURCE_PROVENANCE.txt" \
  "$target_root/SOURCE_HEADWING_READER_SHA256SUMS.txt" \
  "$target_root/HEADWING_READER_SHA256SUMS.txt" \
  "$target_root/HEAD_MODE_VARIANTS.tsv" \
  "$target_root/source-without-normalized-files.sha256" \
  "$target_root/target-without-normalized-files.sha256" \
  "$target_root/source-band-out-comparison.json" \
  "$target_root/source-band-out-comparison.stdout" \
  "$target_root/source-band-out-comparison.stderr" \
  "$target_root/synchronized-band-out-comparison.json" \
  "$target_root/synchronized-band-out-comparison.stdout" \
  "$target_root/synchronized-band-out-comparison.stderr" \
  "$target_root/comparator-unit-test.stdout" \
  "$target_root/comparator-unit-test.stderr" \
  "$target_root/PROVENANCE.txt" \
  "$target_root/normalize-headwing-band-out-a76fd826-v1.slurm" \
  "$tool_dir/compare_qsgw_band_out.py" \
  "$tool_dir/test_compare_qsgw_band_out.py" \
  >"$target_root/OUTPUT_SHA256SUMS.txt"
sha256sum --check --quiet "$target_root/OUTPUT_SHA256SUMS.txt"
touch "$target_root/COMPLETE"
printf 'QSGW_SYNC_BAND_OUT_INPUT=PASS\n'
