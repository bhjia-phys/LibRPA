#!/usr/bin/env bash
#SBATCH --job-name=qsgw-headwing-input
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/headwing-input-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/headwing-input-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/inputs/si-k444-gate3-common-v3-2342408
source_dataset=$source_root/dataset
preparer_source=$root/jobs/prepare_qsgw_headwing_contract-e232cfa6.py
preparer_test_source=$root/jobs/test_prepare_qsgw_headwing_contract-29d80a81.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
target_root=$root/inputs/si-k444-headwing-scf-v1-$SLURM_JOB_ID
dataset=$target_root/dataset
tool_dir=$target_root/tools

expected_source_dataset_manifest_sha=df793f964077ca03de7ad4723975736d0d0fb85d7e548241f3a7a04677cc05f7
expected_source_output_manifest_sha=3b7811c12df23a563e597cae2fb80f592c045d27c803a1afc7ead31d7d7439d4
expected_source_provenance_sha=54e2a4b0f07883f4341d46e41a80fd8ae1ebac7e25d263699eda8819b00ce0d8
expected_source_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_preparer_sha=e232cfa6cd6f769764ff4be963046cb399a75c9c672371d737f8a7f968a76907
expected_preparer_test_sha=29d80a81e5dcda6a37ce9b44ec95ec63a7266f0d1b1718d5785161653ca89fd4

test ! -e "$target_root"
test -e "$source_root/COMPLETE"
test "$(sha256sum "$source_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_dataset_manifest_sha"
test "$(sha256sum "$source_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_output_manifest_sha"
test "$(sha256sum "$source_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_source_contract_sha"
test "$(sha256sum "$preparer_source" | awk '{print $1}')" = \
  "$expected_preparer_sha"
test "$(sha256sum "$preparer_test_source" | awk '{print $1}')" = \
  "$expected_preparer_test_sha"

mkdir -p "$dataset" "$tool_dir"
cp -a --reflink=auto "$source_dataset"/. "$dataset"/
cp "$0" "$target_root/freeze-qsgw-headwing-scf-input.slurm"
cp "$preparer_source" "$tool_dir/prepare_qsgw_headwing_contract.py"
cp "$preparer_test_source" "$tool_dir/test_prepare_qsgw_headwing_contract.py"
cp "$source_root/DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt"
cp "$source_root/OUTPUT_SHA256SUMS.txt" \
  "$target_root/SOURCE_OUTPUT_SHA256SUMS.txt"
cp "$source_root/PROVENANCE.txt" "$target_root/SOURCE_PROVENANCE.txt"

test "$(find "$dataset" -type l | wc -l)" -eq 0
test "$(find "$dataset" -type f | wc -l)" -eq 1367
(
  cd "$source_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

PYTHONPATH="$tool_dir" QSGW_HEADWING_PREPARER="$tool_dir/prepare_qsgw_headwing_contract.py" \
  "$python" -B "$tool_dir/test_prepare_qsgw_headwing_contract.py" \
  >"$target_root/preparer-unit-test.stdout" \
  2>"$target_root/preparer-unit-test.stderr"
cp "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.disabled.contract"
"$python" -B "$tool_dir/prepare_qsgw_headwing_contract.py" \
  "$dataset" "$target_root/preparation.json" \
  --expected-source-contract-sha256 "$expected_source_contract_sha" \
  --velocity-filename velocity_matrix \
  >"$target_root/preparer.stdout" 2>"$target_root/preparer.stderr"
cp "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.head-only.contract"

cmp -s "$source_dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.disabled.contract"
cmp -s "$dataset/qsgw_input.contract" \
  "$dataset/qsgw_input.head-only.contract"

grep -Fq '"passed": true' "$target_root/preparation.json"
grep -Fqx 'n_headwing_kpoints 64' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_grid scf' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_update fixed_basis_rotation' \
  "$dataset/qsgw_input.contract"
test "$(grep -c '^velocity_mf0 ' "$dataset/qsgw_input.contract")" -eq 67
grep -Eq '^velocity_mf0 [0-9a-f]{64} pyatb_librpa_df/k_path_info$' \
  "$dataset/qsgw_input.contract"
grep -Eq '^velocity_mf0 [0-9a-f]{64} pyatb_librpa_df/band_out$' \
  "$dataset/qsgw_input.contract"
grep -Eq '^velocity_mf0 [0-9a-f]{64} pyatb_librpa_df/velocity_matrix$' \
  "$dataset/qsgw_input.contract"
for kpoint in $(seq 0 63); do
  grep -Eq "^velocity_mf0 [0-9a-f]{64} pyatb_librpa_df/KS_eigenvector_${kpoint}\\.dat$" \
    "$dataset/qsgw_input.contract"
done
! grep -Eq '^velocity_mf0 [0-9a-f]{64} velocity_matrix$' \
  "$dataset/qsgw_input.contract"
test "$(grep -c '^headwing_' "$dataset/qsgw_input.contract")" -eq 2

awk '$1 == "velocity_mf0" {print $2 "  " $3}' \
  "$dataset/qsgw_input.contract" \
  >"$target_root/HEADWING_READER_SHA256SUMS.txt"
test "$(wc -l <"$target_root/HEADWING_READER_SHA256SUMS.txt")" -eq 67
(
  cd "$dataset"
  sha256sum --check --quiet "$target_root/HEADWING_READER_SHA256SUMS.txt"
)

(
  cd "$target_root"
  find dataset -type f -print0 | sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
test "$(wc -l <"$target_root/DATASET_SHA256SUMS.txt")" -eq 1369
grep -v ' dataset/qsgw_input.contract$' \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt" \
  >"$target_root/source-without-contract.sha256"
grep -Ev ' dataset/qsgw_input(\.disabled|\.head-only)?\.contract$' \
  "$target_root/DATASET_SHA256SUMS.txt" \
  >"$target_root/target-without-contract.sha256"
cmp -s "$target_root/source-without-contract.sha256" \
  "$target_root/target-without-contract.sha256"

disabled_contract_sha=$(
  sha256sum "$dataset/qsgw_input.disabled.contract" | awk '{print $1}'
)
head_only_contract_sha=$(
  sha256sum "$dataset/qsgw_input.head-only.contract" | awk '{print $1}'
)
test "$disabled_contract_sha" = "$expected_source_contract_sha"
test "$head_only_contract_sha" = \
  "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')"
velocity_reader_manifest_sha=$(
  sha256sum "$target_root/HEADWING_READER_SHA256SUMS.txt" | awk '{print $1}'
)
cat >"$target_root/HEAD_MODE_VARIANTS.tsv" <<EOF
head_mode	replace_w_head	option_dielect_func	qsgw_input_contract
disabled	false	0	qsgw_input.disabled.contract
head_only	true	4	qsgw_input.head-only.contract
EOF
head_mode_variants_sha=$(
  sha256sum "$target_root/HEAD_MODE_VARIANTS.tsv" | awk '{print $1}'
)
cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=freeze_same_grid_live_headwing_input
acceptance_gate=false
controlled_factor=parameters.head_mode
controlled_factor_role=head_mode
source_input_freeze_job=2342408
source_input_root=$source_root
source_dataset_manifest_sha256=$expected_source_dataset_manifest_sha
source_output_manifest_sha256=$expected_source_output_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
source_contract_sha256=$expected_source_contract_sha
disabled_contract=qsgw_input.disabled.contract
disabled_contract_sha256=$disabled_contract_sha
head_only_contract=qsgw_input.head-only.contract
head_only_contract_sha256=$head_only_contract_sha
head_mode_variants=HEAD_MODE_VARIANTS.tsv
head_mode_variants_sha256=$head_mode_variants_sha
velocity_reader_set=pyatb_librpa_df
velocity_file_count=67
velocity_reader_manifest=HEADWING_READER_SHA256SUMS.txt
velocity_reader_manifest_sha256=$velocity_reader_manifest_sha
n_scf_kpoints=64
n_headwing_kpoints=64
headwing_grid=scf
headwing_update=fixed_basis_rotation
all_noncontract_payloads_byte_identical=PASS
shared_physical_payload_file_count=1366
dataset_file_count=1369
preparer_sha256=$expected_preparer_sha
preparer_test_sha256=$expected_preparer_test_sha
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

sha256sum \
  "$target_root/DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_DATASET_SHA256SUMS.txt" \
  "$target_root/SOURCE_OUTPUT_SHA256SUMS.txt" \
  "$target_root/SOURCE_PROVENANCE.txt" \
  "$target_root/HEADWING_READER_SHA256SUMS.txt" \
  "$target_root/HEAD_MODE_VARIANTS.tsv" \
  "$target_root/source-without-contract.sha256" \
  "$target_root/target-without-contract.sha256" \
  "$target_root/preparation.json" \
  "$target_root/PROVENANCE.txt" \
  "$target_root/freeze-qsgw-headwing-scf-input.slurm" \
  "$tool_dir/prepare_qsgw_headwing_contract.py" \
  "$tool_dir/test_prepare_qsgw_headwing_contract.py" \
  "$target_root/preparer-unit-test.stdout" \
  "$target_root/preparer-unit-test.stderr" \
  "$target_root/preparer.stdout" \
  "$target_root/preparer.stderr" \
  >"$target_root/OUTPUT_SHA256SUMS.txt"
sha256sum --check --quiet "$target_root/OUTPUT_SHA256SUMS.txt"
touch "$target_root/COMPLETE"
printf 'QSGW_HEADWING_SCF_INPUT_FREEZE=PASS\n'
