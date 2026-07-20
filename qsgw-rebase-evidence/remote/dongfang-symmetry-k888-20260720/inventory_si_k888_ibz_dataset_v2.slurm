#!/usr/bin/env bash
#SBATCH --job-name=qsgw-sym-k888-inventory
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/qsgw-sym-k888-inventory-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/qsgw-sym-k888-inventory-%j.err

set -euo pipefail

source_root=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-band0-k888-sym-shrink-headwing-iter1-20260514-173351
target_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-symmetry-k888-freeze-20260720-v2
jobs_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-symmetry-k888-jobs-20260720-v1
preflight_root=$jobs_root/preflight-$SLURM_JOB_ID
inventory_tool=$jobs_root/inventory_si_k888_ibz_dataset_v1.py
inventory_test=$jobs_root/test_inventory_si_k888_ibz_dataset_v1.py
slurm_source=$jobs_root/inventory_si_k888_ibz_dataset_v2.slurm
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

expected_inventory_sha=b1c25d0bc3c6460588b97f86d7e241e72e80f0fa0df1e89e8eb9172331b5256c
expected_test_sha=475f25eb8e4c8ef26b4e332b2dcc944313eab97381221e9c6e11752820d38e3c

test -d "$source_root"
test ! -e "$target_root"
test ! -e "$preflight_root"
test -f "$inventory_tool"
test -f "$inventory_test"
test -f "$slurm_source"
test "$(sha256sum "$inventory_tool" | awk '{print $1}')" = "$expected_inventory_sha"
test "$(sha256sum "$inventory_test" | awk '{print $1}')" = "$expected_test_sha"

mkdir "$preflight_root"
PYTHONPATH="$jobs_root" "$python" -B "$inventory_test" \
  >"$preflight_root/inventory-unit-test.stdout" \
  2>"$preflight_root/inventory-unit-test.stderr"

"$python" -B "$inventory_tool" \
  --source "$source_root" \
  --output "$target_root" \
  >"$preflight_root/inventory-command.stdout" \
  2>"$preflight_root/inventory-command.stderr"

mkdir "$target_root/tools"
cp "$inventory_tool" "$target_root/tools/inventory_si_k888_ibz_dataset_v1.py"
cp "$inventory_test" "$target_root/tools/test_inventory_si_k888_ibz_dataset_v1.py"
cp "$slurm_source" "$target_root/inventory_si_k888_ibz_dataset_v2.slurm"
cp "$preflight_root"/* "$target_root/"

grep -Fq '"passed": true' "$target_root/VALIDATION.json"
test "$(grep -c '^' "$target_root/SELECTED_SHA256SUMS.txt")" -gt 0
(
  cd "$source_root"
  sha256sum --check --quiet "$target_root/SELECTED_SHA256SUMS.txt"
)

inventory_sha=$(sha256sum "$target_root/inventory.json" | awk '{print $1}')
selected_manifest_sha=$(
  sha256sum "$target_root/SELECTED_SHA256SUMS.txt" | awk '{print $1}'
)
file_list_sha=$(sha256sum "$target_root/FILE_LIST.tsv" | awk '{print $1}')
slurm_sha=$(sha256sum "$slurm_source" | awk '{print $1}')
cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=freeze_historical_si_k888_ibz_input_inventory
acceptance_gate=false
source_root=$source_root
source_mutated=false
expected_grid=8x8x8
expected_scf_kpoints=29
expected_full_kpoints=512
expected_band_kpoints=143
expected_spins=1
expected_bands=44
expected_aos=44
inventory_sha256=$inventory_sha
selected_manifest_sha256=$selected_manifest_sha
file_list_sha256=$file_list_sha
inventory_tool_sha256=$expected_inventory_sha
inventory_test_sha256=$expected_test_sha
slurm_sha256=$slurm_sha
preflight_root=$preflight_root
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target_root/COMPLETE"
printf 'QSGW_SYM_K888_INVENTORY=PASS\n'
