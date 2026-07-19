#!/usr/bin/env bash
#SBATCH --job-name=qsgw-hw-basis
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/diagnose-headwing-basis-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/diagnose-headwing-basis-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-headwing-scf-v1-2343259
dataset=$input_root/dataset
source_script=$root/jobs/diagnose_headwing_basis.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
output=$root/diagnostics/headwing-basis-b2bc09d0-$SLURM_JOB_ID
expected_dataset_manifest_sha=16b05f98f326c9cf183cd8cc5ea9a5344892aafb77245bb5cadcd2c03d718db6
expected_script_sha=f58854c4a59e54463eefd3ffdf46024537a8376a690a1fceb206fc8bf5a4c892

test ! -e "$output"
test -e "$input_root/COMPLETE"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$source_script" | awk '{print $1}')" = \
  "$expected_script_sha"

mkdir -p "$output"
cp "$source_script" "$output/diagnose_headwing_basis.py"
cp "$0" "$output/diagnose-headwing-basis-b2bc09d0.slurm"
(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$python" -B "$source_script" \
  "$dataset" \
  "$output/headwing-basis-diagnostic.json" \
  "$output/headwing-basis-per-band.tsv" \
  >"$output/diagnostic.stdout" \
  2>"$output/diagnostic.stderr"

cat >"$output/PROVENANCE.txt" <<EOF
analysis=qsgw_headwing_velocity_basis_relation
acceptance_gate=false
candidate_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
failed_gate4_job=2343262
input_root=$input_root
dataset_manifest_sha256=$expected_dataset_manifest_sha
diagnostic_script_sha256=$expected_script_sha
execution_surface=dongfang_sbatch_compute_node
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$output"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$output/COMPLETE"
printf 'QSGW_HEADWING_BASIS_DIAGNOSTIC=PASS\n'
