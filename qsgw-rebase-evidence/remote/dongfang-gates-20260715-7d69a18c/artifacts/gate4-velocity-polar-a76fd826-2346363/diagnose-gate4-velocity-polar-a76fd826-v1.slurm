#!/usr/bin/env bash
#SBATCH --job-name=qsgw-vpolar
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/velocity-polar-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/velocity-polar-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
dataset=$input_root/dataset
output=$root/diagnostics/gate4-velocity-polar-a76fd826-$SLURM_JOB_ID
tool_dir=$output/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
diagnostic_source=$root/jobs/diagnose_qsgw_velocity_polar-c5dc611e.py
test_source=$root/jobs/test_diagnose_qsgw_velocity_polar-d72f121a.py
helper_source=$root/jobs/diagnose_headwing_basis.py

expected_dataset_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_input_output_manifest_sha=b8c26ba28f64666c9f1efa3a840bc1edd1ed15cb84a6ef911990a0c1982de4dc
expected_input_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_reader_manifest_sha=f37bbcfffb92bdef379872c7de20b7555bb1cf08fd39075521baad6ce47cac39
expected_head_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_band_out_sha=f63409ef843dee38d65e3b8d21cc407f9f59eb0a47b9e74d45cdd15ae7bbb56b
expected_diagnostic_sha=c5dc611e53a46efc55d183206188c3e051ffb7b42dab78eacb33b0c687ce84b4
expected_test_sha=d72f121a01c68acbd22e96487869ba8d2e738be23cdaa184df5403c7aff14934
expected_helper_sha=f58854c4a59e54463eefd3ffdf46024537a8376a690a1fceb206fc8bf5a4c892

test ! -e "$output"
test -e "$input_root/COMPLETE"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_input_provenance_sha"
test "$(sha256sum "$input_root/HEADWING_READER_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_reader_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.head-only.contract" | awk '{print $1}')" = \
  "$expected_head_contract_sha"
test "$(sha256sum "$dataset/band_out" | awk '{print $1}')" = \
  "$expected_band_out_sha"
test "$(sha256sum "$dataset/pyatb_librpa_df/band_out" | awk '{print $1}')" = \
  "$expected_band_out_sha"
test "$(sha256sum "$diagnostic_source" | awk '{print $1}')" = \
  "$expected_diagnostic_sha"
test "$(sha256sum "$test_source" | awk '{print $1}')" = \
  "$expected_test_sha"
test "$(sha256sum "$helper_source" | awk '{print $1}')" = \
  "$expected_helper_sha"
cmp -s "$dataset/band_out" "$dataset/pyatb_librpa_df/band_out"
(
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4_VELOCITY_POLAR_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$tool_dir"
cp "$0" "$output/diagnose-gate4-velocity-polar-a76fd826-v1.slurm"
cp "$diagnostic_source" "$tool_dir/diagnose_qsgw_velocity_polar.py"
cp "$test_source" "$tool_dir/test_diagnose_qsgw_velocity_polar.py"
cp "$helper_source" "$tool_dir/diagnose_headwing_basis.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_diagnose_qsgw_velocity_polar.py" \
  >"$output/unit-test.stdout" \
  2>"$output/unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/diagnose_qsgw_velocity_polar.py" \
  "$dataset" \
  "$output/velocity-polar-diagnostic.json" \
  "$output/velocity-polar-per-kpoint.tsv" \
  >"$output/diagnostic.stdout" \
  2>"$output/diagnostic.stderr"

raw_unitarity=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["transform"]["raw_unitarity_maximum_absolute"])' \
    "$output/velocity-polar-diagnostic.json"
)
polar_unitarity=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["transform"]["polar_unitarity_maximum_absolute"])' \
    "$output/velocity-polar-diagnostic.json"
)
raw_reference_velocity=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["velocity"]["raw_vs_reference_maximum_absolute"])' \
    "$output/velocity-polar-diagnostic.json"
)
polar_reference_velocity=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["velocity"]["polar_vs_reference_maximum_absolute"])' \
    "$output/velocity-polar-diagnostic.json"
)
raw_polar_velocity=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["velocity"]["raw_vs_polar_maximum_absolute"])' \
    "$output/velocity-polar-diagnostic.json"
)

cat >"$output/PROVENANCE.txt" <<EOF
analysis=gate4_velocity_polar_projection_diagnostic
acceptance_gate=false
candidate_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
source_failed_gate4_job=2345990
input_root=$input_root
dataset_manifest_sha256=$expected_dataset_manifest_sha
input_output_manifest_sha256=$expected_input_output_manifest_sha
input_provenance_sha256=$expected_input_provenance_sha
diagnostic_script_sha256=$expected_diagnostic_sha
diagnostic_test_sha256=$expected_test_sha
diagnostic_helper_sha256=$expected_helper_sha
raw_unitarity_max_abs=$raw_unitarity
polar_unitarity_max_abs=$polar_unitarity
raw_vs_reference_velocity_max_abs=$raw_reference_velocity
polar_vs_reference_velocity_max_abs=$polar_reference_velocity
raw_vs_polar_velocity_max_abs=$raw_polar_velocity
product_source_changed=false
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
printf 'QSGW_GATE4_VELOCITY_POLAR_DIAGNOSTIC=PASS\n'
