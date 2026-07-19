#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g0w0-repeat-post
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/g0w0-repeat-post-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/g0w0-repeat-post-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate4_run=$root/runs/gate4-head-only-limit-live-a76fd826-2345990
source_g0w0=$gate4_run/g0w0-head-only
source_qsgw=$gate4_run/head-only
repeat_root=$root/diagnostics/gate4-g0w0-repeatability-a76fd826-2346116
repeat_g0w0=$repeat_root/g0w0-repeat
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
target_root=$root/evidence/gate4-g0w0-repeatability-postprocess-a76fd826-$SLURM_JOB_ID
tool_dir=$target_root/tools
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
directory_comparator_source=$root/jobs/compare_g0w0_sigc_dump_directories-15417e56.py
directory_comparator_test_source=$root/jobs/test_compare_g0w0_sigc_dump_directories-5ce8acf1.py

expected_repeat_script_sha=46ae63c02db003145122b883b984248b1b7a4fe2f9327072e627d4c82fcdd493
expected_source_manifest_sha=0489ef13d60ab7db841de38a84bc0eafe7a56d710cda80cf1f5d2098e000a2bc
expected_repeat_manifest_sha=b6ea9015db0ddcf3a107974f5f937033cd71871d1a7d2cd7b86755ebb24e4d21
expected_source_input_sha=41c115083cb2776e0b772571d33becf3968de6a55b39792034099b3830c030c5
expected_source_trace_sha=f4fe1bdea67f86e9b141a02bcb2bee040d8ad093ff58ef4d5cbde34e7ae33c7c
expected_source_qsgw_comparison_sha=fcbafa525de4515145560ee793947bd3e11d733d57edd732c82b43610f76872c
expected_input_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_directory_comparator_sha=15417e56ce7b92fc719eea6788944de8805fba5e5849698fa32a89206cba2eaa
expected_directory_comparator_test_sha=5ce8acf1c8659c3c5e8f20a61d62e8347f987f7b3adeb09b4c7cb3ce5cef2dcd
expected_qsgw_comparator_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_qsgw_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_qsgw_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_component_reader_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc

test ! -e "$target_root"
test ! -e "$repeat_root/COMPLETE"
test ! -e "$gate4_run/COMPLETE"
test "$(sha256sum "$repeat_root/diagnose-gate4-g0w0-repeatability-a76fd826-v1.slurm" | awk '{print $1}')" = \
  "$expected_repeat_script_sha"
test "$(sha256sum "$repeat_root/source-g0w0-dump-manifest.txt" | awk '{print $1}')" = \
  "$expected_source_manifest_sha"
test "$(sha256sum "$repeat_root/repeat-g0w0-dump-manifest.txt" | awk '{print $1}')" = \
  "$expected_repeat_manifest_sha"
test "$(sha256sum "$source_g0w0/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$repeat_g0w0/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$source_qsgw/qsgw_matrices.dat" | awk '{print $1}')" = \
  "$expected_source_trace_sha"
test "$(sha256sum "$gate4_run/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = \
  "$expected_source_qsgw_comparison_sha"
test "$(sha256sum "$input_dir/qsgw_input.head-only.contract" | awk '{print $1}')" = \
  "$expected_input_contract_sha"
test "$(sha256sum "$directory_comparator_source" | awk '{print $1}')" = \
  "$expected_directory_comparator_sha"
test "$(sha256sum "$directory_comparator_test_source" | awk '{print $1}')" = \
  "$expected_directory_comparator_test_sha"
test "$(sha256sum "$repeat_root/tools/compare_qsgw_trace_g0w0_sigc.py" | awk '{print $1}')" = \
  "$expected_qsgw_comparator_sha"
test "$(sha256sum "$repeat_root/tools/summarize_librpa_sigc_matrix_dumps.py" | awk '{print $1}')" = \
  "$expected_qsgw_reader_sha"
test "$(sha256sum "$repeat_root/tools/validate_qsgw_trace_closure.py" | awk '{print $1}')" = \
  "$expected_qsgw_closure_sha"
test "$(sha256sum "$gate4_run/tools/compare_qsgw_component_traces.py" | awk '{print $1}')" = \
  "$expected_component_reader_sha"
test "$(find "$source_g0w0" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
test "$(find "$repeat_g0w0" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
grep -Fq 'libRPA finished successfully' "$source_g0w0/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$repeat_g0w0/librpa.stdout"

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4_G0W0_REPEAT_POST_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$tool_dir"
cp "$0" "$target_root/postprocess-gate4-g0w0-repeatability-a76fd826-v1.slurm"
cp "$directory_comparator_source" \
  "$tool_dir/compare_g0w0_sigc_dump_directories.py"
cp "$directory_comparator_test_source" \
  "$tool_dir/test_compare_g0w0_sigc_dump_directories.py"
cp "$repeat_root/tools/compare_qsgw_trace_g0w0_sigc.py" "$tool_dir/"
cp "$repeat_root/tools/summarize_librpa_sigc_matrix_dumps.py" "$tool_dir/"
cp "$repeat_root/tools/validate_qsgw_trace_closure.py" "$tool_dir/"
cp "$gate4_run/tools/compare_qsgw_component_traces.py" "$tool_dir/"
cp "$repeat_root/source-g0w0-dump-manifest.txt" "$target_root/"
cp "$repeat_root/repeat-g0w0-dump-manifest.txt" "$target_root/"
cp "$gate4_run/head-only-iter1-vs-g0w0-sigc.json" \
  "$target_root/qsgw-vs-source-g0w0.json"
cp "$repeat_root/PROVENANCE.txt" "$target_root/repeat-PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_g0w0_sigc_dump_directories.py" \
  >"$target_root/directory-comparator-unit-test.stdout" \
  2>"$target_root/directory-comparator-unit-test.stderr"

set +e
"$python" -B "$tool_dir/compare_g0w0_sigc_dump_directories.py" \
  "$source_g0w0" \
  "$repeat_g0w0" \
  "$target_root/source-vs-repeat-g0w0.json" \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  >"$target_root/source-vs-repeat-g0w0.stdout" \
  2>"$target_root/source-vs-repeat-g0w0.stderr"
directory_status=$?
set -e
test "$directory_status" -eq 0 -o "$directory_status" -eq 1
! grep -Fq '"error"' "$target_root/source-vs-repeat-g0w0.json"

set +e
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$source_qsgw/qsgw_matrices.dat" \
  "$repeat_g0w0" \
  "$target_root/qsgw-vs-repeat-g0w0.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.head-only.contract" \
  >"$target_root/qsgw-vs-repeat-g0w0.stdout" \
  2>"$target_root/qsgw-vs-repeat-g0w0.stderr"
qsgw_status=$?
set -e
test "$qsgw_status" -eq 0 -o "$qsgw_status" -eq 1
! grep -Fq '"error"' "$target_root/qsgw-vs-repeat-g0w0.json"

g0w0_max=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_abs_difference_ha"])' \
    "$target_root/source-vs-repeat-g0w0.json"
)
g0w0_relative=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["relative_frobenius_difference"])' \
    "$target_root/source-vs-repeat-g0w0.json"
)
g0w0_exact_blocks=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["exact_block_count"])' \
    "$target_root/source-vs-repeat-g0w0.json"
)
qsgw_repeat_max=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_abs_difference_ha"])' \
    "$target_root/qsgw-vs-repeat-g0w0.json"
)
qsgw_repeat_relative=$(
  "$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["relative_frobenius_difference"])' \
    "$target_root/qsgw-vs-repeat-g0w0.json"
)

cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=gate4_g0w0_repeatability_postprocess
acceptance_gate=false
source_gate4_job=2345990
repeat_g0w0_job=2346116
source_g0w0_manifest_sha256=$expected_source_manifest_sha
repeat_g0w0_manifest_sha256=$expected_repeat_manifest_sha
g0w0_byte_identical=false
g0w0_exact_block_count=$g0w0_exact_blocks
g0w0_max_abs_difference_ha=$g0w0_max
g0w0_relative_frobenius_difference=$g0w0_relative
qsgw_vs_source_max_abs_difference_ha=2.479842502490722e-10
qsgw_vs_source_relative_frobenius_difference=2.174636782761653e-11
qsgw_vs_repeat_max_abs_difference_ha=$qsgw_repeat_max
qsgw_vs_repeat_relative_frobenius_difference=$qsgw_repeat_relative
directory_comparator_exit_status=$directory_status
qsgw_comparator_exit_status=$qsgw_status
product_source_changed=false
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target_root/COMPLETE"
printf 'QSGW_GATE4_G0W0_REPEAT_POSTPROCESS=PASS\n'
