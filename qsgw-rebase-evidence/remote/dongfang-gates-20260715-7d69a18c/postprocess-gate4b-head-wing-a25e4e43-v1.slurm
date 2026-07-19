#!/usr/bin/env bash
#SBATCH --job-name=qsgw-g4b-post
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/postprocess-gate4b-head-wing-a25e4e43-v1-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/postprocess-gate4b-head-wing-a25e4e43-v1-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_job=2348059
source_run=$root/runs/gate4b-head-wing-limit-live-a25e4e43-$source_job
gate4a_root=$root/runs/gate4-head-only-limit-live-a25e4e43-2347926
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-a25e4e43-v1
head_only_run=$gate4a_root/head-only
wing_run=$source_run/head-plus-wing
g0w0_run=$source_run/g0w0-head-plus-wing
source_tool_dir=$source_run/tools
packet=$root/evidence/gate4b-head-wing-pass-a25e4e43-$SLURM_JOB_ID
tool_dir=$packet/observers
wing_source=$root/jobs/compare_qsgw_wing_mode_pair-41a383c9.py
wing_test_source=$root/jobs/test_compare_qsgw_wing_mode_pair-5fbe9a6c.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3

expected_candidate_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
expected_candidate_exe_sha=2072fc09a44a5f246513be152a06832dccf777da04b3dd8911b70e0f9a31591e
expected_candidate_build_manifest_sha=e0377a567a3504dbc61ece4e22f2afa392a20ed13640cc1ed5f4dd909dbfe31a
expected_candidate_build_provenance_sha=8041536b8d4f267544db223c5c800ce6ac4467a97782ccb302f2bed4d485b509
expected_gate4a_manifest_sha=1d53d67c15786aae74e2f4bf190ca579827971ee81c5ea8395e5db92938a4334
expected_gate4a_provenance_sha=952fa1e4d09c00b6f0a1fd5d1fb91da492c7891877440dbb1bbf49f1c64161fe
expected_gate4a_script_sha=3bf8531fe86f601c12281f4aa5c5cc64f3b49bdaf02f570181321c68d47c3a86
expected_head_only_input_sha=afdb711940d2f3e90e1b843844b75ff75af798a8bc4f9a13277dd3a1f3b2f3cf
expected_head_only_trace_sha=1e6005087ede05211c20e11b6439fb592ce1f23e7dfabf4eef81d92076f451dd
expected_head_only_stdout_sha=35a6dda5e583326b087881edff9013a3cfafa97a8e743bd025a2388b825806a9
expected_input_dataset_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_input_output_manifest_sha=b8c26ba28f64666c9f1efa3a840bc1edd1ed15cb84a6ef911990a0c1982de4dc
expected_input_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_head_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_source_script_sha=7e5c2d424ab0599730c4deaaffb597e39e79a69a11b2d7958811e16837271e06
expected_source_provenance_sha=3f1fd8e13f78a975c9e65613dbc3de8a9132e1c9a5d814dd480d8491cd102d0f
expected_controlled_pair_sha=5ca6b3786a08117391fb63ca18383f309a3e36a4d292499d0b1c607e8f479807
expected_wing_input_sha=4cb8885afbb1417b8620b7515c20227f3a0c5ec42fa755efdd641a6372fa042c
expected_g0w0_input_sha=cdcedb1f5762f21117fbecb9fbb4a7ec5aef420f41bbc851a30e97b85f85b8de
expected_component_sha=c3daf072f222083a7ebdb9cf45f154d4bef64474f76db05992479a66fe30ebbc
expected_component_test_sha=501ef472ec997a13313081f1f5945ef47e03f83ca16a7603770b32c66dc54ced
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
expected_sigc_sha=f79f9c2cea50a941677ee8f8f18327bdf791b3c02df8b24bec7d9fdd0debbb63
expected_sigc_reader_sha=a2bf9dcec91dc01c221da0fbc84ac514b3a9ca7c83f065fcb260d7f32b5411fb
expected_head_sha=e65d30e7444d868a71d4979b7899327488d92cb5d7ece2371707aead561a1e8a
expected_head_test_sha=1b9095eb0ba32a2fe76f15620fd98d95449d6049400565547a4590b6c3331f58
expected_head_pair_sha=0ec188b1e67a480878940ac3691a4be3888789737271b8cd8049befcd322e79f
expected_wing_sha=41a383c9a61bf865ef3a0f8e3efb5140e813119c233c15f23d6e8ff067881eff
expected_wing_test_sha=5fbe9a6c3dee7550cab7e96d66ce87029af0824dd8065aaed7cc1b3150e46cc0

test ! -e "$packet"
test -d "$source_run"
test -d "$source_tool_dir"
test -f "$wing_source"
test -f "$wing_test_source"
test -e "$gate4a_root/COMPLETE"
test -e "$input_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$candidate_build/chi0_main.exe"

mkdir -p "$packet/source/head-plus-wing" \
  "$packet/source/g0w0-head-plus-wing" \
  "$packet/source/tools" \
  "$tool_dir" \
  "$packet/frozen/gate4a" \
  "$packet/frozen/candidate-build" \
  "$packet/frozen/input"
trap 'rc=$?; printf "exit_code=%s\nline=%s\ncommand=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"$packet/FAILURE.txt"; exit "$rc"' ERR

cp "$source_tool_dir"/*.py "$tool_dir/"
cp "$wing_source" "$tool_dir/compare_qsgw_wing_mode_pair.py"
cp "$wing_test_source" "$tool_dir/test_compare_qsgw_wing_mode_pair.py"

sacct -j "$source_job" \
  --format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End -P \
  >"$packet/SOURCE_JOB_SACCT.txt"
awk -F '|' '$1 == "2348059" && $3 == "FAILED" && $4 == "2:0" {found=1} END {exit !found}' "$packet/SOURCE_JOB_SACCT.txt"
awk -F '|' '$1 == "2348059.1" && $3 == "COMPLETED" && $4 == "0:0" {found=1} END {exit !found}' "$packet/SOURCE_JOB_SACCT.txt"
awk -F '|' '$1 == "2348059.2" && $3 == "COMPLETED" && $4 == "0:0" {found=1} END {exit !found}' "$packet/SOURCE_JOB_SACCT.txt"

test "$(sha256sum "$candidate_build/chi0_main.exe" | awk '{print $1}')" = "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = "$expected_candidate_build_provenance_sha"
grep -Fqx "source_commit=$expected_candidate_commit" "$candidate_build/build-provenance.txt"

test "$(sha256sum "$gate4a_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_gate4a_manifest_sha"
test "$(sha256sum "$gate4a_root/PROVENANCE.txt" | awk '{print $1}')" = "$expected_gate4a_provenance_sha"
test "$(sha256sum "$gate4a_root/gate4-head-only-limit-live-a25e4e43-v1.slurm" | awk '{print $1}')" = "$expected_gate4a_script_sha"
test "$(sha256sum "$head_only_run/librpa.in" | awk '{print $1}')" = "$expected_head_only_input_sha"
test "$(sha256sum "$head_only_run/qsgw_matrices.dat" | awk '{print $1}')" = "$expected_head_only_trace_sha"
test "$(sha256sum "$head_only_run/librpa.stdout" | awk '{print $1}')" = "$expected_head_only_stdout_sha"
(
  cd "$gate4a_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_input_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = "$expected_input_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = "$expected_input_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.head-only.contract" | awk '{print $1}')" = "$expected_head_contract_sha"

test "$(sha256sum "$source_run/gate4b-head-wing-limit-live-a25e4e43-v2.slurm" | awk '{print $1}')" = "$expected_source_script_sha"
test "$(sha256sum "$source_run/PROVENANCE.txt" | awk '{print $1}')" = "$expected_source_provenance_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.json" | awk '{print $1}')" = "$expected_controlled_pair_sha"
test "$(sha256sum "$wing_run/librpa.in" | awk '{print $1}')" = "$expected_wing_input_sha"
test "$(sha256sum "$g0w0_run/librpa.in" | awk '{print $1}')" = "$expected_g0w0_input_sha"
while read -r path expected; do
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$tool_dir/compare_qsgw_component_traces_v3.py $expected_component_sha
$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py $expected_component_test_sha
$tool_dir/validate_qsgw_trace_closure.py $expected_closure_sha
$tool_dir/test_validate_qsgw_trace_closure.py $expected_closure_test_sha
$tool_dir/validate_qsgw_fixed_basis.py $expected_fixed_sha
$tool_dir/validate_qsgw_initial_state.py $expected_initial_sha
$tool_dir/test_validate_qsgw_initial_state.py $expected_initial_test_sha
$tool_dir/compare_qsgw_trace_g0w0_sigc.py $expected_sigc_sha
$tool_dir/summarize_librpa_sigc_matrix_dumps.py $expected_sigc_reader_sha
$tool_dir/validate_qsgw_headwing_trace.py $expected_head_sha
$tool_dir/test_validate_qsgw_headwing_trace.py $expected_head_test_sha
$tool_dir/compare_qsgw_head_mode_pair.py $expected_head_pair_sha
$tool_dir/compare_qsgw_wing_mode_pair.py $expected_wing_sha
$tool_dir/test_compare_qsgw_wing_mode_pair.py $expected_wing_test_sha
EOF
test "$(find "$g0w0_run" -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
test -s "$wing_run/qsgw_matrices.dat"
test -s "$wing_run/qsgw_eigenvalues.dat"
test -s "$wing_run/qsgw_iterations.dat"
test ! -s "$wing_run/librpa.stderr"
test ! -s "$g0w0_run/librpa.stderr"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' "$wing_run/qsgw_iterations.dat")" -eq 2
grep -Fqx '# velocity fixed_basis_rotation' "$wing_run/qsgw_matrices.dat"
grep -Fqx '# headwing scf_grid_analytic_live' "$wing_run/qsgw_matrices.dat"

diff -u "$head_only_run/librpa.in" "$wing_run/librpa.in" >"$packet/HEAD_ONLY_VS_HEAD_WING_INPUT.diff" || test "$?" -eq 1
test "$(grep -c '^-option_dielect_func = 4$' "$packet/HEAD_ONLY_VS_HEAD_WING_INPUT.diff")" -eq 1
test "$(grep -c '^+option_dielect_func = 3$' "$packet/HEAD_ONLY_VS_HEAD_WING_INPUT.diff")" -eq 1
test "$(grep -Ec '^[+-][^+-]' "$packet/HEAD_ONLY_VS_HEAD_WING_INPUT.diff")" -eq 2

(
  cd "$source_run"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$packet/SOURCE_RAW_SHA256SUMS.txt"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$packet/component-unit-test.stdout" 2>"$packet/component-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure.py" \
  >"$packet/closure-unit-test.stdout" 2>"$packet/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$packet/initial-unit-test.stdout" 2>"$packet/initial-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_headwing_trace.py" \
  >"$packet/head-unit-test.stdout" 2>"$packet/head-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_wing_mode_pair.py" \
  >"$packet/wing-unit-test.stdout" 2>"$packet/wing-unit-test.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$wing_run/qsgw_matrices.dat" \
  "$wing_run/qsgw_iterations.dat" \
  "$input_dir/band_out" \
  "$packet/head-plus-wing-initial-state.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$packet/head-plus-wing-initial-state.stdout" \
  2>"$packet/head-plus-wing-initial-state.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$wing_run/qsgw_matrices.dat" \
  "$packet/head-plus-wing-closure.json" \
  --iterations 0:2 \
  --channel 0 \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$packet/head-plus-wing-closure.stdout" \
  2>"$packet/head-plus-wing-closure.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$wing_run/qsgw_matrices.dat" \
  "$wing_run/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$packet/head-plus-wing-fixed-basis.json" \
  --iterations 0:2 \
  --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  >"$packet/head-plus-wing-fixed-basis.stdout" \
  2>"$packet/head-plus-wing-fixed-basis.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$wing_run/qsgw_matrices.dat" \
  "$g0w0_run" \
  "$packet/head-plus-wing-iter1-vs-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.head-only.contract" \
  >"$packet/wing-g0w0-comparator.stdout" \
  2>"$packet/wing-g0w0-comparator.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_wing_mode_pair.py" \
  "$head_only_run/qsgw_matrices.dat" \
  "$wing_run/qsgw_matrices.dat" \
  "$head_only_run/librpa.stdout" \
  "$wing_run/librpa.stdout" \
  "$source_run/CONTROLLED_PAIR.json" \
  "$packet/wing-mode-pair-comparison.json" \
  --initial-tolerance 1e-10 \
  --invariant-tolerance 1e-10 \
  --wing-effect-minimum 1e-12 \
  >"$packet/wing-mode-pair-comparator.stdout" \
  2>"$packet/wing-mode-pair-comparator.stderr"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_headwing_trace.py" \
  "$wing_run/qsgw_matrices.dat" \
  "$packet/head-plus-wing-live-validation.json" \
  --iterations 0:2 \
  --channel 0 \
  --invariant-tolerance 1e-10 \
  --initial-replay-tolerance 1e-10 \
  --head-live-change-minimum 1e-12 \
  >"$packet/head-plus-wing-live-validator.stdout" \
  2>"$packet/head-plus-wing-live-validator.stderr"

for result in \
  "$packet/head-plus-wing-initial-state.json" \
  "$packet/head-plus-wing-closure.json" \
  "$packet/head-plus-wing-fixed-basis.json" \
  "$packet/head-plus-wing-iter1-vs-g0w0-sigc.json" \
  "$packet/wing-mode-pair-comparison.json" \
  "$packet/head-plus-wing-live-validation.json"; do
  grep -Eq '"passed"[[:space:]]*:[[:space:]]*true' "$result"
done

cp "$0" "$packet/postprocess-gate4b-head-wing-a25e4e43-v1.slurm"
cp "$source_run/gate4b-head-wing-limit-live-a25e4e43-v2.slurm" "$packet/source/"
cp "$source_run/PROVENANCE.txt" "$source_run/CONTROLLED_PAIR.json" "$packet/source/"
cp "$wing_run/librpa.in" "$wing_run/librpa.stdout" "$wing_run/librpa.stderr" \
  "$wing_run/qsgw_iterations.dat" "$wing_run/qsgw_eigenvalues.dat" \
  "$packet/source/head-plus-wing/"
cp "$g0w0_run/librpa.in" "$g0w0_run/librpa.stdout" "$g0w0_run/librpa.stderr" \
  "$packet/source/g0w0-head-plus-wing/"
cp "$source_tool_dir"/*.py "$packet/source/tools/"
cp "$root/logs/gate4b-head-wing-a25e4e43-v2-$source_job.out" \
  "$root/logs/gate4b-head-wing-a25e4e43-v2-$source_job.err" \
  "$packet/source/"

cp "$gate4a_root/OUTPUT_SHA256SUMS.txt" "$gate4a_root/PROVENANCE.txt" \
  "$packet/frozen/gate4a/"
cp "$candidate_build/SHA256SUMS.txt" "$candidate_build/build-provenance.txt" \
  "$packet/frozen/candidate-build/"
cp "$input_root/DATASET_SHA256SUMS.txt" "$input_root/OUTPUT_SHA256SUMS.txt" \
  "$input_root/PROVENANCE.txt" "$input_dir/qsgw_input.head-only.contract" \
  "$packet/frozen/input/"

{
  printf 'gate=gate4b-head-plus-wing-g0w0-limit-live-update\n'
  printf 'acceptance_gate=true\n'
  printf 'result=PASS\n'
  printf 'candidate_commit=%s\n' "$expected_candidate_commit"
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'source_numerical_job=%s\n' "$source_job"
  printf 'source_numerical_steps=2348059.1,2348059.2\n'
  printf 'source_batch_failure=observer_cli_only\n'
  printf 'source_script_sha256=%s\n' "$expected_source_script_sha"
  printf 'source_raw_manifest_sha256=%s\n' "$(sha256sum "$packet/SOURCE_RAW_SHA256SUMS.txt" | awk '{print $1}')"
  printf 'wing_observer_sha256=%s\n' "$expected_wing_sha"
  printf 'wing_observer_test_sha256=%s\n' "$expected_wing_test_sha"
  printf 'gate4a_source_job=2347926\n'
  printf 'gate4a_output_manifest_sha256=%s\n' "$expected_gate4a_manifest_sha"
  printf 'controlled_factor=parameters.head_mode\n'
  printf 'side_a=head_only_option_dielect_func_4\n'
  printf 'side_b=head_plus_wing_option_dielect_func_3\n'
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mixing=none\n'
  printf 'qsgw_iterations=0:2\n'
  printf 'observer_thresholds=initial_1e-12,closure_1e-10,hermiticity_1e-10,fixed_basis_1e-10,g0w0_max_abs_1e-10,g0w0_relF_1e-10\n'
  printf 'postprocess_slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'postprocess_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$packet/PROVENANCE.txt"

{
  printf 'QSGW_GATE4B_HEAD_PLUS_WING=PASS\n'
  printf 'source_numerical_job=%s\n' "$source_job"
  printf 'source_numerical_steps=COMPLETED\n'
  printf 'postprocess_job=%s\n' "$SLURM_JOB_ID"
} >"$packet/GATE_STATUS.txt"

touch "$packet/COMPLETE"
(
  cd "$packet"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

printf 'QSGW_GATE4B_HEAD_PLUS_WING=PASS\n'
