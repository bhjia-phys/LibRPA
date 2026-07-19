#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-only
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4-head-only-a25e4e43-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4-head-only-a25e4e43-%j.err

set -euo pipefail

placeholder=REPLACE'_WITH_'
if grep -q "$placeholder" "$0"; then
  printf 'Gate 4A template has unresolved provenance placeholders\n' >&2
  exit 2
fi

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate3_root=$root/evidence/gate3-postprocess-a76fd826-2345163
frozen_gate3_root=$root/evidence/gate3-postprocess-a76fd826-2345163
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-a25e4e43-v1
candidate_exe=$candidate_build/chi0_main.exe
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate4-head-only-limit-live-a25e4e43-$SLURM_JOB_ID
disabled_run=$run_root/disabled
head_run=$run_root/head-only
g0w0_run=$run_root/g0w0-head-only
tool_dir=$run_root/tools

component_source=$root/jobs/compare_qsgw_component_traces-v4-c3daf072.py
component_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v4-501ef472.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$root/jobs/validate_qsgw_fixed_basis.py
initial_source=$root/jobs/validate_qsgw_initial_state-v1.py
initial_test_source=$root/jobs/test_validate_qsgw_initial_state-v1.py
sigc_source=$root/jobs/compare_qsgw_trace_g0w0_sigc.py
sigc_reader_source=$root/jobs/summarize_librpa_sigc_matrix_dumps.py
head_source=$root/jobs/validate_qsgw_headwing_trace-e65d30e7.py
head_test_source=$root/jobs/test_validate_qsgw_headwing_trace-1b9095eb.py
pair_source=$root/jobs/compare_qsgw_head_mode_pair-0ec188b1.py
pair_test_source=$root/jobs/test_compare_qsgw_head_mode_pair-0d3f12f1.py

expected_candidate_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
expected_candidate_exe_sha=2072fc09a44a5f246513be152a06832dccf777da04b3dd8911b70e0f9a31591e
expected_candidate_build_manifest_sha=e0377a567a3504dbc61ece4e22f2afa392a20ed13640cc1ed5f4dd909dbfe31a
expected_candidate_build_provenance_sha=8041536b8d4f267544db223c5c800ce6ac4467a97782ccb302f2bed4d485b509
expected_gate3_output_manifest_sha=95a78dd2102782142ef58aa7f60fc422ad3333aab75d57fdd2ecec015722e5a7
expected_gate3_provenance_sha=1848618f52ba2b3be4c685479d52bb652c3c1e8b0a0e21350114076a50056d35
expected_gate3_comparison_sha=651f4d8f5b06c7e82fb796a80d9326b5e3a45107becf7673ede2f1e923770e16
expected_frozen_gate3_output_sha=95a78dd2102782142ef58aa7f60fc422ad3333aab75d57fdd2ecec015722e5a7
expected_frozen_gate3_provenance_sha=1848618f52ba2b3be4c685479d52bb652c3c1e8b0a0e21350114076a50056d35
expected_frozen_gate3_miniter10_sha=651f4d8f5b06c7e82fb796a80d9326b5e3a45107becf7673ede2f1e923770e16
expected_frozen_gate3_miniter5_sha=3d4e1e1ac6cf6ecc486b0ccfa8c0122ff91d9c5c2328d8e2e94f587632bb35fd
expected_input_dataset_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_input_output_manifest_sha=b8c26ba28f64666c9f1efa3a840bc1edd1ed15cb84a6ef911990a0c1982de4dc
expected_input_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_disabled_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_head_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_head_mode_variants_sha=ba7d46f884e2218a4e4493e3d7ef7fd50e244152adb544c341a6337d3aaee3d7
expected_reader_manifest_sha=f37bbcfffb92bdef379872c7de20b7555bb1cf08fd39075521baad6ce47cac39
expected_source_band_comparison_sha=14db94c9d0d2274710a5db5a198feb8bba089994ff2f2e1fdbb6e62dca0f0502
expected_synced_band_comparison_sha=5b139b0b0d5aaddf2f0ceee632bb01f9110ce9a1a3f5d2746c328229457ef894
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
expected_pair_sha=0ec188b1e67a480878940ac3691a4be3888789737271b8cd8049befcd322e79f
expected_pair_test_sha=0d3f12f1db4d80d298487e31dd730336bd7f1f4eb0214135e4441adbee7cc216

test ! -e "$run_root"
test "${SLURM_NTASKS:-0}" -eq 4
test "${SLURM_CPUS_PER_TASK:-0}" -eq 12
test -e "$gate3_root/COMPLETE"
test -e "$input_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$candidate_exe"
test "$(sha256sum "$gate3_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate3_output_manifest_sha"
test "$(sha256sum "$gate3_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate3_provenance_sha"
test "$(sha256sum "$gate3_root/old-new-comparison.json" | awk '{print $1}')" = \
  "$expected_gate3_comparison_sha"
grep -Fq '"passed": true' "$gate3_root/old-new-comparison.json"
test -e "$frozen_gate3_root/COMPLETE"
test "$(sha256sum "$frozen_gate3_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_frozen_gate3_output_sha"
test "$(sha256sum "$frozen_gate3_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_frozen_gate3_provenance_sha"
test "$(sha256sum "$frozen_gate3_root/old-new-comparison.json" | awk '{print $1}')" = \
  "$expected_frozen_gate3_miniter10_sha"
test "$(sha256sum "$frozen_gate3_root/old-new-miniter5-comparison.json" | awk '{print $1}')" = \
  "$expected_frozen_gate3_miniter5_sha"
grep -Fq '"passed": true' \
  "$frozen_gate3_root/old-new-comparison.json"
grep -Fq '"passed": true' \
  "$frozen_gate3_root/old-new-miniter5-comparison.json"
test "$(sha256sum "$input_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_dataset_manifest_sha"
test "$(sha256sum "$input_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_output_manifest_sha"
test "$(sha256sum "$input_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_input_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.disabled.contract" | awk '{print $1}')" = \
  "$expected_disabled_contract_sha"
test "$(sha256sum "$input_dir/qsgw_input.head-only.contract" | awk '{print $1}')" = \
  "$expected_head_contract_sha"
test "$(sha256sum "$input_root/HEAD_MODE_VARIANTS.tsv" | awk '{print $1}')" = \
  "$expected_head_mode_variants_sha"
test "$(sha256sum "$input_root/HEADWING_READER_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_reader_manifest_sha"
test "$(sha256sum "$input_root/source-band-out-comparison.json" | awk '{print $1}')" = \
  "$expected_source_band_comparison_sha"
test "$(sha256sum "$input_root/synchronized-band-out-comparison.json" | awk '{print $1}')" = \
  "$expected_synced_band_comparison_sha"
grep -Fq '"passed": false' \
  "$input_root/source-band-out-comparison.json"
grep -Fq '"eigenvalues_within_tolerance": false' \
  "$input_root/source-band-out-comparison.json"
grep -Fq '"passed": true' \
  "$input_root/synchronized-band-out-comparison.json"
grep -Fq '"byte_identical": true' \
  "$input_root/synchronized-band-out-comparison.json"
grep -Fqx 'normalization_policy=pyatb_band_out_replaced_with_main_mf0_band_out' \
  "$input_root/PROVENANCE.txt"
grep -Fqx 'all_other_dataset_payloads_byte_identical=PASS' \
  "$input_root/PROVENANCE.txt"
grep -Fqx 'product_source_changed=false' "$input_root/PROVENANCE.txt"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_linux_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_linux_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"

while read -r path expected; do
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$component_source $expected_component_sha
$component_test_source $expected_component_test_sha
$closure_source $expected_closure_sha
$closure_test_source $expected_closure_test_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$initial_test_source $expected_initial_test_sha
$sigc_source $expected_sigc_sha
$sigc_reader_source $expected_sigc_reader_sha
$head_source $expected_head_sha
$head_test_source $expected_head_test_sha
$pair_source $expected_pair_sha
$pair_test_source $expected_pair_test_sha
EOF

test "$(find "$input_dir" -type l | wc -l)" -eq 0
test "$(find "$input_dir" -type f | wc -l)" -eq 1369
grep -Fqx 'n_headwing_kpoints 0' \
  "$input_dir/qsgw_input.disabled.contract"
grep -Fqx 'headwing_grid disabled' \
  "$input_dir/qsgw_input.disabled.contract"
grep -Fqx 'headwing_update none' \
  "$input_dir/qsgw_input.disabled.contract"
test "$(grep -c '^velocity_mf0 ' \
  "$input_dir/qsgw_input.disabled.contract")" -eq 0
grep -Fqx 'n_headwing_kpoints 64' \
  "$input_dir/qsgw_input.head-only.contract"
grep -Fqx 'headwing_grid scf' \
  "$input_dir/qsgw_input.head-only.contract"
grep -Fqx 'headwing_update fixed_basis_rotation' \
  "$input_dir/qsgw_input.head-only.contract"
test "$(grep -c '^velocity_mf0 ' \
  "$input_dir/qsgw_input.head-only.contract")" -eq 67

if [[ ${QSGW_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'QSGW_GATE4A_PREFLIGHT_ONLY=PASS\n'
  exit 0
fi

mkdir -p "$disabled_run" "$head_run" "$g0w0_run" "$tool_dir"
cp "$0" "$run_root/gate4-head-only-limit-live-a25e4e43-v1.slurm"
cp "$component_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$component_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$component_test_source" \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"
cp "$sigc_source" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py"
cp "$sigc_reader_source" "$tool_dir/summarize_librpa_sigc_matrix_dumps.py"
cp "$head_source" "$tool_dir/validate_qsgw_headwing_trace.py"
cp "$head_test_source" "$tool_dir/test_validate_qsgw_headwing_trace.py"
cp "$pair_source" "$tool_dir/compare_qsgw_head_mode_pair.py"
cp "$pair_test_source" "$tool_dir/test_compare_qsgw_head_mode_pair.py"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$input_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/input-OUTPUT_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_root/HEAD_MODE_VARIANTS.tsv" "$run_root/"
cp "$input_root/HEADWING_READER_SHA256SUMS.txt" "$run_root/"
cp "$input_root/source-band-out-comparison.json" \
  "$run_root/input-source-band-out-comparison.json"
cp "$input_root/synchronized-band-out-comparison.json" \
  "$run_root/input-synchronized-band-out-comparison.json"
cp "$input_dir/qsgw_input.disabled.contract" "$run_root/"
cp "$input_dir/qsgw_input.head-only.contract" "$run_root/"
cp "$gate3_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate3-OUTPUT_SHA256SUMS.txt"
cp "$gate3_root/PROVENANCE.txt" "$run_root/gate3-PROVENANCE.txt"
cp "$gate3_root/old-new-comparison.json" \
  "$run_root/gate3-old-new-comparison.json"
cp "$frozen_gate3_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/frozen-gate3-OUTPUT_SHA256SUMS.txt"
cp "$frozen_gate3_root/PROVENANCE.txt" \
  "$run_root/frozen-gate3-PROVENANCE.txt"
cp "$frozen_gate3_root/old-new-comparison.json" \
  "$run_root/frozen-gate3-miniter10.json"
cp "$frozen_gate3_root/old-new-miniter5-comparison.json" \
  "$run_root/frozen-gate3-miniter5.json"
cp "$candidate_build/SHA256SUMS.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt"
cp "$candidate_build/build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt"

(
  cd "$gate3_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd "$frozen_gate3_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd "$input_root"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  cd "$input_dir"
  sha256sum --check --quiet "$input_root/HEADWING_READER_SHA256SUMS.txt"
  cd "$candidate_build"
  sha256sum --check --quiet SHA256SUMS.txt
) >"$run_root/prerequisite-hash-check.log"

write_qsgw_input() {
  local output=$1
  local replace=$2
  local option=$3
  local contract=$4
  cat >"$output" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = false
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = $replace
option_dielect_func = $option
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = $contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF
}

write_qsgw_input "$disabled_run/librpa.in" false 0 \
  qsgw_input.disabled.contract
write_qsgw_input "$head_run/librpa.in" true 4 \
  qsgw_input.head-only.contract
grep -Ev '^(replace_w_head|option_dielect_func|qsgw_input_contract) = ' \
  "$disabled_run/librpa.in" >"$run_root/disabled-normalized.in"
grep -Ev '^(replace_w_head|option_dielect_func|qsgw_input_contract) = ' \
  "$head_run/librpa.in" >"$run_root/head-only-normalized.in"
cmp -s "$run_root/disabled-normalized.in" \
  "$run_root/head-only-normalized.in"

cat >"$g0w0_run/librpa.in" <<EOF
task = g0w0
input_dir = $input_dir
output_dir = .
fn_vxc_scf = vxc_out.dat
fn_band_kpath_info = disabled_for_grid_only_sigc_comparison
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = true
option_dielect_func = 4
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
output_energy_qp = false
EOF

disabled_input_sha=$(sha256sum "$disabled_run/librpa.in" | awk '{print $1}')
head_input_sha=$(sha256sum "$head_run/librpa.in" | awk '{print $1}')
cat >"$run_root/CONTROLLED_PAIR.json" <<EOF
{
  "schema": "qsgw-head-mode-controlled-pair-v1",
  "changed_factor": "parameters.head_mode",
  "factor_role": "head_mode",
  "common": {
    "source_commit": "$expected_candidate_commit",
    "executable_sha256": "$expected_candidate_exe_sha",
    "dataset_manifest_sha256": "$expected_input_dataset_manifest_sha",
    "mpi_ranks": $SLURM_NTASKS,
    "omp_threads": $SLURM_CPUS_PER_TASK,
    "deterministic_reduction": true
  },
  "side_a": {"parameters": {"head_mode": "disabled"}},
  "side_b": {"parameters": {"head_mode": "head_only"}},
  "variants": {
    "disabled": {
      "replace_w_head": false,
      "option_dielect_func": 0,
      "qsgw_input_contract": "qsgw_input.disabled.contract",
      "qsgw_input_contract_sha256": "$expected_disabled_contract_sha",
      "librpa_input_sha256": "$disabled_input_sha"
    },
    "head_only": {
      "replace_w_head": true,
      "option_dielect_func": 4,
      "qsgw_input_contract": "qsgw_input.head-only.contract",
      "qsgw_input_contract_sha256": "$expected_head_contract_sha",
      "librpa_input_sha256": "$head_input_sha"
    }
  }
}
EOF
"$python" -m json.tool "$run_root/CONTROLLED_PAIR.json" \
  >"$run_root/CONTROLLED_PAIR.canonical.json"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/component-unit-test.stdout" \
  2>"$run_root/component-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-unit-test.stdout" \
  2>"$run_root/initial-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_headwing_trace.py" \
  >"$run_root/head-unit-test.stdout" \
  2>"$run_root/head-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_head_mode_pair.py" \
  >"$run_root/pair-unit-test.stdout" \
  2>"$run_root/pair-unit-test.stderr"

{
  printf 'gate=gate4a-head-only-g0w0-limit-live-update\n'
  printf 'acceptance_gate=true\n'
  printf 'historical_headwing_oracle_reliable=false\n'
  printf 'historical_oracle_policy=g0w0_limit_plus_structural_invariants_plus_replay_plus_independent_pair\n'
  printf 'candidate_commit=%s\n' "$expected_candidate_commit"
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'gate3_numerical_job=2344833\n'
  printf 'gate3_acceptance_job=2345163\n'
  printf 'gate3_output_manifest_sha256=%s\n' \
    "$expected_gate3_output_manifest_sha"
  printf 'frozen_gate3_root=%s\n' "$frozen_gate3_root"
  printf 'frozen_gate3_output_manifest_sha256=%s\n' \
    "$expected_frozen_gate3_output_sha"
  printf 'frozen_gate3_provenance_sha256=%s\n' \
    "$expected_frozen_gate3_provenance_sha"
  printf 'input_root=%s\n' "$input_root"
  printf 'input_normalization_job=2345960\n'
  printf 'dataset_manifest_sha256=%s\n' \
    "$expected_input_dataset_manifest_sha"
  printf 'source_band_out_comparison_sha256=%s\n' \
    "$expected_source_band_comparison_sha"
  printf 'synchronized_band_out_comparison_sha256=%s\n' \
    "$expected_synced_band_comparison_sha"
  printf 'controlled_factor=parameters.head_mode\n'
  printf 'controlled_factor_role=head_mode\n'
  printf 'head_mode_variants_sha256=%s\n' \
    "$expected_head_mode_variants_sha"
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mixing=none\n'
  printf 'qsgw_iterations=0:2\n'
  printf 'mpi_ranks=%s\n' "$SLURM_NTASKS"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' \
    "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$run_root/PROVENANCE.txt"

(
  cd "$g0w0_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
)
(
  cd "$disabled_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
)
(
  cd "$head_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
)

test "$(find "$g0w0_run" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
for directory in "$disabled_run" "$head_run"; do
  test -f "$directory/qsgw_matrices.dat"
  test -f "$directory/qsgw_eigenvalues.dat"
  test -f "$directory/qsgw_iterations.dat"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$directory/qsgw_iterations.dat")" = 2
  grep -Fqx '# qsgw_mixer none' "$directory/qsgw_matrices.dat"
  grep -Fqx '# symmetry unsupported_full_bz_only' \
    "$directory/qsgw_matrices.dat"
done
grep -Fqx '# velocity disabled_stage1' \
  "$disabled_run/qsgw_matrices.dat"
grep -Fqx '# headwing disabled_stage1' \
  "$disabled_run/qsgw_matrices.dat"
grep -Fqx '# velocity fixed_basis_rotation' \
  "$head_run/qsgw_matrices.dat"
grep -Fqx '# headwing scf_grid_analytic_live' \
  "$head_run/qsgw_matrices.dat"

for mode in disabled head-only; do
  directory=$run_root/$mode
  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_initial_state.py" \
    "$directory/qsgw_matrices.dat" \
    "$directory/qsgw_iterations.dat" \
    "$input_dir/band_out" \
    "$run_root/$mode-initial-state.json" \
    --efermi-tolerance-ha 1e-12 \
    --occupation-tolerance 1e-12 \
    >"$run_root/$mode-initial-state.stdout" \
    2>"$run_root/$mode-initial-state.stderr"
  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_trace_closure.py" \
    "$directory/qsgw_matrices.dat" \
    "$run_root/$mode-closure.json" \
    --iterations 0:2 \
    --channel 0 \
    --closure-tolerance-ha 1e-10 \
    --hermiticity-tolerance-ha 1e-10 \
    >"$run_root/$mode-closure.stdout" \
    2>"$run_root/$mode-closure.stderr"
  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_fixed_basis.py" \
    "$directory/qsgw_matrices.dat" \
    "$directory/qsgw_eigenvalues.dat" \
    "$input_dir/band_out" \
    "$run_root/$mode-fixed-basis.json" \
    --iterations 0:2 \
    --channel 0 \
    --eigenvalue-tolerance-ha 1e-10 \
    --invariant-tolerance 1e-10 \
    >"$run_root/$mode-fixed-basis.stdout" \
    2>"$run_root/$mode-fixed-basis.stderr"
  grep -Fq '"passed": true' "$run_root/$mode-initial-state.json"
  grep -Fq '"passed": true' "$run_root/$mode-closure.json"
  grep -Fq '"passed": true' "$run_root/$mode-fixed-basis.json"
done

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" \
  "$head_run/qsgw_matrices.dat" \
  "$g0w0_run" \
  "$run_root/head-only-iter1-vs-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  --input-contract "$input_dir/qsgw_input.head-only.contract" \
  >"$run_root/head-g0w0-comparator.stdout" \
  2>"$run_root/head-g0w0-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/head-only-iter1-vs-g0w0-sigc.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_head_mode_pair.py" \
  "$disabled_run/qsgw_matrices.dat" \
  "$head_run/qsgw_matrices.dat" \
  "$run_root/CONTROLLED_PAIR.json" \
  "$run_root/head-mode-pair-comparison.json" \
  --initial-tolerance 1e-10 \
  --exx-tolerance 1e-10 \
  --head-effect-minimum 1e-12 \
  >"$run_root/head-mode-pair-comparator.stdout" \
  2>"$run_root/head-mode-pair-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/head-mode-pair-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_headwing_trace.py" \
  "$head_run/qsgw_matrices.dat" \
  "$run_root/head-live-validation.json" \
  --iterations 0:2 \
  --channel 0 \
  --invariant-tolerance 1e-10 \
  --initial-replay-tolerance 1e-10 \
  --head-live-change-minimum 1e-12 \
  >"$run_root/head-live-validator.stdout" \
  2>"$run_root/head-live-validator.stderr"
grep -Fq '"passed": true' "$run_root/head-live-validation.json"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'QSGW_GATE4A_HEAD_ONLY_LIMIT_LIVE=PASS\n'
