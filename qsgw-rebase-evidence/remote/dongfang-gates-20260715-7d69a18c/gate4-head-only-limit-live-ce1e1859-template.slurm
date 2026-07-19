#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-only
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4-head-only-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4-head-only-%j.err

set -euo pipefail

placeholder=REPLACE'_WITH_'
if grep -q "$placeholder" "$0"; then
  printf 'Gate 4A template has unresolved provenance placeholders\n' >&2
  exit 2
fi

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate3_root=$root/runs/gate3-old-new-direct-miniter10-ce1e1859-2342954
input_root=$root/inputs/REPLACE_WITH_HEADWING_INPUT_ROOT
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-ce1e1859-v2
candidate_exe=$candidate_build/chi0_main.exe
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate4-head-only-limit-live-ce1e1859-$SLURM_JOB_ID
disabled_run=$run_root/disabled
head_run=$run_root/head-only
g0w0_run=$run_root/g0w0-head-only
tool_dir=$run_root/tools

component_source=$root/jobs/compare_qsgw_component_traces-v3-9f9ec9be.py
component_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v3-72a79b04.py
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

expected_candidate_commit=ce1e1859316e1b659df0178b553bbbfba2ca1173
expected_candidate_exe_sha=4ec3efb01e7e706356dac9294e5564edf439753e50f21cf98c8f4f37ed006ee0
expected_candidate_build_manifest_sha=056dbe1942bed448922d716abcd31500d6574a984d664cd79b11eb6c333d5bfc
expected_candidate_build_provenance_sha=82ed702891ef6c4e0149e63ec7b9362cf4205a4c14b0ce48fa098166543a47c0
expected_gate3_output_manifest_sha=REPLACE_WITH_GATE3_OUTPUT_MANIFEST_SHA
expected_gate3_provenance_sha=REPLACE_WITH_GATE3_PROVENANCE_SHA
expected_gate3_comparison_sha=REPLACE_WITH_GATE3_COMPARISON_SHA
expected_input_dataset_manifest_sha=REPLACE_WITH_HEADWING_DATASET_MANIFEST_SHA
expected_input_output_manifest_sha=REPLACE_WITH_HEADWING_OUTPUT_MANIFEST_SHA
expected_input_provenance_sha=REPLACE_WITH_HEADWING_PROVENANCE_SHA
expected_disabled_contract_sha=REPLACE_WITH_DISABLED_CONTRACT_SHA
expected_head_contract_sha=REPLACE_WITH_HEAD_ONLY_CONTRACT_SHA
expected_head_mode_variants_sha=REPLACE_WITH_HEAD_MODE_VARIANTS_SHA
expected_reader_manifest_sha=REPLACE_WITH_HEADWING_READER_MANIFEST_SHA
expected_component_sha=9f9ec9beb0d08754d1f9a39cb528df9aa4000df7f3443479984ec2df6144bf89
expected_component_test_sha=72a79b04274c4c1dd13ef9ee3dcfffea7d0d972d8ad7cfe928ad0c23dc5cc9f8
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
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' "$candidate_build/build-provenance.txt"

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

mkdir -p "$disabled_run" "$head_run" "$g0w0_run" "$tool_dir"
cp "$0" "$run_root/gate4-head-only-limit-live-ce1e1859.slurm"
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
cp "$input_dir/qsgw_input.disabled.contract" "$run_root/"
cp "$input_dir/qsgw_input.head-only.contract" "$run_root/"
cp "$gate3_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate3-OUTPUT_SHA256SUMS.txt"
cp "$gate3_root/PROVENANCE.txt" "$run_root/gate3-PROVENANCE.txt"
cp "$gate3_root/old-new-comparison.json" \
  "$run_root/gate3-old-new-comparison.json"
cp "$candidate_build/SHA256SUMS.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt"
cp "$candidate_build/build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt"

(
  cd "$gate3_root"
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
  printf 'gate3_job=2342954\n'
  printf 'gate3_output_manifest_sha256=%s\n' \
    "$expected_gate3_output_manifest_sha"
  printf 'input_root=%s\n' "$input_root"
  printf 'dataset_manifest_sha256=%s\n' \
    "$expected_input_dataset_manifest_sha"
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
