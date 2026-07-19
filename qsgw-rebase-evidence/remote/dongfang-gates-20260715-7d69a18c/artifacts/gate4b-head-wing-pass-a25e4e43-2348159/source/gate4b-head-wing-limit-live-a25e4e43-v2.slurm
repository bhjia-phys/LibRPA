#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-wing
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4b-head-wing-a25e4e43-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate4b-head-wing-a25e4e43-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
gate4a_root=$root/runs/gate4-head-only-limit-live-a25e4e43-2347926
input_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
input_dir=$input_root/dataset
candidate_build=$root/build-candidate-a25e4e43-v1
candidate_exe=$candidate_build/chi0_main.exe
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate4b-head-wing-limit-live-a25e4e43-$SLURM_JOB_ID
head_only_run=$gate4a_root/head-only
wing_run=$run_root/head-plus-wing
g0w0_run=$run_root/g0w0-head-plus-wing
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
head_pair_source=$root/jobs/compare_qsgw_head_mode_pair-0ec188b1.py
wing_source=$root/jobs/compare_qsgw_wing_mode_pair-88651fe4.py
wing_test_source=$root/jobs/test_compare_qsgw_wing_mode_pair-c9fb2fad.py

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
expected_wing_sha=88651fe43f41741f9f56ce5a601f1ecbc85303654abaac700e561da7de2b461a
expected_wing_test_sha=c9fb2fad97581b19bcec581a0a11ca97e02271af78a6cf043bbea4fac89f0e25

test ! -e "$run_root"
test "${SLURM_NTASKS:-0}" -eq 4
test "${SLURM_CPUS_PER_TASK:-0}" -eq 12
test -e "$gate4a_root/COMPLETE"
test -e "$input_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$expected_candidate_exe_sha"
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
$head_pair_source $expected_head_pair_sha
$wing_source $expected_wing_sha
$wing_test_source $expected_wing_test_sha
EOF

mkdir -p "$wing_run" "$g0w0_run" "$tool_dir"
cp "$0" "$run_root/gate4b-head-wing-limit-live-a25e4e43-v2.slurm"
cp "$gate4a_root/OUTPUT_SHA256SUMS.txt" "$run_root/gate4a-OUTPUT_SHA256SUMS.txt"
cp "$gate4a_root/PROVENANCE.txt" "$run_root/gate4a-PROVENANCE.txt"
cp "$candidate_build/SHA256SUMS.txt" "$run_root/candidate-build-SHA256SUMS.txt"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$input_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$input_root/OUTPUT_SHA256SUMS.txt" "$run_root/input-OUTPUT_SHA256SUMS.txt"
cp "$input_root/PROVENANCE.txt" "$run_root/input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.head-only.contract" "$run_root/"
cp "$component_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$component_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$component_test_source" "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"
cp "$sigc_source" "$tool_dir/compare_qsgw_trace_g0w0_sigc.py"
cp "$sigc_reader_source" "$tool_dir/summarize_librpa_sigc_matrix_dumps.py"
cp "$head_source" "$tool_dir/validate_qsgw_headwing_trace.py"
cp "$head_test_source" "$tool_dir/test_validate_qsgw_headwing_trace.py"
cp "$head_pair_source" "$tool_dir/compare_qsgw_head_mode_pair.py"
cp "$wing_source" "$tool_dir/compare_qsgw_wing_mode_pair.py"
cp "$wing_test_source" "$tool_dir/test_compare_qsgw_wing_mode_pair.py"

cat >"$wing_run/librpa.in" <<EOF
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
replace_w_head = true
option_dielect_func = 3
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.head-only.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

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
option_dielect_func = 3
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
output_energy_qp = false
EOF

grep -Ev '^option_dielect_func = ' "$head_only_run/librpa.in" >"$run_root/head-only-normalized.in"
grep -Ev '^option_dielect_func = ' "$wing_run/librpa.in" >"$run_root/head-plus-wing-normalized.in"
cmp -s "$run_root/head-only-normalized.in" "$run_root/head-plus-wing-normalized.in"
wing_input_sha=$(sha256sum "$wing_run/librpa.in" | awk '{print $1}')
cat >"$run_root/CONTROLLED_PAIR.json" <<EOF
{
  "schema": "qsgw-wing-mode-controlled-pair-v1",
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
  "side_a": {"parameters": {"head_mode": "head_only"}},
  "side_b": {"parameters": {"head_mode": "head_plus_wing"}},
  "variants": {
    "head_only": {
      "replace_w_head": true,
      "option_dielect_func": 4,
      "qsgw_input_contract": "qsgw_input.head-only.contract",
      "qsgw_input_contract_sha256": "$expected_head_contract_sha",
      "librpa_input_sha256": "$expected_head_only_input_sha"
    },
    "head_plus_wing": {
      "replace_w_head": true,
      "option_dielect_func": 3,
      "qsgw_input_contract": "qsgw_input.head-only.contract",
      "qsgw_input_contract_sha256": "$expected_head_contract_sha",
      "librpa_input_sha256": "$wing_input_sha"
    }
  }
}
EOF
"$python" -m json.tool "$run_root/CONTROLLED_PAIR.json" >"$run_root/CONTROLLED_PAIR.canonical.json"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1

PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" >"$run_root/component-unit-test.stdout" 2>"$run_root/component-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/test_validate_qsgw_trace_closure.py" >"$run_root/closure-unit-test.stdout" 2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/test_validate_qsgw_initial_state.py" >"$run_root/initial-unit-test.stdout" 2>"$run_root/initial-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/test_validate_qsgw_headwing_trace.py" >"$run_root/head-unit-test.stdout" 2>"$run_root/head-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/test_compare_qsgw_wing_mode_pair.py" >"$run_root/wing-unit-test.stdout" 2>"$run_root/wing-unit-test.stderr"

{
  printf 'gate=gate4b-head-plus-wing-g0w0-limit-live-update\n'
  printf 'acceptance_gate=true\n'
  printf 'candidate_commit=%s\n' "$expected_candidate_commit"
  printf 'candidate_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'gate4a_source_job=2347926\n'
  printf 'gate4a_output_manifest_sha256=%s\n' "$expected_gate4a_manifest_sha"
  printf 'controlled_factor=parameters.head_mode\n'
  printf 'side_a=head_only\n'
  printf 'side_b=head_plus_wing\n'
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'mixing=none\n'
  printf 'qsgw_iterations=0:2\n'
  printf 'mpi_ranks=%s\n' "$SLURM_NTASKS"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
} >"$run_root/PROVENANCE.txt"

(
  cd "$g0w0_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" >librpa.stdout 2>librpa.stderr
)
(
  cd "$wing_run"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" >librpa.stdout 2>librpa.stderr
)

test "$(find "$g0w0_run" -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 384
test -f "$wing_run/qsgw_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' "$wing_run/qsgw_iterations.dat")" -eq 2
grep -Fqx '# velocity fixed_basis_rotation' "$wing_run/qsgw_matrices.dat"
grep -Fqx '# headwing scf_grid_analytic_live' "$wing_run/qsgw_matrices.dat"

PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/validate_qsgw_initial_state.py" "$wing_run/qsgw_iterations.dat" "$input_dir/band_out" "$run_root/head-plus-wing-initial-state.json" --iteration 0 --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 >"$run_root/head-plus-wing-initial-state.stdout" 2>"$run_root/head-plus-wing-initial-state.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/validate_qsgw_trace_closure.py" "$wing_run/qsgw_matrices.dat" "$run_root/head-plus-wing-closure.json" --iterations 1:2 --channel 0 --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 >"$run_root/head-plus-wing-closure.stdout" 2>"$run_root/head-plus-wing-closure.stderr"
PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/validate_qsgw_fixed_basis.py" "$wing_run/qsgw_matrices.dat" "$wing_run/qsgw_eigenvalues.dat" "$input_dir/band_out" "$run_root/head-plus-wing-fixed-basis.json" --iterations 0:2 --channel 0 --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 >"$run_root/head-plus-wing-fixed-basis.stdout" 2>"$run_root/head-plus-wing-fixed-basis.stderr"
grep -Fq '"passed": true' "$run_root/head-plus-wing-initial-state.json"
grep -Fq '"passed": true' "$run_root/head-plus-wing-closure.json"
grep -Fq '"passed": true' "$run_root/head-plus-wing-fixed-basis.json"

PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/compare_qsgw_trace_g0w0_sigc.py" "$wing_run/qsgw_matrices.dat" "$g0w0_run" "$run_root/head-plus-wing-iter1-vs-g0w0-sigc.json" --iteration 1 --channel 0 --source kgrid --max-abs-tolerance-ha 1e-10 --relative-frobenius-tolerance 1e-10 --input-contract "$input_dir/qsgw_input.head-only.contract" >"$run_root/wing-g0w0-comparator.stdout" 2>"$run_root/wing-g0w0-comparator.stderr"
grep -Fq '"passed": true' "$run_root/head-plus-wing-iter1-vs-g0w0-sigc.json"

PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/compare_qsgw_wing_mode_pair.py" "$head_only_run/qsgw_matrices.dat" "$wing_run/qsgw_matrices.dat" "$head_only_run/librpa.stdout" "$wing_run/librpa.stdout" "$run_root/CONTROLLED_PAIR.json" "$run_root/wing-mode-pair-comparison.json" --initial-tolerance 1e-10 --invariant-tolerance 1e-10 --wing-effect-minimum 1e-12 >"$run_root/wing-mode-pair-comparator.stdout" 2>"$run_root/wing-mode-pair-comparator.stderr"
grep -Fq '"passed": true' "$run_root/wing-mode-pair-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B "$tool_dir/validate_qsgw_headwing_trace.py" "$wing_run/qsgw_matrices.dat" "$run_root/head-plus-wing-live-validation.json" --iterations 0:2 --channel 0 --invariant-tolerance 1e-10 --initial-replay-tolerance 1e-10 --head-live-change-minimum 1e-12 >"$run_root/head-plus-wing-live-validator.stdout" 2>"$run_root/head-plus-wing-live-validator.stderr"
grep -Fq '"passed": true' "$run_root/head-plus-wing-live-validation.json"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'QSGW_GATE4B_HEAD_PLUS_WING=PASS\n'
