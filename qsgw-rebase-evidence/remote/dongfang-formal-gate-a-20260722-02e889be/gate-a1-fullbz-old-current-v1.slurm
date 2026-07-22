#!/usr/bin/env bash
#SBATCH --job-name=qsgw-a1-fullbz
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-formal-gate-a-20260722-02e889be/logs/gate-a1-fullbz-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-formal-gate-a-20260722-02e889be/logs/gate-a1-fullbz-%j.err

set -Eeuo pipefail
trap 'status=$?; printf "ERROR line=%s status=%s command=%s\n" \
  "$LINENO" "$status" "$BASH_COMMAND" >&2' ERR

: "${RUNNER_COMMIT:?submit with RUNNER_COMMIT=<40-hex commit>}"
test "${#RUNNER_COMMIT}" = 40

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-formal-gate-a-20260722-02e889be
historical_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
common_root=$historical_root/inputs/si-k444-gate3-common-v3-2342408
input_dir=$common_root/dataset
old_build=/data/home/df_iopcas_bhj/qsgw-builds/old-new-a03a4c0-majorfix-v14-20260713
old_exe=$old_build/old/build/chi0_main.exe
candidate_source_parent=$root/src-candidate-02e889be-v1
candidate_source=$candidate_source_parent/librpa-02e889be
candidate_build=$root/build-candidate-02e889be-v1
candidate_exe=$candidate_build/chi0_main.exe
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate-a1-fullbz-old-current-02e889be-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_old_build_provenance_sha=8dfa0742e5618be14c27a96764ddf240af2ed8fc19662bac4d506bd44e77c96b
expected_candidate_commit=02e889be3eb45e7cfd5de393a0cd3bb63763d51e
expected_product_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=c400d45f8a0fd19251e6fb2ed1ffe7a0b9edd55cf69e3e8b744ffd93afcf5cfc
expected_candidate_manifest_sha=10f0a089844ec9db74122d4a9c4c5131bdf50c5a8638f09c2dbab15aef440679
expected_candidate_provenance_sha=96daa7a97430f8683d6cad4f71d54418b8bae960b182c51feb28231a5b58faea
expected_source_archive_sha=1cc873d7fd8f6e91a2d39a159b0c03647fbacc95ed05a962f276423f54134c46
expected_build_runner_sha=5de81e3ffc40705a832bc2d2c2577829fb92b95c3446d979a5806b905455b006
expected_common_dataset_manifest_sha=df793f964077ca03de7ad4723975736d0d0fb85d7e548241f3a7a04677cc05f7
expected_common_output_manifest_sha=3b7811c12df23a563e597cae2fb80f592c045d27c803a1afc7ead31d7d7439d4
expected_common_provenance_sha=54e2a4b0f07883f4341d46e41a80fd8ae1ebac7e25d263699eda8819b00ce0d8
expected_common_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_adapter_sha=99730f36d687017f545f5170e20f283be584cdf599afd2056a34ef00bb1b3f50
expected_adapter_test_sha=3fcbea1e560e287d8bc22dc3d1fdbdb393ed6335ffc7bcf6d465b60ddcd0e647
expected_base_comparator_sha=0d42b8135788ba1b7dd60299f5a1131537a46a880f03448382886414d7bfb5f4
expected_closure_adapter_sha=97494001b8824f86508e353edc4e73f97d7befc179c842221079733333482528
expected_closure_sha=ee9a3d1479f1cfd5cf0621f072d82470070910e5a914de021543c4f8bcadd469
expected_fixed_sha=0b3e5f1c7833e541f27882d25ec8c8b3ab01a658311bdbd331eb26106290c5a1
expected_initial_sha=725919900d32045bed29744d7a864e72fcfd660da60f0fedf57453900f27fc9b
expected_closure_test_sha=64428228923ea4ace7ef54cf8a4e5ceef8c7258011f2f5fc6863db1e7b9e29fb
expected_initial_test_sha=3617af57bb6bc93810ce178197b119ae00315f9dd61f2ae6c8226de449e35955
expected_current_parser_sha=9f83e07780bc832864abf68f3a2d82ee5238d8c13eaa2b568f68d9cf91269401

test ! -e "$run_root"
test -e "$common_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$old_exe"
test -x "$candidate_exe"
mkdir -p "$tool_dir"
cp "$0" "$run_root/gate-a1-fullbz-old-current-v1.slurm"

printf 'preflight=legacy_provenance\n'
test "$(sha256sum "$old_exe" | awk '{print $1}')" = \
  "$expected_old_exe_sha"
test "$(sha256sum "$old_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_old_build_provenance_sha"
grep -Fqx "old_oracle_commit=$expected_old_commit" \
  "$old_build/build-provenance.txt"

printf 'preflight=candidate_build_provenance\n'
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_provenance_sha"
test "$(sha256sum "$candidate_build/build-candidate-02e889be-v1.slurm" | awk '{print $1}')" = \
  "$expected_build_runner_sha"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx "product_source_equivalent_commit=$expected_product_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx "source_archive_sha256=$expected_source_archive_sha" \
  "$candidate_build/build-provenance.txt"
(
  cd "$candidate_build"
  sha256sum --check --quiet SHA256SUMS.txt
)
printf 'preflight=candidate_source_manifest\n'
(
  cd "$candidate_source_parent"
  sha256sum --check --quiet "$candidate_build/source-files.sha256"
)

printf 'preflight=common_input_bundle\n'
test "$(sha256sum "$common_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_dataset_manifest_sha"
test "$(sha256sum "$common_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_output_manifest_sha"
test "$(sha256sum "$common_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_common_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_common_contract_sha"
(
  cd "$common_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_band_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_update none' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"

printf 'preflight=observer_tools\n'
adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_legacy_v4_current_v6.py
adapter_test_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/test_compare_qsgw_legacy_v4_current_v6.py
base_comparator_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/compare_qsgw_component_traces-v4-c3daf072.py
closure_adapter_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/compare_qsgw_component_traces_v6_adapter.py
observer_root=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/observer-tools-v1
closure_source=$observer_root/validate_qsgw_trace_closure.py
fixed_source=$observer_root/validate_qsgw_fixed_basis.py
initial_source=$observer_root/validate_qsgw_initial_state.py
closure_test_source=$observer_root/test_validate_qsgw_trace_closure-v3-38de02fa.py
initial_test_source=$observer_root/test_validate_qsgw_initial_state-v1.py
current_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py

while read -r path expected
do
  actual=$(sha256sum "$path" | awk '{print $1}')
  printf 'observer_sha256=%s actual=%s expected=%s\n' \
    "$path" "$actual" "$expected"
  test "$actual" = "$expected"
done <<EOF
$adapter_source $expected_adapter_sha
$adapter_test_source $expected_adapter_test_sha
$base_comparator_source $expected_base_comparator_sha
$closure_adapter_source $expected_closure_adapter_sha
$closure_source $expected_closure_sha
$fixed_source $expected_fixed_sha
$initial_source $expected_initial_sha
$closure_test_source $expected_closure_test_sha
$initial_test_source $expected_initial_test_sha
$current_parser_source $expected_current_parser_sha
EOF

cp "$adapter_source" "$tool_dir/compare_qsgw_legacy_v4_current_v6.py"
cp "$base_comparator_source" "$tool_dir/base_comparator.py"
cp "$base_comparator_source" "$tool_dir/compare_qsgw_component_traces_v4.py"
cp "$closure_adapter_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$current_parser_source" "$tool_dir/cmp_qsgw_v6.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$closure_test_source" "$tool_dir/test_validate_qsgw_trace_closure.py"
cp "$initial_test_source" "$tool_dir/test_validate_qsgw_initial_state.py"

cp "$old_build/build-provenance.txt" "$run_root/old-build-provenance.txt"
cp "$candidate_build/build-provenance.txt" "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" "$run_root/candidate-build-SHA256SUMS.txt"
cp "$common_root/DATASET_SHA256SUMS.txt" "$run_root/DATASET_SHA256SUMS.txt"
cp "$common_root/OUTPUT_SHA256SUMS.txt" "$run_root/common-input-OUTPUT_SHA256SUMS.txt"
cp "$common_root/PROVENANCE.txt" "$run_root/common-input-PROVENANCE.txt"
cp "$input_dir/qsgw_input.contract" "$run_root/qsgw_input.contract"

printf 'preflight=observer_unit_tests\n'
PYTHONPATH="$tool_dir" "$python" -B "$adapter_test_source" \
  >"$run_root/adapter-unit-test.stdout" \
  2>"$run_root/adapter-unit-test.stderr"
(
  cd "$tool_dir"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_trace_closure.py \
    >"$run_root/closure-unit-test.stdout" \
    2>"$run_root/closure-unit-test.stderr"
  PYTHONPATH="$tool_dir" "$python" -B test_validate_qsgw_initial_state.py \
    >"$run_root/initial-unit-test.stdout" \
    2>"$run_root/initial-unit-test.stderr"
)

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
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/CONTROLLED_VARIABLES.txt" <<EOF
gate=A1
oracle=legacy_symmetry_off_full_bz
candidate=current_symmetry_off_full_bz
dataset=$input_dir
dataset_manifest_sha256=$expected_common_dataset_manifest_sha
input_contract_sha256=$expected_common_contract_sha
scf_kpoints=64
headwing=off
hartree=off
band=off
h_qsgw_cut=disabled_non_band
use_shrink_abfs=false
use_fullcoul_exx=false
use_fullcoul_eps=true
use_fullcoul_wc=false
mpi_ranks=4
omp_threads=12
deterministic_reduction=1
mode_1=legacy_linear_beta_1_direct_vs_current_none_miniter2
mode_2=legacy_linear_beta_0.2_vs_current_linear_beta_0.2_miniter5
EOF

printf 'numerics=begin\n'
run_mode() {
  local mode_name=$1
  local target_iter=$2
  local legacy_beta=$3
  local candidate_mode=$4
  local candidate_beta=$5
  local mode_root=$run_root/$mode_name
  local legacy_run=$mode_root/legacy
  local candidate_run=$mode_root/current
  mkdir -p "$legacy_run" "$candidate_run"

  cat >"$legacy_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

  cat >"$candidate_run/librpa.in" <<EOF
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
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = $candidate_mode
qsgw_mixing_beta = $candidate_beta
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

  (
    cd "$legacy_run"
    export QSGW_ORACLE_TRACE=1
    export QSGW_ORACLE_UPDATE_HARTREE=0
    export QSGW_HROUND_SCALE=0
    export LIBRPA_QSGW_MIXING_BETA=$legacy_beta
    export LD_LIBRARY_PATH="$old_build/old/build/qsgw:$old_build/old/build/src:$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    mpirun -np "$SLURM_NTASKS" "$old_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )

  test -s "$legacy_run/qsgw_oracle_matrices.dat"
  test -s "$legacy_run/homo_lumo_vs_iterations.dat"
  grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# qsgw_mixer linear' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx "# qsgw_min_iter $target_iter" "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx "# qsgw_max_iter $target_iter" "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# qsgw_update_hartree 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_gw 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_symmetry_exx 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# replace_w_head 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# option_dielect_func 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_shrink_abfs 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_exx 0' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_eps 1' "$legacy_run/qsgw_oracle_matrices.dat"
  grep -Fqx '# use_fullcoul_wc 0' "$legacy_run/qsgw_oracle_matrices.dat"
  test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
    "$legacy_run/qsgw_oracle_matrices.dat")" = "$target_iter"

  (
    cd "$candidate_run"
    unset QSGW_ORACLE_TRACE
    unset QSGW_ORACLE_UPDATE_HARTREE
    unset QSGW_HROUND_SCALE
    unset LIBRPA_QSGW_MIXING_BETA
    export LD_LIBRARY_PATH="$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )

  test -s "$candidate_run/qsgw_matrices.dat"
  test -s "$candidate_run/qsgw_eigenvalues.dat"
  test -s "$candidate_run/qsgw_iterations.dat"
  test -s "$candidate_run/homo_lumo_vs_iterations.dat"
  grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat
  do
    grep -Fqx '# qsgw_contract_version 6' "$candidate_run/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$candidate_run/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$candidate_run/$trace"
    grep -Fqx '# symmetry exx_off_gw_off_rpa_off' "$candidate_run/$trace"
    grep -Fqx '# headwing disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# hartree disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# band disabled_stage1' "$candidate_run/$trace"
    grep -Fqx '# h_qsgw_cut disabled_non_band' "$candidate_run/$trace"
    grep -Fqx "# qsgw_mixer $candidate_mode" "$candidate_run/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $expected_common_contract_sha" \
      "$candidate_run/$trace"
  done
  test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
    "$candidate_run/qsgw_matrices.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/qsgw_eigenvalues.dat")" = "$target_iter"
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$candidate_run/qsgw_iterations.dat")" = "$target_iter"

  "$python" -B "$tool_dir/compare_qsgw_legacy_v4_current_v6.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_eigenvalues.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$mode_root/legacy-current-comparison.json" \
    --base-comparator "$tool_dir/base_comparator.py" \
    --current-contract-parser "$tool_dir/cmp_qsgw_v6.py" \
    --iterations "0:$target_iter" \
    --expected-mode "$candidate_mode" \
    --expected-legacy-beta "$legacy_beta" \
    --expected-current-beta "$candidate_beta" \
    --expected-legacy-symmetry off \
    --expected-current-symmetry off \
    --expected-legacy-use-shrink-abfs 0 \
    --frequency-tolerance 1e-10 \
    --matrix-max-abs-tolerance-ha 1e-8 \
    --matrix-relative-tolerance 1e-8 \
    --eigenvalue-tolerance-ha 1e-6 \
    --gap-tolerance-ev 1e-5 \
    --degeneracy-tolerance-ha 1e-8 \
    --state-tolerance 1e-10 \
    --normalized-current-matrix "$mode_root/current-v5-self-matrices.dat" \
    --normalized-current-eigenvalues "$mode_root/current-v5-self-eigenvalues.dat" \
    --normalized-current-iterations "$mode_root/current-v5-self-iterations.dat" \
    >"$mode_root/legacy-current-comparison.stdout" \
    2>"$mode_root/legacy-current-comparison.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-current-comparison.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_trace_closure.py" \
    "$legacy_run/qsgw_oracle_matrices.dat" \
    "$mode_root/legacy-closure.json" \
    --iterations "0:$target_iter" --channel 0 --legacy-contract \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/legacy-closure.stdout" \
    2>"$mode_root/legacy-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/legacy-closure.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_trace_closure.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-closure.json" \
    --iterations "0:$target_iter" --channel 0 \
    --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
    >"$mode_root/current-closure.stdout" \
    2>"$mode_root/current-closure.stderr"
  grep -Fq '"passed": true' "$mode_root/current-closure.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_fixed_basis.py" \
    "$mode_root/current-v5-self-matrices.dat" \
    "$mode_root/current-v5-self-eigenvalues.dat" \
    "$input_dir/band_out" "$mode_root/current-fixed-basis.json" \
    --iterations "0:$target_iter" --channel 0 \
    --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
    >"$mode_root/current-fixed-basis.stdout" \
    2>"$mode_root/current-fixed-basis.stderr"
  grep -Fq '"passed": true' "$mode_root/current-fixed-basis.json"

  PYTHONPATH="$tool_dir" "$python" -B \
    "$tool_dir/validate_qsgw_initial_state.py" \
    "$candidate_run/qsgw_matrices.dat" \
    "$candidate_run/qsgw_iterations.dat" \
    "$input_dir/band_out" "$mode_root/current-initial-state.json" \
    --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
    >"$mode_root/current-initial-state.stdout" \
    2>"$mode_root/current-initial-state.stderr"
  grep -Fq '"passed": true' "$mode_root/current-initial-state.json"

  awk -v tolerance=1e-10 '
    NF && $1 !~ /^#/ {
      value = $7 + 0.0
      if (!seen) { reference = value; max_abs = 0.0; seen = 1 }
      delta = value - reference
      if (delta < 0.0) delta = -delta
      if (delta > max_abs) max_abs = delta
      if (delta > tolerance) failed = 1
      count += 1
    }
    END {
      printf "reference_electron_count=%.17g\n", reference
      printf "max_abs_delta=%.17g\n", max_abs
      printf "tolerance=%.17g\n", tolerance
      printf "iteration_count=%d\n", count
      if (!seen || failed) exit 1
    }
  ' "$candidate_run/qsgw_iterations.dat" \
    >"$mode_root/current-electron-count.txt"

  cat >"$mode_root/ACCEPTANCE.txt" <<EOF
accepted=true
gate=A1
mode=$mode_name
iterations=0:$target_iter
legacy_symmetry=off_full_bz
current_symmetry=off_full_bz
legacy_current_matrix_relative_tolerance=1e-8
legacy_current_eigenvalue_tolerance_ha=1e-6
legacy_current_gap_tolerance_ev=1e-5
closure_tolerance_ha=1e-10
hermiticity_tolerance_ha=1e-10
fixed_basis_tolerance=1e-10
electron_count_tolerance=1e-10
EOF
}

run_mode no-mix-miniter2 2 1 none 0.2
run_mode linear-beta-0.2-miniter5 5 0.2 linear 0.2

(
  cd "$common_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$candidate_source_parent"
  sha256sum --check --quiet "$candidate_build/source-files.sha256"
)

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=A1_legacy_fullbz_vs_current_fullbz
acceptance=true
runner_commit=$RUNNER_COMMIT
runner_sha256=$(sha256sum "$run_root/gate-a1-fullbz-old-current-v1.slurm" | awk '{print $1}')
legacy_commit=$expected_old_commit
legacy_executable=$old_exe
legacy_executable_sha256=$expected_old_exe_sha
candidate_commit=$expected_candidate_commit
candidate_product_equivalent_commit=$expected_product_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
candidate_build_manifest_sha256=$expected_candidate_manifest_sha
candidate_build_provenance_sha256=$expected_candidate_provenance_sha
source_archive_sha256=$expected_source_archive_sha
dataset=$input_dir
dataset_manifest_sha256=$expected_common_dataset_manifest_sha
input_contract_sha256=$expected_common_contract_sha
crystal_symmetry=off
time_reversal_reduction=off_full_64_kpoint_input
headwing=off
hartree=off
band=off
h_qsgw_cut=disabled_non_band
mpi_ranks=$SLURM_NTASKS
omp_threads=$OMP_NUM_THREADS
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cat >"$run_root/A1_ACCEPTANCE.txt" <<EOF
accepted=true
oracle=legacy_symmetry_off_full_bz
candidate=current_symmetry_off_full_bz
no_mix_miniter2=passed
linear_beta_0.2_miniter5=passed
iteration_zero_through_final=compared
matrix_state_and_invariants=passed
EOF

(
  cd "$run_root"
  find no-mix-miniter2 linear-beta-0.2-miniter5 -type f -print0 | \
    sort -z | xargs -0 sha256sum >MODE_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet MODE_OUTPUT_SHA256SUMS.txt
)
sha256sum \
  "$run_root/PROVENANCE.txt" \
  "$run_root/CONTROLLED_VARIABLES.txt" \
  "$run_root/A1_ACCEPTANCE.txt" \
  "$run_root/MODE_OUTPUT_SHA256SUMS.txt" \
  "$run_root/old-build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt" \
  "$run_root/DATASET_SHA256SUMS.txt" \
  "$run_root/common-input-OUTPUT_SHA256SUMS.txt" \
  "$run_root/common-input-PROVENANCE.txt" \
  "$run_root/qsgw_input.contract" \
  "$run_root/gate-a1-fullbz-old-current-v1.slurm" \
  "$run_root/adapter-unit-test.stdout" \
  "$run_root/adapter-unit-test.stderr" \
  "$run_root/closure-unit-test.stdout" \
  "$run_root/closure-unit-test.stderr" \
  "$run_root/initial-unit-test.stdout" \
  "$run_root/initial-unit-test.stderr" \
  "$run_root/oneapi-setvars.log" \
  >"$run_root/SHA256SUMS.txt"
(
  cd "$run_root"
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
touch "$run_root/GREEN_CONFIRMED"
printf 'GREEN_CONFIRMED %s\n' "$run_root"
