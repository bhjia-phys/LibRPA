#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a1-symmetry-nomix-miniter2-20260720-v3
gate_a0=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-847reader-ibzocc-build-20260720-v7
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
dataset_input_dir=$dataset/
legacy_build=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-847reader-ibzocc-20260720-v7/build
candidate_build=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13/build
legacy_exe=$legacy_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
legacy_run=$run_root/legacy
candidate_run=$run_root/candidate

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
legacy_exe_sha=4e56549ef9e45ae79de4518262280f3552877642b7e76d86ef257740bb23b95d
candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
gate_a0_provenance_sha=35e9aafe010609653a3f6fa289aabe7e3aa47d1a492d61412d4155300c2fbf2f
gate_a0_output_manifest_sha=315e44d2c5288d76c74619b459f0f3dc532fb9d096b951fdf3587d5fa5c08e43
dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
bundle_output_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179
target_iter=2
mpi_ranks=4
omp_threads=8

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
test -e "$gate_a0/GREEN_CONFIRMED"
test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test -x "$legacy_exe"
test -x "$candidate_exe"
test "$(sha256sum "$legacy_exe" | awk '{print $1}')" = "$legacy_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$candidate_exe_sha"
test "$(sha256sum "$gate_a0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$gate_a0_provenance_sha"
test "$(sha256sum "$gate_a0/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$gate_a0_output_manifest_sha"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$bundle_output_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$contract_sha"
test "$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$vxc_manifest_sha"
test "${dataset_input_dir: -1}" = "/"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
mkdir -p "$legacy_run" "$candidate_run"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$gate_a0/PROVENANCE.txt" "$run_root/gate-a0-PROVENANCE.txt"
cp "$bundle/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"

cat >"$legacy_run/librpa.in" <<EOF
task = qsgw
input_dir = $dataset_input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

cat >"$candidate_run/librpa.in" <<EOF
task = qsgw
input_dir = $dataset_input_dir
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
use_shrink_abfs = true
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_iterative_headwing = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=legacy_v4_oracle_harness_to_candidate_v5_symmetry_on
iterations=0:$target_iter
legacy_mixing=linear_beta_1_direct
candidate_mixing=none_direct
legacy_oracle_reader=historical_847_shrinked_basis_contract
legacy_oracle_ibz_occupation=stored_weights_include_geometric_kpoint_weight
legacy_symmetry=abacus_exx_on_abacus_gw_on
candidate_symmetry=exx_on_gw_on_rpa_on
use_shrink_abfs=true
headwing=off
hartree=off
band=off
nfreq=6
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
dataset_manifest_sha256=$dataset_manifest_sha
contract_sha256=$contract_sha
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a1_symmetry_nomix_miniter2_run_v3
acceptance=false_pending_numerical_observers
legacy_commit=$legacy_commit
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
candidate_commit=$candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$candidate_exe_sha
dataset=$dataset
input_dir=$dataset_input_dir
input_dir_trailing_slash=true
dataset_manifest_sha256=$dataset_manifest_sha
contract_sha256=$contract_sha
vxc_manifest_sha256=$vxc_manifest_sha
target_iteration=$target_iter
crystal_symmetry=on_ibz_8_to_full_bz_64
mixing=legacy_linear_beta_1_direct,candidate_none_direct
legacy_oracle_patch_scope=reader_and_ibz_occupation_harness_only
use_shrink_abfs=true
headwing=off
hartree=off
band=off
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
sha256sum "$legacy_run/librpa.in" "$candidate_run/librpa.in" \
  "$dataset/band_out" "$dataset/bz_sampling_out" \
  "$dataset/qsgw_input.contract" "$dataset/qsgw_vxc_scf.manifest" \
  >>"$run_root/PROVENANCE.txt"

(
  cd "$legacy_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=1
  export LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$legacy_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
test -s "$legacy_run/qsgw_oracle_matrices.dat"
test -s "$legacy_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixer linear' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixing_beta 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_gw 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_exx 1' "$legacy_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' "$legacy_run/qsgw_oracle_matrices.dat"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$legacy_run/homo_lumo_vs_iterations.dat")" = "$target_iter"
grep -Fq 'libRPA finished successfully' "$legacy_run/librpa.stdout"
if grep -Fq 'Cannot find overlap matrix file' \
  "$legacy_run/librpa.stdout" "$legacy_run/librpa.stderr"; then
  echo 'legacy run did not consume all overlap matrices' >&2
  exit 1
fi
if grep -Fq 'occupation exceeds storage capacity' \
  "$legacy_run/librpa.stdout" "$legacy_run/librpa.stderr"; then
  echo 'legacy run rejected geometrically weighted IBZ occupations' >&2
  exit 1
fi

(
  cd "$candidate_run"
  unset QSGW_ORACLE_TRACE QSGW_ORACLE_UPDATE_HARTREE QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA
  export LD_LIBRARY_PATH="$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >runtime.txt
  timeout 14400 mpirun -np "$mpi_ranks" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>runtime.txt
)
test -s "$candidate_run/qsgw_matrices.dat"
test -s "$candidate_run/qsgw_eigenvalues.dat"
test -s "$candidate_run/qsgw_iterations.dat"
test -s "$candidate_run/homo_lumo_vs_iterations.dat"
for trace in "$candidate_run/qsgw_matrices.dat" \
  "$candidate_run/qsgw_eigenvalues.dat" "$candidate_run/qsgw_iterations.dat"; do
  grep -Fqx '# qsgw_contract_version 5' "$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$trace"
  grep -Fqx '# live_update eigenvalues_wfc' "$trace"
  grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$trace"
  grep -Fqx '# headwing disabled_stage1' "$trace"
  grep -Fqx '# hartree disabled_stage1' "$trace"
  grep -Fqx '# band disabled_stage1' "$trace"
  grep -Fqx '# qsgw_mixer none' "$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $contract_sha" "$trace"
done
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$candidate_run/homo_lumo_vs_iterations.dat")" = "$target_iter"
grep -Fq 'libRPA finished successfully' "$candidate_run/librpa.stdout"

printf 'run_complete=true\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find legacy candidate -maxdepth 1 -type f -print0 | sort -z | \
    xargs -0 sha256sum >RUN_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet RUN_OUTPUT_SHA256SUMS.txt
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name RUN_COMPLETE \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/RUN_COMPLETE"
echo GATE_A1_SYMMETRY_NOMIX_MINITER2_RUN_V3=PASS
cat "$legacy_run/homo_lumo_vs_iterations.dat"
cat "$candidate_run/homo_lumo_vs_iterations.dat"
exit 0
