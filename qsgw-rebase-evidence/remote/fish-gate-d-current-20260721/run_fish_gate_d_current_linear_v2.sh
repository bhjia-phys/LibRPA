#!/usr/bin/env bash
set -euo pipefail

: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must identify the clean candidate}"
: "${CANDIDATE_GATE0_ROOT:?CANDIDATE_GATE0_ROOT must identify an accepted Gate 0}"
: "${CANDIDATE_GATE0_PROVENANCE_SHA256:?Gate 0 provenance hash is required}"
: "${CANDIDATE_EXE_SHA256:?Candidate executable hash is required}"
: "${BAND_BUNDLE_ROOT:?BAND_BUNDLE_ROOT must identify the immutable band bundle}"
: "${BAND_BUNDLE_PROVENANCE_SHA256:?Band bundle provenance hash is required}"
: "${BAND_BUNDLE_OUTPUT_SHA256:?Band bundle output-manifest hash is required}"
: "${BAND_DATASET_MANIFEST_SHA256:?Band dataset manifest hash is required}"
: "${BAND_CONTRACT_SHA256:?Band input-contract hash is required}"
: "${BAND_VXC_MANIFEST_SHA256:?Band Vxc manifest hash is required}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the run directory immutable}"

require_sha() {
  local value=$1
  local length=$2
  local label=$3
  [[ "$value" =~ ^[0-9a-f]+$ ]]
  test "${#value}" -eq "$length" || {
    echo "$label has the wrong length" >&2
    exit 2
  }
}

require_sha "$CANDIDATE_COMMIT" 40 CANDIDATE_COMMIT
for specification in \
  "CANDIDATE_GATE0_PROVENANCE_SHA256:$CANDIDATE_GATE0_PROVENANCE_SHA256" \
  "CANDIDATE_EXE_SHA256:$CANDIDATE_EXE_SHA256" \
  "BAND_BUNDLE_PROVENANCE_SHA256:$BAND_BUNDLE_PROVENANCE_SHA256" \
  "BAND_BUNDLE_OUTPUT_SHA256:$BAND_BUNDLE_OUTPUT_SHA256" \
  "BAND_DATASET_MANIFEST_SHA256:$BAND_DATASET_MANIFEST_SHA256" \
  "BAND_CONTRACT_SHA256:$BAND_CONTRACT_SHA256" \
  "BAND_VXC_MANIFEST_SHA256:$BAND_VXC_MANIFEST_SHA256" \
  "RUNNER_SHA256:$RUNNER_SHA256"; do
  require_sha "${specification#*:}" 64 "${specification%%:*}"
done
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-d-linear-${RUN_TAG}
dataset=$BAND_BUNDLE_ROOT/dataset
dataset_input_dir=$dataset/
mpi_ranks=1
omp_threads=32
python_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
python=$python_env/bin/python

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

provenance_value() {
  local key=$1
  local file=$2
  awk -F= -v key="$key" '
    $1 == key {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count != 1) exit 2
      print value
    }
  ' "$file"
}

test ! -e "$run_root"
test -x "$python"
test -e "$CANDIDATE_GATE0_ROOT/GREEN_CONFIRMED"
test ! -e "$CANDIDATE_GATE0_ROOT/FAILED"
test "$(sha256sum "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$CANDIDATE_GATE0_PROVENANCE_SHA256"
(
  cd "$CANDIDATE_GATE0_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(provenance_value candidate_commit "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_COMMIT"
candidate_source=$(provenance_value candidate_source "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_build=$(provenance_value candidate_build "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")
test "$(provenance_value candidate_executable_sha256 "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt")" = \
  "$CANDIDATE_EXE_SHA256"
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = "$CANDIDATE_EXE_SHA256"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$candidate_source" status --porcelain)"

runner_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/run_fish_gate_d_current_linear_v2.sh
validator_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/validate_qsgw_band_v6.py
validator_test_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/test_validate_qsgw_band_v6.py
cut_comparator_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/compare_qsgw_band_cut_modes_v1.py
cut_comparator_test_source=$candidate_source/qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/test_compare_qsgw_band_cut_modes_v1.py
contract_parser_source=$candidate_source/regression_tests/backend/comparisons/cmp_qsgw.py
for path in "$runner_source" "$validator_source" "$validator_test_source" \
  "$cut_comparator_source" "$cut_comparator_test_source" \
  "$contract_parser_source"; do
  test -f "$path"
done
test "$(sha256sum "$runner_source" | awk '{print $1}')" = "$RUNNER_SHA256"

test -e "$BAND_BUNDLE_ROOT/COMPLETE"
test ! -e "$BAND_BUNDLE_ROOT/FAILED"
test -z "$(find "$BAND_BUNDLE_ROOT" -type l -print -quit)"
test "$(sha256sum "$BAND_BUNDLE_ROOT/PROVENANCE.txt" | awk '{print $1}')" = \
  "$BAND_BUNDLE_PROVENANCE_SHA256"
test "$(sha256sum "$BAND_BUNDLE_ROOT/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$BAND_BUNDLE_OUTPUT_SHA256"
test "$(sha256sum "$BAND_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$BAND_DATASET_MANIFEST_SHA256"
test "$(sha256sum "$dataset/qsgw_band_input.contract" | awk '{print $1}')" = \
  "$BAND_CONTRACT_SHA256"
test "$(sha256sum "$dataset/qsgw_vxc_band.manifest" | awk '{print $1}')" = \
  "$BAND_VXC_MANIFEST_SHA256"
test "${dataset_input_dir: -1}" = "/"
(
  cd "$BAND_BUNDLE_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
for expected in \
  'gate=pinned_abacus_si_k444_symmetry_band_bundle_v1' \
  'grid=4x4x4' \
  'scf_kpoints=8' \
  'full_bz_kpoints=64' \
  'band_kpoints=143' \
  'n_bands=44' \
  'n_basis=44' \
  'symmetry=on_grid' \
  'headwing=off_fail_fast' \
  'hartree=off' \
  'band_update=operator_fourier' \
  "dataset_manifest_sha256=$BAND_DATASET_MANIFEST_SHA256" \
  "qsgw_band_contract_sha256=$BAND_CONTRACT_SHA256" \
  "qsgw_band_vxc_manifest_sha256=$BAND_VXC_MANIFEST_SHA256"; do
  grep -Fqx "$expected" "$BAND_BUNDLE_ROOT/PROVENANCE.txt"
done
grep -Fqx 'band_update operator_fourier' "$dataset/qsgw_band_input.contract"
grep -Fqx 'n_band_kpoints 143' "$dataset/qsgw_band_input.contract"

mkdir -p "$run_root/tools" "$run_root/unit-band" \
  "$run_root/unit-cut/mode0" "$run_root/unit-cut/mode1" \
  "$run_root/unit-cut/mode2"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$CANDIDATE_GATE0_ROOT/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$CANDIDATE_GATE0_ROOT/OUTPUT_SHA256SUMS.txt" "$run_root/gate0-OUTPUT_SHA256SUMS.txt"
cp "$BAND_BUNDLE_ROOT/PROVENANCE.txt" "$run_root/bundle-PROVENANCE.txt"
cp "$BAND_BUNDLE_ROOT/DATASET_SHA256SUMS.txt" "$run_root/bundle-DATASET_SHA256SUMS.txt"
cp "$validator_source" "$run_root/tools/validate_qsgw_band_v6.py"
cp "$cut_comparator_source" "$run_root/tools/compare_qsgw_band_cut_modes_v1.py"
cp "$contract_parser_source" "$run_root/tools/cmp_qsgw.py"

LIBRPA_QSGW_TEST_TMP="$run_root/unit-band" \
  "$python" -B "$validator_test_source" \
  >"$run_root/validator-unit-test.stdout" \
  2>"$run_root/validator-unit-test.stderr"
LIBRPA_QSGW_CUT_TEST_TMP="$run_root/unit-cut" \
  "$python" -B "$cut_comparator_test_source" \
  >"$run_root/cut-comparator-unit-test.stdout" \
  2>"$run_root/cut-comparator-unit-test.stderr"

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

cat >"$run_root/CONTROLLED_TRIO.txt" <<EOF
comparison=current_qsgw_band_cut_modes_0_1_2_linear_beta_0.2
acceptance_scope=structural_postmix_cut_and_metamorphic_not_legacy_numerical_acceptance
changed_factor=qsgw_band0_cut_mode
fixed_qsgw_band0_unoccupied_keep=10
fixed_qsgw_band0_cut_shift_ha=20.0
fixed_qsgw_mixer=linear
fixed_qsgw_mixing_beta=0.2
fixed_iterations=0:2
dataset_manifest_sha256=$BAND_DATASET_MANIFEST_SHA256
candidate_commit=$CANDIDATE_COMMIT
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
symmetry=on
headwing=off
hartree=off
band=operator_fourier
EOF

run_mode() {
  local cut_mode=$1
  local mode_root=$run_root/mode$cut_mode
  mkdir -p "$mode_root"
  cat >"$mode_root/librpa.in" <<EOF
task = qsgw_band
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
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_band_input.contract
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
qsgw_export_hamiltonian_for_pyatb = true
qsgw_band0_unoccupied_keep = 10
qsgw_band0_cut_mode = $cut_mode
qsgw_band0_cut_shift_ha = 20.0
EOF
  cat >"$mode_root/PARAMETERS.txt" <<EOF
cut_mode=$cut_mode
iterations=0:2
unoccupied_keep=10
cut_shift_ha=20.0
mixer=linear
mixing_beta=0.2
mixing_order=mix_then_reapply_cut
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=fixed_reference_operator_fourier_live
dataset=$dataset
dataset_manifest_sha256=$BAND_DATASET_MANIFEST_SHA256
EOF

  (
    cd "$mode_root"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
    test -s "$mode_root/$trace"
    grep -Fqx '# qsgw_contract_version 6' "$mode_root/$trace"
    grep -Fqx '# fixed_basis immutable_mf0' "$mode_root/$trace"
    grep -Fqx '# live_update eigenvalues_wfc' "$mode_root/$trace"
    grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$mode_root/$trace"
    grep -Fqx '# headwing disabled_stage1' "$mode_root/$trace"
    grep -Fqx '# hartree disabled_stage1' "$mode_root/$trace"
    grep -Fqx '# band fixed_reference_operator_fourier_live' "$mode_root/$trace"
    grep -Fqx '# h_qsgw_cut band_postprocess' "$mode_root/$trace"
    grep -Fqx '# qsgw_band0_unoccupied_keep 10' "$mode_root/$trace"
    grep -Fqx "# qsgw_band0_cut_mode $cut_mode" "$mode_root/$trace"
    grep -Fqx '# qsgw_band0_cut_shift_ha 20' "$mode_root/$trace"
    grep -Fqx '# qsgw_mixer linear' "$mode_root/$trace"
    grep -Fqx "# qsgw_input_contract_sha256 $BAND_CONTRACT_SHA256" "$mode_root/$trace"
  done
  test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
    "$mode_root/qsgw_iterations.dat")" = 2
  test "$(find "$mode_root" -maxdepth 1 -type f -name 'KS_band_spin_*_*.dat' | wc -l)" -eq 2
  test "$(find "$mode_root" -maxdepth 1 -type f -name 'EXX_band_spin_*_*.dat' | wc -l)" -eq 2
  test "$(find "$mode_root" -maxdepth 1 -type f -name 'QSGW_band_spin_*_*.dat' | wc -l)" -eq 2
  test "$(find "$mode_root" -maxdepth 1 -type f -name 'hrs*_nao_qsgw_iter_*.csr' | wc -l)" -eq 2
  grep -Fq 'libRPA finished successfully' "$mode_root/librpa.stdout"

  PYTHONPATH="$run_root/tools" "$python" -B \
    "$run_root/tools/validate_qsgw_band_v6.py" \
    "$mode_root/qsgw_matrices.dat" \
    "$mode_root/qsgw_eigenvalues.dat" \
    "$mode_root/qsgw_iterations.dat" \
    "$mode_root" "$dataset/bz_sampling_out" \
    "$mode_root/band-validation.json" \
    --expected-iterations 2 \
    --expected-cut-mode "$cut_mode" \
    --expected-unoccupied-keep 10 \
    --expected-shift-ha 20.0 \
    >"$mode_root/band-validation.stdout" \
    2>"$mode_root/band-validation.stderr"
  grep -Fq '"passed": true' "$mode_root/band-validation.json"
}

run_mode 0
run_mode 1
run_mode 2

PYTHONPATH="$run_root/tools" "$python" -B \
  "$run_root/tools/compare_qsgw_band_cut_modes_v1.py" \
  "$run_root/mode0" "$run_root/mode1" "$run_root/mode2" \
  "$run_root/cut-mode-comparison.json" \
  >"$run_root/cut-mode-comparison.stdout" \
  2>"$run_root/cut-mode-comparison.stderr"
grep -Fq '"passed": true' "$run_root/cut-mode-comparison.json"

cat >"$run_root/ACCEPTANCE.txt" <<EOF
accepted=true
scope=current_qsgw_band_linear_postmix_cut_structural_and_metamorphic_gate
legacy_same_dataset_acceptance=false
iterations=0:2
cut_modes=0,1,2
mixing=linear_beta_0.2
mixing_order=mix_then_reapply_cut
raw_closure_tolerance_ha=1e-10
matrix_relative_frobenius_tolerance=1e-8
fixed_basis_tolerance=1e-10
fourier_invariant_tolerance=1e-10
csr_roundtrip_absolute_tolerance_ha=1e-8
csr_roundtrip_relative_frobenius_tolerance=1e-8
band_table_tolerance_ev=1e-5
EOF

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate_d_current_linear_v2
acceptance=true
acceptance_scope=current_linear_postmix_cut_structural_and_metamorphic_not_legacy_numerical
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
candidate_commit=$CANDIDATE_COMMIT
candidate_gate0_root=$CANDIDATE_GATE0_ROOT
candidate_gate0_provenance_sha256=$CANDIDATE_GATE0_PROVENANCE_SHA256
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_exe
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
band_bundle_root=$BAND_BUNDLE_ROOT
band_bundle_provenance_sha256=$BAND_BUNDLE_PROVENANCE_SHA256
band_bundle_output_manifest_sha256=$BAND_BUNDLE_OUTPUT_SHA256
dataset_manifest_sha256=$BAND_DATASET_MANIFEST_SHA256
band_contract_sha256=$BAND_CONTRACT_SHA256
band_vxc_manifest_sha256=$BAND_VXC_MANIFEST_SHA256
symmetry=on
headwing=off
hartree=off
band=operator_fourier
cut_modes=0,1,2
iterations=0:2
mixing=linear_beta_0.2
mixing_order=mix_then_reapply_cut
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
LIBRI_DETERMINISTIC_REDUCTION=$LIBRI_DETERMINISTIC_REDUCTION
python=$($python --version 2>&1)
numpy=$($python -c 'import numpy; print(numpy.__version__)')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
printf 'FISH_GATE_D_CURRENT_LINEAR_V2=PASS\n'
cat "$run_root/ACCEPTANCE.txt"
