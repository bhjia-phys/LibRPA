#!/usr/bin/env bash
set -Eeuo pipefail

: "${CANDIDATE_SOURCE:?clean candidate checkout is required}"
: "${CANDIDATE_COMMIT:?candidate commit is required}"
: "${CANDIDATE_EXE:?candidate executable is required}"
: "${CANDIDATE_EXE_SHA256:?candidate executable hash is required}"
: "${LEGACY_EXE:?legacy executable is required}"
: "${LEGACY_EXE_SHA256:?legacy executable hash is required}"
: "${DATASET_DIR:?frozen Si k444 dataset is required}"
: "${DATASET_MANIFEST_SHA256:?dataset manifest hash is required}"
: "${RUN_TAG:?immutable run tag is required}"

case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_TAG contains unsafe characters" >&2
    exit 2
    ;;
esac

iterations=${ITERATIONS:-5}
mpi_ranks=${MPI_RANKS:-4}
omp_threads=${OMP_THREADS:-12}
mpiexec=${MPIEXEC:-mpirun}
run_root=/home/bhj/ai-runs/librpa-qsgw-si-k444-head-band-$RUN_TAG
overlay=$run_root/input
legacy=$run_root/legacy
candidate=$run_root/candidate
tools_dir=$run_root/tools
runner_relative=qsgw-rebase-evidence/remote/fish-formal-gate-a-20260723
head_contract=$DATASET_DIR/qsgw_input.head-only.contract
band_contract=$DATASET_DIR/qsgw_input.band.contract
merged_contract=$overlay/qsgw_input.head-band.contract
python=${PYTHON:-python3}

expected_legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_head_contract_sha=dd255667efae3e27a435f48684b801b30b3f2a68bfa6c7e792fa4a1de848921d
expected_band_contract_sha=94c1832de2c73fc3d8f44c93bf206a39f80b9e2233d40398af075b19dd40cdd3

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
require_sha "$CANDIDATE_EXE_SHA256" 64 CANDIDATE_EXE_SHA256
require_sha "$LEGACY_EXE_SHA256" 64 LEGACY_EXE_SHA256
require_sha "$DATASET_MANIFEST_SHA256" 64 DATASET_MANIFEST_SHA256
test "$iterations" -ge 2
test "$mpi_ranks" -ge 1
test "$omp_threads" -ge 1

record_failure() {
  local rc=$?
  trap - EXIT
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure EXIT

test ! -e "$run_root"
test -d "$CANDIDATE_SOURCE/.git"
test "$(git -C "$CANDIDATE_SOURCE" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$CANDIDATE_SOURCE" status --porcelain)"
test -x "$CANDIDATE_EXE"
test -x "$LEGACY_EXE"
test "$(sha256sum "$CANDIDATE_EXE" | awk '{print $1}')" = \
  "$CANDIDATE_EXE_SHA256"
test "$(sha256sum "$LEGACY_EXE" | awk '{print $1}')" = \
  "$LEGACY_EXE_SHA256"
test -d "$DATASET_DIR"
test -f "$DATASET_DIR/DATASET_SHA256SUMS.txt"
test "$(sha256sum "$DATASET_DIR/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$DATASET_MANIFEST_SHA256"
test "$(sha256sum "$head_contract" | awk '{print $1}')" = \
  "$expected_head_contract_sha"
test "$(sha256sum "$band_contract" | awk '{print $1}')" = \
  "$expected_band_contract_sha"
grep -Fqx 'n_scf_kpoints 64' "$head_contract"
grep -Fqx 'n_headwing_kpoints 64' "$head_contract"
grep -Fqx 'n_band_kpoints 0' "$head_contract"
grep -Fqx 'headwing_grid scf' "$head_contract"
grep -Fqx 'headwing_update fixed_basis_rotation' "$head_contract"
grep -Fqx 'hartree_update off' "$head_contract"
grep -Fqx 'band_update off' "$head_contract"
grep -Fqx 'n_scf_kpoints 64' "$band_contract"
grep -Fqx 'n_headwing_kpoints 0' "$band_contract"
grep -Fqx 'n_band_kpoints 201' "$band_contract"
grep -Fqx 'headwing_grid disabled' "$band_contract"
grep -Fqx 'headwing_update none' "$band_contract"
grep -Fqx 'hartree_update off' "$band_contract"
grep -Fqx 'band_update operator_fourier' "$band_contract"
(
  cd "$DATASET_DIR"
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

mkdir -p "$overlay" "$legacy" "$candidate" "$tools_dir"
for entry in "$DATASET_DIR"/*; do
  ln -s "$entry" "$overlay/$(basename "$entry")"
done

merge_tool=$CANDIDATE_SOURCE/$runner_relative/merge_qsgw_head_band_contracts_v1.py
merge_test=$CANDIDATE_SOURCE/$runner_relative/test_merge_qsgw_head_band_contracts_v1.py
compare_tool=$CANDIDATE_SOURCE/$runner_relative/compare_qsgw_band_iterations_v1.py
compare_test=$CANDIDATE_SOURCE/$runner_relative/test_compare_qsgw_band_iterations_v1.py
cp "$merge_tool" "$merge_test" "$compare_tool" "$compare_test" "$tools_dir/"
(
  cd "$tools_dir"
  "$python" -B -m unittest -v \
    test_merge_qsgw_head_band_contracts_v1.py \
    test_compare_qsgw_band_iterations_v1.py \
    >tool-tests.stdout 2>tool-tests.stderr
)
"$python" -B "$tools_dir/merge_qsgw_head_band_contracts_v1.py" \
  --head-contract "$head_contract" \
  --band-contract "$band_contract" \
  --output "$merged_contract"
grep -Fqx 'n_scf_kpoints 64' "$merged_contract"
grep -Fqx 'n_headwing_kpoints 64' "$merged_contract"
grep -Fqx 'n_band_kpoints 201' "$merged_contract"
grep -Fqx 'headwing_grid scf' "$merged_contract"
grep -Fqx 'headwing_update fixed_basis_rotation' "$merged_contract"
grep -Fqx 'hartree_update off' "$merged_contract"
grep -Fqx 'band_update operator_fourier' "$merged_contract"

cat >"$legacy/librpa.in" <<EOF
task = qsgw_band
input_dir = $overlay
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
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
use_kpara_scf_eigvec = false
output_energy_qp = false
max_iter = $iterations
EOF

cat >"$candidate/librpa.in" <<EOF
task = qsgw_band
input_dir = $overlay
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
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
use_kpara_scf_eigvec = false
output_energy_qp = false
qsgw_input_contract = qsgw_input.head-band.contract
qsgw_mixer = none
qsgw_min_iter = $iterations
qsgw_max_iter = $iterations
qsgw_write_iteration_matrices = false
qsgw_update_hartree = false
EOF

iteration_csv=$(seq -s, 1 "$iterations")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=si_k444_head_only_qsgw_band_old_vs_new_v1
result=PENDING
legacy_commit=$expected_legacy_commit
legacy_executable=$LEGACY_EXE
legacy_executable_sha256=$LEGACY_EXE_SHA256
candidate_commit=$CANDIDATE_COMMIT
candidate_executable=$CANDIDATE_EXE
candidate_executable_sha256=$CANDIDATE_EXE_SHA256
dataset_dir=$DATASET_DIR
dataset_manifest_sha256=$DATASET_MANIFEST_SHA256
head_contract_sha256=$expected_head_contract_sha
band_contract_sha256=$expected_band_contract_sha
merged_contract_sha256=$(sha256sum "$merged_contract" | awk '{print $1}')
scf_kpoints=64
band_kpoints=201
bands=26
occupied_bands=4
iterations=0:$iterations
head=analytic_same_grid_live
wing=off
hartree=off
legacy_update=direct
candidate_mixer=none
energy_tolerance_ev=1e-4
gap_tolerance_ev=2e-4
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

export OMP_NUM_THREADS=$omp_threads
export MKL_NUM_THREADS=$omp_threads
export OPENBLAS_NUM_THREADS=$omp_threads
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export LIBRI_DETERMINISTIC_REDUCTION=1

(
  cd "$legacy"
  unset LIBRPA_QSGW_MIXING_BETA
  "$mpiexec" -np "$mpi_ranks" "$LEGACY_EXE" \
    >librpa.stdout 2>librpa.stderr
)
(
  cd "$candidate"
  "$mpiexec" -np "$mpi_ranks" "$CANDIDATE_EXE" \
    >librpa.stdout 2>librpa.stderr
)

for iteration in $(seq 1 "$iterations"); do
  test -s "$legacy/QSGW_band_spin_1_${iteration}.dat"
  test -s "$candidate/QSGW_band_spin_1_${iteration}.dat"
done
test "$(find "$legacy" -maxdepth 1 -type f \
  -name 'QSGW_band_spin_1_*.dat' | wc -l)" -eq "$iterations"
test "$(find "$candidate" -maxdepth 1 -type f \
  -name 'QSGW_band_spin_1_*.dat' | wc -l)" -eq "$iterations"

"$python" -B "$tools_dir/compare_qsgw_band_iterations_v1.py" \
  --legacy-dir "$legacy" \
  --candidate-dir "$candidate" \
  --iterations "$iteration_csv" \
  --spins 1 \
  --occupied-bands 4 \
  --coordinate-tolerance 1e-7 \
  --energy-tolerance-ev 1e-4 \
  --gap-tolerance-ev 2e-4 \
  --output "$run_root/band-gap-comparison.json" \
  >"$run_root/comparator.stdout" 2>"$run_root/comparator.stderr"

"$python" -B - "$run_root/band-gap-comparison.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
assert report["schema"] == "librpa-qsgw-band-iteration-comparison-v1"
assert report["passed"] is True
PY

printf 'result=PASS\ncompleted_utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | \
    xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
run_succeeded=1
trap - EXIT
echo SI_K444_HEAD_ONLY_QSGW_BAND_OLD_VS_NEW_V1=PASS
