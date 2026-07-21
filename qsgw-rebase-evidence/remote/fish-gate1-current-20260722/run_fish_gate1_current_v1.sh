#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_COMMIT:?set RUNNER_COMMIT}"
: "${RUNNER_SOURCE:?set RUNNER_SOURCE}"
: "${RUNNER_SHA256:?set RUNNER_SHA256}"
: "${RUN_TAG:?set RUN_TAG}"

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274
expected_upstream_exe_sha=ae86432a32bdd9d2cc6e7d2f80f7e99e4de053329eb233c69d6e7e833aa006c3
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260722-66bfe1cf-v1
bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
expected_dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
expected_bundle_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
expected_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179

gate_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260722
vxc_source=$RUNNER_SOURCE/$gate_dir/vxc_out
vxc_provenance_source=$RUNNER_SOURCE/$gate_dir/VXC_SOURCE_PROVENANCE.txt
comparator_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/compare_g0w0_sigc_dump_directories.py
comparator_test_source=$RUNNER_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/test_compare_g0w0_sigc_dump_directories.py
expected_vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd
expected_vxc_provenance_sha=f0ffd643620324363a7cd8c2ba8660d1ff1efb9b78763b1ecdb74b2542af5490
expected_comparator_sha=df621f74e83955c6fb62da4762dd3b4fae1044f66559380fd8c3cac26be7d1fb
expected_comparator_test_sha=66fdd37687b4c98e06b14086d1ba2070134ffa40624912afe96f2d5a71146baa

python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
mpi_ranks=1
omp_threads=32
run_root=/home/bhj/ai-runs/librpa-qsgw-gate1-current-$RUN_TAG
upstream_run=$run_root/upstream
candidate_run=$run_root/candidate
overlay=$run_root/input-overlay
tool_dir=$run_root/tools

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

[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test ! -e "$run_root"
test -d "$RUNNER_SOURCE/.git"
test "$(git -C "$RUNNER_SOURCE" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$RUNNER_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"

test -e "$gate0/GREEN_CONFIRMED"
test ! -e "$gate0/FAILED"
test "$(sha256sum "$gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate0_provenance_sha"
test "$(sha256sum "$gate0/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate0_manifest_sha"
(
  cd "$gate0"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
grep -Fqx 'acceptance=true' "$gate0/PROVENANCE.txt"
grep -Fqx 'protected_diff=empty' "$gate0/PROVENANCE.txt"
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
upstream_exe=$(provenance_value upstream_executable "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test -x "$upstream_exe"
test -x "$candidate_exe"
test "$(sha256sum "$upstream_exe" | awk '{print $1}')" = \
  "$expected_upstream_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"

test -e "$bundle/COMPLETE"
test ! -e "$bundle/FAILED"
test "$(sha256sum "$bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_dataset_manifest_sha"
test "$(sha256sum "$bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_bundle_manifest_sha"
test "$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_contract_sha"
test "$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')" = \
  "$expected_vxc_manifest_sha"
(
  cd "$bundle"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
grep -Fqx 'n_scf_kpoints 8' "$dataset/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$dataset/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'use_shrink_abfs=true' "$bundle/PROVENANCE.txt"
test "$(find "$dataset" -maxdepth 1 -type f | wc -l)" -eq 56

test -f "$vxc_source"
test -f "$vxc_provenance_source"
test -f "$comparator_source"
test -f "$comparator_test_source"
test "$(sha256sum "$vxc_source" | awk '{print $1}')" = "$expected_vxc_sha"
test "$(sha256sum "$vxc_provenance_source" | awk '{print $1}')" = \
  "$expected_vxc_provenance_sha"
test "$(sha256sum "$comparator_source" | awk '{print $1}')" = \
  "$expected_comparator_sha"
test "$(sha256sum "$comparator_test_source" | awk '{print $1}')" = \
  "$expected_comparator_test_sha"
test -x "$python"
"$python" -c 'import numpy; print(numpy.__version__)' >/dev/null

mkdir -p "$upstream_run" "$candidate_run" "$overlay" "$tool_dir"
cp "$0" "$run_root/run_fish_gate1_current_v1.sh"
cp "$vxc_provenance_source" "$run_root/VXC_SOURCE_PROVENANCE.txt"
cp "$comparator_source" "$tool_dir/compare_g0w0_sigc_dump_directories.py"
cp "$comparator_test_source" "$tool_dir/test_compare_g0w0_sigc_dump_directories.py"
for path in "$dataset"/*; do
  test -f "$path"
  ln -s "$path" "$overlay/$(basename "$path")"
done
cp "$vxc_source" "$overlay/vxc_out"
test "$(find "$overlay" -maxdepth 1 -type l | wc -l)" -eq 56
test "$(find "$overlay" -maxdepth 1 -type f | wc -l)" -eq 1
(
  cd "$overlay"
  for path in *; do sha256sum "$path"; done
) >"$run_root/input-overlay.sha256"
find "$overlay" -maxdepth 1 -type l -printf '%f -> %l\n' | \
  sort >"$run_root/input-overlay-links.txt"

"$python" -B "$tool_dir/test_compare_g0w0_sigc_dump_directories.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"

cat >"$run_root/librpa.in" <<EOF
task = g0w0
input_dir = $overlay/
output_dir = .
fn_vxc_scf = vxc_out
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
output_energy_qp = true
EOF
cp "$run_root/librpa.in" "$upstream_run/librpa.in"
cp "$run_root/librpa.in" "$candidate_run/librpa.in"
cmp "$upstream_run/librpa.in" "$candidate_run/librpa.in"

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

cat >"$run_root/CONTROLLED_PAIR.txt" <<EOF
pair=upstream_g0w0_vs_candidate_g0w0
changed_factor=source_and_executable_only
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
dataset_manifest_sha256=$expected_dataset_manifest_sha
vxc_out_sha256=$expected_vxc_sha
librpa_input_sha256=$(sha256sum "$run_root/librpa.in" | awk '{print $1}')
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
EOF

run_one() {
  local label=$1
  local executable=$2
  local work=$run_root/$label
  (
    cd "$work"
    export LD_LIBRARY_PATH="$base_ld_library_path"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
    timeout 21600 mpirun -np "$mpi_ranks" "$executable" \
      >librpa.stdout 2>librpa.stderr
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
  )
  grep -Fq 'libRPA finished successfully' "$work/librpa.stdout"
  test -s "$work/energy_qp"
  test "$(find "$work" -maxdepth 1 \
    -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 48
}

run_one upstream "$upstream_exe"
run_one candidate "$candidate_exe"

"$python" -B "$tool_dir/compare_g0w0_sigc_dump_directories.py" \
  "$upstream_run" "$candidate_run" "$run_root/g0w0-comparison.json" \
  --source kgrid \
  --max-abs-tolerance-ha 1e-12 \
  --relative-frobenius-tolerance 1e-12 \
  >"$run_root/comparator.stdout" \
  2>"$run_root/comparator.stderr"
"$python" - "$run_root/g0w0-comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["passed"] is True
assert report["block_count"] == 48
assert report["spin_count"] == 1
assert report["kpoint_count"] == 8
assert report["frequency_count"] == 6
assert report["matrix_dimensions"] == [44]
PY
cmp "$upstream_run/energy_qp" "$candidate_run/energy_qp"
sha256sum "$upstream_run/energy_qp" "$candidate_run/energy_qp" \
  >"$run_root/energy-qp.sha256"

(
  cd "$upstream_run"
  find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/upstream-sigc.sha256"
(
  cd "$candidate_run"
  find . -maxdepth 1 -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' \
    -print0 | sort -z | xargs -0 sha256sum
) >"$run_root/candidate-sigc.sha256"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate1_current_g0w0_ab_v1
acceptance=true
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
gate0_provenance_sha256=$expected_gate0_provenance_sha
gate0_output_manifest_sha256=$expected_gate0_manifest_sha
upstream_executable=$upstream_exe
upstream_executable_sha256=$expected_upstream_exe_sha
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
dataset=$dataset
dataset_manifest_sha256=$expected_dataset_manifest_sha
vxc_out_source_provenance_sha256=$expected_vxc_provenance_sha
vxc_out_sha256=$expected_vxc_sha
qsgw_input_contract_sha256=$expected_contract_sha
qsgw_vxc_scf_manifest_sha256=$expected_vxc_manifest_sha
librpa_input_sha256=$(sha256sum "$run_root/librpa.in" | awk '{print $1}')
comparison_sha256=$(sha256sum "$run_root/g0w0-comparison.json" | awk '{print $1}')
sigc_block_count=48
sigc_max_abs_tolerance_ha=1e-12
sigc_relative_frobenius_tolerance=1e-12
energy_qp=byte_identical
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
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
printf 'FISH_GATE1_CURRENT_G0W0_AB_V1=PASS\n'
cat "$run_root/g0w0-comparison.json"
