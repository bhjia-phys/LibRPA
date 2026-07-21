#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUNNER_COMMIT:?set RUNNER_COMMIT}"
: "${RUNNER_SOURCE:?set RUNNER_SOURCE}"
: "${RUNNER_SHA256:?set RUNNER_SHA256}"
: "${RUN_TAG:?set RUN_TAG}"

[[ "$RUNNER_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260722-66bfe1cf-v1
gate1_accept=/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-20260722-c91a4305-v1
gate1_source=/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260722-ca542aca-v1
expected_gate1_provenance_sha=e9f7dd5104696456c5e64699dd38834c2d563ddbc41e77987b27dc7b8adb41b1
expected_gate1_manifest_sha=5a9c7d2c027d09b20c8a16956200e26ee4c6a122dc7ab226099e249f2958a9f4
expected_gate1_source_manifest_sha=2cddd17a864560d9d004d43a6d5f2fe46d9b7798ad33c4f7195005624758cc21
expected_gate1_comparison_sha=02073d2d049a7ba3ba0a09986f1246beb8b25a19981686d91abd64ff46c64f0b

bundle=/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3
dataset=$bundle/dataset
expected_dataset_manifest_sha=869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690
expected_bundle_manifest_sha=b3be3227d0aea82492e92085340d567b47a6ba3ebb0976e1f23d542e9d5db648
expected_contract_sha=5b90f7314d7231e0d1cc3d272957e26b9aa89d9b204c71d0378d1940098f0d46
expected_overlay_contract_sha=7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7
expected_vxc_manifest_sha=714af7a617cdf971651a21e2b599819b7a53f9eac6b76cf8e3ead8c9a4890179

gate2_dir=qsgw-rebase-evidence/remote/fish-gate2-current-20260722
gate1_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260722
compare_source=$RUNNER_SOURCE/$gate2_dir/compare_qsgw_iter1_g0w0_v1.py
validate_source=$RUNNER_SOURCE/$gate2_dir/validate_qsgw_iter1_v6.py
fixture_source=$RUNNER_SOURCE/$gate2_dir/gate2_test_fixture_v1.py
compare_test_source=$RUNNER_SOURCE/$gate2_dir/test_compare_qsgw_iter1_g0w0_v1.py
validate_test_source=$RUNNER_SOURCE/$gate2_dir/test_validate_qsgw_iter1_v6.py
contract_builder_source=$RUNNER_SOURCE/$gate2_dir/build_qsgw_contract_overlay_v1.py
contract_builder_test_source=$RUNNER_SOURCE/$gate2_dir/test_build_qsgw_contract_overlay_v1.py
cmp_qsgw_source=$RUNNER_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py
stru_builder_source=$RUNNER_SOURCE/$gate1_dir/build_stru_symmetry_overlay_v1.py
stru_builder_test_source=$RUNNER_SOURCE/$gate1_dir/test_build_stru_symmetry_overlay_v1.py
stru_tail_source=$RUNNER_SOURCE/$gate1_dir/stru_symmetry_tail.dd421665.si444.txt
stru_provenance_source=$RUNNER_SOURCE/$gate1_dir/STRU_SYMMETRY_SOURCE_PROVENANCE.txt
vxc_source=$RUNNER_SOURCE/$gate1_dir/vxc_out
vxc_provenance_source=$RUNNER_SOURCE/$gate1_dir/VXC_SOURCE_PROVENANCE.txt

expected_compare_sha=40eb6e8f61353d5ff08e1d4892a3ed09c2931dcee2d2854829e580f3d6692a8d
expected_validate_sha=31519c1b0a2d6d17dfb1f2bc38b4c28b95897638109f251c8b0a4829107293c3
expected_fixture_sha=686c2bff3558d12228eb79e3a93cd9a1ebb760e0e1587cb9205623f1f220268f
expected_compare_test_sha=7aafb90b1765082891528d6db9d2d220583e0df69315151efb420f48b800d8bf
expected_validate_test_sha=dd72cc67540bd6b7550502f234bd4cb0e013cab0b7ac7e234950749bf29de94a
expected_contract_builder_sha=5791d893cd3d1cf2771d0ba7883911db153c243ace3d3ac9a1c2f8384145bd29
expected_contract_builder_test_sha=6bab5f704147079994254b5e47631b2f99b8f20df64e2da7b621257336010117
expected_cmp_qsgw_sha=f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb
expected_stru_builder_sha=d90e22d671c9b5ade4b799d987dfa471d1c0b65747c96748cfbb832998938c80
expected_stru_builder_test_sha=1ef3e79588c1c45e5ffb8ed93e99784cc7464c7683ff6c5518be4cbdeac1d57f
expected_stru_tail_sha=9863bfb6be3234e0ea050f45d7b2245f5529e357a6f87567998632fc3461ed8e
expected_stru_provenance_sha=3252fd881b65ca662d9105ddfbaea48711ce085d7e733293a3bcf3257c04f37e
expected_vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd
expected_vxc_provenance_sha=f0ffd643620324363a7cd8c2ba8660d1ff1efb9b78763b1ecdb74b2542af5490
expected_source_stru_sha=5d943ee64376bc4e3315cc7ae779a1a91785b42e515290d37dacd78147947b5a
expected_overlay_stru_sha=e756fd9551bfa9df748473880259ba019de904867c1aaff126b1b3a9c51a8873

python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
mpi_ranks=1
omp_threads=32
run_root=/home/bhj/ai-runs/librpa-qsgw-gate2-current-$RUN_TAG
run_dir=$run_root/candidate-qsgw
overlay=$run_root/input-overlay
tool_dir=$run_root/tools

run_succeeded=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $run_succeeded -ne 1 && -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

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
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test -x "$candidate_exe"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"

test -e "$gate1_accept/GREEN_CONFIRMED"
test ! -e "$gate1_accept/FAILED"
test "$(sha256sum "$gate1_accept/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate1_provenance_sha"
test "$(sha256sum "$gate1_accept/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate1_manifest_sha"
test "$(sha256sum "$gate1_accept/SOURCE_RUN_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate1_source_manifest_sha"
test "$(sha256sum "$gate1_accept/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_gate1_comparison_sha"
grep -Fqx 'acceptance=true' "$gate1_accept/PROVENANCE.txt"
grep -Fqx 'source_run_status=rejected_observer_threshold_only' \
  "$gate1_accept/PROVENANCE.txt"
grep -Fqx 'symmetry=exx_on_gw_on_rpa_on' "$gate1_accept/PROVENANCE.txt"
grep -Fqx 'headwing=off' "$gate1_accept/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$gate1_accept/PROVENANCE.txt"
grep -Fqx 'band=off' "$gate1_accept/PROVENANCE.txt"
(
  cd "$gate1_accept"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$gate1_source"
  sha256sum --check --quiet "$gate1_accept/SOURCE_RUN_SHA256SUMS.txt"
)
test "$(find "$gate1_source/upstream" -maxdepth 1 \
  -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 48

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

while read -r path expected; do
  test -f "$path"
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done <<EOF
$compare_source $expected_compare_sha
$validate_source $expected_validate_sha
$fixture_source $expected_fixture_sha
$compare_test_source $expected_compare_test_sha
$validate_test_source $expected_validate_test_sha
$contract_builder_source $expected_contract_builder_sha
$contract_builder_test_source $expected_contract_builder_test_sha
$cmp_qsgw_source $expected_cmp_qsgw_sha
$stru_builder_source $expected_stru_builder_sha
$stru_builder_test_source $expected_stru_builder_test_sha
$stru_tail_source $expected_stru_tail_sha
$stru_provenance_source $expected_stru_provenance_sha
$vxc_source $expected_vxc_sha
$vxc_provenance_source $expected_vxc_provenance_sha
EOF
test "$(sha256sum "$dataset/stru_out" | awk '{print $1}')" = \
  "$expected_source_stru_sha"
test -x "$python"
"$python" -c 'import numpy; print(numpy.__version__)' >/dev/null

mkdir -p "$run_dir" "$overlay" "$tool_dir"
cp "$0" "$run_root/run_fish_gate2_current_v1.sh"
cp "$gate0/PROVENANCE.txt" "$run_root/gate0-PROVENANCE.txt"
cp "$gate0/OUTPUT_SHA256SUMS.txt" "$run_root/gate0-OUTPUT_SHA256SUMS.txt"
cp "$gate1_accept/PROVENANCE.txt" "$run_root/gate1-accepted-PROVENANCE.txt"
cp "$gate1_accept/OUTPUT_SHA256SUMS.txt" "$run_root/gate1-accepted-OUTPUT_SHA256SUMS.txt"
cp "$gate1_accept/SOURCE_RUN_SHA256SUMS.txt" "$run_root/gate1-SOURCE_RUN_SHA256SUMS.txt"
cp "$gate1_accept/g0w0-comparison.json" "$run_root/gate1-g0w0-comparison.json"
cp "$stru_provenance_source" "$run_root/STRU_SYMMETRY_SOURCE_PROVENANCE.txt"
cp "$vxc_provenance_source" "$run_root/VXC_SOURCE_PROVENANCE.txt"
cp "$compare_source" "$tool_dir/compare_qsgw_iter1_g0w0_v1.py"
cp "$validate_source" "$tool_dir/validate_qsgw_iter1_v6.py"
cp "$fixture_source" "$tool_dir/gate2_test_fixture_v1.py"
cp "$compare_test_source" "$tool_dir/test_compare_qsgw_iter1_g0w0_v1.py"
cp "$validate_test_source" "$tool_dir/test_validate_qsgw_iter1_v6.py"
cp "$contract_builder_source" "$tool_dir/build_qsgw_contract_overlay_v1.py"
cp "$contract_builder_test_source" "$tool_dir/test_build_qsgw_contract_overlay_v1.py"
cp "$cmp_qsgw_source" "$tool_dir/cmp_qsgw.py"
cp "$stru_builder_source" "$tool_dir/build_stru_symmetry_overlay_v1.py"
cp "$stru_builder_test_source" "$tool_dir/test_build_stru_symmetry_overlay_v1.py"
cp "$stru_tail_source" "$tool_dir/stru_symmetry_tail.dd421665.si444.txt"

for path in "$dataset"/*; do
  test -f "$path"
  if [[ $(basename "$path") == stru_out || \
        $(basename "$path") == qsgw_input.contract ]]; then
    continue
  fi
  ln -s "$path" "$overlay/$(basename "$path")"
done
cp "$vxc_source" "$overlay/vxc_out"
"$python" -B "$tool_dir/build_stru_symmetry_overlay_v1.py" \
  "$dataset/stru_out" \
  "$tool_dir/stru_symmetry_tail.dd421665.si444.txt" \
  "$overlay/stru_out" \
  "$run_root/stru-symmetry-overlay-validation.json" \
  --expected-grid 4 4 4 \
  --n-scf-kpoints 8 \
  --metric-tolerance 1e-10 \
  --atom-tolerance 1e-5 \
  >"$run_root/stru-symmetry-overlay-builder.stdout" \
  2>"$run_root/stru-symmetry-overlay-builder.stderr"
test "$(sha256sum "$overlay/stru_out" | awk '{print $1}')" = \
  "$expected_overlay_stru_sha"
"$python" -B "$tool_dir/build_qsgw_contract_overlay_v1.py" \
  "$dataset/qsgw_input.contract" \
  "$dataset/stru_out" \
  "$overlay/stru_out" \
  "$overlay/qsgw_input.contract" \
  "$run_root/qsgw-contract-overlay-validation.json" \
  >"$run_root/qsgw-contract-overlay-builder.stdout" \
  2>"$run_root/qsgw-contract-overlay-builder.stderr"
test "$(sha256sum "$overlay/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_overlay_contract_sha"
test "$(find "$overlay" -maxdepth 1 -type l | wc -l)" -eq 54
test "$(find "$overlay" -maxdepth 1 -type f | wc -l)" -eq 3
(
  cd "$overlay"
  for path in *; do sha256sum "$path"; done
) >"$run_root/input-overlay.sha256"
find "$overlay" -maxdepth 1 -type l -printf '%f -> %l\n' | \
  sort >"$run_root/input-overlay-links.txt"

(
  cd "$tool_dir"
  CMP_QSGW="$tool_dir/cmp_qsgw.py" "$python" -B -m unittest -v \
    test_build_qsgw_contract_overlay_v1.py \
    test_compare_qsgw_iter1_g0w0_v1.py \
    test_validate_qsgw_iter1_v6.py \
    >"$run_root/gate2-observer-unit-test.stdout" \
    2>"$run_root/gate2-observer-unit-test.stderr"
  "$python" -B -m unittest -v test_build_stru_symmetry_overlay_v1.py \
    >"$run_root/stru-builder-unit-test.stdout" \
    2>"$run_root/stru-builder-unit-test.stderr"
)

cat >"$run_root/librpa.in" <<EOF
task = qsgw
input_dir = $overlay/
output_dir = .
fn_vxc_scf = vxc_out
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
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF
cp "$run_root/librpa.in" "$run_dir/librpa.in"

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
pair=candidate_qsgw_first_self_energy_vs_accepted_upstream_g0w0
changed_factor=task_g0w0_to_qsgw_trace_iteration_1_channel_0
semantic_iteration_zero=immutable_initial_state
semantic_first_self_energy=trace_iteration_1_channel_0
runner_commit=$RUNNER_COMMIT
candidate_source_commit=$expected_candidate_commit
candidate_executable_sha256=$expected_candidate_exe_sha
upstream_source_commit=$expected_upstream_commit
gate1_accepted_provenance_sha256=$expected_gate1_provenance_sha
gate1_source_manifest_sha256=$expected_gate1_source_manifest_sha
dataset_manifest_sha256=$expected_dataset_manifest_sha
source_qsgw_input_contract_sha256=$expected_contract_sha
overlay_qsgw_input_contract_sha256=$expected_overlay_contract_sha
source_stru_out_sha256=$expected_source_stru_sha
symmetry_tail_sha256=$expected_stru_tail_sha
overlay_stru_out_sha256=$expected_overlay_stru_sha
vxc_out_sha256=$expected_vxc_sha
mpi_ranks=$mpi_ranks
omp_threads=$omp_threads
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mixing=none
iterations=0:1
EOF

(
  cd "$run_dir"
  export LD_LIBRARY_PATH="$base_ld_library_path"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >runtime.txt
  timeout 21600 mpirun -np "$mpi_ranks" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>runtime.txt
)
grep -Fq 'libRPA finished successfully' "$run_dir/librpa.stdout"
test ! -s "$run_dir/librpa.stderr"
for trace in qsgw_matrices.dat qsgw_eigenvalues.dat qsgw_iterations.dat; do
  test -s "$run_dir/$trace"
  grep -Fqx '# qsgw_contract_version 6' "$run_dir/$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$run_dir/$trace"
  grep -Fqx '# live_update eigenvalues_wfc' "$run_dir/$trace"
  grep -Fqx '# velocity disabled_stage1' "$run_dir/$trace"
  grep -Fqx '# headwing disabled_stage1' "$run_dir/$trace"
  grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$run_dir/$trace"
  grep -Fqx '# hartree disabled_stage1' "$run_dir/$trace"
  grep -Fqx '# band disabled_stage1' "$run_dir/$trace"
  grep -Fqx '# h_qsgw_cut disabled_non_band' "$run_dir/$trace"
  grep -Fqx '# qsgw_mixer none' "$run_dir/$trace"
  grep -Fqx "# qsgw_input_contract_sha256 $expected_overlay_contract_sha" "$run_dir/$trace"
done

"$python" -B "$tool_dir/validate_qsgw_iter1_v6.py" \
  "$run_dir/qsgw_matrices.dat" \
  "$run_dir/qsgw_eigenvalues.dat" \
  "$run_dir/qsgw_iterations.dat" \
  "$overlay/band_out" \
  "$overlay/qsgw_input.contract" \
  "$tool_dir/cmp_qsgw.py" \
  "$run_root/qsgw-iter1-invariants.json" \
  --closure-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  --eigenvalue-tolerance-ha 1e-10 \
  --initial-tolerance 1e-10 \
  >"$run_root/qsgw-iter1-validator.stdout" \
  2>"$run_root/qsgw-iter1-validator.stderr"

"$python" -B "$tool_dir/compare_qsgw_iter1_g0w0_v1.py" \
  "$run_dir/qsgw_matrices.dat" \
  "$gate1_source/upstream" \
  "$overlay/qsgw_input.contract" \
  "$tool_dir/cmp_qsgw.py" \
  "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  >"$run_root/qsgw-g0w0-comparator.stdout" \
  2>"$run_root/qsgw-g0w0-comparator.stderr"

"$python" - "$run_root/qsgw-iter1-invariants.json" \
  "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    invariants = json.load(handle)
assert invariants["passed"] is True
assert invariants["qsgw_contract_version"] == 6
assert invariants["iterations"] == [0, 1]
assert invariants["qsgw_input_contract_sha256"] == "7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7"
assert invariants["matrix_block_count"] == 8
assert invariants["matrix_dimensions"] == [44]
assert invariants["raw_h_closure_max_abs_ha"] <= 1.0e-10
assert invariants["raw_h_closure_relative_frobenius"] <= 1.0e-8
assert invariants["none_mixer_max_abs_ha"] <= 1.0e-10
assert invariants["hermiticity_max_abs_ha"] <= 1.0e-10
assert invariants["rotation_unitarity_max_abs"] <= 1.0e-10
assert invariants["diagonalization_offdiagonal_max_abs_ha"] <= 1.0e-10
assert invariants["diagonalization_eigenvalue_max_abs_ha"] <= 1.0e-10
assert invariants["fixed_basis_wfc_rotation_relative_frobenius"] <= 1.0e-10
assert invariants["trace_eigenvalue_max_abs_ha"] <= 1.0e-10
assert invariants["input_eigenvalue_max_abs_ha"] <= 1.0e-10
assert invariants["input_occupation_max_abs"] <= 1.0e-10
assert invariants["input_efermi_max_abs_ha"] <= 1.0e-10
assert invariants["trace_summary_electron_count_max_abs"] <= 1.0e-10

with open(sys.argv[2], encoding="utf-8") as handle:
    sigc = json.load(handle)
assert sigc["passed"] is True
assert sigc["qsgw_contract_version"] == 6
assert sigc["qsgw_input_contract_sha256"] == "7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7"
assert sigc["iteration"] == 1
assert sigc["channel"] == 0
assert sigc["block_count"] == 48
assert sigc["spin_count"] == 1
assert sigc["kpoint_count"] == 8
assert sigc["frequency_count"] == 6
assert sigc["matrix_dimensions"] == [44]
assert sigc["max_abs_difference_ha"] <= 1.0e-10
assert sigc["relative_frobenius_difference"] <= 1.0e-10
PY

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate2_current_qsgw_first_self_energy_v1
acceptance=true
runner_commit=$RUNNER_COMMIT
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
gate0_provenance_sha256=$expected_gate0_provenance_sha
gate1_accepted_provenance_sha256=$expected_gate1_provenance_sha
gate1_accepted_output_manifest_sha256=$expected_gate1_manifest_sha
gate1_source_manifest_sha256=$expected_gate1_source_manifest_sha
gate1_upstream_g0w0_source=$gate1_source/upstream
dataset=$dataset
dataset_manifest_sha256=$expected_dataset_manifest_sha
source_qsgw_input_contract_sha256=$expected_contract_sha
overlay_qsgw_input_contract_sha256=$expected_overlay_contract_sha
qsgw_contract_overlay_validation_sha256=$(sha256sum "$run_root/qsgw-contract-overlay-validation.json" | awk '{print $1}')
source_stru_out_sha256=$expected_source_stru_sha
symmetry_tail_sha256=$expected_stru_tail_sha
overlay_stru_out_sha256=$expected_overlay_stru_sha
vxc_out_sha256=$expected_vxc_sha
librpa_input_sha256=$(sha256sum "$run_root/librpa.in" | awk '{print $1}')
qsgw_iter1_invariants_sha256=$(sha256sum "$run_root/qsgw-iter1-invariants.json" | awk '{print $1}')
qsgw_iter1_g0w0_sigc_comparison_sha256=$(sha256sum "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" | awk '{print $1}')
semantic_iteration_zero=immutable_initial_state
semantic_first_self_energy=trace_iteration_1_channel_0
sigc_block_count=48
sigc_max_abs_tolerance_ha=1e-10
sigc_relative_frobenius_tolerance=1e-10
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mixing=none
iterations=0:1
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
run_succeeded=1
trap - EXIT
printf 'FISH_GATE2_CURRENT_QSGW_FIRST_SELF_ENERGY_V1=PASS\n'
cat "$run_root/qsgw-iter1-vs-upstream-g0w0-sigc.json"
cat "$run_root/qsgw-iter1-invariants.json"
