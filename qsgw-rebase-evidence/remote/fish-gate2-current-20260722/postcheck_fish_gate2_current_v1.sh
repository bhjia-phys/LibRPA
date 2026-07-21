#!/usr/bin/env bash
set -Eeuo pipefail

: "${POSTCHECK_COMMIT:?set POSTCHECK_COMMIT}"
: "${POSTCHECK_SOURCE:?set POSTCHECK_SOURCE}"
: "${POSTCHECK_SHA256:?set POSTCHECK_SHA256}"
: "${POSTCHECK_TAG:?set POSTCHECK_TAG}"

[[ "$POSTCHECK_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$POSTCHECK_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$POSTCHECK_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "POSTCHECK_TAG contains unsafe characters" >&2; exit 2 ;;
esac

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260722-66bfe1cf-v1
gate1_accept=/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-20260722-c91a4305-v1
gate1_source=/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260722-ca542aca-v1
source_run=/home/bhj/ai-runs/librpa-qsgw-gate2-current-20260722-aca53374-v1
source_checkout=/tmp/librpa-qsgw-gate2-runner-aca53374
postcheck_root=/home/bhj/ai-runs/librpa-qsgw-gate2-current-postcheck-$POSTCHECK_TAG
gate_dir=qsgw-rebase-evidence/remote/fish-gate2-current-20260722
tool_dir=$postcheck_root/tools

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274
expected_gate1_provenance_sha=e9f7dd5104696456c5e64699dd38834c2d563ddbc41e77987b27dc7b8adb41b1
expected_gate1_manifest_sha=5a9c7d2c027d09b20c8a16956200e26ee4c6a122dc7ab226099e249f2958a9f4
expected_gate1_source_manifest_sha=2cddd17a864560d9d004d43a6d5f2fe46d9b7798ad33c4f7195005624758cc21

expected_source_runner_commit=aca5337411a80a46db5fd05fd0d60ed043e540b9
expected_source_runner_sha=8f1bf992b412b37faf2ed2d409291ea7e33c34e466f60b70e23c0f5bc1c0c808
expected_source_failed_sha=898e1ddd3f4503ef20814330849b37125be8c7af7f22350e7d249bef016c9991
expected_source_input_sha=35ecbf7db21471645723fd031a21f67fe51afb581f9940e8f99dbb5f96f07aea
expected_source_pair_sha=d33dd8fc9d97f083fe5c4a600bf4ed1ee81a1d27a3f3963a9ba76ed07e10996d
expected_source_stru_report_sha=2f5a8bac3f7c1eecee395be74286e970107070fc91c15f6af7155b5765c94b72
expected_source_contract_report_sha=df4dc8d5c29ec3c8cfc52849570d32c0468eb25cdb636b64ffcb180a7e7f4768
expected_source_overlay_manifest_sha=37b4055c8237a83eda778cbb60307184732bcf483794a1a08272ff0a9e92cceb
expected_source_rejected_report_sha=902d264fa3f047caded9161dc1e379e86e2a81d9270280382531724fc88522fa
expected_matrix_trace_sha=85a8e04bd29780630a6720832898e06e18953810c2b183f87e5be4a9af93af37
expected_eigenvalue_trace_sha=1ee2b4e9afe21987a95bfc3315e9945b0dd9dfa69d8f41e1734f4e4607e009f8
expected_iteration_trace_sha=e7ae3485578958d1ead4bead993e27b508bbe32f7ff7ab688f1b82d5d7837ef9
expected_stdout_sha=4e6ff99fe181a0ff6e3c764cce267600e235959023f3271923201ab3481c10a1
expected_stderr_sha=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
expected_runtime_sha=044bc482630759c4d06b8dc02ea5719e78caae9343a75a03fe15058b7108af64
expected_overlay_contract_sha=7dcd3a99dc081f3321a709d84e8f29a4ffb35371c6170ba1cdb5a76a43ef1fe7
expected_overlay_stru_sha=e756fd9551bfa9df748473880259ba019de904867c1aaff126b1b3a9c51a8873
expected_vxc_sha=7928a0bd99f3a58b78881fa72861da2ccbfb2d4f4a47338a77d140d1adaff1dd

compare_source=$POSTCHECK_SOURCE/$gate_dir/compare_qsgw_iter1_g0w0_v1.py
validate_source=$POSTCHECK_SOURCE/$gate_dir/validate_qsgw_iter1_v6.py
fixture_source=$POSTCHECK_SOURCE/$gate_dir/gate2_test_fixture_v1.py
compare_test_source=$POSTCHECK_SOURCE/$gate_dir/test_compare_qsgw_iter1_g0w0_v1.py
validate_test_source=$POSTCHECK_SOURCE/$gate_dir/test_validate_qsgw_iter1_v6.py
cmp_qsgw_source=$POSTCHECK_SOURCE/regression_tests/backend/comparisons/cmp_qsgw.py
expected_compare_sha=40eb6e8f61353d5ff08e1d4892a3ed09c2931dcee2d2854829e580f3d6692a8d
expected_validate_sha=e6871fc4b8c5519df89c39cf005f3ca2dc8f03713ea2848a5224676a9386eeb2
expected_fixture_sha=14524959b4774f3a87e965ec03715e5185c99efa6bef62c6083b9539c0483be7
expected_compare_test_sha=7aafb90b1765082891528d6db9d2d220583e0df69315151efb420f48b800d8bf
expected_validate_test_sha=e0910205e35a6f01f729d2369c57060d9b1c7970c0a2a7f0147a7380fa5786fe
expected_cmp_qsgw_sha=f1e2b6f19250b0ff8b18785d3d29072f5f423fb4fdc2ae2b35381900f1282dbb

python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
postcheck_succeeded=0

record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $postcheck_succeeded -ne 1 && -d ${postcheck_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$postcheck_root/FAILED"
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

test ! -e "$postcheck_root"
test -d "$POSTCHECK_SOURCE/.git"
test "$(git -C "$POSTCHECK_SOURCE" rev-parse HEAD)" = "$POSTCHECK_COMMIT"
test -z "$(git -C "$POSTCHECK_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$POSTCHECK_SHA256"
test -x "$python"

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
test "$(provenance_value upstream_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_upstream_commit"
test "$(provenance_value candidate_commit "$gate0/PROVENANCE.txt")" = \
  "$expected_candidate_commit"
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
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
(
  cd "$gate1_accept"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$gate1_source"
  sha256sum --check --quiet "$gate1_accept/SOURCE_RUN_SHA256SUMS.txt"
)

test -e "$source_run/FAILED"
test ! -e "$source_run/GREEN_CONFIRMED"
test ! -e "$source_run/PROVENANCE.txt"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = \
  "$expected_source_failed_sha"
grep -Fqx 'exit_code=2' "$source_run/FAILED"
test "$(sha256sum "$source_run/run_fish_gate2_current_v1.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"
test "$(git -C "$source_checkout" rev-parse HEAD)" = \
  "$expected_source_runner_commit"
test -z "$(git -C "$source_checkout" status --porcelain)"
test "$(sha256sum "$source_checkout/$gate_dir/run_fish_gate2_current_v1.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"

declare -A source_hashes=(
  [librpa.in]=$expected_source_input_sha
  [CONTROLLED_PAIR.txt]=$expected_source_pair_sha
  [stru-symmetry-overlay-validation.json]=$expected_source_stru_report_sha
  [qsgw-contract-overlay-validation.json]=$expected_source_contract_report_sha
  [input-overlay.sha256]=$expected_source_overlay_manifest_sha
  [qsgw-iter1-invariants.json]=$expected_source_rejected_report_sha
  [candidate-qsgw/qsgw_matrices.dat]=$expected_matrix_trace_sha
  [candidate-qsgw/qsgw_eigenvalues.dat]=$expected_eigenvalue_trace_sha
  [candidate-qsgw/qsgw_iterations.dat]=$expected_iteration_trace_sha
  [candidate-qsgw/librpa.stdout]=$expected_stdout_sha
  [candidate-qsgw/librpa.stderr]=$expected_stderr_sha
  [candidate-qsgw/runtime.txt]=$expected_runtime_sha
  [input-overlay/qsgw_input.contract]=$expected_overlay_contract_sha
  [input-overlay/stru_out]=$expected_overlay_stru_sha
  [input-overlay/vxc_out]=$expected_vxc_sha
)
for relative_path in "${!source_hashes[@]}"; do
  test "$(sha256sum "$source_run/$relative_path" | awk '{print $1}')" = \
    "${source_hashes[$relative_path]}"
done
(
  cd "$source_run"
  sha256sum --check --quiet input-overlay.sha256
)
grep -Fq 'libRPA finished successfully' "$source_run/candidate-qsgw/librpa.stdout"
test ! -s "$source_run/candidate-qsgw/librpa.stderr"
grep -Fq 'completed_utc=' "$source_run/candidate-qsgw/runtime.txt"
grep -Fqx '# qsgw_contract_version 6' "$source_run/candidate-qsgw/qsgw_matrices.dat"
grep -Fqx '# symmetry exx_on_gw_on_rpa_on' "$source_run/candidate-qsgw/qsgw_matrices.dat"
grep -Fqx '# headwing disabled_stage1' "$source_run/candidate-qsgw/qsgw_matrices.dat"
grep -Fqx '# hartree disabled_stage1' "$source_run/candidate-qsgw/qsgw_matrices.dat"
grep -Fqx '# band disabled_stage1' "$source_run/candidate-qsgw/qsgw_matrices.dat"
grep -Fqx '# qsgw_mixer none' "$source_run/candidate-qsgw/qsgw_matrices.dat"

for pair in \
  "$compare_source:$expected_compare_sha" \
  "$validate_source:$expected_validate_sha" \
  "$fixture_source:$expected_fixture_sha" \
  "$compare_test_source:$expected_compare_test_sha" \
  "$validate_test_source:$expected_validate_test_sha" \
  "$cmp_qsgw_source:$expected_cmp_qsgw_sha"; do
  path=${pair%%:*}
  expected=${pair##*:}
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
done

mkdir -p "$postcheck_root" "$tool_dir"
cp "$0" "$postcheck_root/postcheck_fish_gate2_current_v1.sh"
cp "$compare_source" "$tool_dir/compare_qsgw_iter1_g0w0_v1.py"
cp "$validate_source" "$tool_dir/validate_qsgw_iter1_v6.py"
cp "$fixture_source" "$tool_dir/gate2_test_fixture_v1.py"
cp "$compare_test_source" "$tool_dir/test_compare_qsgw_iter1_g0w0_v1.py"
cp "$validate_test_source" "$tool_dir/test_validate_qsgw_iter1_v6.py"
cp "$cmp_qsgw_source" "$tool_dir/cmp_qsgw.py"
(
  cd "$source_run"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$postcheck_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$source_run"
  sha256sum --check --quiet "$postcheck_root/SOURCE_RUN_SHA256SUMS.txt"
)

(
  cd "$tool_dir"
  CMP_QSGW="$tool_dir/cmp_qsgw.py" "$python" -B -m unittest -v \
    test_compare_qsgw_iter1_g0w0_v1.py \
    test_validate_qsgw_iter1_v6.py \
    >"$postcheck_root/observer-unit-test.stdout" \
    2>"$postcheck_root/observer-unit-test.stderr"
)

"$python" - "$source_run/qsgw-iter1-invariants.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["passed"] is False
assert report["raw_h_closure_max_abs_ha"] == 8.519558093667001e-06
assert report["raw_h_closure_relative_frobenius"] == 4.447157099356966e-09
assert report["hermiticity_max_abs_ha"] == 8.51955812610286e-06
assert report["none_mixer_max_abs_ha"] <= 1.0e-10
assert report["rotation_unitarity_max_abs"] <= 1.0e-10
assert report["diagonalization_offdiagonal_max_abs_ha"] <= 1.0e-10
assert report["diagonalization_eigenvalue_max_abs_ha"] <= 1.0e-10
assert report["fixed_basis_wfc_rotation_relative_frobenius"] <= 1.0e-10
assert report["trace_eigenvalue_max_abs_ha"] <= 1.0e-10
assert report["input_eigenvalue_max_abs_ha"] <= 1.0e-10
assert report["input_occupation_max_abs"] <= 1.0e-10
assert report["input_efermi_max_abs_ha"] <= 1.0e-10
assert report["trace_summary_electron_count_max_abs"] <= 1.0e-10
PY

"$python" -B "$tool_dir/validate_qsgw_iter1_v6.py" \
  "$source_run/candidate-qsgw/qsgw_matrices.dat" \
  "$source_run/candidate-qsgw/qsgw_eigenvalues.dat" \
  "$source_run/candidate-qsgw/qsgw_iterations.dat" \
  "$source_run/input-overlay/band_out" \
  "$source_run/input-overlay/qsgw_input.contract" \
  "$tool_dir/cmp_qsgw.py" \
  "$postcheck_root/qsgw-iter1-invariants.json" \
  --closure-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  --eigenvalue-tolerance-ha 1e-10 \
  --initial-tolerance 1e-10 \
  >"$postcheck_root/qsgw-iter1-validator.stdout" \
  2>"$postcheck_root/qsgw-iter1-validator.stderr"

"$python" -B "$tool_dir/compare_qsgw_iter1_g0w0_v1.py" \
  "$source_run/candidate-qsgw/qsgw_matrices.dat" \
  "$gate1_source/upstream" \
  "$source_run/input-overlay/qsgw_input.contract" \
  "$tool_dir/cmp_qsgw.py" \
  "$postcheck_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  --iteration 1 \
  --channel 0 \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  >"$postcheck_root/qsgw-g0w0-comparator.stdout" \
  2>"$postcheck_root/qsgw-g0w0-comparator.stderr"

"$python" - "$postcheck_root/qsgw-iter1-invariants.json" \
  "$postcheck_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" <<'PY'
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
assert invariants["component_hermiticity_max_abs_ha"]["exx"] == 8.51955812610286e-06
assert invariants["component_hermiticity_max_abs_ha"]["vc"] == 0.0
assert invariants["component_hermiticity_max_abs_ha"]["vxc_dft"] == 1.0069671667484314e-12
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
assert sigc["max_abs_difference_ha"] == 2.0267304319323924e-11
assert sigc["relative_frobenius_difference"] == 2.125913085616405e-11
assert sigc["max_abs_tolerance_ha"] == 1.0e-10
assert sigc["relative_frobenius_tolerance"] == 1.0e-10
PY

source_manifest_sha=$(sha256sum "$postcheck_root/SOURCE_RUN_SHA256SUMS.txt" | awk '{print $1}')
invariants_sha=$(sha256sum "$postcheck_root/qsgw-iter1-invariants.json" | awk '{print $1}')
sigc_sha=$(sha256sum "$postcheck_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" | awk '{print $1}')
cat >"$postcheck_root/PROVENANCE.txt" <<EOF
gate=fish_gate2_current_qsgw_first_self_energy_postcheck_v1
acceptance=true
postcheck_commit=$POSTCHECK_COMMIT
postcheck_runner_sha256=$POSTCHECK_SHA256
postcheck_tag=$POSTCHECK_TAG
source_run=$source_run
source_run_status=rejected_observer_semantics_only
source_runner_commit=$expected_source_runner_commit
source_runner_sha256=$expected_source_runner_sha
source_run_manifest_sha256=$source_manifest_sha
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
matrix_trace_sha256=$expected_matrix_trace_sha
eigenvalue_trace_sha256=$expected_eigenvalue_trace_sha
iteration_trace_sha256=$expected_iteration_trace_sha
qsgw_input_contract_sha256=$expected_overlay_contract_sha
semantic_iteration_zero=immutable_initial_state
semantic_first_self_energy=trace_iteration_1_channel_0
hamiltonian_closure=legacy_upper_triangle_authoritative
sigc_block_count=48
sigc_max_abs_difference_ha=2.0267304319323924e-11
sigc_relative_frobenius_difference=2.125913085616405e-11
sigc_max_abs_tolerance_ha=1e-10
sigc_relative_frobenius_tolerance=1e-10
qsgw_iter1_invariants_sha256=$invariants_sha
qsgw_iter1_g0w0_sigc_comparison_sha256=$sigc_sha
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mixing=none
iterations=0:1
mpi_ranks=1
omp_threads=32
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$source_run"
  sha256sum --check --quiet "$postcheck_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$postcheck_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
printf 'FISH_GATE2_CURRENT_QSGW_FIRST_SELF_ENERGY_POSTCHECK_V1=PASS\n'
cat "$postcheck_root/qsgw-iter1-invariants.json"
cat "$postcheck_root/qsgw-iter1-vs-upstream-g0w0-sigc.json"
touch "$postcheck_root/GREEN_CONFIRMED"
postcheck_succeeded=1
trap - EXIT
