#!/usr/bin/env bash
set -Eeuo pipefail

: "${RECOVERY_COMMIT:?set RECOVERY_COMMIT}"
: "${RECOVERY_SOURCE:?set RECOVERY_SOURCE}"
: "${RECOVERY_SHA256:?set RECOVERY_SHA256}"
: "${RECOVERY_TAG:?set RECOVERY_TAG}"

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260722-66bfe1cf-v1
source_run=/home/bhj/ai-runs/librpa-qsgw-gate1-current-20260722-ca542aca-v1
source_checkout=/tmp/librpa-qsgw-gate1-runner-ca542aca
recovery_root=/home/bhj/ai-runs/librpa-qsgw-gate1-current-postcheck-$RECOVERY_TAG
gate_dir=qsgw-rebase-evidence/remote/fish-gate1-current-20260722
tool_dir=$recovery_root/tools

expected_source_runner_commit=ca542aca1fe59a4e917cce93a04d886e8c8fd346
expected_source_runner_sha=d37503f1196ec9a28fffb7da695d6f5ada033c6cf3bd2ea498c7162c60b92eeb
expected_source_failed_sha=f861ba7366cb3db14b631505a388a4a79b09d4f53fb8119d83bdf3c6583b3813
expected_source_input_sha=72c03a69a67ef62f207907b16cbac2f32fc9d741d21ceaee4dc71882181a2f35
expected_source_pair_sha=12dbf5ac786983dbb3b089606ee1eeecd378afcef151b08dfec95e177997b3dc
expected_source_overlay_report_sha=8ebde1b139fc79a2ca73089443233d3853e37873d46728b31762ae78d8e4d57c
expected_source_overlay_manifest_sha=17e12adc1af5db67ad19c95bdecc9da41d65b97d89aab7731cce555abbf0620d
expected_source_g0w0_comparison_sha=d23012c91c9ed77d611007201196b1f1a7d939e8b8796cf0052b030a31cf2cfa
expected_upstream_energy_sha=cd0a1462b419abc823cae31efb0990a12a969bc5979b4a2e65969696cda6246f
expected_candidate_energy_sha=8fc06df11d7382dd8491a890270ee35d79b2aaa218667d58154ecaafb0ce88a8

expected_upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_commit=66bfe1cfd35c983222935d250039a8fe5c4b7af1
expected_gate0_provenance_sha=a2dade04eafdc6c1f5af9bf28c4fbbf5a54b4b31e393b05352b6941be69210a2
expected_gate0_manifest_sha=4f65a0eb309c7d8bc4ee52a2e90ca1107c9688465234e6f45cbbdcfbbc60d274
expected_upstream_exe_sha=ae86432a32bdd9d2cc6e7d2f80f7e99e4de053329eb233c69d6e7e833aa006c3
expected_candidate_exe_sha=b5f9ea21e15c583db47644d4f71513d79fb3a7e06ea65bfd3e891b61923f6c16

g0w0_comparator_source=$RECOVERY_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/compare_g0w0_sigc_dump_directories.py
g0w0_comparator_test_source=$RECOVERY_SOURCE/qsgw-rebase-evidence/remote/dongfang-gates-20260715-7d69a18c/test_compare_g0w0_sigc_dump_directories.py
energy_comparator_source=$RECOVERY_SOURCE/$gate_dir/compare_energy_qp_v1.py
energy_comparator_test_source=$RECOVERY_SOURCE/$gate_dir/test_compare_energy_qp_v1.py
expected_g0w0_comparator_sha=15417e56ce7b92fc719eea6788944de8805fba5e5849698fa32a89206cba2eaa
expected_g0w0_comparator_test_sha=5ce8acf1c8659c3c5e8f20a61d62e8347f987f7b3adeb09b4c7cb3ce5cef2dcd
expected_energy_comparator_sha=3b0cda0c4287ff1b483fe26c511dbebd755f9c4e1a8bd0e85fa2660bfc596bbb
expected_energy_comparator_test_sha=c172118a9a735342468c4eef009baafe9b18db4c10cbc89e190bb967cbf64cd5

python=${PYTHON:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python}
recovery_succeeded=0

record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $recovery_succeeded -ne 1 && -d ${recovery_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$recovery_root/FAILED"
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

[[ "$RECOVERY_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RECOVERY_SHA256" =~ ^[0-9a-f]{64}$ ]]
test ! -e "$recovery_root"
test -d "$RECOVERY_SOURCE/.git"
test "$(git -C "$RECOVERY_SOURCE" rev-parse HEAD)" = "$RECOVERY_COMMIT"
test -z "$(git -C "$RECOVERY_SOURCE" status --porcelain)"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RECOVERY_SHA256"
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
upstream_exe=$(provenance_value upstream_executable "$gate0/PROVENANCE.txt")
candidate_exe=$(provenance_value candidate_executable "$gate0/PROVENANCE.txt")
test "$(sha256sum "$upstream_exe" | awk '{print $1}')" = \
  "$expected_upstream_exe_sha"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"

test -e "$source_run/FAILED"
test ! -e "$source_run/GREEN_CONFIRMED"
test ! -e "$source_run/PROVENANCE.txt"
test "$(sha256sum "$source_run/FAILED" | awk '{print $1}')" = \
  "$expected_source_failed_sha"
grep -Fqx 'exit_code=1' "$source_run/FAILED"
test "$(sha256sum "$source_run/run_fish_gate1_current_v1.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"
test "$(git -C "$source_checkout" rev-parse HEAD)" = \
  "$expected_source_runner_commit"
test -z "$(git -C "$source_checkout" status --porcelain)"
test "$(sha256sum "$source_checkout/$gate_dir/run_fish_gate1_current_v1.sh" | awk '{print $1}')" = \
  "$expected_source_runner_sha"
test "$(sha256sum "$source_run/librpa.in" | awk '{print $1}')" = \
  "$expected_source_input_sha"
test "$(sha256sum "$source_run/CONTROLLED_PAIR.txt" | awk '{print $1}')" = \
  "$expected_source_pair_sha"
test "$(sha256sum "$source_run/stru-symmetry-overlay-validation.json" | awk '{print $1}')" = \
  "$expected_source_overlay_report_sha"
test "$(sha256sum "$source_run/input-overlay.sha256" | awk '{print $1}')" = \
  "$expected_source_overlay_manifest_sha"
test "$(sha256sum "$source_run/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_source_g0w0_comparison_sha"
test "$(sha256sum "$source_run/upstream/energy_qp" | awk '{print $1}')" = \
  "$expected_upstream_energy_sha"
test "$(sha256sum "$source_run/candidate/energy_qp" | awk '{print $1}')" = \
  "$expected_candidate_energy_sha"
cmp "$source_run/librpa.in" "$source_run/upstream/librpa.in"
cmp "$source_run/librpa.in" "$source_run/candidate/librpa.in"
for label in upstream candidate; do
  test ! -s "$source_run/$label/librpa.stderr"
  grep -Fq 'libRPA finished successfully' "$source_run/$label/librpa.stdout"
  grep -Fq 'completed_utc=' "$source_run/$label/runtime.txt"
  test "$(find "$source_run/$label" -maxdepth 1 \
    -name 'Sigc_fk_mn_kgrid_ispin_*_ik_*_ifreq_*.bin' | wc -l)" -eq 48
done

test "$(sha256sum "$g0w0_comparator_source" | awk '{print $1}')" = \
  "$expected_g0w0_comparator_sha"
test "$(sha256sum "$g0w0_comparator_test_source" | awk '{print $1}')" = \
  "$expected_g0w0_comparator_test_sha"
test "$(sha256sum "$energy_comparator_source" | awk '{print $1}')" = \
  "$expected_energy_comparator_sha"
test "$(sha256sum "$energy_comparator_test_source" | awk '{print $1}')" = \
  "$expected_energy_comparator_test_sha"

mkdir -p "$recovery_root" "$tool_dir"
cp "$0" "$recovery_root/recover_fish_gate1_current_v1.sh"
cp "$g0w0_comparator_source" "$tool_dir/compare_g0w0_sigc_dump_directories.py"
cp "$g0w0_comparator_test_source" "$tool_dir/test_compare_g0w0_sigc_dump_directories.py"
cp "$energy_comparator_source" "$tool_dir/compare_energy_qp_v1.py"
cp "$energy_comparator_test_source" "$tool_dir/test_compare_energy_qp_v1.py"
(
  cd "$source_run"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$source_run"
  sha256sum --check --quiet "$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)

(
  cd "$tool_dir"
  "$python" -B -m unittest -v \
    test_compare_g0w0_sigc_dump_directories.py \
    test_compare_energy_qp_v1.py \
    >"$recovery_root/comparator-unit-test.stdout" \
    2>"$recovery_root/comparator-unit-test.stderr"
)

"$python" - "$source_run/g0w0-comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["passed"] is False
assert report["block_count"] == 48
assert report["max_abs_difference_ha"] == 1.8186075345471608e-11
assert report["relative_frobenius_difference"] == 2.1449190585723414e-11
assert report["thresholds"]["max_abs_tolerance_ha"] == 1.0e-12
assert report["thresholds"]["relative_frobenius_tolerance"] == 1.0e-12
PY

"$python" -B "$tool_dir/compare_g0w0_sigc_dump_directories.py" \
  "$source_run/upstream" "$source_run/candidate" \
  "$recovery_root/g0w0-comparison.json" \
  --source kgrid \
  --max-abs-tolerance-ha 1e-10 \
  --relative-frobenius-tolerance 1e-10 \
  >"$recovery_root/g0w0-comparator.stdout" \
  2>"$recovery_root/g0w0-comparator.stderr"
"$python" -B "$tool_dir/compare_energy_qp_v1.py" \
  "$source_run/upstream/energy_qp" "$source_run/candidate/energy_qp" \
  "$recovery_root/energy-qp-comparison.json" \
  --max-abs-tolerance-ha 1e-9 \
  >"$recovery_root/energy-qp-comparator.stdout" \
  2>"$recovery_root/energy-qp-comparator.stderr"

"$python" - "$recovery_root/g0w0-comparison.json" \
  "$recovery_root/energy-qp-comparison.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    sigma = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    energy = json.load(handle)
assert sigma["passed"] is True
assert sigma["block_count"] == 48
assert sigma["spin_count"] == 1
assert sigma["kpoint_count"] == 8
assert sigma["frequency_count"] == 6
assert sigma["matrix_dimensions"] == [44]
assert sigma["max_abs_difference_ha"] == 1.8186075345471608e-11
assert sigma["relative_frobenius_difference"] == 2.1449190585723414e-11
assert sigma["thresholds"]["max_abs_tolerance_ha"] == 1.0e-10
assert sigma["thresholds"]["relative_frobenius_tolerance"] == 1.0e-10
assert energy["passed"] is True
assert energy["kpoint_count"] == 64
assert energy["state_count"] == 2816
assert energy["kpoint_coordinate_max_abs_difference"] == 0.0
assert energy["occupation_max_abs_difference"] == 0.0
assert energy["ks_energy_max_abs_difference_ha"] == 0.0
assert energy["qp_energy_max_abs_difference_ha"] == 2.000000165480742e-10
assert energy["qp_energy_nonzero_difference_count"] == 47
assert energy["thresholds"]["qp_energy_max_abs_tolerance_ha"] == 1.0e-9
PY

source_manifest_sha=$(sha256sum "$recovery_root/SOURCE_RUN_SHA256SUMS.txt" | awk '{print $1}')
g0w0_comparison_sha=$(sha256sum "$recovery_root/g0w0-comparison.json" | awk '{print $1}')
energy_comparison_sha=$(sha256sum "$recovery_root/energy-qp-comparison.json" | awk '{print $1}')
cat >"$recovery_root/PROVENANCE.txt" <<EOF
gate=fish_gate1_current_g0w0_ab_recovery_v1
acceptance=true
recovery_commit=$RECOVERY_COMMIT
recovery_runner_sha256=$RECOVERY_SHA256
recovery_tag=$RECOVERY_TAG
source_run=$source_run
source_run_status=rejected_observer_threshold_only
source_runner_commit=$expected_source_runner_commit
source_runner_sha256=$expected_source_runner_sha
source_run_manifest_sha256=$source_manifest_sha
upstream_commit=$expected_upstream_commit
candidate_commit=$expected_candidate_commit
upstream_executable=$upstream_exe
upstream_executable_sha256=$expected_upstream_exe_sha
candidate_executable=$candidate_exe
candidate_executable_sha256=$expected_candidate_exe_sha
librpa_input_sha256=$expected_source_input_sha
sigc_block_count=48
sigc_max_abs_difference_ha=1.8186075345471608e-11
sigc_relative_frobenius_difference=2.1449190585723414e-11
sigc_max_abs_tolerance_ha=1e-10
sigc_relative_frobenius_tolerance=1e-10
g0w0_comparison_sha256=$g0w0_comparison_sha
energy_qp_kpoint_count=64
energy_qp_state_count=2816
energy_qp_occupation_and_ks_exact=true
energy_qp_nonzero_difference_count=47
energy_qp_max_abs_difference_ha=2.000000165480742e-10
energy_qp_max_abs_tolerance_ha=1e-9
energy_qp_comparison_sha256=$energy_comparison_sha
symmetry=exx_on_gw_on_rpa_on
headwing=off
hartree=off
band=off
mpi_ranks=1
omp_threads=32
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$source_run"
  sha256sum --check --quiet "$recovery_root/SOURCE_RUN_SHA256SUMS.txt"
)
(
  cd "$recovery_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
printf 'FISH_GATE1_CURRENT_G0W0_AB_RECOVERY_V1=PASS\n'
cat "$recovery_root/g0w0-comparison.json"
cat "$recovery_root/energy-qp-comparison.json"
touch "$recovery_root/GREEN_CONFIRMED"
recovery_succeeded=1
trap - EXIT
