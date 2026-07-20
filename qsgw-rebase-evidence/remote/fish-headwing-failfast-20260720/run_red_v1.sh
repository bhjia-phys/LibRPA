#!/usr/bin/env bash

set -euo pipefail

source_root=/tmp/librpa-qsgw-headwing-failfast-red-v1
build_root=$source_root/build
run_root=/home/bhj/ai-runs/librpa-qsgw-headwing-failfast-red-20260720-v1
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
expected_head=4b3bd906
expected_test_sha=83e93e99423666bc216b984de1ac9c7f32c031fb108c8d4c356c00aa4924b1c1

test ! -e "$run_root"
test -d "$source_root/.git"
test -x "$pytest_env/bin/pytest"
test "$(git -C "$source_root" rev-parse --short=8 HEAD)" = "$expected_head"
test "$(sha256sum "$source_root/driver/test/test_qsgw_inputfile.cpp" | awk '{print $1}')" = "$expected_test_sha"

mkdir -p "$run_root"
cp "$0" "$run_root/run_red_v1.sh"
git -C "$source_root" status --short --branch >"$run_root/git-status.txt"
git -C "$source_root" diff -- driver/test/test_qsgw_inputfile.cpp \
  >"$run_root/red-test.patch"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export PATH="$pytest_env/bin:$PATH"

cmake -S "$source_root" -B "$build_root" \
  >"$run_root/configure.stdout" \
  2>"$run_root/configure.stderr"
cmake --build "$build_root" --target test_qsgw_inputfile -j4 \
  >"$run_root/build.stdout" \
  2>"$run_root/build.stderr"

set +e
ctest --test-dir "$build_root" -R '^test_qsgw_inputfile$' \
  --output-on-failure \
  >"$run_root/ctest.stdout" \
  2>"$run_root/ctest.stderr"
ctest_status=$?
set -e
printf '%s\n' "$ctest_status" >"$run_root/ctest.exit-code.txt"
test "$ctest_status" -ne 0
grep -Fq 'test_qsgw_inputfile' "$run_root/ctest.stdout"

cat >"$run_root/PROVENANCE.txt" <<EOF
analysis=qsgw_headwing_failfast_tdd_red
acceptance_gate=false
expected_failure=true
source_root=$source_root
source_commit=$(git -C "$source_root" rev-parse HEAD)
test_sha256=$expected_test_sha
ctest_exit_code=$ctest_status
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name RED_CONFIRMED -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/RED_CONFIRMED"
printf 'QSGW_HEADWING_FAILFAST_TDD_RED=PASS\n'
