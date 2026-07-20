#!/usr/bin/env bash

set -euo pipefail

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
source_root=/tmp/librpa-qsgw-headwing-failfast-green-v1-44059ebf
build_root=$source_root/build
run_root=/home/bhj/ai-runs/librpa-qsgw-headwing-failfast-green-20260720-v1
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
expected_commit=44059ebf4dbe2c76f2bfa7598baa388e14b0205e
upstream_base=42d3863c1d865194d382a085851d1e2e8a39764f

test -d "$bare_repo"
test ! -e "$source_root"
test ! -e "$run_root"
test -x "$pytest_env/bin/pytest"

git --git-dir="$bare_repo" worktree add --detach "$source_root" "$expected_commit"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"

mkdir -p "$run_root"
cp "$0" "$run_root/run_green_v1.sh"
git -C "$source_root" status --short --branch >"$run_root/git-status.txt"
git -C "$source_root" rev-parse HEAD >"$run_root/source-commit.txt"

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
  >"$run_root/focused-build.stdout" \
  2>"$run_root/focused-build.stderr"
ctest --test-dir "$build_root" -R '^test_qsgw_inputfile$' \
  --output-on-failure \
  >"$run_root/focused-ctest.stdout" \
  2>"$run_root/focused-ctest.stderr"

cmake --build "$build_root" -j4 \
  >"$run_root/full-build.stdout" \
  2>"$run_root/full-build.stderr"
ctest --test-dir "$build_root" --output-on-failure \
  >"$run_root/full-ctest.stdout" \
  2>"$run_root/full-ctest.stderr"

git -C "$source_root" diff --exit-code "$upstream_base" -- \
  src/core driver/tasks/g0w0.cpp driver/tasks/rpa.cpp \
  >"$run_root/protected-diff.patch" \
  2>"$run_root/protected-diff.stderr"

grep -Fq '100% tests passed' "$run_root/focused-ctest.stdout"
grep -Fq '100% tests passed' "$run_root/full-ctest.stdout"
grep -Fq 'QSGW iterative head/wing is unsupported' \
  "$source_root/driver/inputfile.cpp"
grep -Fq 'QSGW iterative head/wing is unsupported' \
  "$source_root/driver/tasks/qsgw.cpp"
test ! -e "$source_root/src/qsgw/headwing_update.cpp"
test ! -e "$source_root/src/qsgw/headwing_update.h"
test ! -e "$source_root/src/test/test_qsgw_headwing_update.cpp"
test ! -s "$run_root/protected-diff.patch"

printf '%s\n' \
  'analysis=qsgw_headwing_failfast_tdd_green' \
  'acceptance_gate=true' \
  "source_root=$source_root" \
  "source_commit=$expected_commit" \
  "upstream_base=$upstream_base" \
  'focused_ctest=pass' \
  'full_ctest=pass' \
  'protected_diff=empty' \
  "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
printf 'QSGW_HEADWING_FAILFAST_TDD_GREEN=PASS\n'
