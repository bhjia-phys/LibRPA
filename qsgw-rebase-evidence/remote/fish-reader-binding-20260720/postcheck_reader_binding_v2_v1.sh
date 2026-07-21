#!/usr/bin/env bash
set -euo pipefail

source_root=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
build_root=$source_root/build
source_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2
run_root=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1
expected_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_base=42d3863c1d865194d382a085851d1e2e8a39764f
expected_test_count=60
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

test -d "$source_root/.git"
test -d "$build_root"
test -d "$source_evidence"
test ! -e "$run_root"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"
git -C "$source_root" diff --exit-code
git -C "$source_root" diff --cached --exit-code

mkdir -p "$run_root"
cp "$0" "$run_root/postcheck_reader_binding_v2_v1.sh"
git -C "$source_root" status --short --branch >"$run_root/git-status.txt"
git -C "$source_root" rev-parse HEAD >"$run_root/source-commit.txt"
git -C "$source_root" diff --name-status "$upstream_base" \
  >"$run_root/candidate-name-status.txt"
git -C "$source_root" diff --exit-code "$upstream_base" -- \
  src/core driver/tasks/g0w0.cpp driver/tasks/rpa.cpp \
  >"$run_root/protected-diff.patch" \
  2>"$run_root/protected-diff.stderr"

cp "$source_evidence/configure.stdout" "$run_root/"
cp "$source_evidence/configure.stderr" "$run_root/"
cp "$source_evidence/focused-build.stdout" "$run_root/"
cp "$source_evidence/focused-build.stderr" "$run_root/"
cp "$source_evidence/focused-ctest.stdout" "$run_root/"
cp "$source_evidence/focused-ctest.stderr" "$run_root/"
cp "$source_evidence/full-build.stdout" "$run_root/"
cp "$source_evidence/full-build.stderr" "$run_root/"
cp "$source_evidence/full-ctest.stdout" "$run_root/"
cp "$source_evidence/full-ctest.stderr" "$run_root/"
cp "$source_evidence/full-ctest.xml" "$run_root/"
cp "$source_evidence/ctest-list.stdout" "$run_root/"
cp "$source_evidence/ctest-list.stderr" "$run_root/"
cp "$build_root/CMakeCache.txt" "$run_root/CMakeCache.txt"

grep -Fq '100% tests passed, 0 tests failed out of 1' \
  "$run_root/focused-ctest.stdout"
grep -Fq "100% tests passed, 0 tests failed out of $expected_test_count" \
  "$run_root/full-ctest.stdout"
grep -Fq "Total Tests: $expected_test_count" "$run_root/ctest-list.stdout"
test ! -s "$run_root/focused-build.stderr"
test ! -s "$run_root/full-build.stderr"
test ! -s "$run_root/protected-diff.patch"
test "$(awk -F= '/^CMAKE_CXX_COMPILER:/{print $2}' "$build_root/CMakeCache.txt")" = \
  "$expected_cxx"
test "$(awk -F= '/^CMAKE_Fortran_COMPILER:/{print $2}' "$build_root/CMakeCache.txt")" = \
  "$expected_fortran"
test -x "$build_root/chi0_main.exe"

executable_sha=$(sha256sum "$build_root/chi0_main.exe" | awk '{print $1}')
cmake_cache_sha=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
runner_sha=$(sha256sum "$0" | awk '{print $1}')
printf '%s\n' \
  'analysis=qsgw_reader_binding_v2_postcheck' \
  'acceptance_gate=true' \
  "runner_sha256=$runner_sha" \
  "source_evidence=$source_evidence" \
  "source_root=$source_root" \
  "source_commit=$expected_commit" \
  "upstream_base=$upstream_base" \
  "expected_test_count=$expected_test_count" \
  "passed_test_count=$expected_test_count" \
  'failed_test_count=0' \
  'not_run_test_count=0' \
  'focused_ctest=pass' \
  'full_ctest=pass' \
  'protected_diff=empty' \
  "cxx_compiler=$expected_cxx" \
  "fortran_compiler=$expected_fortran" \
  "executable_sha256=$executable_sha" \
  "cmake_cache_sha256=$cmake_cache_sha" \
  "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
printf 'QSGW_READER_BINDING_V2_POSTCHECK=PASS\n'
