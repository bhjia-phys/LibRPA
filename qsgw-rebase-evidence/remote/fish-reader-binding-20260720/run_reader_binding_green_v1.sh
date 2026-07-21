#!/usr/bin/env bash
set -euo pipefail

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
source_root=/tmp/librpa-qsgw-reader-binding-green-v1-b7273e13
build_root=$source_root/build
run_root=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v1
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
expected_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_base=42d3863c1d865194d382a085851d1e2e8a39764f
expected_test_count=60
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

test -d "$bare_repo"
test ! -e "$source_root"
test ! -e "$run_root"
test -x "$pytest_env/bin/pytest"
test -x "$expected_cxx"
test -x "$expected_fortran"

git clone --no-checkout "$bare_repo" "$source_root"
git -C "$source_root" checkout --detach "$expected_commit"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"

mkdir -p "$run_root"
cp "$0" "$run_root/run_reader_binding_green_v1.sh"
git -C "$source_root" status --short --branch >"$run_root/git-status.txt"
git -C "$source_root" rev-parse HEAD >"$run_root/source-commit.txt"
git -C "$source_root" diff --name-status "$upstream_base" \
  >"$run_root/candidate-name-status.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export PATH="$pytest_env/bin:$PATH"

cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="$expected_cxx" \
  -DCMAKE_Fortran_COMPILER="$expected_fortran" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG" \
  -DLIBRPA_ENABLE_DRIVER=ON \
  -DLIBRPA_USE_LIBRI=ON \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_CPP_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_BIND=OFF \
  -DLIBRPA_ENABLE_FORTRAN_TEST=ON \
  -DLIBRPA_USE_CMAKE_INC=OFF \
  -DENABLE_GREENX_CTEST=ON \
  >"$run_root/configure.stdout" \
  2>"$run_root/configure.stderr"

cmake --build "$build_root" --target test_qsgw_input_contract -j4 \
  >"$run_root/focused-build.stdout" \
  2>"$run_root/focused-build.stderr"
ctest --test-dir "$build_root" -R '^test_qsgw_input_contract$' \
  --output-on-failure \
  >"$run_root/focused-ctest.stdout" \
  2>"$run_root/focused-ctest.stderr"

cmake --build "$build_root" -j4 \
  >"$run_root/full-build.stdout" \
  2>"$run_root/full-build.stderr"
ctest --test-dir "$build_root" -N \
  >"$run_root/ctest-list.stdout" \
  2>"$run_root/ctest-list.stderr"
grep -Fq "Total Tests: $expected_test_count" "$run_root/ctest-list.stdout"
ctest --test-dir "$build_root" --output-on-failure -j4 \
  --output-junit "$run_root/full-ctest.xml" \
  >"$run_root/full-ctest.stdout" \
  2>"$run_root/full-ctest.stderr"

git -C "$source_root" diff --exit-code "$upstream_base" -- \
  src/core driver/tasks/g0w0.cpp driver/tasks/rpa.cpp \
  >"$run_root/protected-diff.patch" \
  2>"$run_root/protected-diff.stderr"

grep -Fq '100% tests passed' "$run_root/focused-ctest.stdout"
grep -Fq "100% tests passed, 0 tests failed out of $expected_test_count" \
  "$run_root/full-ctest.stdout"
test ! -s "$run_root/protected-diff.patch"
test -x "$build_root/chi0_main.exe"

executable_sha=$(sha256sum "$build_root/chi0_main.exe" | awk '{print $1}')
cmake_cache_sha=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
printf '%s\n' \
  'analysis=qsgw_reader_binding_green' \
  'acceptance_gate=true' \
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

cp "$build_root/CMakeCache.txt" "$run_root/CMakeCache.txt"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
printf 'QSGW_READER_BINDING_GREEN=PASS\n'
