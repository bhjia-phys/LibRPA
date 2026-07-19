#!/usr/bin/env bash

set -euo pipefail

root=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-red-20260716-785ac390
source_dir=$root/source
build_dir=$root/build-oneapi-v2
evidence=$root/evidence-oneapi-v2
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_test_sha=785ac390524e051dc7c47552ccf4cf6eea68b8993c6d4bfa678ea66d8475a5f1
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

test ! -e "$evidence"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$source_dir/src/test/test_qsgw_fixed_basis.cpp" | awk '{print $1}')" = \
  "$expected_test_sha"
test "$(git -C "$source_dir" diff --name-only)" = \
  "src/test/test_qsgw_fixed_basis.cpp"

mkdir -p "$evidence"
cp "$0" "$evidence/run-fish-polar-tdd-red-oneapi-v2.sh"
git -C "$source_dir" diff -- src/test/test_qsgw_fixed_basis.cpp \
  >"$evidence/tdd-red-test.patch"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$evidence/oneapi-setvars.log" 2>&1
set -u

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="$expected_cxx" \
  -DCMAKE_Fortran_COMPILER="$expected_fortran" \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_CPP_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_TEST=ON \
  -DLIBRPA_ENABLE_DRIVER=ON \
  >"$evidence/configure.stdout" \
  2>"$evidence/configure.stderr"
test "$(awk -F= '/^CMAKE_CXX_COMPILER:/{print $2}' "$build_dir/CMakeCache.txt")" = \
  "$expected_cxx"
test "$(awk -F= '/^CMAKE_Fortran_COMPILER:/{print $2}' "$build_dir/CMakeCache.txt")" = \
  "$expected_fortran"
grep -Fq 'OpenMP_CXX_FLAGS:STRING=-fiopenmp' "$build_dir/CMakeCache.txt"
cp "$build_dir/CMakeCache.txt" "$evidence/"

cmake --build "$build_dir" --target test_qsgw_fixed_basis -j32 \
  >"$evidence/build.stdout" \
  2>"$evidence/build.stderr"
set +e
ctest --test-dir "$build_dir" \
  -R '^test_qsgw_fixed_basis$' \
  --output-on-failure \
  >"$evidence/ctest.stdout" \
  2>"$evidence/ctest.stderr"
ctest_status=$?
set -e
test "$ctest_status" -ne 0
grep -Fq 'rounded_alignment.maximum_relative_wfc_residual > 1.0e-11' \
  "$evidence/ctest.stdout"

cat >"$evidence/PROVENANCE.txt" <<EOF
analysis=qsgw_velocity_polar_tdd_red_oneapi_v2
acceptance_gate=false
expected_failure_confirmed=true
candidate_commit=$expected_commit
test_file_sha256=$expected_test_sha
product_source_changed=false
test_only_change=true
target=test_qsgw_fixed_basis
ctest_exit_status=$ctest_status
cmake_cxx_compiler=$expected_cxx
cmake_fortran_compiler=$expected_fortran
cmake_openmp_cxx_flags=-fiopenmp
execution_surface=fish_direct_small_build_test
supersedes_misconfigured_evidence=$root/evidence
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$evidence"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$evidence/TDD_RED_CONFIRMED"
printf 'QSGW_VELOCITY_POLAR_TDD_RED_ONEAPI_V2=PASS\n'
