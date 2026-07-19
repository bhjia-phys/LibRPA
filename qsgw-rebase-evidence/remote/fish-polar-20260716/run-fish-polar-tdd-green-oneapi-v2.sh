#!/usr/bin/env bash

set -euo pipefail

root=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-green-20260716-262a6424
source_dir=$root/source
build_dir=$root/build-oneapi-v2
evidence=$root/evidence-oneapi-v2
patch_file=$root/polar-unitary-projection-green.patch
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_patch_sha=262a6424d4128248bbedd65976a734105bb757d095f501e508bc1789688b226e
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

test ! -e "$evidence"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  "$expected_patch_sha"
test "$(git -C "$source_dir" diff --name-only | wc -l)" -eq 3
git -C "$source_dir" diff --check

mkdir -p "$evidence"
cp "$0" "$evidence/run-fish-polar-tdd-green-oneapi-v2.sh"
cp "$patch_file" "$evidence/"
git -C "$source_dir" diff -- \
  src/qsgw/fixed_basis.cpp \
  src/qsgw/fixed_basis.h \
  src/test/test_qsgw_fixed_basis.cpp \
  >"$evidence/applied.patch"
test "$(sha256sum "$evidence/applied.patch" | awk '{print $1}')" = \
  "$expected_patch_sha"

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

cmake --build "$build_dir" \
  --target test_qsgw_fixed_basis test_qsgw_fixed_basis_mpi \
  -j32 \
  >"$evidence/build-targeted.stdout" \
  2>"$evidence/build-targeted.stderr"
export OMP_NUM_THREADS=1
ctest --test-dir "$build_dir" \
  -R '^test_qsgw_fixed_basis$' \
  --output-on-failure \
  >"$evidence/ctest-serial.stdout" \
  2>"$evidence/ctest-serial.stderr"
ctest --test-dir "$build_dir" \
  -R '^test_qsgw_fixed_basis_mpi$' \
  --output-on-failure \
  >"$evidence/ctest-mpi.stdout" \
  2>"$evidence/ctest-mpi.stderr"
grep -Fq '100% tests passed' "$evidence/ctest-serial.stdout"
grep -Fq '100% tests passed' "$evidence/ctest-mpi.stdout"

cmake --build "$build_dir" -j32 \
  >"$evidence/build-full.stdout" \
  2>"$evidence/build-full.stderr"
ctest --test-dir "$build_dir" -N \
  >"$evidence/ctest-list.stdout" \
  2>"$evidence/ctest-list.stderr"
grep -Fq 'Total Tests: 59' "$evidence/ctest-list.stdout"
ctest --test-dir "$build_dir" --output-on-failure -j16 \
  >"$evidence/ctest-full.stdout" \
  2>"$evidence/ctest-full.stderr"
grep -Fq '100% tests passed, 0 tests failed out of 59' \
  "$evidence/ctest-full.stdout"

cat >"$evidence/PROVENANCE.txt" <<EOF
analysis=qsgw_velocity_polar_tdd_green_oneapi_v2
acceptance_gate=false
candidate_parent_commit=$expected_commit
patch_sha256=$expected_patch_sha
changed_files=src/qsgw/fixed_basis.cpp,src/qsgw/fixed_basis.h,src/test/test_qsgw_fixed_basis.cpp
protected_shared_numerical_files_changed=false
serial_fixed_basis_test=PASS
mpi4_fixed_basis_test=PASS
candidate_ctest_count=59
candidate_ctest_result=PASS
omp_threads=1
ctest_parallelism=16
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
touch "$evidence/COMPLETE"
printf 'QSGW_VELOCITY_POLAR_TDD_GREEN_ONEAPI_V2=PASS\n'
