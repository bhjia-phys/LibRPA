#!/usr/bin/env bash

set -euo pipefail

root=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-green-20260716-262a6424
source_dir=$root/source
build_dir=$root/build-oneapi-v2
failed_v2=$root/evidence-oneapi-v2
evidence=$root/evidence-oneapi-v3
python_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
patch_file=$root/polar-unitary-projection-green.patch
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_patch_sha=262a6424d4128248bbedd65976a734105bb757d095f501e508bc1789688b226e
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

test ! -e "$evidence"
test -d "$failed_v2"
test ! -e "$failed_v2/COMPLETE"
test -x "$python_env/bin/python3"
test -x "$python_env/bin/pytest"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  "$expected_patch_sha"
test "$(awk -F= '/^CMAKE_CXX_COMPILER:/{print $2}' "$build_dir/CMakeCache.txt")" = \
  "$expected_cxx"
test "$(awk -F= '/^CMAKE_Fortran_COMPILER:/{print $2}' "$build_dir/CMakeCache.txt")" = \
  "$expected_fortran"
grep -Fq 'OpenMP_CXX_FLAGS:STRING=-fiopenmp' "$build_dir/CMakeCache.txt"
git -C "$source_dir" diff --check

mkdir -p "$evidence"
cp "$0" "$evidence/run-fish-polar-oneapi-ctest-v3.sh"
cp "$patch_file" "$evidence/"
cp "$build_dir/CMakeCache.txt" "$evidence/"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$evidence/oneapi-setvars.log" 2>&1
set -u
export PATH="$python_env/bin:$PATH"
test "$(command -v python3)" = "$python_env/bin/python3"
test "$(command -v pytest)" = "$python_env/bin/pytest"
python3 --version >"$evidence/python-version.txt" 2>&1
pytest --version >"$evidence/pytest-version.txt" 2>&1

cmake --build "$build_dir" -j32 \
  >"$evidence/build-full.stdout" \
  2>"$evidence/build-full.stderr"
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
analysis=qsgw_velocity_polar_oneapi_ctest_v3
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
python_environment=$python_env
python_executable=$(command -v python3)
pytest_executable=$(command -v pytest)
execution_surface=fish_direct_small_build_test
supersedes_misconfigured_evidence=$root/evidence
supersedes_missing_pytest_evidence=$failed_v2
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
printf 'QSGW_VELOCITY_POLAR_ONEAPI_CTEST_V3=PASS\n'
