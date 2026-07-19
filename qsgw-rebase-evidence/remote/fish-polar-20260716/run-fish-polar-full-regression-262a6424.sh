#!/usr/bin/env bash

set -euo pipefail

root=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-green-20260716-262a6424
source_dir=$root/source
build_dir=$root/build
green_evidence=$root/evidence
evidence=$root/full-regression
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_patch_sha=262a6424d4128248bbedd65976a734105bb757d095f501e508bc1789688b226e
expected_green_manifest_sha=c9b09c2617fe266ae679280527f0fab7bf169cefdfd094a66cd48d00c5b7dbf2

test ! -e "$evidence"
test -e "$green_evidence/TDD_GREEN_CONFIRMED"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$root/polar-unitary-projection-green.patch" | awk '{print $1}')" = \
  "$expected_patch_sha"
test "$(sha256sum "$green_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_green_manifest_sha"
(
  cd "$green_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
git -C "$source_dir" diff --check

mkdir -p "$evidence"
cp "$0" "$evidence/run-fish-polar-full-regression-262a6424.sh"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$evidence/oneapi-setvars.log" 2>&1
set -u

cmake --build "$build_dir" -j32 \
  >"$evidence/build.stdout" \
  2>"$evidence/build.stderr"
ctest --test-dir "$build_dir" -N \
  >"$evidence/ctest-list.stdout" \
  2>"$evidence/ctest-list.stderr"
grep -Fq 'Total Tests: 59' "$evidence/ctest-list.stdout"

export OMP_NUM_THREADS=1
ctest --test-dir "$build_dir" --output-on-failure -j16 \
  >"$evidence/ctest.stdout" \
  2>"$evidence/ctest.stderr"
grep -Fq '100% tests passed, 0 tests failed out of 59' \
  "$evidence/ctest.stdout"

cat >"$evidence/PROVENANCE.txt" <<EOF
analysis=qsgw_velocity_polar_full_candidate_regression
acceptance_gate=false
candidate_parent_commit=$expected_commit
patch_sha256=$expected_patch_sha
changed_files=src/qsgw/fixed_basis.cpp,src/qsgw/fixed_basis.h,src/test/test_qsgw_fixed_basis.cpp
protected_shared_numerical_files_changed=false
candidate_ctest_count=59
candidate_ctest_result=PASS
omp_threads=1
ctest_parallelism=16
compiler_environment=oneapi
cxx_compiler=$(command -v mpiicpx)
fortran_compiler=$(command -v mpiifx)
execution_surface=fish_direct_small_build_test
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
printf 'QSGW_VELOCITY_POLAR_FULL_REGRESSION=PASS\n'
