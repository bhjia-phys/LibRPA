#!/usr/bin/env bash

set -euo pipefail

root=/home/bhj/ai-runs/librpa-qsgw-polar-tdd-green-20260716-262a6424
source_dir=$root/source
build_dir=$root/build
evidence=$root/evidence
patch_file=$root/polar-unitary-projection-green.patch
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_patch_sha=262a6424d4128248bbedd65976a734105bb757d095f501e508bc1789688b226e

test ! -e "$evidence"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = \
  "$expected_patch_sha"
test "$(git -C "$source_dir" diff --name-only | wc -l)" -eq 3
test "$(git -C "$source_dir" diff --name-only | sed -n '1p')" = \
  "src/qsgw/fixed_basis.cpp"
test "$(git -C "$source_dir" diff --name-only | sed -n '2p')" = \
  "src/qsgw/fixed_basis.h"
test "$(git -C "$source_dir" diff --name-only | sed -n '3p')" = \
  "src/test/test_qsgw_fixed_basis.cpp"
git -C "$source_dir" diff --check

mkdir -p "$evidence"
cp "$0" "$evidence/run-fish-polar-tdd-green-262a6424.sh"
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
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_CPP_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_TEST=ON \
  -DLIBRPA_ENABLE_DRIVER=ON \
  >"$evidence/configure.stdout" \
  2>"$evidence/configure.stderr"
cmake --build "$build_dir" \
  --target test_qsgw_fixed_basis test_qsgw_fixed_basis_mpi \
  -j32 \
  >"$evidence/build.stdout" \
  2>"$evidence/build.stderr"

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

cat >"$evidence/PROVENANCE.txt" <<EOF
analysis=qsgw_velocity_polar_tdd_green
acceptance_gate=false
candidate_parent_commit=$expected_commit
patch_sha256=$expected_patch_sha
changed_files=src/qsgw/fixed_basis.cpp,src/qsgw/fixed_basis.h,src/test/test_qsgw_fixed_basis.cpp
protected_shared_numerical_files_changed=false
serial_fixed_basis_test=PASS
mpi4_fixed_basis_test=PASS
omp_threads=1
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
touch "$evidence/TDD_GREEN_CONFIRMED"
printf 'QSGW_VELOCITY_POLAR_TDD_GREEN=PASS\n'
