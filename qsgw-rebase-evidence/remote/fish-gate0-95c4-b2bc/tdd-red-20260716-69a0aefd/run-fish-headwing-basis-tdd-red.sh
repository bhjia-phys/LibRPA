#!/usr/bin/env bash
set -euo pipefail

run_root=/home/bhj/ai-runs/librpa-qsgw-headwing-basis-tdd-red-20260716-69a0aefd
repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
source_root=$run_root/source
build_root=$run_root/build
evidence=$run_root/evidence
test_source=/home/bhj/ai-runs/librpa-qsgw-headwing-basis-stage/test_qsgw_fixed_basis-red.cpp
expected_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
expected_test_sha=69a0aefdd1c69c577d2d2c83ace0748e6f7401461d07088c24c67387e8641556

test ! -e "$run_root"
test -d "$repo"
test "$(sha256sum "$test_source" | awk '{print $1}')" = \
  "$expected_test_sha"
mkdir -p "$run_root" "$evidence"
git clone -q --no-checkout "$repo" "$source_root"
git -C "$source_root" checkout -q --detach "$expected_commit"
test -z "$(git -C "$source_root" status --porcelain)"
cp "$test_source" "$source_root/src/test/test_qsgw_fixed_basis.cpp"
test -n "$(git -C "$source_root" status --porcelain -- src/test/test_qsgw_fixed_basis.cpp)"
git -C "$source_root" diff -- src/test/test_qsgw_fixed_basis.cpp \
  >"$evidence/test-red.diff"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$evidence/oneapi-setvars.log" 2>&1
set -u
export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx

cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx \
  -DCMAKE_Fortran_COMPILER=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG" \
  -DLIBRPA_ENABLE_DRIVER=ON \
  -DLIBRPA_USE_LIBRI=ON \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_CPP_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_BIND=OFF \
  -DLIBRPA_ENABLE_FORTRAN_TEST=ON \
  -DLIBRPA_USE_CMAKE_INC=OFF \
  -DENABLE_GREENX_CTEST=ON \
  >"$evidence/configure.log" 2>&1
cmake --build "$build_root" --target test_qsgw_fixed_basis --parallel 32 \
  >"$evidence/build.log" 2>&1

set +e
"$build_root/src/test/test_qsgw_fixed_basis" \
  >"$evidence/test.stdout" 2>"$evidence/test.stderr"
test_status=$?
set -e
test "$test_status" -ne 0
grep -Fq \
  'QSGW head/wing velocity WFC basis is not phase-equivalent to the fixed reference' \
  "$evidence/test.stderr"

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=qsgw_headwing_complete_unitary_tdd_red
acceptance_gate=false
source_commit=$expected_commit
test_source_sha256=$expected_test_sha
expected_failure=phase_only_alignment_rejects_complete_unitary_basis_rotation
observed_exit_code=$test_status
compiler_environment=oneapi-2025.2.1
host=$(hostname)
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
cp "$0" "$evidence/run-fish-headwing-basis-tdd-red.sh"
(
  cd "$run_root"
  find evidence -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >evidence/SHA256SUMS.txt
  sha256sum --check --quiet evidence/SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
printf 'QSGW_HEADWING_COMPLETE_UNITARY_TDD_RED=PASS\n'
