#!/usr/bin/env bash
set -euo pipefail

run_root=/home/bhj/ai-runs/librpa-qsgw-headwing-basis-tdd-green-v2-20260716-555f24a0
repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
source_root=$run_root/source
build_root=$run_root/build
evidence=$run_root/evidence
stage=/home/bhj/ai-runs/librpa-qsgw-headwing-basis-stage
expected_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
implementation_patch=$stage/headwing-basis-unitary-green.patch
expected_patch_sha=555f24a029ae77e0ecb45460a436e96d2c5a9ef15bfaf804e0a1842eafcfb3fa

test ! -e "$run_root"
test -d "$repo"
test "$(sha256sum "$implementation_patch" | awk '{print $1}')" = \
  "$expected_patch_sha"

mkdir -p "$run_root" "$evidence"
git clone -q --no-checkout "$repo" "$source_root"
git -C "$source_root" checkout -q --detach "$expected_commit"
test -z "$(git -C "$source_root" status --porcelain)"
git -C "$source_root" apply --check "$implementation_patch"
git -C "$source_root" apply "$implementation_patch"
git -C "$source_root" diff --check
git -C "$source_root" diff -- \
  src/qsgw/fixed_basis.cpp \
  src/qsgw/fixed_basis.h \
  src/test/test_qsgw_fixed_basis.cpp \
  src/test/test_qsgw_fixed_basis_mpi.cpp \
  >"$evidence/implementation.diff"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$evidence/oneapi-setvars.log" 2>&1
set -u
export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export I_MPI_FABRICS=shm

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
cmake --build "$build_root" \
  --target test_qsgw_fixed_basis test_qsgw_fixed_basis_mpi \
  --parallel 32 >"$evidence/build.log" 2>&1

"$build_root/src/test/test_qsgw_fixed_basis" \
  >"$evidence/test.stdout" 2>"$evidence/test.stderr"
test ! -s "$evidence/test.stderr"
grep -Fqx 'test_qsgw_fixed_basis: all tests passed' \
  "$evidence/test.stdout"
mpirun -np 4 "$build_root/src/test/test_qsgw_fixed_basis_mpi" \
  >"$evidence/mpi-test.stdout" 2>"$evidence/mpi-test.stderr"
test ! -s "$evidence/mpi-test.stderr"

cat >"$evidence/PROVENANCE.txt" <<EOF
gate=qsgw_headwing_complete_unitary_tdd_green
acceptance_gate=false
source_commit=$expected_commit
implementation_patch_sha256=$expected_patch_sha
fixed_basis_cpp_sha256=$(sha256sum "$source_root/src/qsgw/fixed_basis.cpp" | awk '{print $1}')
fixed_basis_h_sha256=$(sha256sum "$source_root/src/qsgw/fixed_basis.h" | awk '{print $1}')
test_source_sha256=$(sha256sum "$source_root/src/test/test_qsgw_fixed_basis.cpp" | awk '{print $1}')
mpi_test_source_sha256=$(sha256sum "$source_root/src/test/test_qsgw_fixed_basis_mpi.cpp" | awk '{print $1}')
serial_test=pass
mpi_test_ranks=4
mpi_test=pass
compiler_environment=oneapi-2025.2.1
host=$(hostname)
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
cp "$0" "$evidence/run-fish-headwing-basis-tdd-green.sh"
(
  cd "$run_root"
  find evidence -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >evidence/SHA256SUMS.txt
  sha256sum --check --quiet evidence/SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
printf 'QSGW_HEADWING_COMPLETE_UNITARY_TDD_GREEN=PASS\n'
