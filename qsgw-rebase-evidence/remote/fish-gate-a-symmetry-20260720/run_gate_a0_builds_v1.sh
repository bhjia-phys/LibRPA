#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-builds-20260720-v1
source_root=/tmp/librpa-qsgw-gate-a0-builds-20260720-v1
legacy_source=$source_root/legacy-e08f4a13
candidate_source=$source_root/candidate-b7273e13
legacy_build=$legacy_source/build
candidate_build=$candidate_source/build
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_candidate_tests=60
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

record_failure() {
  local rc=$?
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test -d "$bare_repo"
test -x "$pytest_env/bin/pytest"
test -x "$expected_cxx"
test -x "$expected_fortran"
test ! -e "$run_root"
test ! -e "$source_root"
mkdir -p "$run_root" "$source_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

git clone --no-checkout "$bare_repo" "$legacy_source"
git -C "$legacy_source" checkout --detach "$legacy_commit"
git clone --no-checkout "$bare_repo" "$candidate_source"
git -C "$candidate_source" checkout --detach "$candidate_commit"
test "$(git -C "$legacy_source" rev-parse HEAD)" = "$legacy_commit"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$candidate_commit"
test "$(git --git-dir="$bare_repo" rev-parse "$upstream_commit^{commit}")" = "$upstream_commit"
test -z "$(git -C "$legacy_source" status --porcelain)"
test -z "$(git -C "$candidate_source" status --porcelain)"

git -C "$legacy_source" status --short --branch >"$run_root/legacy-git-status.txt"
git -C "$candidate_source" status --short --branch >"$run_root/candidate-git-status.txt"
git -C "$candidate_source" diff --name-status "$upstream_commit" \
  >"$run_root/candidate-name-status.txt"
git -C "$candidate_source" diff --exit-code "$upstream_commit" -- \
  driver/tasks/g0w0.cpp driver/tasks/rpa.cpp src/core \
  >"$run_root/candidate-protected-diff.patch" \
  2>"$run_root/candidate-protected-diff.stderr"
test ! -s "$run_root/candidate-protected-diff.patch"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export PATH="$pytest_env/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'runner_sha256=%s\n' "$RUNNER_SHA256"
  printf 'legacy_commit=%s\n' "$legacy_commit"
  printf 'candidate_commit=%s\n' "$candidate_commit"
  printf 'upstream_commit=%s\n' "$upstream_commit"
  "$expected_cxx" --version | head -n 1
  "$expected_fortran" --version | head -n 1
  cmake --version | head -n 1
  mpirun --version | head -n 2
} >"$run_root/toolchain.txt"

cmake -S "$legacy_source" -B "$legacy_build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="$expected_cxx" \
  -DCMAKE_Fortran_COMPILER="$expected_fortran" \
  -DMPI_CXX_COMPILER="$expected_cxx" \
  -DMPI_Fortran_COMPILER="$expected_fortran" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG" \
  -DUSE_LIBRI=ON \
  -DUSE_CMAKE_INC=OFF \
  -DUSE_GREENX_API=ON \
  -DUSE_EXTERNAL_GREENX=OFF \
  -DENABLE_TEST=OFF \
  -DENABLE_DRIVER=ON \
  -DBUILD_LIBRPA_SHARED=ON \
  >"$run_root/legacy-configure.stdout" \
  2>"$run_root/legacy-configure.stderr"
cmake --build "$legacy_build" -j4 \
  >"$run_root/legacy-build.stdout" \
  2>"$run_root/legacy-build.stderr"

cmake -S "$candidate_source" -B "$candidate_build" \
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
  >"$run_root/candidate-configure.stdout" \
  2>"$run_root/candidate-configure.stderr"
cmake --build "$candidate_build" -j4 \
  >"$run_root/candidate-build.stdout" \
  2>"$run_root/candidate-build.stderr"
ctest --test-dir "$candidate_build" -N \
  >"$run_root/candidate-ctest-list.stdout" \
  2>"$run_root/candidate-ctest-list.stderr"
grep -Fq "Total Tests: $expected_candidate_tests" \
  "$run_root/candidate-ctest-list.stdout"
ctest --test-dir "$candidate_build" --output-on-failure -j4 \
  --output-junit "$run_root/candidate-ctest.xml" \
  >"$run_root/candidate-ctest.stdout" \
  2>"$run_root/candidate-ctest.stderr"
grep -Fq "100% tests passed, 0 tests failed out of $expected_candidate_tests" \
  "$run_root/candidate-ctest.stdout"

legacy_exe=$legacy_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
test -x "$legacy_exe"
test -x "$candidate_exe"
strings "$legacy_exe" >"$run_root/legacy-executable.strings"
grep -Fq 'QSGW_ORACLE_TRACE' "$run_root/legacy-executable.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
sha256sum "$candidate_exe" >"$run_root/candidate-executable.sha256"
LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$legacy_exe" >"$run_root/legacy-ldd.txt"
ldd "$candidate_exe" >"$run_root/candidate-ldd.txt"
cp "$legacy_build/CMakeCache.txt" "$run_root/legacy-CMakeCache.txt"
cp "$candidate_build/CMakeCache.txt" "$run_root/candidate-CMakeCache.txt"

legacy_exe_sha=$(awk '{print $1}' "$run_root/legacy-executable.sha256")
candidate_exe_sha=$(awk '{print $1}' "$run_root/candidate-executable.sha256")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_executable_freeze
acceptance=true
legacy_role=merge_before_qsgw_oracle
legacy_commit=$legacy_commit
legacy_source=$legacy_source
legacy_build=$legacy_build
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
legacy_tests=not_configured_oracle_build
candidate_role=merge_after_latest_upstream_candidate
candidate_commit=$candidate_commit
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_exe
candidate_executable_sha256=$candidate_exe_sha
candidate_tests_passed=$expected_candidate_tests
candidate_tests_failed=0
candidate_tests_not_run=0
upstream_commit=$upstream_commit
candidate_protected_diff=empty
cxx_compiler=$expected_cxx
fortran_compiler=$expected_fortran
build_type=RelWithDebInfo
cxx_flags=-O2_-g_-DNDEBUG
omp_num_threads_build=$OMP_NUM_THREADS
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
    ! -name FAILED -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
echo GATE_A0_BUILDS=PASS
cat "$run_root/PROVENANCE.txt"
exit 0
