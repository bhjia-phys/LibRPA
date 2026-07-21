#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-builds-20260720-v2
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-20260720-v2
legacy_build=$legacy_source/build
candidate_source=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
candidate_build=$candidate_source/build
candidate_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_libcomm=c46a34d7b76d0f317ccd1718740f8169d8aa3fa4
expected_libri=d4f732011c6c2d115f651c38796cf7ecf823393b
expected_candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
expected_candidate_cache_sha=c7065a59207b293e6c94aa4ac670d7a7be68384aba2f12ab49049413edb9b0bb
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
test -d "$candidate_source/.git"
test -d "$candidate_build"
test -e "$candidate_evidence/GREEN_CONFIRMED"
test -x "$candidate_build/chi0_main.exe"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$candidate_commit"
test -z "$(git -C "$candidate_source" status --porcelain)"
test "$(sha256sum "$candidate_build/chi0_main.exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/CMakeCache.txt" | awk '{print $1}')" = \
  "$expected_candidate_cache_sha"
grep -Fq '100% tests passed, 0 tests failed out of 60' \
  "$candidate_evidence/full-ctest.stdout"
test ! -s "$candidate_evidence/protected-diff.patch"
test ! -e "$run_root"
test ! -e "$legacy_source"
mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

git clone --no-checkout "$bare_repo" "$legacy_source"
git -C "$legacy_source" checkout --detach "$legacy_commit"
git -C "$legacy_source" submodule update --init --recursive \
  >"$run_root/legacy-submodule-update.stdout" \
  2>"$run_root/legacy-submodule-update.stderr"
test "$(git -C "$legacy_source" rev-parse HEAD)" = "$legacy_commit"
test "$(git -C "$legacy_source/thirdparty/LibComm" rev-parse HEAD)" = \
  "$expected_libcomm"
test "$(git -C "$legacy_source/thirdparty/LibRI" rev-parse HEAD)" = \
  "$expected_libri"
test -z "$(git -C "$legacy_source" status --porcelain)"
git -C "$legacy_source" status --short --branch >"$run_root/legacy-git-status.txt"
git -C "$legacy_source" submodule status --recursive \
  >"$run_root/legacy-submodule-status.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

legacy_exe=$legacy_build/chi0_main.exe
candidate_exe=$candidate_build/chi0_main.exe
test -x "$legacy_exe"
strings "$legacy_exe" >"$run_root/legacy-executable.strings"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 strings >>"$run_root/legacy-executable.strings"
grep -Fq 'QSGW_ORACLE_TRACE' "$run_root/legacy-executable.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
sha256sum "$candidate_exe" >"$run_root/candidate-executable.sha256"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 sha256sum >"$run_root/legacy-runtime-libraries.sha256"
LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$legacy_exe" >"$run_root/legacy-ldd.txt"
ldd "$candidate_exe" >"$run_root/candidate-ldd.txt"
cp "$legacy_build/CMakeCache.txt" "$run_root/legacy-CMakeCache.txt"
cp "$candidate_evidence/PROVENANCE.txt" "$run_root/candidate-PROVENANCE.txt"
cp "$candidate_evidence/OUTPUT_SHA256SUMS.txt" \
  "$run_root/candidate-OUTPUT_SHA256SUMS.txt"
cp "$candidate_evidence/CMakeCache.txt" "$run_root/candidate-CMakeCache.txt"
cp "$candidate_evidence/full-ctest.stdout" "$run_root/candidate-full-ctest.stdout"
cp "$candidate_evidence/full-ctest.xml" "$run_root/candidate-full-ctest.xml"
cp "$candidate_evidence/protected-diff.patch" \
  "$run_root/candidate-protected-diff.patch"

legacy_exe_sha=$(awk '{print $1}' "$run_root/legacy-executable.sha256")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_executable_freeze
acceptance=true
legacy_role=merge_before_qsgw_oracle
legacy_commit=$legacy_commit
legacy_libcomm_commit=$expected_libcomm
legacy_libri_commit=$expected_libri
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
candidate_executable_sha256=$expected_candidate_exe_sha
candidate_tests_passed=60
candidate_tests_failed=0
candidate_tests_not_run=0
upstream_commit=$upstream_commit
candidate_protected_diff=empty
cxx_compiler=$expected_cxx
fortran_compiler=$expected_fortran
build_type=RelWithDebInfo
cxx_flags=-O2_-g_-DNDEBUG
failed_attempt_v1=/home/bhj/ai-runs/librpa-qsgw-gate-a0-builds-20260720-v1
failed_attempt_v1_cause=legacy_submodules_not_initialized
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
