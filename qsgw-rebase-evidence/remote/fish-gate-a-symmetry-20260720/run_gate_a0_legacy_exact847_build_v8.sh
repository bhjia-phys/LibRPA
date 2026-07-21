#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-build-20260720-v8
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8
legacy_build=$legacy_source/build
candidate_source=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
candidate_build=$candidate_source/build
candidate_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1
source_archive=/tmp/librpa-8476213f-source-20260720.tar.gz
libcomm_seed=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-847reader-ibzocc-20260720-v7/thirdparty/LibComm
libri_archive=/tmp/libri-cvclr-source-bf27c1c3.tar.gz

legacy_commit=8476213f66c68efb43404713eacbd04966820f26
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_libcomm=c46a34d7b76d0f317ccd1718740f8169d8aa3fa4
historical_libri_gitlink=d4f732011c6c2d115f651c38796cf7ecf823393b
historical_libri_base=f164e202334fff2703998fc10c401d185bca268c
expected_source_archive_sha=6fba457fe619386ff3702975b197dc77cedd21cf00cf5817a0dda510480b6804
expected_libri_archive_sha=bf27c1c332c572b7efaa204b5af54e701cf08797f648c57bff0b1bc20dcfc706
expected_libri_manifest_sha=985ca7f829b516f36c5aa86900cae1153bbd834a1abd10f989857e711c352e13
expected_read_data_sha=62b80bec72ed200dd961e44de46933d0a06934696e74a1e2950aa76ca5f30b17
expected_task_qsgw_sha=3a40b53f8d0393b07a82bf5fc1ef86b1022138660c72b72b9eadee87e32f0f8b
expected_gw_sha=c520c12ab2eb77289fcb6cf2763bf032990874e05345516000c6d4d6ecd056e3
expected_librpa_sha=ea39aa6ea1b295298156cbf83f721f411d29975feea35012dde7d93ac7ff329b
expected_candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
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

test -f "$source_archive"
test -f "$libri_archive"
test -d "$libcomm_seed"
test "$(sha256sum "$source_archive" | awk '{print $1}')" = \
  "$expected_source_archive_sha"
test "$(sha256sum "$libri_archive" | awk '{print $1}')" = \
  "$expected_libri_archive_sha"
test "$(git -C "$libcomm_seed" rev-parse HEAD)" = "$expected_libcomm"
git -C "$libcomm_seed" diff --exit-code
git -C "$libcomm_seed" diff --cached --exit-code
test -e "$candidate_evidence/GREEN_CONFIRMED"
test -x "$candidate_build/chi0_main.exe"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$candidate_commit"
git -C "$candidate_source" diff --exit-code
git -C "$candidate_source" diff --cached --exit-code
test "$(sha256sum "$candidate_build/chi0_main.exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
grep -Fq '100% tests passed, 0 tests failed out of 60' \
  "$candidate_evidence/full-ctest.stdout"
test ! -s "$candidate_evidence/protected-diff.patch"
test ! -e "$run_root"
test ! -e "$legacy_source"
mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"

mkdir -p "$legacy_source"
tar -xzf "$source_archive" -C "$legacy_source" --strip-components=1
test "$(sha256sum "$legacy_source/driver/read_data.cpp" | awk '{print $1}')" = \
  "$expected_read_data_sha"
test "$(sha256sum "$legacy_source/driver/task_qsgw.cpp" | awk '{print $1}')" = \
  "$expected_task_qsgw_sha"
test "$(sha256sum "$legacy_source/src/gw.cpp" | awk '{print $1}')" = \
  "$expected_gw_sha"
test "$(sha256sum "$legacy_source/src/librpa.cpp" | awk '{print $1}')" = \
  "$expected_librpa_sha"

mkdir -p "$legacy_source/thirdparty/LibComm"
git -C "$libcomm_seed" archive --format=tar "$expected_libcomm" | \
  tar -xf - -C "$legacy_source/thirdparty/LibComm"
test -f "$legacy_source/thirdparty/LibComm/include/Comm/Comm_Tools.h"
mkdir -p "$legacy_source/thirdparty/LibRI"
tar -xzf "$libri_archive" -C "$legacy_source/thirdparty/LibRI"
test -f "$legacy_source/thirdparty/LibRI/include/RI/physics/Hartree.h"
test -f "$legacy_source/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp"

(
  cd "$legacy_source/thirdparty/LibRI"
  find . -path './.git' -prune -o -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"$run_root/legacy-libri-SHA256SUMS.txt"
test "$(sha256sum "$run_root/legacy-libri-SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_libri_manifest_sha"
sha256sum "$source_archive" >"$run_root/legacy-source-archive.sha256"
(
  cd "$legacy_source"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$run_root/legacy-source-SHA256SUMS.txt"
)

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
strings "$legacy_exe" >"$run_root/legacy-runtime.strings"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 strings >>"$run_root/legacy-runtime.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
sha256sum "$candidate_exe" >"$run_root/candidate-executable.sha256"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$run_root/legacy-runtime-libraries.sha256"
LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$legacy_exe" >"$run_root/legacy-ldd.txt"
cp "$legacy_build/CMakeCache.txt" "$run_root/legacy-CMakeCache.txt"
cp "$candidate_evidence/PROVENANCE.txt" "$run_root/candidate-PROVENANCE.txt"
cp "$candidate_evidence/protected-diff.patch" \
  "$run_root/candidate-protected-diff.patch"

legacy_exe_sha=$(awk '{print $1}' "$run_root/legacy-executable.sha256")
legacy_source_manifest_sha=$(sha256sum "$run_root/legacy-source-SHA256SUMS.txt" | awk '{print $1}')
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_exact847_executable_freeze_v8
acceptance=true
legacy_role=exact_committed_8476213_reproduction_without_observers
legacy_commit=$legacy_commit
legacy_source_archive=$source_archive
legacy_source_archive_sha256=$expected_source_archive_sha
legacy_source_manifest_sha256=$legacy_source_manifest_sha
legacy_source_patch=none
legacy_libcomm_commit=$expected_libcomm
legacy_libri_gitlink=$historical_libri_gitlink
legacy_libri_base=$historical_libri_base
legacy_libri_state=frozen_historical_cvclr_worktree
legacy_libri_archive_sha256=$expected_libri_archive_sha
legacy_libri_manifest_sha256=$expected_libri_manifest_sha
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
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name GREEN_CONFIRMED \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
echo GATE_A0_LEGACY_EXACT847_BUILD_V8=PASS
cat "$run_root/PROVENANCE.txt"
