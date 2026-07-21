#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-847reader-build-20260720-v5
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-847reader-20260720-v5
legacy_build=$legacy_source/build
candidate_source=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
candidate_build=$candidate_source/build
candidate_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1
libri_archive=/tmp/libri-cvclr-source-bf27c1c3.tar.gz
legacy_reader_patch=/tmp/legacy_oracle_847_shrink_reader_restore.patch

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_libcomm=c46a34d7b76d0f317ccd1718740f8169d8aa3fa4
historical_libri_base=f164e202334fff2703998fc10c401d185bca268c
expected_libri_archive_sha=bf27c1c332c572b7efaa204b5af54e701cf08797f648c57bff0b1bc20dcfc706
expected_libri_manifest_sha=985ca7f829b516f36c5aa86900cae1153bbd834a1abd10f989857e711c352e13
expected_legacy_reader_patch_sha=dec81c5f0ebde6604f1e8e90af17b0d7ef5d0d2ec09c5a927506a4f9c07d1469
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
test -f "$libri_archive"
test -f "$legacy_reader_patch"
test "$(sha256sum "$libri_archive" | awk '{print $1}')" = \
  "$expected_libri_archive_sha"
test "$(sha256sum "$legacy_reader_patch" | awk '{print $1}')" = \
  "$expected_legacy_reader_patch_sha"
test -e "$candidate_evidence/GREEN_CONFIRMED"
test -x "$candidate_build/chi0_main.exe"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$candidate_commit"
git -C "$candidate_source" diff --exit-code
git -C "$candidate_source" diff --cached --exit-code
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
git -C "$legacy_source" submodule update --init thirdparty/LibComm \
  >"$run_root/legacy-libcomm-update.stdout" \
  2>"$run_root/legacy-libcomm-update.stderr"
test "$(git -C "$legacy_source/thirdparty/LibComm" rev-parse HEAD)" = \
  "$expected_libcomm"
test ! -e "$legacy_source/thirdparty/LibRI/include"
mkdir -p "$legacy_source/thirdparty/LibRI"
tar -xzf "$libri_archive" -C "$legacy_source/thirdparty/LibRI"
test -f "$legacy_source/thirdparty/LibRI/include/RI/physics/Hartree.h"
test -f "$legacy_source/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp"

git -C "$legacy_source" apply --check "$legacy_reader_patch"
git -C "$legacy_source" apply "$legacy_reader_patch"
git -C "$legacy_source" diff --check
test "$(git -C "$legacy_source" diff --name-only -- . ':!thirdparty/LibRI')" = \
  $'driver/read_data.cpp\nsrc/librpa.cpp'
git -C "$legacy_source" diff -- driver/read_data.cpp src/librpa.cpp \
  >"$run_root/legacy-847-shrink-reader-restore.patch"
test -s "$run_root/legacy-847-shrink-reader-restore.patch"
grep -Fq 'atom_mu = atom_mu_s;' "$legacy_source/driver/read_data.cpp"
grep -Fq 'else if (keyword == "Cs_shrinked_data")' \
  "$legacy_source/src/librpa.cpp"
grep -Fq 'atom_mu_s.insert(pair<atom_t, size_t>(I, naux_mu));' \
  "$legacy_source/src/librpa.cpp"
if grep -Fq 'std::vector<size_t> shrinked_mu;' \
  "$legacy_source/driver/read_data.cpp"; then
  echo 'legacy shrink reader restore did not remove shard-local inference' >&2
  exit 1
fi
cp "$legacy_reader_patch" "$run_root/legacy-reader-input.patch"

(
  cd "$legacy_source/thirdparty/LibRI"
  find . -path './.git' -prune -o -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"$run_root/legacy-libri-SHA256SUMS.txt"
test "$(sha256sum "$run_root/legacy-libri-SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_libri_manifest_sha"
git -C "$legacy_source" diff --cached --exit-code
git -C "$legacy_source" status --short --branch \
  >"$run_root/legacy-git-status.txt"
git -C "$legacy_source" submodule status \
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
strings "$legacy_exe" >"$run_root/legacy-runtime.strings"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 strings >>"$run_root/legacy-runtime.strings"
grep -Fq 'QSGW_ORACLE_TRACE' "$run_root/legacy-runtime.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
sha256sum "$candidate_exe" >"$run_root/candidate-executable.sha256"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$run_root/legacy-runtime-libraries.sha256"
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
legacy_role=merge_before_qsgw_oracle_with_historical_847_shrink_reader_restore
legacy_commit=$legacy_commit
legacy_reader_behavior_source=8476213f66c68efb43404713eacbd04966820f26
legacy_reader_historical_run=si-k444-qsgw-old-exactinput-wcfq-20260704-014854
legacy_reader_patch_sha256=$expected_legacy_reader_patch_sha
legacy_reader_patch_scope=driver/read_data.cpp_and_src/librpa.cpp_only_oracle_build
legacy_libcomm_commit=$expected_libcomm
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
failed_attempt_v1=missing_gitlink_submodules
failed_attempt_v2=gitlink_libri_lacks_historical_hartree_api
failed_attempt_a1_v1=input_dir_missing_trailing_slash
failed_attempt_a1_v2=e08_shard_local_shrink_dimension_inference_sigfpe
failed_attempt_smoke_v4=reader_restore_missing_atom_mu_s_population_sigsegv
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
echo GATE_A0_LEGACY_847READER_BUILD_V5=PASS
cat "$run_root/PROVENANCE.txt"
exit 0
