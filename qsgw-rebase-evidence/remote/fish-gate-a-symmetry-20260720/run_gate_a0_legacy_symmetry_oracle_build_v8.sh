#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-symmetry-oracle-build-20260720-v8
base_evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-847reader-ibzocc-build-20260720-v7
base_source=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-847reader-ibzocc-20260720-v7
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-e08f4a13-symmetry-oracle-20260720-v8
legacy_build=$legacy_source/build
candidate_source=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
candidate_build=$candidate_source/build
candidate_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1
qlist_patch=/tmp/legacy_oracle_symmetry_qlist_restore_v1.patch

legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
historical_behavior_commit=8476213f66c68efb43404713eacbd04966820f26
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
base_patch_sha=d34bf83fb394f1f58c86b2c387b8e1fbba8a69bbbc783ebc770d1c2de69903ed
qlist_patch_sha=94d1c4503762d26218b25a58a863650eab77e41b2a4faa21b8afe4f669c4fa9a
base_provenance_sha=35e9aafe010609653a3f6fa289aabe7e3aa47d1a492d61412d4155300c2fbf2f
base_manifest_sha=315e44d2c5288d76c74619b459f0f3dc532fb9d096b951fdf3587d5fa5c08e43
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

test ! -e "$run_root"
test ! -e "$legacy_source"
test -e "$base_evidence/GREEN_CONFIRMED"
test ! -e "$base_evidence/FAILED"
test -d "$base_source"
test -f "$qlist_patch"
test "$(sha256sum "$qlist_patch" | awk '{print $1}')" = "$qlist_patch_sha"
test "$(sha256sum "$base_evidence/legacy-oracle-restore.patch" | awk '{print $1}')" = \
  "$base_patch_sha"
test "$(sha256sum "$base_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$base_provenance_sha"
test "$(sha256sum "$base_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$base_manifest_sha"
(
  cd "$base_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
test "$(git -C "$base_source" rev-parse HEAD)" = "$legacy_commit"
git -C "$base_source" diff --cached --exit-code
test "$(git -C "$base_source" diff --name-only -- . ':!thirdparty/LibRI')" = \
  $'driver/read_data.cpp\nqsgw/fermi_energy_occupation.cpp\nsrc/librpa.cpp'
test "$(git -C "$base_source" diff -- \
  driver/read_data.cpp qsgw/fermi_energy_occupation.cpp src/librpa.cpp | \
  sha256sum | awk '{print $1}')" = "$base_patch_sha"

test -e "$candidate_evidence/GREEN_CONFIRMED"
test ! -e "$candidate_evidence/FAILED"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$candidate_commit"
git -C "$candidate_source" diff --exit-code
git -C "$candidate_source" diff --cached --exit-code
test "$(sha256sum "$candidate_build/chi0_main.exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
grep -Fq '100% tests passed, 0 tests failed out of 60' \
  "$candidate_evidence/full-ctest.stdout"
test ! -s "$candidate_evidence/protected-diff.patch"

mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp -a "$base_source" "$legacy_source"
rm -rf "$legacy_build"
git -C "$legacy_source" apply --check "$qlist_patch"
git -C "$legacy_source" apply "$qlist_patch"
git -C "$legacy_source" diff --check
test "$(git -C "$legacy_source" diff --name-only -- . ':!thirdparty/LibRI')" = \
  $'driver/read_data.cpp\ndriver/task_qsgw.cpp\nqsgw/fermi_energy_occupation.cpp\nsrc/librpa.cpp'
git -C "$legacy_source" diff -- \
  driver/read_data.cpp driver/task_qsgw.cpp \
  qsgw/fermi_energy_occupation.cpp src/librpa.cpp \
  >"$run_root/legacy-symmetry-oracle.patch"
cp "$base_evidence/legacy-oracle-restore.patch" "$run_root/base-oracle-restore.patch"
cp "$qlist_patch" "$run_root/qlist-restore.patch"

grep -Fq 'vector<Vector3_Order<double>> qlist = klist;' \
  "$legacy_source/driver/task_qsgw.cpp"
grep -Fq 'atom_mu = atom_mu_s;' "$legacy_source/driver/read_data.cpp"
grep -Fq 'electrons += stored;' \
  "$legacy_source/qsgw/fermi_energy_occupation.cpp"
if sed -n '874,884p' "$legacy_source/driver/task_qsgw.cpp" | \
  grep -Fq 'for (auto q_weight : irk_weight)'; then
  echo 'legacy symmetry oracle still derives qlist from irk_weight coordinates' >&2
  exit 1
fi

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
test -x "$legacy_exe"
strings "$legacy_exe" >"$run_root/legacy-runtime.strings"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 strings >>"$run_root/legacy-runtime.strings"
grep -Fq 'QSGW_ORACLE_TRACE' "$run_root/legacy-runtime.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$run_root/legacy-runtime-libraries.sha256"
LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$legacy_exe" >"$run_root/legacy-ldd.txt"
cp "$legacy_build/CMakeCache.txt" "$run_root/legacy-CMakeCache.txt"

legacy_exe_sha=$(awk '{print $1}' "$run_root/legacy-executable.sha256")
combined_patch_sha=$(sha256sum "$run_root/legacy-symmetry-oracle.patch" | awk '{print $1}')
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_symmetry_oracle_build_v8
acceptance=true_oracle_harness_build
legacy_role=merge_before_qsgw_oracle_with_historical_reader_ibz_occupation_and_qlist_restore
legacy_commit=$legacy_commit
historical_behavior_commit=$historical_behavior_commit
base_oracle_patch_sha256=$base_patch_sha
qlist_patch_sha256=$qlist_patch_sha
combined_oracle_patch_sha256=$combined_patch_sha
legacy_patch_scope=driver/read_data.cpp_driver/task_qsgw.cpp_qsgw/fermi_energy_occupation.cpp_src/librpa.cpp_only
qlist_contract=historical_qsgw_band0_and_upstream_g0w0_klist
ibz_occupation_contract=stored_weights_already_include_geometric_kpoint_weight
legacy_source=$legacy_source
legacy_build=$legacy_build
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
candidate_commit=$candidate_commit
candidate_executable=$candidate_build/chi0_main.exe
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
echo GATE_A0_LEGACY_SYMMETRY_ORACLE_BUILD_V8=PASS
cat "$run_root/PROVENANCE.txt"
