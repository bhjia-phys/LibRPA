#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-build-20260720-v12
base_evidence=/home/bhj/ai-runs/librpa-qsgw-gate-a0-legacy-exact847-build-20260720-v8
base_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-20260720-v8
legacy_source=/tmp/librpa-qsgw-gate-a0-legacy-exact847-band0-scheme-a-20260720-v12
legacy_build=$legacy_source/build
candidate_source=/tmp/librpa-qsgw-reader-binding-green-v2-b7273e13
candidate_build=$candidate_source/build
candidate_evidence=/home/bhj/ai-runs/librpa-qsgw-reader-binding-20260720-v2-postcheck-v1
staging=/tmp/librpa-qsgw-exact847-band0-oracle-occupation-v3
call_patch=$staging/legacy_exact847_band0_scheme_a_occupation_calls_v3.patch
corrected_fermi_cpp=$staging/fermi_energy_occupation.cpp
corrected_fermi_h=$staging/fermi_energy_occupation.h

legacy_commit=8476213f66c68efb43404713eacbd04966820f26
occupation_origin_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
candidate_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_commit=42d3863c1d865194d382a085851d1e2e8a39764f
expected_base_provenance_sha=4a9f54c6fc03445efefe96be60885ab43e1b03d34208950b41904d8171711c80
expected_base_output_sha=bbf6e45c2f1cf799ec9c36ac3b0a872041ac955842db76af2c752e8faf73d8fb
expected_base_source_manifest_sha=213e69d5391d1fb12e7ca37f7f7cbf9f41c46f7d2da14abf85459a37e0eecf60
expected_base_exe_sha=481ec33b3118747eb33ff3c252ab23fe23f7202c3cee7ee7147ac60b2e5cedaa
expected_base_task_sha=a932e60fa1b96e10a44eaa2f07da23bf57bbbb1c36600b1d4f9861a5d843e68e
expected_base_fermi_cpp_sha=01c7a1341c36cf47835ef60118943a94330582f4beb26d10142539bb1f2a99c8
expected_base_fermi_h_sha=476483f35950e3ca39335df8d19b70bff1c0e4ca2a373250e1775ae15005b31a
expected_call_patch_sha=dfb6f4060bb99b75b098596d376657e060e7d5fcd2b79f1fd9a6af3c7292b373
expected_corrected_fermi_cpp_sha=e4dcf3cb0998f312eaeab2e530c1cb306c0b9790608034784dad0ca3ea1437f3
expected_corrected_fermi_h_sha=6965348b51d698720ce8b3ef9bc27314e66c5fb82a82ef8b31d524dc0dc1f0d3
expected_modified_task_sha=34e5c93fe12259f4838469b0e19b2c2316d4b6871ba9d6c9da5b3c057a01ab34
expected_candidate_exe_sha=e45ca971c77d32309236a78e90ddd95aa0f37f3414befdb050ae3089eb9dc4c9
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx

record_failure() {
  local rc=$?
  trap - ERR
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
test -x "$base_source/build/chi0_main.exe"
test "$(sha256sum "$base_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_base_provenance_sha"
test "$(sha256sum "$base_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_base_output_sha"
test "$(sha256sum "$base_evidence/legacy-source-SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_base_source_manifest_sha"
test "$(sha256sum "$base_source/build/chi0_main.exe" | awk '{print $1}')" = \
  "$expected_base_exe_sha"
test "$(sha256sum "$base_source/driver/task_qsgw_band_0.cpp" | awk '{print $1}')" = \
  "$expected_base_task_sha"
test "$(sha256sum "$base_source/qsgw/fermi_energy_occupation.cpp" | awk '{print $1}')" = \
  "$expected_base_fermi_cpp_sha"
test "$(sha256sum "$base_source/qsgw/fermi_energy_occupation.h" | awk '{print $1}')" = \
  "$expected_base_fermi_h_sha"
(
  cd "$base_evidence"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$base_source"
  sha256sum --check --quiet "$base_evidence/legacy-source-SHA256SUMS.txt"
)

test -f "$call_patch"
test -f "$corrected_fermi_cpp"
test -f "$corrected_fermi_h"
test "$(sha256sum "$call_patch" | awk '{print $1}')" = "$expected_call_patch_sha"
test "$(sha256sum "$corrected_fermi_cpp" | awk '{print $1}')" = \
  "$expected_corrected_fermi_cpp_sha"
test "$(sha256sum "$corrected_fermi_h" | awk '{print $1}')" = \
  "$expected_corrected_fermi_h_sha"

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

mkdir -p "$run_root" "$legacy_source"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
(
  cd "$base_source"
  tar --exclude='./build' -cf - .
) | tar -xf - -C "$legacy_source"
(
  cd "$legacy_source"
  sha256sum --check --quiet "$base_evidence/legacy-source-SHA256SUMS.txt"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$run_root/source-before-SHA256SUMS.txt"
)
test "$(sha256sum "$run_root/source-before-SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_base_source_manifest_sha"

install -m 0644 "$corrected_fermi_cpp" \
  "$legacy_source/qsgw/fermi_energy_occupation.cpp"
install -m 0644 "$corrected_fermi_h" \
  "$legacy_source/qsgw/fermi_energy_occupation.h"
(
  cd "$legacy_source"
  git apply --check "$call_patch"
  git apply "$call_patch"
)
test "$(sha256sum "$legacy_source/driver/task_qsgw_band_0.cpp" | awk '{print $1}')" = \
  "$expected_modified_task_sha"
test "$(sha256sum "$legacy_source/qsgw/fermi_energy_occupation.cpp" | awk '{print $1}')" = \
  "$expected_corrected_fermi_cpp_sha"
test "$(sha256sum "$legacy_source/qsgw/fermi_energy_occupation.h" | awk '{print $1}')" = \
  "$expected_corrected_fermi_h_sha"
(
  cd "$legacy_source"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$run_root/source-after-SHA256SUMS.txt"
)
awk '{print $2}' "$run_root/source-before-SHA256SUMS.txt" \
  >"$run_root/source-before-paths.txt"
awk '{print $2}' "$run_root/source-after-SHA256SUMS.txt" \
  >"$run_root/source-after-paths.txt"
cmp -s "$run_root/source-before-paths.txt" "$run_root/source-after-paths.txt"
grep -Ev '  \./(driver/task_qsgw_band_0\.cpp|qsgw/fermi_energy_occupation\.(cpp|h))$' \
  "$run_root/source-before-SHA256SUMS.txt" >"$run_root/protected-before-SHA256SUMS.txt"
grep -Ev '  \./(driver/task_qsgw_band_0\.cpp|qsgw/fermi_energy_occupation\.(cpp|h))$' \
  "$run_root/source-after-SHA256SUMS.txt" >"$run_root/protected-after-SHA256SUMS.txt"
cmp -s "$run_root/protected-before-SHA256SUMS.txt" \
  "$run_root/protected-after-SHA256SUMS.txt"

{
  diff -u --label a/driver/task_qsgw_band_0.cpp \
    --label b/driver/task_qsgw_band_0.cpp \
    "$base_source/driver/task_qsgw_band_0.cpp" \
    "$legacy_source/driver/task_qsgw_band_0.cpp" || [[ $? -eq 1 ]]
  diff -u --label a/qsgw/fermi_energy_occupation.cpp \
    --label b/qsgw/fermi_energy_occupation.cpp \
    "$base_source/qsgw/fermi_energy_occupation.cpp" \
    "$legacy_source/qsgw/fermi_energy_occupation.cpp" || [[ $? -eq 1 ]]
  diff -u --label a/qsgw/fermi_energy_occupation.h \
    --label b/qsgw/fermi_energy_occupation.h \
    "$base_source/qsgw/fermi_energy_occupation.h" \
    "$legacy_source/qsgw/fermi_energy_occupation.h" || [[ $? -eq 1 ]]
} >"$run_root/legacy-corrected-oracle-harness.patch"
cp "$call_patch" "$run_root/qsgw-band0-occupation-calls.patch"
cp "$corrected_fermi_cpp" "$run_root/corrected-fermi-energy-occupation.cpp"
cp "$corrected_fermi_h" "$run_root/corrected-fermi-energy-occupation.h"
grep -Fq 'read_qsgw_kpoint_weights' \
  "$legacy_source/driver/task_qsgw_band_0.cpp"
test "$(grep -Fc 'update_qsgw_zero_temperature_occupations(' \
  "$legacy_source/driver/task_qsgw_band_0.cpp")" -eq 3
grep -Fq 'electrons += stored;' \
  "$legacy_source/qsgw/fermi_energy_occupation.cpp"

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
grep -Fq 'Legacy Scheme-A' "$run_root/legacy-runtime.strings"
grep -Fq 'QSGW band0 grid and band references have different electron counts' \
  "$run_root/legacy-runtime.strings"
grep -Fq 'QSGW band0: band reference electrons' \
  "$run_root/legacy-runtime.strings"
grep -Fq 'QSGW band0: grid updated electrons' \
  "$run_root/legacy-runtime.strings"
grep -Fq 'QSGW band0: band updated electrons' \
  "$run_root/legacy-runtime.strings"
sha256sum "$legacy_exe" >"$run_root/legacy-executable.sha256"
find "$legacy_build" -type f \( -name 'libqsgw.so*' -o -name 'librpa.so*' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$run_root/legacy-runtime-libraries.sha256"
LD_LIBRARY_PATH="$legacy_build/src:$legacy_build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$legacy_exe" >"$run_root/legacy-ldd.txt"
cp "$legacy_build/CMakeCache.txt" "$run_root/legacy-CMakeCache.txt"

legacy_exe_sha=$(awk '{print $1}' "$run_root/legacy-executable.sha256")
combined_patch_sha=$(sha256sum \
  "$run_root/legacy-corrected-oracle-harness.patch" | awk '{print $1}')
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_legacy_exact847_band0_scheme_a_build_v12
acceptance=true_corrected_multi_iteration_oracle_harness_build
legacy_role=raw_exact847_qsgw_band0_plus_historical_corrected_scheme_a_occupation_only
legacy_commit=$legacy_commit
raw_exact847_evidence=$base_evidence
raw_exact847_provenance_sha256=$expected_base_provenance_sha
raw_exact847_output_manifest_sha256=$expected_base_output_sha
raw_exact847_source_manifest_sha256=$expected_base_source_manifest_sha
raw_exact847_executable_sha256=$expected_base_exe_sha
occupation_origin_commit=$occupation_origin_commit
occupation_contract=global_degeneracy_aware_zero_temperature_ibz_stored_weight
band_occupation_serialization_tolerance=1e-8
occupation_cpp_sha256=$expected_corrected_fermi_cpp_sha
occupation_h_sha256=$expected_corrected_fermi_h_sha
qsgw_band0_call_patch_sha256=$expected_call_patch_sha
combined_oracle_harness_patch_sha256=$combined_patch_sha
legacy_patch_scope=driver/task_qsgw_band_0.cpp_qsgw/fermi_energy_occupation.cpp_qsgw/fermi_energy_occupation.h_only
protected_source_manifest=identical
raw_historical_binary=false
corrected_oracle_harness=true
legacy_source=$legacy_source
legacy_build=$legacy_build
legacy_executable=$legacy_exe
legacy_executable_sha256=$legacy_exe_sha
legacy_tests=build_only_pending_miniter2_runtime
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
echo GATE_A0_LEGACY_EXACT847_BAND0_SCHEME_A_BUILD_V12=PASS
cat "$run_root/PROVENANCE.txt"
