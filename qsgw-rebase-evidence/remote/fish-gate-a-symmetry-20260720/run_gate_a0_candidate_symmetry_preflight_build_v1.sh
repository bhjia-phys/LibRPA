#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

bare_repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
staging=/tmp/librpa-qsgw-symmetry-preflight-staging-20260720-v1
source_root=/tmp/librpa-qsgw-symmetry-preflight-b7273e13-20260720-v1
build_root=$source_root/build
run_root=/home/bhj/ai-runs/librpa-qsgw-gate-a0-candidate-symmetry-preflight-build-20260720-v1
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
driver_name=qsgw.cpp
test_name=test_qsgw_driver_wiring.py

expected_commit=b7273e13c77d5ea781f192cea3c4201710b6f9fa
upstream_base=42d3863c1d865194d382a085851d1e2e8a39764f
expected_base_driver_sha=ce3213a827e4a3dbb66a40945211954b0ec4dd8c96e329d142aa1de2e7b7caca
expected_modified_driver_sha=3c64a161c0a06d5fd5f3a1ae2a3f2d2173b347a057eb0a71d672e623602e099f
expected_test_sha=e03bd2be164e39c34d208f682fe21887ef7e1ceea1bb7174b7f0d7a48db247af
expected_test_count=60
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

test -d "$bare_repo"
test -d "$staging"
test ! -e "$source_root"
test ! -e "$run_root"
test -x "$pytest_env/bin/pytest"
test -x "$expected_cxx"
test -x "$expected_fortran"
test "$(sha256sum "$staging/$driver_name" | awk '{print $1}')" = \
  "$expected_modified_driver_sha"
test "$(sha256sum "$staging/$test_name" | awk '{print $1}')" = \
  "$expected_test_sha"

mkdir -p "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
git clone --no-checkout "$bare_repo" "$source_root" \
  >"$run_root/clone.stdout" 2>"$run_root/clone.stderr"
git -C "$source_root" checkout --detach "$expected_commit" \
  >"$run_root/checkout.stdout" 2>"$run_root/checkout.stderr"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"
test "$(sha256sum "$source_root/driver/tasks/$driver_name" | awk '{print $1}')" = \
  "$expected_base_driver_sha"
test ! -e "$source_root/regression_tests/backend/$test_name"

install -m 0644 "$staging/$driver_name" \
  "$source_root/driver/tasks/$driver_name"
install -m 0644 "$staging/$test_name" \
  "$source_root/regression_tests/backend/$test_name"
test "$(sha256sum "$source_root/driver/tasks/$driver_name" | awk '{print $1}')" = \
  "$expected_modified_driver_sha"
test "$(sha256sum "$source_root/regression_tests/backend/$test_name" | awk '{print $1}')" = \
  "$expected_test_sha"
test "$(git -C "$source_root" diff --name-only)" = \
  "driver/tasks/$driver_name"
test "$(git -C "$source_root" ls-files --others --exclude-standard)" = \
  "regression_tests/backend/$test_name"
git -C "$source_root" diff --check
git -C "$source_root" diff --exit-code -- \
  src/core driver/tasks/g0w0.cpp driver/tasks/rpa.cpp \
  >"$run_root/protected-working-tree-diff.patch"
test ! -s "$run_root/protected-working-tree-diff.patch"
git -C "$source_root" diff -- driver/tasks/qsgw.cpp \
  >"$run_root/qsgw-driver.patch"
git -C "$source_root" status --short --branch \
  >"$run_root/git-status.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export PATH="$pytest_env/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

(
  cd "$source_root"
  python3 -m pytest -p no:cacheprovider \
    "regression_tests/backend/$test_name" -q
) >"$run_root/wiring-test.stdout" 2>"$run_root/wiring-test.stderr"

cmake -S "$source_root" -B "$build_root" \
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
  >"$run_root/configure.stdout" 2>"$run_root/configure.stderr"

cmake --build "$build_root" -j4 \
  >"$run_root/build.stdout" 2>"$run_root/build.stderr"
ctest --test-dir "$build_root" -N \
  >"$run_root/ctest-list.stdout" 2>"$run_root/ctest-list.stderr"
grep -Fq "Total Tests: $expected_test_count" "$run_root/ctest-list.stdout"
ctest --test-dir "$build_root" --output-on-failure -j4 \
  --output-junit "$run_root/full-ctest.xml" \
  >"$run_root/full-ctest.stdout" 2>"$run_root/full-ctest.stderr"
grep -Fq "100% tests passed, 0 tests failed out of $expected_test_count" \
  "$run_root/full-ctest.stdout"
test -x "$build_root/chi0_main.exe"

executable_sha=$(sha256sum "$build_root/chi0_main.exe" | awk '{print $1}')
cache_sha=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_candidate_symmetry_preflight_build_v1
acceptance=true
claim=build_and_unit_tests_only_numerical_gate_pending
base_commit=$expected_commit
upstream_base=$upstream_base
source_root=$source_root
build_root=$build_root
base_driver_sha256=$expected_base_driver_sha
modified_driver_sha256=$expected_modified_driver_sha
wiring_test_sha256=$expected_test_sha
working_tree_scope=driver/tasks/qsgw.cpp
untracked_test_scope=regression_tests/backend/test_qsgw_driver_wiring.py
protected_working_tree_diff=empty
tests_passed=$expected_test_count
tests_failed=0
tests_not_run=0
cxx_compiler=$expected_cxx
fortran_compiler=$expected_fortran
executable_sha256=$executable_sha
cmake_cache_sha256=$cache_sha
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
cp "$build_root/CMakeCache.txt" "$run_root/CMakeCache.txt"
sha256sum "$build_root/chi0_main.exe" >"$run_root/executable.sha256"
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
touch "$run_root/COMPLETE"

echo GATE_A0_CANDIDATE_SYMMETRY_PREFLIGHT_BUILD_V1=PASS
cat "$run_root/PROVENANCE.txt"
