#!/usr/bin/env bash
set -euo pipefail

: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must be the exact clean candidate commit}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact LF-normalized runner}"
: "${RUN_TAG:?RUN_TAG must make the remote run directory immutable}"

bare_repo=${BARE_REPO:-/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git}
pytest_env=${PYTEST_ENV:-/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv}
docs_python=${DOCS_PYTHON:-/usr/bin/python3}
upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff
expected_upstream_tests=39
expected_candidate_tests=63
expected_cxx=/opt/intel/oneapi/mpi/2021.16/bin/mpiicpx
expected_fortran=/opt/intel/oneapi/mpi/2021.16/bin/mpiifx
source_root=/tmp/librpa-qsgw-gate0-${RUN_TAG}
upstream_source=$source_root/upstream
candidate_source=$source_root/candidate
upstream_build=$source_root/build-upstream
candidate_build=$source_root/build-candidate
run_root=/home/bhj/ai-runs/librpa-qsgw-gate0-${RUN_TAG}

case "$CANDIDATE_COMMIT" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) echo "CANDIDATE_COMMIT must be a 40-character lowercase Git hash" >&2; exit 2 ;;
esac
if [[ ! $RUNNER_SHA256 =~ ^[0-9a-f]{64}$ ]]; then
  echo "RUNNER_SHA256 must be a 64-character lowercase SHA-256" >&2
  exit 2
fi
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

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
test -x "$pytest_env/bin/python"
test -x "$docs_python"
"$docs_python" -c 'import yaml'
test -x "$expected_cxx"
test -x "$expected_fortran"
test ! -e "$source_root"
test ! -e "$run_root"
git --git-dir="$bare_repo" cat-file -e "$upstream_commit^{commit}"
git --git-dir="$bare_repo" cat-file -e "$CANDIDATE_COMMIT^{commit}"

mkdir -p "$source_root" "$run_root"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
git clone --no-checkout "$bare_repo" "$upstream_source" \
  >"$run_root/upstream-clone.stdout" 2>"$run_root/upstream-clone.stderr"
git clone --no-checkout "$bare_repo" "$candidate_source" \
  >"$run_root/candidate-clone.stdout" 2>"$run_root/candidate-clone.stderr"
git -C "$upstream_source" checkout --detach "$upstream_commit"
git -C "$candidate_source" checkout --detach "$CANDIDATE_COMMIT"
test "$(git -C "$upstream_source" rev-parse HEAD)" = "$upstream_commit"
test "$(git -C "$candidate_source" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$upstream_source" status --porcelain)"
test -z "$(git -C "$candidate_source" status --porcelain)"

git -C "$upstream_source" rev-parse HEAD HEAD^{tree} \
  >"$run_root/upstream-source.txt"
git -C "$candidate_source" rev-parse HEAD HEAD^{tree} \
  >"$run_root/candidate-source.txt"
git -C "$upstream_source" submodule status \
  >"$run_root/upstream-submodules.txt"
git -C "$candidate_source" submodule status \
  >"$run_root/candidate-submodules.txt"
git -C "$candidate_source" diff --name-status "$upstream_commit" \
  >"$run_root/candidate-name-status.txt"

set +eu
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
oneapi_rc=$?
set -eu
test "$oneapi_rc" -eq 0
export PATH="$pytest_env/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'os=%s\n' "$(uname -a)"
  "$expected_cxx" --version | head -n 1
  "$expected_fortran" --version | head -n 1
  cmake --version | head -n 1
  ctest --version | head -n 1
  mpirun --version | head -n 2
  "$pytest_env/bin/python" --version
  "$docs_python" --version
  "$docs_python" -c 'import yaml; print("PyYAML", yaml.__version__)'
} >"$run_root/toolchain.txt"

configure_build_test() {
  local label=$1
  local source=$2
  local build=$3
  local expected_count=$4

  cmake -S "$source" -B "$build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="$expected_cxx" \
    -DCMAKE_Fortran_COMPILER="$expected_fortran" \
    -DMPI_CXX_COMPILER="$expected_cxx" \
    -DMPI_Fortran_COMPILER="$expected_fortran" \
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG" \
    -DLIBRPA_ENABLE_DRIVER=ON \
    -DLIBRPA_USE_LIBRI=ON \
    -DLIBRPA_ENABLE_TEST=ON \
    -DLIBRPA_ENABLE_CPP_TEST=ON \
    -DLIBRPA_ENABLE_FORTRAN_BIND=OFF \
    -DLIBRPA_ENABLE_FORTRAN_TEST=ON \
    -DLIBRPA_USE_CMAKE_INC=OFF \
    -DENABLE_GREENX_CTEST=ON \
    >"$run_root/$label-configure.stdout" \
    2>"$run_root/$label-configure.stderr"
  cmake --build "$build" --parallel 4 \
    >"$run_root/$label-build.stdout" \
    2>"$run_root/$label-build.stderr"
  ctest --test-dir "$build" -N \
    >"$run_root/$label-ctest-list.stdout" \
    2>"$run_root/$label-ctest-list.stderr"
  ctest --test-dir "$build" --show-only=json-v1 \
    >"$run_root/$label-ctest-inventory.json" \
    2>"$run_root/$label-ctest-inventory.stderr"
  grep -Fq "Total Tests: $expected_count" \
    "$run_root/$label-ctest-list.stdout"
  ctest --test-dir "$build" --output-on-failure --parallel 4 \
    --output-junit "$run_root/$label-ctest.xml" \
    >"$run_root/$label-ctest.stdout" \
    2>"$run_root/$label-ctest.stderr"
  grep -Fq "100% tests passed, 0 tests failed out of $expected_count" \
    "$run_root/$label-ctest.stdout"
  test -x "$build/chi0_main.exe"
  cp "$build/CMakeCache.txt" "$run_root/$label-CMakeCache.txt"
  sha256sum "$build/chi0_main.exe" \
    >"$run_root/$label-executable.sha256"
  ldd "$build/chi0_main.exe" >"$run_root/$label-ldd.txt"
}

configure_build_test upstream "$upstream_source" "$upstream_build" \
  "$expected_upstream_tests"
configure_build_test candidate "$candidate_source" "$candidate_build" \
  "$expected_candidate_tests"

focus_regex='^(test_qsgw_inputfile|test_qsgw_abacus_csr|test_qsgw_band_output|test_qsgw_hamiltonian_cut|test_qsgw_hamiltonian_mixing|test_qsgw_hartree_dump|test_qsgw_hartree_workflow|test_qsgw_operator_fourier|test_qsgw_sha256|test_qsgw_vxc_io)$'
ctest --test-dir "$candidate_build" -R "$focus_regex" --output-on-failure \
  --output-junit "$run_root/candidate-focused-ctest.xml" \
  >"$run_root/candidate-focused-ctest.stdout" \
  2>"$run_root/candidate-focused-ctest.stderr"
grep -Fq '100% tests passed, 0 tests failed out of 10' \
  "$run_root/candidate-focused-ctest.stdout"

"$pytest_env/bin/python" -B -m pytest \
  "$candidate_source/regression_tests/backend/comparisons/test_cmp_qsgw.py" \
  "$candidate_source/regression_tests/backend/test_qsgw_driver_wiring.py" \
  -q -p no:cacheprovider --basetemp "$run_root/pytest-tmp" \
  >"$run_root/candidate-python-tests.stdout" \
  2>"$run_root/candidate-python-tests.stderr"

{
  "$docs_python" -B \
    "$candidate_source/docs/user_guide/generate_runtime_parameters.py" \
    --output "$run_root/candidate-runtime-parameters.md" \
    --check-defaults
  "$docs_python" -B \
    "$candidate_source/docs/user_guide/generate_runtime_parameters.py" \
    --output "$run_root/candidate-runtime-parameters.md" \
    --check --check-defaults
} >"$run_root/candidate-docs-check.stdout" \
  2>"$run_root/candidate-docs-check.stderr"

git -C "$candidate_source" diff --exit-code "$upstream_commit" -- \
  src/core/dielecmodel.cpp src/core/dielecmodel.h \
  src/core/gw.cpp src/core/gw.h \
  src/core/exx.cpp src/core/exx.h \
  src/api/compute_g0w0.cpp src/api/compute_exx.cpp \
  driver/tasks/g0w0.cpp driver/tasks/g0w0_band.cpp \
  >"$run_root/protected-diff.patch" \
  2>"$run_root/protected-diff.stderr"
test ! -s "$run_root/protected-diff.patch"
test -z "$(git -C "$upstream_source" status --porcelain)"
test -z "$(git -C "$candidate_source" status --porcelain)"

upstream_exe_sha=$(awk '{print $1}' "$run_root/upstream-executable.sha256")
candidate_exe_sha=$(awk '{print $1}' "$run_root/candidate-executable.sha256")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=fish_gate0_current_v2
acceptance=true
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
upstream_commit=$upstream_commit
candidate_commit=$CANDIDATE_COMMIT
upstream_test_count=$expected_upstream_tests
candidate_test_count=$expected_candidate_tests
candidate_focused_test_count=10
failed_test_count=0
not_run_test_count=0
protected_diff=empty
upstream_source=$upstream_source
upstream_build=$upstream_build
upstream_executable=$upstream_build/chi0_main.exe
upstream_executable_sha256=$upstream_exe_sha
candidate_source=$candidate_source
candidate_build=$candidate_build
candidate_executable=$candidate_build/chi0_main.exe
candidate_executable_sha256=$candidate_exe_sha
cxx_compiler=$expected_cxx
fortran_compiler=$expected_fortran
build_type=RelWithDebInfo
cxx_flags=-O2_-g_-DNDEBUG
omp_threads=1
python_qsgw_tests=pass
runtime_parameter_docs_check=pass
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/GREEN_CONFIRMED"
printf 'FISH_GATE0_CURRENT_V2=PASS\n'
cat "$run_root/PROVENANCE.txt"
