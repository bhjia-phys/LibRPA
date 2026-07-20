#!/usr/bin/env bash

set -euo pipefail

source_root=/home/bhj/LibRPA-qsgw
build_root=$source_root/build
run_root=/home/bhj/ai-runs/librpa-qsgw-symmetry-k888-ctest-20260720-v2
pytest_env=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv
expected_head=76f5a144c26314b480f0eb69717d707831ebb115
expected_test_count=61

test ! -e "$run_root"
test -d "$source_root/.git"
test -d "$build_root"
test -x "$pytest_env/bin/pytest"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$source_root" status --porcelain --untracked-files=no)"

mkdir -p "$run_root"
cp "$0" "$run_root/run_fish_full_ctest_v2.sh"
git -C "$source_root" status --short --branch >"$run_root/git-status.txt"
git -C "$source_root" log -1 --format=fuller >"$run_root/git-head.txt"
git -C "$source_root" diff --stat >"$run_root/tracked-diff-stat.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" \
  2>"$run_root/oneapi-setvars.stderr"
set -u
export PATH="$pytest_env/bin:$PATH"

command -v mpirun >"$run_root/mpirun-path.txt"
command -v pytest >"$run_root/pytest-path.txt"
mpirun --version >"$run_root/mpirun-version.txt" 2>&1
pytest --version >"$run_root/pytest-version.txt" 2>&1
"$pytest_env/bin/python" --version \
  >"$run_root/python-version.txt" 2>&1
"$pytest_env/bin/python" -m pip freeze \
  >"$run_root/python-freeze.txt" 2>&1
cmake --version >"$run_root/cmake-version.txt"
ctest --version >"$run_root/ctest-version.txt"

cmake --build "$build_root" -j4 \
  >"$run_root/build.stdout" 2>"$run_root/build.stderr"
ctest --test-dir "$build_root" -N \
  >"$run_root/ctest-list.txt" 2>"$run_root/ctest-list.stderr"
grep -Fq "Total Tests: $expected_test_count" "$run_root/ctest-list.txt"
ctest --test-dir "$build_root" --output-on-failure -j4 \
  --output-junit "$run_root/ctest.xml" \
  >"$run_root/ctest.stdout" 2>"$run_root/ctest.stderr"
grep -Fq "100% tests passed, 0 tests failed out of $expected_test_count" \
  "$run_root/ctest.stdout"

executable_sha=$(sha256sum "$build_root/chi0_main.exe" | awk '{print $1}')
cmake_cache_sha=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
pytest_sha=$(sha256sum "$pytest_env/bin/pytest" | awk '{print $1}')
cat >"$run_root/PROVENANCE.txt" <<EOF
analysis=fish_full_ctest_after_ibz_operator_restore
acceptance_gate=false
host=Fisherd-Server100.96.1.64
source_root=$source_root
build_root=$build_root
source_commit=$expected_head
expected_test_count=$expected_test_count
passed_test_count=$expected_test_count
failed_test_count=0
oneapi_setvars=/opt/intel/oneapi/setvars.sh
pytest_environment=$pytest_env
pytest_sha256=$pytest_sha
executable_sha256=$executable_sha
cmake_cache_sha256=$cmake_cache_sha
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE -print0 |
    sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'FISH_FULL_CTEST=PASS\n'
