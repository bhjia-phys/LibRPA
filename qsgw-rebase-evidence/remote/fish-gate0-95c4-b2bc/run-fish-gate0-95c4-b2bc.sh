#!/usr/bin/env bash
set -euo pipefail

run_root=/home/bhj/ai-runs/librpa-qsgw-gate0-20260716T0500-b2bc09d0
repo=/tmp/librpa-qsgw-cb294020-84e6b0a/repo.git
prior_gate=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c
comparator=/home/bhj/ai-runs/librpa-qsgw-gate0-stage-ce1e1859/compare_fish_g0w0.py
upstream_src=$run_root/source-upstream
candidate_src=$run_root/source-candidate
upstream_build=$run_root/build-upstream-oneapi
candidate_build=$run_root/build-candidate-oneapi
evidence=$run_root/evidence

expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_candidate=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
candidate_ref=refs/heads/codex/qsgw-independent-upstream-95c4-20260716
expected_comparator_sha=df08fb32df01398f0d7f1abf50d7d4ed35ba3c79c43dd5e0b5797b1211d17b00
expected_input_manifest_sha=e04c5712b1677fb01814abf9fc25c7b7e5905b63c0243a24ba5034bffd56189f
expected_official_output_sha=b3723d92576a5a6ece4ea7943226a1f719ed956e84d4e290c232c699fc2fa4ef

test ! -e "$run_root"
test -d "$repo"
test -d "$prior_gate/venv"
test -f "$comparator"
test "$(git --git-dir="$repo" rev-parse "$candidate_ref")" = "$expected_candidate"
git --git-dir="$repo" cat-file -e "$expected_upstream^{commit}"
test "$(sha256sum "$comparator" | awk '{print $1}')" = "$expected_comparator_sha"
test "$(sha256sum "$prior_gate/evidence/g0w0-si-input-files.sha256" | awk '{print $1}')" = \
  "$expected_input_manifest_sha"
test "$(sha256sum "$prior_gate/evidence/g0w0-official-reference.stdout" | awk '{print $1}')" = \
  "$expected_official_output_sha"

mkdir -p "$run_root" "$evidence"
git clone -q --no-checkout "$repo" "$upstream_src"
git -C "$upstream_src" checkout -q --detach "$expected_upstream"
git clone -q --no-checkout "$repo" "$candidate_src"
git -C "$candidate_src" checkout -q --detach "$expected_candidate"
test -z "$(git -C "$upstream_src" status --porcelain)"
test -z "$(git -C "$candidate_src" status --porcelain)"
git -C "$upstream_src" rev-parse HEAD >"$evidence/upstream-source-commit.txt"
git -C "$candidate_src" rev-parse HEAD >"$evidence/candidate-source-commit.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force >"$evidence/oneapi-setvars.log" 2>&1
set -u
export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx
export PATH="$prior_gate/venv/bin:$PATH"
export PYTHONPATH="$prior_gate/venv/lib/python3.13/site-packages"

configure_and_build() {
  local source=$1
  local build=$2
  local label=$3
  cmake -S "$source" -B "$build" \
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
    >"$evidence/$label-configure.log" 2>&1
  cmake --build "$build" --parallel 32 \
    >"$evidence/$label-build.log" 2>&1
  ctest --test-dir "$build" -N >"$evidence/$label-ctest-list.txt"
  ctest --test-dir "$build" --output-on-failure \
    --output-junit "$evidence/$label-ctest.xml" -j 4 \
    >"$evidence/$label-ctest-full.log" 2>&1
  test -x "$build/chi0_main.exe"
}

configure_and_build "$upstream_src" "$upstream_build" upstream
configure_and_build "$candidate_src" "$candidate_build" candidate

upstream_count=$(awk '/Total Tests:/ {print $3}' \
  "$evidence/upstream-ctest-list.txt")
candidate_count=$(awk '/Total Tests:/ {print $3}' \
  "$evidence/candidate-ctest-list.txt")
test -n "$upstream_count"
test -n "$candidate_count"
test "$((candidate_count - upstream_count))" -eq 21
test -z "$(git -C "$upstream_src" status --porcelain)"
test -z "$(git -C "$candidate_src" status --porcelain)"
git -C "$upstream_src" status --porcelain=v1 \
  >"$evidence/upstream-source-status-after.txt"
git -C "$candidate_src" status --porcelain=v1 \
  >"$evidence/candidate-source-status-after.txt"

mkdir -p "$run_root/g0w0/input"
ln -s "$prior_gate/g0w0-ab/input/dataset" \
  "$run_root/g0w0/input/dataset"
cp "$prior_gate/g0w0-ab/input/librpa.in" \
  "$run_root/g0w0/input/librpa.in"
(
  cd "$run_root/g0w0/input"
  find -L . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$evidence/g0w0-si-input-files.sha256"
cmp "$prior_gate/evidence/g0w0-si-input-files.sha256" \
  "$evidence/g0w0-si-input-files.sha256"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export I_MPI_FABRICS=shm
for side in upstream candidate; do
  run=$run_root/g0w0/$side/librpa
  mkdir -p "$run"
  ln -s ../input/dataset "$run_root/g0w0/$side/dataset"
  cp "$run_root/g0w0/input/librpa.in" "$run/librpa.in"
  if [ "$side" = upstream ]; then
    exe=$upstream_build/chi0_main.exe
  else
    exe=$candidate_build/chi0_main.exe
  fi
  (
    cd "$run"
    mpirun -np 4 "$exe" >runner.stdout 2>runner.stderr
  )
  test ! -s "$run/runner.stderr"
  grep -Fq 'libRPA finished successfully' "$run/runner.stdout"
done

/usr/bin/python3 -B "$comparator" \
  --source "$candidate_src" \
  --candidate "$run_root/g0w0/candidate/librpa/runner.stdout" \
  --upstream "$run_root/g0w0/upstream/librpa/runner.stdout" \
  --official "$prior_gate/evidence/g0w0-official-reference.stdout" \
  --output "$evidence/g0w0-comparison.json" \
  >"$evidence/g0w0-comparator.stdout" \
  2>"$evidence/g0w0-comparator.stderr"
grep -Fq '"passed": true' "$evidence/g0w0-comparison.json"

/usr/bin/python3 -B "$comparator" \
  --source "$upstream_src" \
  --candidate "$run_root/g0w0/upstream/librpa/runner.stdout" \
  --upstream "$prior_gate/evidence/g0w0-official-reference.stdout" \
  --official "$prior_gate/evidence/g0w0-official-reference.stdout" \
  --output "$evidence/upstream-vs-official-comparison.json" \
  >"$evidence/upstream-vs-official-comparator.stdout" \
  2>"$evidence/upstream-vs-official-comparator.stderr"
grep -Fq '"passed": true' \
  "$evidence/upstream-vs-official-comparison.json"

cp "$0" "$evidence/run-fish-gate0-95c4-b2bc.sh"
cp "$comparator" "$evidence/compare_fish_g0w0.py"
cp "$prior_gate/evidence/g0w0-official-reference.stdout" "$evidence/"
cp "$run_root/g0w0/upstream/librpa/runner.stdout" \
  "$evidence/g0w0-upstream.stdout"
cp "$run_root/g0w0/upstream/librpa/runner.stderr" \
  "$evidence/g0w0-upstream.stderr"
cp "$run_root/g0w0/candidate/librpa/runner.stdout" \
  "$evidence/g0w0-candidate.stdout"
cp "$run_root/g0w0/candidate/librpa/runner.stderr" \
  "$evidence/g0w0-candidate.stderr"

{
  printf 'gate=fish-latest-upstream-candidate-build-unit-g0w0\n'
  printf 'upstream_commit=%s\n' "$expected_upstream"
  printf 'candidate_commit=%s\n' "$expected_candidate"
  printf 'source_status=clean_detached_checkouts\n'
  printf 'upstream_ctest_count=%s\n' "$upstream_count"
  printf 'candidate_ctest_count=%s\n' "$candidate_count"
  printf 'compiler_environment=oneapi-2025.2.1\n'
  printf 'cxx_compiler=%s\n' "$(command -v mpiicpx)"
  printf 'fortran_compiler=%s\n' "$(command -v mpiifx)"
  printf 'cmake=%s\n' "$(cmake --version | head -n 1)"
  printf 'g0w0_case=g0w0_aims_Si_libri\n'
  printf 'g0w0_mpi_ranks=4\n'
  printf 'g0w0_omp_threads=1\n'
  printf 'g0w0_input_manifest_sha256=%s\n' \
    "$expected_input_manifest_sha"
  printf 'upstream_executable=%s\n' "$upstream_build/chi0_main.exe"
  printf 'upstream_executable_sha256=%s\n' \
    "$(sha256sum "$upstream_build/chi0_main.exe" | awk '{print $1}')"
  printf 'candidate_executable=%s\n' "$candidate_build/chi0_main.exe"
  printf 'candidate_executable_sha256=%s\n' \
    "$(sha256sum "$candidate_build/chi0_main.exe" | awk '{print $1}')"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$evidence/PROVENANCE.txt"

(
  cd "$run_root"
  sha256sum \
    evidence/PROVENANCE.txt \
    evidence/upstream-source-commit.txt \
    evidence/candidate-source-commit.txt \
    evidence/upstream-source-status-after.txt \
    evidence/candidate-source-status-after.txt \
    evidence/oneapi-setvars.log \
    evidence/upstream-configure.log \
    evidence/upstream-build.log \
    evidence/upstream-ctest-list.txt \
    evidence/upstream-ctest-full.log \
    evidence/upstream-ctest.xml \
    evidence/candidate-configure.log \
    evidence/candidate-build.log \
    evidence/candidate-ctest-list.txt \
    evidence/candidate-ctest-full.log \
    evidence/candidate-ctest.xml \
    evidence/g0w0-si-input-files.sha256 \
    evidence/g0w0-comparison.json \
    evidence/upstream-vs-official-comparison.json \
    evidence/g0w0-upstream.stdout \
    evidence/g0w0-upstream.stderr \
    evidence/g0w0-candidate.stdout \
    evidence/g0w0-candidate.stderr \
    evidence/g0w0-official-reference.stdout \
    evidence/run-fish-gate0-95c4-b2bc.sh \
    evidence/compare_fish_g0w0.py \
    build-upstream-oneapi/chi0_main.exe \
    build-candidate-oneapi/chi0_main.exe \
    >evidence/SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
printf 'FISH_LATEST_UPSTREAM_GATE0=PASS\n'
