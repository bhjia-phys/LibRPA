#!/usr/bin/env bash
set -euo pipefail

gate0=/home/bhj/ai-runs/librpa-qsgw-gate0-20260716T0500-b2bc09d0
run_root=/home/bhj/ai-runs/librpa-qsgw-regression-bn-headwing-20260716T0515-b2bc09d0
source_root=$gate0/source-candidate
exe=$gate0/build-candidate-oneapi/chi0_main.exe
python=/home/bhj/ai-runs/librpa-qsgw-gate0-20260715T1731-7d69a18c/venv/bin/python3
case_name=g0w0_band_abacus_BN_headwing_sym_kpara_shrink_v1_libri
regression_root=$source_root/regression_tests
case_root=$regression_root/testcases/$case_name
reference_root=$regression_root/refs/$case_name

expected_candidate=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
expected_exe_sha=955586f0b653e5ff53c1553e08b1d24db6954b86cebec87f4350cd6c9dddc0d6
expected_gate0_manifest_sha=b94e92356bbfb6d7e4ed544de2d2d4b4b509bcc6fbfdd02f714011de9fb1d17d
expected_gate0_provenance_sha=57dedf7092e1e8c07cc97d7bd77652583d0a48a435b424d3668a9bb2d6681101

test ! -e "$run_root"
test -e "$gate0/evidence/COMPLETE"
test "$(sha256sum "$gate0/evidence/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate0_manifest_sha"
test "$(sha256sum "$gate0/evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate0_provenance_sha"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_candidate"
test -z "$(git -C "$source_root" status --porcelain)"
test -x "$exe"
test "$(sha256sum "$exe" | awk '{print $1}')" = "$expected_exe_sha"
test -x "$python"
test -f "$case_root/dataset.tar.gz"
test -f "$case_root/librpa/librpa.in"
test -f "$reference_root/librpa/librpa.out"
test -f "$reference_root/librpa/EXX_band_spin_1.dat"
test -f "$reference_root/librpa/GW_band_spin_1.dat"

mkdir -p "$run_root"
cp "$0" "$run_root/run-fish-bn-headwing-regression-b2bc.sh"
(
  cd "$regression_root"
  sha256sum \
    "testcases/$case_name/dataset.tar.gz" \
    "testcases/$case_name/librpa/librpa.in" \
    "refs/$case_name/librpa/librpa.out" \
    "refs/$case_name/librpa/EXX_band_spin_1.dat" \
    "refs/$case_name/librpa/GW_band_spin_1.dat" \
    >"$run_root/INPUT_REFERENCE_SHA256SUMS.txt"
)

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export I_MPI_FABRICS=shm

(
  cd "$regression_root"
  "$python" -B run_regression.py full "$exe" \
    --use-libri \
    -n 4 \
    --nthreads 1 \
    --mpiexec mpirun \
    --verbose \
    -d "$run_root/workspace" \
    -o "$run_root/regression.log" \
    --only "$case_name"
) >"$run_root/runner.stdout" 2>"$run_root/runner.stderr"

test ! -s "$run_root/runner.stderr"
grep -Fq '1 tests run, 1 PASSED, 0 FAILED' "$run_root/runner.stdout"
grep -Fq '1 tests run, 1 PASSED, 0 FAILED' "$run_root/regression.log"
test -f "$run_root/workspace/testcases/$case_name/librpa/librpa.out"
test ! -s "$run_root/workspace/testcases/$case_name/librpa/librpa.err"
test -f "$run_root/workspace/testcases/$case_name/librpa/EXX_band_spin_1.dat"
test -f "$run_root/workspace/testcases/$case_name/librpa/GW_band_spin_1.dat"
test -z "$(git -C "$source_root" status --porcelain)"

(
  cd "$run_root"
  find workspace -type f -print0 | sort -z | xargs -0 sha256sum \
    >WORKSPACE_SHA256SUMS.txt
)

{
  printf 'gate=fish-upstream-bn-symmetry-kpara-headwing-regression\n'
  printf 'candidate_commit=%s\n' "$expected_candidate"
  printf 'candidate_executable=%s\n' "$exe"
  printf 'candidate_executable_sha256=%s\n' "$expected_exe_sha"
  printf 'parent_gate0_manifest_sha256=%s\n' "$expected_gate0_manifest_sha"
  printf 'case=%s\n' "$case_name"
  printf 'mpi_ranks=4\n'
  printf 'omp_threads=1\n'
  printf 'mkl_threads=1\n'
  printf 'i_mpi_fabrics=shm\n'
  printf 'source_status=clean_detached_checkout\n'
  printf 'result=PASS\n'
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$run_root/PROVENANCE.txt"

(
  cd "$run_root"
  sha256sum \
    PROVENANCE.txt \
    INPUT_REFERENCE_SHA256SUMS.txt \
    WORKSPACE_SHA256SUMS.txt \
    regression.log \
    runner.stdout \
    runner.stderr \
    oneapi-setvars.log \
    run-fish-bn-headwing-regression-b2bc.sh \
    >OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
printf 'FISH_BN_HEADWING_REGRESSION=PASS\n'
