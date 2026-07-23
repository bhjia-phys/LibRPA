#!/usr/bin/env bash
set -euo pipefail

run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-current-read-old-sigcrf-20260723-v1
source_run=/home/bhj/ai-runs/librpa-qsgw-regression-aims-si-k333-headonly-20260723-v2
source_case=$source_run/workspace/testcases/qsgw_aims_Si_k333_headonly_libri
legacy_run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-old-sigcrf-20260723-v2
legacy_sigcrf=$legacy_run/work/sigcrf
current_source=/tmp/librpa-qsgw-head-band-23e03a8d
current_exe=$current_source/build/chi0_main.exe
expected_current_exe_sha=f0e2814c20846edf8b1703556f78fc2466155fbfcbabc4e3a745055e14b61b6a

test ! -e "$run"
test -x "$current_exe"
test "$(sha256sum "$current_exe" | awk '{print $1}')" = \
  "$expected_current_exe_sha"
test -f "$legacy_run/COMPLETE"
test "$(find "$legacy_sigcrf" -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 6
test -f "$source_case/librpa/librpa.in"
test -d "$source_case/dataset"

mkdir -p "$run/work/librpa.d"
complete=0
record_exit() {
  local rc=$?
  trap - EXIT
  if [[ $complete -ne 1 ]]; then
    printf 'exit_code=%s\nfailed_utc=%s\n' \
      "$rc" "$(date --iso-8601=seconds)" >"$run/FAILED"
  fi
  exit "$rc"
}
trap record_exit EXIT

ln -s "$source_case/dataset" "$run/dataset"
cp "$source_case/librpa/librpa.in" "$run/work/librpa.in"
sed -i \
  -e 's|^input_dir = .*|input_dir = ../dataset|' \
  -e 's/^n_params_anacon = .*/n_params_anacon = 6/' \
  -e 's/^qsgw_min_iter = .*/qsgw_min_iter = 1/' \
  -e 's/^qsgw_max_iter = .*/qsgw_max_iter = 1/' \
  "$run/work/librpa.in"
printf '%s\n' \
  'output_dir = ./librpa.d/' \
  "restart_from_dir = $legacy_sigcrf/" \
  'read_sigc_mat_rf = true' \
  >>"$run/work/librpa.in"

grep -Fqx 'task = qsgw' "$run/work/librpa.in"
grep -Fqx 'n_params_anacon = 6' "$run/work/librpa.in"
grep -Fqx 'qsgw_mixer = none' "$run/work/librpa.in"
grep -Fqx 'qsgw_max_iter = 1' "$run/work/librpa.in"
grep -Fqx 'qsgw_update_hartree = false' "$run/work/librpa.in"
grep -Fqx 'qsgw_iterative_headwing = false' "$run/work/librpa.in"
grep -Fqx 'read_sigc_mat_rf = true' "$run/work/librpa.in"

cat >"$run/PROVENANCE.txt" <<EOF
run=$run
purpose=current_qsgw_adapter_against_fixed_legacy_sigcrf
source_run=$source_run
dataset=$source_case/dataset
legacy_sigcrf=$legacy_sigcrf
legacy_sigcrf_checksums=$legacy_run/SIGCRF_SHA256SUMS.txt
current_source=$current_source
current_commit=d18d65964cd34920e5efedaa0b6fb3a501255429
current_binary=$current_exe
current_binary_sha256=$expected_current_exe_sha
mpi=1
omp=4
mkl=1
mixer=none
hartree=off
head_recompute=off
wing=off
iterations=1
read_sigc_mat_rf=true
EOF

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$run/oneapi-setvars.stdout" 2>"$run/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$current_source/build/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$run/work"
date --iso-8601=seconds >"$run/STARTED"
set +e
/usr/bin/time -v timeout 10800 mpirun -np 1 "$current_exe" \
  >librpa.stdout 2>librpa.stderr
rc=$?
set -e
printf '%s\n' "$rc" >"$run/EXIT_CODE"
date --iso-8601=seconds >"$run/FINISHED"
test "$rc" -eq 0
grep -Fq 'Finished reading real-space imaginary-frequency NAO sigma_c matrices.' \
  librpa.stdout
grep -Fq 'QSGW completed iterations: 1' librpa.stdout
grep -Fq 'libRPA finished successfully' librpa.stdout
test -s librpa.d/qsgw_matrices.dat
test -s librpa.d/qsgw_eigenvalues.dat
test -s librpa.d/qsgw_iterations.dat

(
  cd "$run"
  sha256sum \
    work/librpa.in \
    work/librpa.d/qsgw_matrices.dat \
    work/librpa.d/qsgw_eigenvalues.dat \
    work/librpa.d/qsgw_iterations.dat \
    >CURRENT_SHA256SUMS.txt
)
touch "$run/COMPLETE"
complete=1
trap - EXIT
