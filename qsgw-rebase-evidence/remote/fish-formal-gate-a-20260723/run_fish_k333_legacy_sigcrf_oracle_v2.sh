#!/usr/bin/env bash
set -euo pipefail

run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-old-sigcrf-20260723-v2
source_run=/home/bhj/ai-runs/qsgw-regression-aims-si-k333-headonly-old-task-qsgw-20260723-v1
old_build=/tmp/librpa-qsgw-formal-gate-a-20260723/legacy/old/build
old_exe=$old_build/chi0_main.exe
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788

test ! -e "$run"
test -x "$old_exe"
test "$(sha256sum "$old_exe" | awk '{print $1}')" = "$expected_old_exe_sha"
test -f "$source_run/work/librpa.in"
test -L "$source_run/dataset"

mkdir -p "$run/work/sigcrf"
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

ln -s "$(readlink -f "$source_run/dataset")" "$run/dataset"
cp "$source_run/work/librpa.in" "$run/work/librpa.in"
sed -i \
  -e 's|^output_dir = .*|output_dir = ./sigcrf/|' \
  -e 's/^max_iter = .*/max_iter = 1/' \
  "$run/work/librpa.in"
printf '%s\n' 'output_gw_sigc_mat_rf = true' >>"$run/work/librpa.in"

grep -Fqx 'task = qsgw' "$run/work/librpa.in"
grep -Fqx 'max_iter = 1' "$run/work/librpa.in"
grep -Fqx 'output_gw_sigc_mat_rf = true' "$run/work/librpa.in"

cat >"$run/PROVENANCE.txt" <<EOF
run=$run
purpose=legacy_qsgw_adapter_oracle_with_written_sigcrf
source_run=$source_run
dataset=$(readlink -f "$run/dataset")
legacy_binary=$old_exe
legacy_binary_sha256=$expected_old_exe_sha
legacy_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
mpi=1
omp=4
mkl=1
mixing_beta=1
hartree=off
head=on
wing=off
iterations=1
output_gw_sigc_mat_rf=true
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
export QSGW_ORACLE_UPDATE_HARTREE=0
export QSGW_ORACLE_TRACE=1
export LIBRPA_QSGW_MIXING_BETA=1
export LIBRPA_QSGW_MIXING_HISTORY=12
export LD_LIBRARY_PATH="$old_build/qsgw:$old_build/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$run/work"
date --iso-8601=seconds >"$run/STARTED"
set +e
/usr/bin/time -v timeout 10800 mpirun -np 1 "$old_exe" \
  >librpa.stdout 2>librpa.stderr
rc=$?
set -e
printf '%s\n' "$rc" >"$run/EXIT_CODE"
date --iso-8601=seconds >"$run/FINISHED"
test "$rc" -eq 0
grep -Fq 'libRPA finished successfully' librpa.stdout
test -s sigcrf/qsgw_oracle_matrices.dat
test "$(find sigcrf -maxdepth 1 -type f -name 'SigcRF*' | wc -l)" -eq 6

(
  cd "$run"
  find work/sigcrf -maxdepth 1 -type f -name 'SigcRF*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >SIGCRF_SHA256SUMS.txt
  sha256sum work/librpa.in work/sigcrf/qsgw_oracle_matrices.dat \
    >ORACLE_SHA256SUMS.txt
)
touch "$run/COMPLETE"
complete=1
trap - EXIT
