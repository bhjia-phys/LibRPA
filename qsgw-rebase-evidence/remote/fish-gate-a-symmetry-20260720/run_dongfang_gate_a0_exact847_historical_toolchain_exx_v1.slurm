#!/bin/bash
#SBATCH -J q847-exx-tool
#SBATCH -p 48cp3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH -o /data/home/df_iopcas_bhj/q847-exx-tool-%j.out
#SBATCH -e /data/home/df_iopcas_bhj/q847-exx-tool-%j.err

set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

home=/data/home/df_iopcas_bhj
run_root=$home/ai-runs/librpa-qsgw-gate-a0-exact847-historical-toolchain-exx-20260720-v1
scratch_root=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/librpa-qsgw-gate-a0-exact847-historical-toolchain-exx-20260720-v1-${SLURM_JOB_ID}
source_dir=$scratch_root/source
build_dir=$source_dir/build
work=$scratch_root/work
historical=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa

source_archive=$home/librpa-8476213f-source-20260720.tar.gz
libri_archive=$home/libri-cvclr-source-bf27c1c3.tar.gz
libcomm_archive=$home/libcomm-c46a34d7-source-20260720.tar.gz
bundle_manifest=$home/librpa-qsgw-gate-a-symmetry-band0-DATASET_SHA256SUMS-20260720.txt
source_verifier=$home/verify_historical_band0_source_against_bundle_v1.py
source_verifier_test=$home/test_verify_historical_band0_source_against_bundle_v1.py
analyzer=$home/analyze_legacy_band0_text_diff_v1.py
analyzer_test=$home/test_analyze_legacy_band0_text_diff_v1.py

expected_source_archive_sha=6fba457fe619386ff3702975b197dc77cedd21cf00cf5817a0dda510480b6804
expected_libri_archive_sha=bf27c1c332c572b7efaa204b5af54e701cf08797f648c57bff0b1bc20dcfc706
expected_libri_manifest_sha=985ca7f829b516f36c5aa86900cae1153bbd834a1abd10f989857e711c352e13
expected_libcomm_archive_sha=101ffdbed832e0409728ed06716e241ad043b724f2bd54d5e0ccbb475aa79e06
expected_bundle_manifest_sha=7fe17e43e20978833cd3c4bead17943c1ed9f231e5b03b3e46d9f342e9904956
expected_source_verifier_sha=25238d020dad3056840ce0290836580679ea27180777e80326dd50566eb44266
expected_source_verifier_test_sha=8d731a434e2044540c0ae070310a9e06dd36ea29e869d2f7a54c99a33b71ba71
expected_analyzer_sha=d50d699ccf54dce8b17288f0a5e30bbfe51cd68173c6412d4fe2013abe5f6a41
expected_analyzer_test_sha=448778bca260be5bb364ad2ba079453232bbf52dc77974bf132b01ed86ad7c8f
expected_read_data_sha=62b80bec72ed200dd961e44de46933d0a06934696e74a1e2950aa76ca5f30b17
expected_task_qsgw_sha=3a40b53f8d0393b07a82bf5fc1ef86b1022138660c72b72b9eadee87e32f0f8b
expected_gw_sha=c520c12ab2eb77289fcb6cf2763bf032990874e05345516000c6d4d6ecd056e3
expected_librpa_sha=ea39aa6ea1b295298156cbf83f721f411d29975feea35012dde7d93ac7ff329b
expected_oracle_exx_sha=a95d0a6b6a68483a483e499dc3060d127dfe12c50b04edfb9a43dad4795ce4b8
expected_oracle_qsgw_sha=835d4e35a346cd5cc1639fa1696ca3aa9283e877c2ff10c8ee309ed162a66059
expected_cxx=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiicpx
expected_fortran=/data/app/intel/oneapi-2024.2/compiler/2024.2/bin/ifx
stage=preflight

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${run_root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\nstage=%s\nslurm_job_id=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "${stage:-unknown}" \
      "${SLURM_JOB_ID:-unknown}" >"$run_root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$run_root"
test ! -e "$scratch_root"
test -d "$historical"
test -s "$historical/EXX_band_spin_1_1.dat"
test -s "$historical/QSGW_band_spin_1_1.dat"
test "$(sha256sum "$0" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$source_archive" | awk '{print $1}')" = "$expected_source_archive_sha"
test "$(sha256sum "$libri_archive" | awk '{print $1}')" = "$expected_libri_archive_sha"
test "$(sha256sum "$libcomm_archive" | awk '{print $1}')" = "$expected_libcomm_archive_sha"
test "$(sha256sum "$bundle_manifest" | awk '{print $1}')" = "$expected_bundle_manifest_sha"
test "$(sha256sum "$source_verifier" | awk '{print $1}')" = "$expected_source_verifier_sha"
test "$(sha256sum "$source_verifier_test" | awk '{print $1}')" = "$expected_source_verifier_test_sha"
test "$(sha256sum "$analyzer" | awk '{print $1}')" = "$expected_analyzer_sha"
test "$(sha256sum "$analyzer_test" | awk '{print $1}')" = "$expected_analyzer_test_sha"
test "$(sha256sum "$historical/EXX_band_spin_1_1.dat" | awk '{print $1}')" = "$expected_oracle_exx_sha"
test "$(sha256sum "$historical/QSGW_band_spin_1_1.dat" | awk '{print $1}')" = "$expected_oracle_qsgw_sha"
test -x "$expected_cxx"
test -x "$expected_fortran"

mkdir -p "$run_root" "$scratch_root" "$run_root/runtime" \
  "$run_root/output" "$run_root/tools"
printf '%s\n' "$RUNNER_SHA256" >"$run_root/runner-sha256.txt"
cp "$0" "$run_root/runner.slurm"
cp "$source_verifier" "$source_verifier_test" "$analyzer" "$analyzer_test" \
  "$run_root/tools/"

stage=tool_tests
(
  cd "$home"
  python3 -m unittest "$(basename "$source_verifier_test")" \
    "$(basename "$analyzer_test")"
) >"$run_root/tool-tests.stdout" 2>"$run_root/tool-tests.stderr"

stage=input_manifest
python3 "$source_verifier" "$bundle_manifest" "$historical" \
  "$run_root/historical-source-vs-bundle.json" --minimum-existing 1000 \
  >"$run_root/historical-source-vs-bundle.stdout" \
  2>"$run_root/historical-source-vs-bundle.stderr"

stage=source_prepare
mkdir -p "$source_dir"
tar -xzf "$source_archive" -C "$source_dir" --strip-components=1
test "$(sha256sum "$source_dir/driver/read_data.cpp" | awk '{print $1}')" = "$expected_read_data_sha"
test "$(sha256sum "$source_dir/driver/task_qsgw.cpp" | awk '{print $1}')" = "$expected_task_qsgw_sha"
test "$(sha256sum "$source_dir/src/gw.cpp" | awk '{print $1}')" = "$expected_gw_sha"
test "$(sha256sum "$source_dir/src/librpa.cpp" | awk '{print $1}')" = "$expected_librpa_sha"
mkdir -p "$source_dir/thirdparty/LibComm" "$source_dir/thirdparty/LibRI"
tar -xzf "$libcomm_archive" -C "$source_dir/thirdparty/LibComm"
tar -xzf "$libri_archive" -C "$source_dir/thirdparty/LibRI"
(
  cd "$source_dir/thirdparty/LibRI"
  find . -path './.git' -prune -o -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"$run_root/libri-SHA256SUMS.txt"
test "$(sha256sum "$run_root/libri-SHA256SUMS.txt" | awk '{print $1}')" = "$expected_libri_manifest_sha"
(
  cd "$source_dir"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) >"$run_root/source-SHA256SUMS.txt"

stage=build
set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.stdout" 2>"$run_root/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
{
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$expected_cxx" --version | head -n 1
  "$expected_fortran" --version | head -n 1
  cmake --version | head -n 1
  mpirun --version | head -n 2
} >"$run_root/toolchain.txt"

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE= \
  -DCMAKE_CXX_COMPILER="$expected_cxx" \
  -DCMAKE_Fortran_COMPILER="$expected_fortran" \
  -DCMAKE_CXX_FLAGS="-g -O2 -std=c++17 -DLIBRPA_VERBOSE -fiopenmp" \
  -DUSE_LIBRI=ON \
  -DUSE_CMAKE_INC=OFF \
  -DUSE_GREENX_API=ON \
  -DUSE_EXTERNAL_GREENX=OFF \
  -DENABLE_TEST=OFF \
  -DENABLE_DRIVER=ON \
  -DBUILD_LIBRPA_SHARED=ON \
  >"$run_root/configure.stdout" 2>"$run_root/configure.stderr"
cmake --build "$build_dir" -j8 \
  >"$run_root/build.stdout" 2>"$run_root/build.stderr"

exe=$build_dir/chi0_main.exe
test -x "$exe"
cp "$exe" "$run_root/runtime/chi0_main.exe"
find "$build_dir/src" "$build_dir/qsgw" -maxdepth 1 -type f \
  \( -name 'librpa.so*' -o -name 'libqsgw.so*' \) \
  -exec cp -L {} "$run_root/runtime/" \;
sha256sum "$run_root/runtime/"* >"$run_root/runtime-SHA256SUMS.txt"
LD_LIBRARY_PATH="$build_dir/src:$build_dir/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$exe" >"$run_root/runtime-ldd.txt"
cp "$build_dir/CMakeCache.txt" "$run_root/CMakeCache.txt"
strings "$exe" >"$run_root/runtime.strings"

stage=work_prepare
mkdir -p "$work"
while IFS= read -r name; do
  case "$name" in
    librpa.in|librpa.d|logs|librpa.stdout|librpa.stderr|librpa_para_*|\
    EXX_band_spin_*.dat|KS_band_spin_*.dat|QSGW_band_spin_*.dat|\
    homo_lumo_history*|hrs*_qsgw_iter_*.csr|qsgw_dump_old)
      continue
      ;;
  esac
  ln -s "$historical/$name" "$work/$name"
done < <(find "$historical" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)

# exx_band requires this legacy scalar reader before its band path. The values
# affect only the unused grid report; the band result reads the frozen matrices.
{
  printf '8 1 44\n'
  for _ in $(seq 1 $((8 * 44))); do
    printf '0.0 0.0\n'
  done
} >"$work/vxc_out"

cat >"$work/librpa.in" <<'EOF'
task = exx_band
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_shrink_abfs = t
use_abacus_exx_symmetry = t
use_abacus_gw_symmetry = t
use_fullcoul_exx = t
libri_exx_threshold_V = 1e-1
libri_exx_threshold_C = 1e-4
libri_exx_threshold_D = 1e-4
output_dir = ./
EOF

stage=librpa
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
export LD_LIBRARY_PATH="$build_dir/src:$build_dir/qsgw:${LD_LIBRARY_PATH:-}"
(
  cd "$work"
  timeout 3600 mpirun -np 1 "$exe" 16 1e-12 \
    >librpa.stdout 2>librpa.stderr
)
grep -Fq 'Task work begins: exx_band' "$work/librpa.stdout"
grep -Fq 'libRPA finished successfully' "$work/librpa.stdout"
! grep -Fq 'task failed' "$work/librpa.stdout"
test -s "$work/EXX_band_spin_1.dat"
test -s "$work/KS_band_spin_1.dat"

stage=compare
python3 "$analyzer" \
  "$historical/EXX_band_spin_1_1.dat" "$work/EXX_band_spin_1.dat" \
  "$historical/QSGW_band_spin_1_1.dat" "$historical/QSGW_band_spin_1_1.dat" \
  "$run_root/band-comparison.json" \
  >"$run_root/band-comparison.stdout" 2>"$run_root/band-comparison.stderr"
python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); e=r["exx"]; print("exx_energy_max_abs_diff_ev="+repr(e["energy_max_abs_diff_ev"])); print("exx_energy_relative_frobenius="+repr(e["energy_relative_frobenius"])); print("exx_max_location="+json.dumps(e["energy_max_location"],sort_keys=True))' \
  "$run_root/band-comparison.json" >"$run_root/diagnostic-summary.txt"
if python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r["exx"]["energy_max_abs_diff_ev"] <= 1.0000001e-5 else 1)' \
  "$run_root/band-comparison.json"; then
  touch "$run_root/HISTORICAL_EXX_REPRODUCED_TO_FILE_PRECISION"
fi

cp "$work/librpa.in" "$work/vxc_out" "$work/librpa.stdout" \
  "$work/librpa.stderr" "$work/EXX_band_spin_1.dat" \
  "$work/KS_band_spin_1.dat" "$run_root/output/"
find "$work" -maxdepth 1 -type f -name 'librpa_para_*' \
  -exec cp {} "$run_root/output/" \;

stage=freeze
input_map_summary=$(python3 -c \
  'import json,sys; r=json.load(open(sys.argv[1])); print(str(r["matching_entries"])+"/"+str(r["existing_entries"]))' \
  "$run_root/historical-source-vs-bundle.json")
cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate_a0_exact847_historical_toolchain_standalone_exx_v1
acceptance_scope=diagnostic_only
legacy_commit=8476213f66c68efb43404713eacbd04966820f26
source_patch=none
source_archive_sha256=$expected_source_archive_sha
libri_archive_sha256=$expected_libri_archive_sha
libri_manifest_sha256=$expected_libri_manifest_sha
libcomm_archive_sha256=$expected_libcomm_archive_sha
historical_input_root=$historical
frozen_bundle_manifest_sha256=$expected_bundle_manifest_sha
historical_existing_input_matches=$input_map_summary
oracle_exx_sha256=$expected_oracle_exx_sha
oracle_qsgw_sha256=$expected_oracle_qsgw_sha
task=exx_band
grid_vxc_placeholder=zero_hartree
grid_vxc_placeholder_scope=reader_gate_and_unused_grid_report_only
band_vxc_source=historical_frozen_band_matrix_files
cxx_compiler=$expected_cxx
fortran_compiler=$expected_fortran
cxx_flags=-g_-O2_-std=c++17_-DLIBRPA_VERBOSE_-fiopenmp
mpi_ranks=1
omp_threads=${SLURM_CPUS_PER_TASK:-32}
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=${SLURM_NODELIST:-unknown}
runner_sha256=$RUNNER_SHA256
scratch_root=$scratch_root
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
    ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
find "$run_root" -type d -exec chmod 0555 {} +
find "$run_root" -type f -exec chmod 0444 {} +

echo GATE_A0_EXACT847_HISTORICAL_TOOLCHAIN_EXX_V1=COMPLETE
cat "$run_root/diagnostic-summary.txt"
