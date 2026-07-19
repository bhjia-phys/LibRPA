#!/usr/bin/env bash
#SBATCH --job-name=qsgw-reader-head-v5
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/resume-reader-head-v5-2e446af2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/resume-reader-head-v5-2e446af2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_parent=$root/src-diagnostic-reader-head-2e446af2-v4
src=$source_parent/source
source_evidence=$root/build-diagnostic-reader-head-2e446af2-v4
build=$root/build-diagnostic-reader-head-2e446af2-v5
patch_file=$root/jobs/diagnostic-gate4-preserve-reader-head-2e446af2.patch

expected_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418
expected_patch_sha=ac06b195fa4985b4059a80dcd3ebae17813450ef4cf465f6f4044f64e7aa50e7
expected_source_manifest_sha=a2230d8a1387b8abfad94ccba52ccee748911b6b446e4fa1131f2146a245886e
expected_patched_qsgw_sha=c6f7674a5ff7459ad45fb4f4dbbb90f6e2c2ea568d61a41b4bd4ad20f0e9756d

test -d "$src"
test -f "$source_evidence/source-files.sha256"
test ! -e "$build"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = "$expected_patch_sha"
test "$(sha256sum "$source_evidence/source-files.sha256" | awk '{print $1}')" = "$expected_source_manifest_sha"
test "$(sha256sum "$src/driver/tasks/qsgw.cpp" | awk '{print $1}')" = "$expected_patched_qsgw_sha"
test "$(find "$src" -type f \( -name '*.orig' -o -name '*.rej' \) | wc -l)" -eq 0
(
  cd "$source_parent"
  sha256sum --check --quiet "$source_evidence/source-files.sha256"
)

mkdir -p "$build"
cp "$0" "$build/resume-build-diagnostic-reader-head-2e446af2-v5.slurm"
cp "$patch_file" "$build/diagnostic-gate4-preserve-reader-head-2e446af2.patch"
cp "$source_evidence/source-files.sha256" "$build/source-files.sha256"

set +u
source /data/app/intel/oneapi-2024.2/compiler/2024.2/env/vars.sh \
  >"$build/compiler-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/mpi/2021.13/env/vars.sh \
  >"$build/mpi-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/mkl/2024.2/env/vars.sh \
  >"$build/mkl-vars.log" 2>&1
source /data/app/intel/oneapi-2024.2/tbb/2021.13/env/vars.sh \
  >"$build/tbb-vars.log" 2>&1
set -u
test -x /data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiicpx
test -x /data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiifx
export CC=mpiicx
export CXX=mpiicpx
export FC=mpiifx

cmake -S "$src" -B "$build" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiicpx \
  -DCMAKE_Fortran_COMPILER=/data/app/intel/oneapi-2024.2/mpi/2021.13/bin/mpiifx \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG" \
  -DLIBRPA_ENABLE_DRIVER=ON \
  -DLIBRPA_USE_LIBRI=ON \
  -DLIBRPA_ENABLE_TEST=OFF \
  -DLIBRPA_ENABLE_CPP_TEST=OFF \
  -DLIBRPA_ENABLE_FORTRAN_BIND=OFF \
  -DLIBRPA_ENABLE_FORTRAN_TEST=OFF \
  -DLIBRPA_USE_CMAKE_INC=OFF \
  >"$build/configure.log" 2>&1
cmake --build "$build" --parallel "${SLURM_CPUS_PER_TASK:-6}" \
  >"$build/build.log" 2>&1

exe=$build/chi0_main.exe
test -x "$exe"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = "$expected_patch_sha"
test "$(sha256sum "$src/driver/tasks/qsgw.cpp" | awk '{print $1}')" = "$expected_patched_qsgw_sha"
(
  cd "$source_parent"
  sha256sum --check --quiet "$source_evidence/source-files.sha256"
)

{
  printf 'side=diagnostic-reader-head-ablation\n'
  printf 'acceptance_candidate=false\n'
  printf 'base_source_commit=%s\n' "$expected_commit"
  printf 'diagnostic_patch_sha256=%s\n' "$expected_patch_sha"
  printf 'patched_qsgw_cpp_sha256=%s\n' "$expected_patched_qsgw_sha"
  printf 'diagnostic_variable=preserve_reader_head_for_scf_grid_iteration_one\n'
  printf 'shared_gw_source_changed=false\n'
  printf 'source_resume_from_failed_build_job=2347450\n'
  printf 'environment_initialization=component_vars_compiler_mpi_mkl_tbb\n'
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unknown}"
  printf 'build_type=RelWithDebInfo\n'
  printf 'compiler_environment=oneapi-2024.2\n'
  printf 'cxx_compiler=%s\n' "$(command -v mpiicpx)"
  printf 'fortran_compiler=%s\n' "$(command -v mpiifx)"
  printf 'executable=%s\n' "$exe"
  printf 'executable_sha256=%s\n' "$(sha256sum "$exe" | awk '{print $1}')"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$build/build-provenance.txt"

sha256sum \
  "$build/CMakeCache.txt" \
  "$build/configure.log" \
  "$build/build.log" \
  "$build/build-provenance.txt" \
  "$build/resume-build-diagnostic-reader-head-2e446af2-v5.slurm" \
  "$build/diagnostic-gate4-preserve-reader-head-2e446af2.patch" \
  "$build/compiler-vars.log" \
  "$build/mpi-vars.log" \
  "$build/mkl-vars.log" \
  "$build/tbb-vars.log" \
  "$build/source-files.sha256" \
  "$exe" \
  "$patch_file" \
  >"$build/SHA256SUMS.txt"
sha256sum --check --quiet "$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
