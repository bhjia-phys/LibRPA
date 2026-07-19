#!/usr/bin/env bash
#SBATCH --job-name=qsgw-reader-head-build
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-reader-head-2e446af2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-reader-head-2e446af2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-2e446af2-linux-git-archive.tar.gz
patch_file=$root/jobs/diagnostic-gate4-preserve-reader-head-2e446af2.patch
source_parent=$root/src-diagnostic-reader-head-2e446af2-v3
src=$source_parent/source
build=$root/build-diagnostic-reader-head-2e446af2-v3

expected_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418
expected_archive_sha=433d971ab428d221ce122d295b784f5bd7530c4dbd9643ad417a36948beb82ce
expected_archive_entries=1838
expected_patch_sha=ac06b195fa4985b4059a80dcd3ebae17813450ef4cf465f6f4044f64e7aa50e7

test -f "$archive"
test -f "$patch_file"
test ! -e "$source_parent"
test ! -e "$build"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_archive_sha"
test "$(tar -tzf "$archive" | wc -l)" -eq "$expected_archive_entries"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = "$expected_patch_sha"
command -v patch >/dev/null

mkdir -p "$source_parent" "$build"
cp "$0" "$build/build-diagnostic-reader-head-2e446af2-v1.slurm"
cp "$patch_file" "$build/diagnostic-gate4-preserve-reader-head-2e446af2.patch"
tar -xzf "$archive" -C "$source_parent"
test -f "$src/CMakeLists.txt"
sha256sum "$src/driver/tasks/qsgw.cpp" >"$build/qsgw-before-patch.sha256"
(
  cd "$src"
  patch --binary --batch --forward -p1 <"$patch_file" \
    >"$build/patch.stdout" 2>"$build/patch.stderr"
)
sha256sum "$src/driver/tasks/qsgw.cpp" >"$build/qsgw-after-patch.sha256"
test "$(cut -d ' ' -f1 "$build/qsgw-before-patch.sha256")" != \
  "$(cut -d ' ' -f1 "$build/qsgw-after-patch.sha256")"

(
  cd "$source_parent"
  find source -type f -print0 | sort -z | xargs -0 sha256sum
) >"$build/source-files.sha256"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$build/oneapi-setvars.log" 2>&1
set -u
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
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_archive_sha"
test "$(sha256sum "$patch_file" | awk '{print $1}')" = "$expected_patch_sha"
(
  cd "$source_parent"
  sha256sum --check --quiet "$build/source-files.sha256"
)

{
  printf 'side=diagnostic-reader-head-ablation\n'
  printf 'acceptance_candidate=false\n'
  printf 'base_source_commit=%s\n' "$expected_commit"
  printf 'source_archive_sha256=%s\n' "$expected_archive_sha"
  printf 'diagnostic_patch_sha256=%s\n' "$expected_patch_sha"
  printf 'diagnostic_variable=preserve_reader_head_for_scf_grid_iteration_one\n'
  printf 'shared_gw_source_changed=false\n'
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
  "$build/build-diagnostic-reader-head-2e446af2-v1.slurm" \
  "$build/diagnostic-gate4-preserve-reader-head-2e446af2.patch" \
  "$build/qsgw-before-patch.sha256" \
  "$build/qsgw-after-patch.sha256" \
  "$build/source-files.sha256" \
  "$exe" \
  "$archive" \
  "$patch_file" \
  >"$build/SHA256SUMS.txt"
sha256sum --check --quiet "$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
