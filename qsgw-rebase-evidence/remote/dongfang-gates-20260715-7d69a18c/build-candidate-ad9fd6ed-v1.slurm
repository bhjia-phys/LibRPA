#!/usr/bin/env bash
#SBATCH --job-name=librpa-ad9fd6ed
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-ad9fd6ed-v1-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-ad9fd6ed-v1-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-qsgw-ad9fd6ed.tar.gz
source_parent=$root/src-candidate-ad9fd6ed-v1
src=$source_parent
build=$root/build-candidate-ad9fd6ed-v1

expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_parent=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
expected_commit=ad9fd6ed65552b5ad2b3e80483aaf56538d15805
expected_archive_sha=29faf202859ecc044e381301ecce4040aa08e0d738f391a3d8c20292278bcdbe
fish_executable_sha=a18bcfa20c2e0c2bf5fe1154517c1a41411eea638b082c4981b8d814c8f3b958

test -f "$archive"
test ! -e "$source_parent"
test ! -e "$build"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"

mkdir -p "$source_parent" "$build"
cp "$0" "$build/build-candidate-ad9fd6ed-v1.slurm"
tar -xzf "$archive" -C "$source_parent"
test -f "$src/CMakeLists.txt"
test -f "$src/src/qsgw/hartree_dump.cpp"
test -f "$src/src/test/test_qsgw_hartree_dump.cpp"
grep -q "hartree_dump" "$src/src/qsgw/CMakeLists.txt"
(
  cd "$source_parent"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
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
cmake --build "$build" --parallel "${SLURM_CPUS_PER_TASK:-12}" \
  >"$build/build.log" 2>&1

exe=$build/chi0_main.exe
test -x "$exe"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
(
  cd "$source_parent"
  sha256sum --check --quiet "$build/source-files.sha256"
)

cat >"$build/build-provenance.txt" <<EOF
side=candidate-clean-commit
upstream_base=$expected_upstream
source_parent_commit=$expected_parent
source_commit=$expected_commit
source_status=clean_linux_git_archive_from_commit
source_archive=$archive
source_archive_sha256=$expected_archive_sha
fish_gate0_result=60_of_60_pass
fish_gate0_executable_sha256=$fish_executable_sha
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
build_type=RelWithDebInfo
compiler_environment=oneapi-2024.2
cxx_compiler=$(command -v mpiicpx)
fortran_compiler=$(command -v mpiifx)
cmake=$(cmake --version | head -n 1)
executable=$exe
executable_sha256=$(sha256sum "$exe" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

sha256sum \
  "$build/CMakeCache.txt" \
  "$build/configure.log" \
  "$build/build.log" \
  "$build/build-provenance.txt" \
  "$build/build-candidate-ad9fd6ed-v1.slurm" \
  "$build/source-files.sha256" \
  "$build/oneapi-setvars.log" \
  "$exe" \
  "$archive" \
  >"$build/SHA256SUMS.txt"
sha256sum --check --quiet "$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
