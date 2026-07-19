#!/bin/bash
#SBATCH -J qsgw-old-vh-v35
#SBATCH -p 48cp3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -o /data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.out
#SBATCH -e /data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.err

set -euo pipefail

seed=/data/home/df_iopcas_bhj/qsgw-builds/old-new-a03a4c0-majorfix-v14-20260713/old
libri_git=/data/home/df_iopcas_bhj/software-stack/src/LibRPA-kouxiang/thirdparty/LibRI
libri_snapshot=/data/home/df_iopcas_bhj/tmp/librpa_kouxiang_hr_selfcheck_src_20260623/thirdparty/LibRI
root=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-oracle-v35-20260717
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_libri_base=f164e202334fff2703998fc10c401d185bca268c
expected_hartree_header_sha=9e5cb86ea3bf8cfc50f419cff32119c9fef9f56d8626bfd22dcbd4076aa44c86

test -d "$seed/src"
test -x "$seed/build/chi0_main.exe"
test "$(sha256sum "$seed/build/chi0_main.exe" | awk '{print $1}')" = "$expected_old_exe_sha"
test -e "$libri_git/.git"
test -d "$libri_snapshot/include/RI/physics"
test "$(git -C "$libri_git" rev-parse HEAD)" = "$expected_libri_base"
test "$(sha256sum "$libri_snapshot/include/RI/physics/Hartree.h" | awk '{print $1}')" = "$expected_hartree_header_sha"
test ! -e "$root"

mkdir -p "$root/src" "$root/provenance"

# Freeze the source contents without copying mutable Git metadata.
(
  cd "$seed/src"
  tar --exclude='./.git' --exclude='./thirdparty/LibRI' -cf - .
) | tar -xf - -C "$root/src"
mkdir -p "$root/src/thirdparty/LibRI"
(
  cd "$libri_snapshot"
  tar --exclude='./.git' -cf - .
) | tar -xf - -C "$root/src/thirdparty/LibRI"

manifest_tree() {
  local tree=$1
  local output=$2
  (
    cd "$tree"
    find . -path './.git' -prune -o -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) >"$output"
}

manifest_tree "$libri_git" "$root/provenance/libri-git-worktree-SHA256SUMS.txt"
manifest_tree "$libri_snapshot" "$root/provenance/libri-snapshot-SHA256SUMS.txt"
cmp "$root/provenance/libri-git-worktree-SHA256SUMS.txt" \
  "$root/provenance/libri-snapshot-SHA256SUMS.txt"
manifest_tree "$root/src/thirdparty/LibRI" \
  "$root/provenance/libri-build-copy-SHA256SUMS.txt"
cmp "$root/provenance/libri-snapshot-SHA256SUMS.txt" \
  "$root/provenance/libri-build-copy-SHA256SUMS.txt"

git -C "$libri_git" status --short >"$root/provenance/libri-git-status.txt"
git -C "$libri_git" diff --binary >"$root/provenance/libri-dirty.patch"
test -s "$root/provenance/libri-dirty.patch"
git -C "$libri_git" diff --quiet --cached
printf '%s\n' "$expected_libri_base" >"$root/provenance/libri-base-commit.txt"

cmake_file="$root/src/src/CMakeLists.txt"
test "$(grep -Fc 'hartree_oracle_stub.cpp' "$cmake_file")" -eq 1
sed -i 's/hartree_oracle_stub\.cpp/hartree.cpp/' "$cmake_file"
test "$(grep -Fc 'hartree.cpp' "$cmake_file")" -eq 1
test "$(grep -Fc 'hartree_oracle_stub.cpp' "$cmake_file")" -eq 0

manifest_tree "$root/src" "$root/provenance/legacy-source-SHA256SUMS.txt"

{
  printf 'seed=%s\n' "$seed"
  printf 'root=%s\n' "$root"
  printf 'legacy_oracle_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01\n'
  printf 'legacy_librpa_parent=7a7ff17f8a1609c0b7763961ecec9f97ffeb87ef\n'
  printf 'legacy_libri_base=%s\n' "$expected_libri_base"
  printf 'legacy_libri_snapshot=%s\n' "$libri_snapshot"
  printf 'hartree_build_mode=real_legacy_hartree_cpp\n'
  sha256sum \
    "$seed/build/chi0_main.exe" \
    "$root/src/src/hartree.cpp" \
    "$root/src/src/hartree.h" \
    "$root/src/src/CMakeLists.txt" \
    "$root/src/driver/task_qsgw.cpp" \
    "$root/src/thirdparty/LibRI/include/RI/physics/Hartree.h" \
    "$root/provenance/libri-dirty.patch" \
    "$root/provenance/libri-snapshot-SHA256SUMS.txt" \
    "$root/provenance/legacy-source-SHA256SUMS.txt"
} >"$root/source-provenance.txt"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$root/setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cmake -S "$root/src" -B "$root/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_Fortran_COMPILER=ifx \
  -DMPI_CXX_COMPILER=mpiicpx \
  -DMPI_Fortran_COMPILER=mpiifx \
  -DUSE_LIBRI=ON \
  -DUSE_CMAKE_INC=OFF \
  -DUSE_GREENX_API=ON \
  -DUSE_EXTERNAL_GREENX=OFF \
  -DENABLE_TEST=OFF \
  -DENABLE_DRIVER=ON \
  -DBUILD_LIBRPA_SHARED=ON \
  >"$root/configure.log" 2>&1

cmake --build "$root/build" -j 16 >"$root/build.log" 2>&1

exe="$root/build/chi0_main.exe"
librpa="$root/build/src/librpa.so.0.3.0"
libqsgw="$root/build/qsgw/libqsgw.so.0.3.0"
test -x "$exe"
test -f "$librpa"
test -f "$libqsgw"

strings "$exe" "$libqsgw" | grep -Fq 'QSGW_ORACLE_UPDATE_HARTREE'
if strings "$librpa" | grep -Fq 'Hartree-off compatibility mode'; then
  echo 'Stub Hartree diagnostic leaked into real legacy oracle' >&2
  exit 4
fi

ldd "$exe" >"$root/ldd-default.txt"
LD_LIBRARY_PATH="$root/build/src:$root/build/qsgw:${LD_LIBRARY_PATH:-}" \
  ldd "$exe" >"$root/ldd-local-runtime.txt"
readelf -d "$exe" >"$root/readelf-dynamic.txt"

{
  sha256sum "$exe" "$librpa" "$libqsgw"
  mpiicpx --version | head -n 1
  ifx --version | head -n 1
  cmake --version | head -n 1
} >"$root/build-provenance.txt"

sha256sum \
  "$root/source-provenance.txt" \
  "$root/build-provenance.txt" \
  "$root/ldd-default.txt" \
  "$root/ldd-local-runtime.txt" \
  "$root/readelf-dynamic.txt" \
  "$root/configure.log" \
  "$root/build.log" \
  >"$root/OUTPUT_SHA256SUMS.txt"

touch "$root/COMPLETE"
printf 'LEGACY_HARTREE_ORACLE_BUILD=PASS root=%s\n' "$root"
