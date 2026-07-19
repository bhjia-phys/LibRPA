#!/bin/bash
#SBATCH -J qsgw-old-vh-v36
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
libri_evidence=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/provenance/legacy-hartree-libri-cvclr-f164e202-v1
root=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-oracle-v36-20260717
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_libri_archive_sha=bf27c1c332c572b7efaa204b5af54e701cf08797f648c57bff0b1bc20dcfc706
expected_libri_manifest_sha=985ca7f829b516f36c5aa86900cae1153bbd834a1abd10f989857e711c352e13
expected_libri_patch_sha=60f833dd92ec79c0d30d2786c1be14e97f45c8318d027a62c4997b084ec7100d
expected_libri_provenance_sha=dbf8a6dee2a9d3bfeab95de6518be829061942dc8c897bc801ae2004c86c9d8d
expected_libri_output_manifest_sha=eabc17f4857b55ca0629f5757b95911a0ea3ab7b7a55a91def66e652989e5b5f

test -d "$seed/src"
test -x "$seed/build/chi0_main.exe"
test "$(sha256sum "$seed/build/chi0_main.exe" | awk '{print $1}')" = "$expected_old_exe_sha"
test -f "$libri_evidence/COMPLETE"
test "$(sha256sum "$libri_evidence/libri-cvclr-source.tar.gz" | awk '{print $1}')" = \
  "$expected_libri_archive_sha"
test "$(sha256sum "$libri_evidence/source-SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_libri_manifest_sha"
test "$(sha256sum "$libri_evidence/libri-dirty.patch" | awk '{print $1}')" = \
  "$expected_libri_patch_sha"
test "$(sha256sum "$libri_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_libri_provenance_sha"
test "$(sha256sum "$libri_evidence/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_libri_output_manifest_sha"
test ! -e "$root"

mkdir -p "$root/src/thirdparty/LibRI" "$root/provenance"

# Reconstruct the old source with the complete historical cvclr LibRI tree.
(
  cd "$seed/src"
  tar --exclude='./.git' --exclude='./thirdparty/LibRI' -cf - .
) | tar -xf - -C "$root/src"
tar -xzf "$libri_evidence/libri-cvclr-source.tar.gz" \
  -C "$root/src/thirdparty/LibRI"

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

manifest_tree "$root/src/thirdparty/LibRI" \
  "$root/provenance/libri-build-copy-SHA256SUMS.txt"
cmp "$libri_evidence/source-SHA256SUMS.txt" \
  "$root/provenance/libri-build-copy-SHA256SUMS.txt"
cp "$libri_evidence/PROVENANCE.txt" \
  "$libri_evidence/base-commit.txt" \
  "$libri_evidence/git-status.txt" \
  "$libri_evidence/libri-dirty.patch" \
  "$libri_evidence/OUTPUT_SHA256SUMS.txt" \
  "$root/provenance/"

cmake_file="$root/src/src/CMakeLists.txt"
test "$(grep -Fc 'hartree_oracle_stub.cpp' "$cmake_file")" -eq 1
sed -i 's/hartree_oracle_stub\.cpp/hartree.cpp/' "$cmake_file"
test "$(grep -Fc 'hartree.cpp' "$cmake_file")" -eq 1
test "$(grep -Fc 'hartree_oracle_stub.cpp' "$cmake_file")" -eq 0
grep -Fq 'cal_cvcd_k_hartree' \
  "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp"

manifest_tree "$root/src" "$root/provenance/legacy-source-SHA256SUMS.txt"

{
  printf 'seed=%s\n' "$seed"
  printf 'root=%s\n' "$root"
  printf 'legacy_oracle_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01\n'
  printf 'legacy_librpa_parent=7a7ff17f8a1609c0b7763961ecec9f97ffeb87ef\n'
  printf 'legacy_libri_base=f164e202334fff2703998fc10c401d185bca268c\n'
  printf 'legacy_libri_archive_sha256=%s\n' "$expected_libri_archive_sha"
  printf 'hartree_build_mode=real_legacy_hartree_cpp_cvclr\n'
  sha256sum \
    "$seed/build/chi0_main.exe" \
    "$root/src/src/hartree.cpp" \
    "$root/src/src/hartree.h" \
    "$root/src/src/CMakeLists.txt" \
    "$root/src/driver/task_qsgw.cpp" \
    "$root/src/thirdparty/LibRI/include/RI/physics/Hartree.h" \
    "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" \
    "$root/provenance/libri-dirty.patch" \
    "$root/provenance/libri-build-copy-SHA256SUMS.txt" \
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

strings "$exe" "$libqsgw" >"$root/strings-exe-libqsgw.txt"
strings "$librpa" >"$root/strings-librpa.txt"
grep -Fq 'QSGW_ORACLE_UPDATE_HARTREE' "$root/strings-exe-libqsgw.txt"
if grep -Fq 'Hartree-off compatibility mode' "$root/strings-librpa.txt"; then
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
