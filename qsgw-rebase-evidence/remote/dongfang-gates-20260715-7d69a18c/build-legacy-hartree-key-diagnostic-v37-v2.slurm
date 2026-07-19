#!/usr/bin/env bash
#SBATCH --job-name=qsgw-old-vh-v37d
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.err

set -euo pipefail

seed=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-oracle-v36-20260717
root=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-key-diagnostic-v37-20260717
patch_file=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/jobs/legacy-libri-hartree-key-diagnostic-v2.patch

expected_seed_exe_sha=97aca5a73f50c4d1f7e48018a2c7ac3b5fb712401ebb8ed79fcf23a0e6128c05
expected_seed_build_provenance_sha=c733c00c5da083436179d9bf64345effa0ef4398d6e4826ad073d9b45a9db4e0
expected_seed_source_provenance_sha=fb05da6aee4c6f0d8bb671523d015c44808e08b9cad6a9da9aa47df9eb0f6938
expected_seed_output_manifest_sha=e323a42a09db2b41ab70f77e18a0406043a7c397d79871792df9a317559f535c
expected_original_hartree_impl_sha=d734909b7a60d1d77235cb909c903584a2226e8b523cb5715a86f627cf7c3c93
expected_patch_sha=1b2a3698d249d8a0b456c270e2d3f4eb43ebd556ca2c4146e9fb69dd896eb22c

test -d "$seed/src"
test -x "$seed/build/chi0_main.exe"
test -f "$seed/COMPLETE"
test ! -e "$root"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$seed/build/chi0_main.exe $expected_seed_exe_sha
$seed/build-provenance.txt $expected_seed_build_provenance_sha
$seed/source-provenance.txt $expected_seed_source_provenance_sha
$seed/OUTPUT_SHA256SUMS.txt $expected_seed_output_manifest_sha
$seed/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp $expected_original_hartree_impl_sha
$patch_file $expected_patch_sha
EOF

mkdir -p "$root/src" "$root/provenance"
(
  cd "$seed/src"
  tar --exclude='./.git' -cf - .
) | tar -xf - -C "$root/src"

cp "$patch_file" "$root/provenance/"
cp "$seed/source-provenance.txt" "$root/provenance/seed-source-provenance.txt"
cp "$seed/OUTPUT_SHA256SUMS.txt" "$root/provenance/seed-OUTPUT_SHA256SUMS.txt"

sed -i 's/\r$//' \
  "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp"
(
  cd "$root/src"
  patch --batch --forward -p1 <"$patch_file"
)
grep -Fq 'HARTREE_KEY_DIAGNOSTIC preflight_pass' \
  "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp"
test "$(sha256sum "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" | awk '{print $1}')" \
  != "$expected_original_hartree_impl_sha"

manifest_tree() {
  local tree=$1
  local output=$2
  (
    cd "$tree"
    find . -path './.git' -prune -o -type f -print0 \
      | LC_ALL=C sort -z | xargs -0 sha256sum
  ) >"$output"
}

manifest_tree "$root/src" "$root/provenance/diagnostic-source-SHA256SUMS.txt"

{
  printf 'seed=%s\n' "$seed"
  printf 'root=%s\n' "$root"
  printf 'diagnostic_only=true\n'
  printf 'numerical_behavior_change=none_before_failed_access\n'
  printf 'diagnostic_scope=preflight_outer_keys_and_k_indices\n'
  printf 'patch_sha256=%s\n' "$expected_patch_sha"
  printf 'original_hartree_impl_sha256=%s\n' "$expected_original_hartree_impl_sha"
  sha256sum \
    "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" \
    "$root/provenance/legacy-libri-hartree-key-diagnostic-v2.patch" \
    "$root/provenance/diagnostic-source-SHA256SUMS.txt"
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

exe=$root/build/chi0_main.exe
librpa=$root/build/src/librpa.so.0.3.0
libqsgw=$root/build/qsgw/libqsgw.so.0.3.0
test -x "$exe"
test -f "$librpa"
test -f "$libqsgw"
strings "$libqsgw" | grep -Fq 'HARTREE_KEY_DIAGNOSTIC preflight_pass'

{
  sha256sum "$exe" "$librpa" "$libqsgw"
  mpiicpx --version | head -n 1
  ifx --version | head -n 1
  cmake --version | head -n 1
} >"$root/build-provenance.txt"

sha256sum \
  "$root/source-provenance.txt" \
  "$root/build-provenance.txt" \
  "$root/configure.log" \
  "$root/build.log" \
  >"$root/OUTPUT_SHA256SUMS.txt"
touch "$root/COMPLETE"
echo LEGACY_HARTREE_KEY_DIAGNOSTIC_BUILD=PASS
