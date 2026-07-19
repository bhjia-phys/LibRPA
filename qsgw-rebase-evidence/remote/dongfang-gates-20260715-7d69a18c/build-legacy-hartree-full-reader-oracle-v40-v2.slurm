#!/usr/bin/env bash
#SBATCH --job-name=qsgw-old-vh-v40fr2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/qsgw-builds/slurm-%x-%j.err

set -euo pipefail

seed=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-oracle-v36-20260717
root=/data/home/df_iopcas_bhj/qsgw-builds/legacy-hartree-full-reader-oracle-v40-20260717
patch_file=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/jobs/legacy-qsgw-hartree-full-coulomb-reader-v1.patch

expected_seed_exe_sha=97aca5a73f50c4d1f7e48018a2c7ac3b5fb712401ebb8ed79fcf23a0e6128c05
expected_seed_build_provenance_sha=c733c00c5da083436179d9bf64345effa0ef4398d6e4826ad073d9b45a9db4e0
expected_seed_source_provenance_sha=fb05da6aee4c6f0d8bb671523d015c44808e08b9cad6a9da9aa47df9eb0f6938
expected_seed_output_manifest_sha=e323a42a09db2b41ab70f77e18a0406043a7c397d79871792df9a317559f535c
expected_original_task_qsgw_sha=6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1
expected_normalized_task_qsgw_sha=50215d59480b46bea7e3edb34ac5e453165afcf0a66bc6a48053b17b491b303f
expected_patched_task_qsgw_sha=2f408db3bda4930bbe3ae89eb5bcc6b00efb7c8f5b85c08e2aec70a063ff49f3
expected_original_hartree_impl_sha=d734909b7a60d1d77235cb909c903584a2226e8b523cb5715a86f627cf7c3c93
expected_patch_sha=9a5e57eeaeac389c84b377b930cab5b2538cb44b7c596030b6b8ca8bdb6c6c37

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
$seed/src/driver/task_qsgw.cpp $expected_original_task_qsgw_sha
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

/usr/bin/dos2unix \
  "$root/src/driver/task_qsgw.cpp" \
  >"$root/provenance/dos2unix.log" 2>&1
test "$(sha256sum "$root/src/driver/task_qsgw.cpp" | awk '{print $1}')" \
  = "$expected_normalized_task_qsgw_sha"
(
  cd "$root/src"
  patch --batch --forward -p1 <"$patch_file"
)
grep -Fq '|| oracle_env_bool("QSGW_ORACLE_UPDATE_HARTREE", false))' \
  "$root/src/driver/task_qsgw.cpp"
test "$(sha256sum "$root/src/driver/task_qsgw.cpp" | awk '{print $1}')" \
  = "$expected_patched_task_qsgw_sha"
test "$(sha256sum "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" | awk '{print $1}')" \
  = "$expected_original_hartree_impl_sha"

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
  printf 'legacy_oracle_adapter=true\n'
  printf 'adapter_scope=task_qsgw_cut_coulomb_reader_only_when_hartree_env_enabled\n'
  printf 'hartree_off_reader_behavior=unchanged\n'
  printf 'patch_sha256=%s\n' "$expected_patch_sha"
  printf 'original_task_qsgw_sha256=%s\n' "$expected_original_task_qsgw_sha"
  printf 'normalized_task_qsgw_sha256=%s\n' "$expected_normalized_task_qsgw_sha"
  printf 'patched_task_qsgw_sha256=%s\n' "$expected_patched_task_qsgw_sha"
  printf 'original_hartree_impl_sha256=%s\n' "$expected_original_hartree_impl_sha"
  sha256sum \
    "$root/src/driver/task_qsgw.cpp" \
    "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" \
    "$root/provenance/legacy-qsgw-hartree-full-coulomb-reader-v1.patch" \
    "$root/provenance/dos2unix.log" \
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

cmake --build "$root/build" -j 6 >"$root/build.log" 2>&1

exe=$root/build/chi0_main.exe
librpa=$root/build/src/librpa.so.0.3.0
libqsgw=$root/build/qsgw/libqsgw.so.0.3.0
test -x "$exe"
test -f "$librpa"
test -f "$libqsgw"
if grep -aFq 'HARTREE_KEY_DIAGNOSTIC' "$librpa"; then
  echo 'LibRI key diagnostic leaked into v40 oracle adapter' >&2
  exit 4
fi

mpiicpx --version >"$root/provenance/mpiicpx-version.txt"
ifx --version >"$root/provenance/ifx-version.txt"
cmake --version >"$root/provenance/cmake-version.txt"

{
  sha256sum "$exe" "$librpa" "$libqsgw"
  sed -n '1p' "$root/provenance/mpiicpx-version.txt"
  sed -n '1p' "$root/provenance/ifx-version.txt"
  sed -n '1p' "$root/provenance/cmake-version.txt"
} >"$root/build-provenance.txt"

sha256sum \
  "$root/source-provenance.txt" \
  "$root/build-provenance.txt" \
  "$root/configure.log" \
  "$root/build.log" \
  >"$root/OUTPUT_SHA256SUMS.txt"
touch "$root/COMPLETE"
echo LEGACY_HARTREE_FULL_READER_ORACLE_BUILD=PASS
