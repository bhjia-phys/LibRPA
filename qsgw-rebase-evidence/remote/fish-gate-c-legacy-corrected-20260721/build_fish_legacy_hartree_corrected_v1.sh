#!/usr/bin/env bash
set -euo pipefail

: "${LEGACY_V36_ROOT:?LEGACY_V36_ROOT must identify the transferred v36 seed}"
: "${CANDIDATE_SOURCE:?CANDIDATE_SOURCE must identify the clean candidate source}"
: "${CANDIDATE_COMMIT:?CANDIDATE_COMMIT must identify the candidate commit}"
: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"
: "${RUN_TAG:?RUN_TAG must make the build directory immutable}"

expected_seed_exe_sha=97aca5a73f50c4d1f7e48018a2c7ac3b5fb712401ebb8ed79fcf23a0e6128c05
expected_seed_build_provenance_sha=c733c00c5da083436179d9bf64345effa0ef4398d6e4826ad073d9b45a9db4e0
expected_seed_source_provenance_sha=fb05da6aee4c6f0d8bb671523d015c44808e08b9cad6a9da9aa47df9eb0f6938
expected_seed_output_manifest_sha=e323a42a09db2b41ab70f77e18a0406043a7c397d79871792df9a317559f535c
expected_original_task_sha=6c79c6731d3420b3c9f70c97dea8c63236bca2952216bb67d9e6b9839552c9d1
expected_normalized_task_sha=50215d59480b46bea7e3edb34ac5e453165afcf0a66bc6a48053b17b491b303f
expected_patched_task_sha=54a5eae4f96b7a39e1400881ce96824f40b8e4ac026dc819b26e519db3e02ef9
expected_hartree_impl_sha=d734909b7a60d1d77235cb909c903584a2226e8b523cb5715a86f627cf7c3c93
expected_patch_sha=127b8c3927de328f095ce1be287fb492b32c7e6e3bd09fc23526252da56dd234

[[ "$CANDIDATE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]]
case "$RUN_TAG" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_TAG contains unsafe characters" >&2; exit 2 ;;
esac

runner_source=$CANDIDATE_SOURCE/qsgw-rebase-evidence/remote/fish-gate-c-legacy-corrected-20260721/build_fish_legacy_hartree_corrected_v1.sh
patch_source=$CANDIDATE_SOURCE/qsgw-rebase-evidence/remote/fish-gate-c-legacy-corrected-20260721/legacy-qsgw-hartree-isolated-full-reader-v1.patch
root=/home/bhj/ai-runs/librpa-qsgw-legacy-hartree-corrected-build-${RUN_TAG}

record_failure() {
  local rc=$?
  if [[ -d ${root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test ! -e "$root"
test -d "$LEGACY_V36_ROOT/src"
test -x "$LEGACY_V36_ROOT/build/chi0_main.exe"
test -e "$LEGACY_V36_ROOT/COMPLETE"
test ! -e "$LEGACY_V36_ROOT/FAILED"
test -f "$runner_source"
test -f "$patch_source"
test "$(sha256sum "$runner_source" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$patch_source" | awk '{print $1}')" = "$expected_patch_sha"
test "$(git -C "$CANDIDATE_SOURCE" rev-parse HEAD)" = "$CANDIDATE_COMMIT"
test -z "$(git -C "$CANDIDATE_SOURCE" status --porcelain)"

while read -r path expected; do
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ $observed != "$expected" ]]; then
    printf 'hash_mismatch path=%s expected=%s observed=%s\n' \
      "$path" "$expected" "$observed" >&2
    exit 1
  fi
done <<EOF
$LEGACY_V36_ROOT/build/chi0_main.exe $expected_seed_exe_sha
$LEGACY_V36_ROOT/build-provenance.txt $expected_seed_build_provenance_sha
$LEGACY_V36_ROOT/source-provenance.txt $expected_seed_source_provenance_sha
$LEGACY_V36_ROOT/OUTPUT_SHA256SUMS.txt $expected_seed_output_manifest_sha
$LEGACY_V36_ROOT/src/driver/task_qsgw.cpp $expected_original_task_sha
$LEGACY_V36_ROOT/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp $expected_hartree_impl_sha
EOF
(
  cd "$LEGACY_V36_ROOT"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$root/src" "$root/provenance"
(
  cd "$LEGACY_V36_ROOT/src"
  tar --exclude='./.git' -cf - .
) | tar -xf - -C "$root/src"
cp "$runner_source" "$root/provenance/"
cp "$patch_source" "$root/provenance/"
cp "$LEGACY_V36_ROOT/source-provenance.txt" \
  "$root/provenance/seed-source-provenance.txt"
cp "$LEGACY_V36_ROOT/build-provenance.txt" \
  "$root/provenance/seed-build-provenance.txt"
cp "$LEGACY_V36_ROOT/OUTPUT_SHA256SUMS.txt" \
  "$root/provenance/seed-OUTPUT_SHA256SUMS.txt"

dos2unix "$root/src/driver/task_qsgw.cpp" \
  >"$root/provenance/dos2unix.log" 2>&1
test "$(sha256sum "$root/src/driver/task_qsgw.cpp" | awk '{print $1}')" = \
  "$expected_normalized_task_sha"
(
  cd "$root/src"
  git apply --check "$patch_source"
  git apply "$patch_source"
)
test "$(sha256sum "$root/src/driver/task_qsgw.cpp" | awk '{print $1}')" = \
  "$expected_patched_task_sha"
test "$(sha256sum "$root/src/thirdparty/LibRI/include/RI/ri/LRI-cal_hartree.hpp" | awk '{print $1}')" = \
  "$expected_hartree_impl_sha"
test "$(grep -Fc 'read_Vq_row(driver_params.input_dir, "coulomb_cut_"' \
  "$root/src/driver/task_qsgw.cpp")" -eq 1
test "$(grep -Fc 'hartree_full_vq_cut, meanfield.get_n_kpoints(), Rlist,' \
  "$root/src/driver/task_qsgw.cpp")" -eq 1
test "$(grep -Fc '|| oracle_env_bool("QSGW_ORACLE_UPDATE_HARTREE", false))' \
  "$root/src/driver/task_qsgw.cpp")" -eq 0

(
  cd "$root/src"
  find . -path './.git' -prune -o -type f -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"$root/provenance/patched-source-SHA256SUMS.txt"

set +u
source /opt/intel/oneapi/setvars.sh --force \
  >"$root/oneapi-setvars.stdout" 2>"$root/oneapi-setvars.stderr"
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
  >"$root/configure.stdout" 2>"$root/configure.stderr"
cmake --build "$root/build" -j 16 \
  >"$root/build.stdout" 2>"$root/build.stderr"

exe=$root/build/chi0_main.exe
librpa=$root/build/src/librpa.so.0.3.0
libqsgw=$root/build/qsgw/libqsgw.so.0.3.0
test -x "$exe"
test -f "$librpa"
test -f "$libqsgw"
strings "$exe" "$libqsgw" >"$root/provenance/exe-libqsgw.strings"
grep -Fq 'QSGW_LEGACY_HARTREE_READER mode=isolated_full' \
  "$root/provenance/exe-libqsgw.strings"
if grep -Fq 'HARTREE_KEY_DIAGNOSTIC' "$root/provenance/exe-libqsgw.strings"; then
  echo "LibRI key diagnostic leaked into corrected legacy build" >&2
  exit 4
fi

mpiicpx --version >"$root/provenance/mpiicpx-version.txt"
ifx --version >"$root/provenance/ifx-version.txt"
cmake --version >"$root/provenance/cmake-version.txt"
ldd "$exe" >"$root/provenance/ldd.txt"

cat >"$root/PROVENANCE.txt" <<EOF
gate=fish_legacy_hartree_corrected_build_v1
acceptance=build_only_no_numerical_oracle_acceptance
runner_sha256=$RUNNER_SHA256
run_tag=$RUN_TAG
candidate_commit=$CANDIDATE_COMMIT
seed_root=$LEGACY_V36_ROOT
seed_executable_sha256=$expected_seed_exe_sha
seed_task_qsgw_sha256=$expected_original_task_sha
normalized_task_qsgw_sha256=$expected_normalized_task_sha
patch_sha256=$expected_patch_sha
patched_task_qsgw_sha256=$expected_patched_task_sha
reader_mode=isolated_full_hartree_map
gw_vq_reader=distributed_row_unchanged
reader_state_restored=Vq_cut,n_irk_points,irk_points,irk_weight
legacy_hartree_normalization=legacy_extra_inverse_nk
executable=$exe
executable_sha256=$(sha256sum "$exe" | awk '{print $1}')
librpa_sha256=$(sha256sum "$librpa" | awk '{print $1}')
libqsgw_sha256=$(sha256sum "$libqsgw" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt \
    ! -name GREEN_CONFIRMED ! -name FAILED -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/GREEN_CONFIRMED"
trap - ERR
echo "FISH_LEGACY_HARTREE_CORRECTED_BUILD_V1=PASS"
echo "root=$root"
