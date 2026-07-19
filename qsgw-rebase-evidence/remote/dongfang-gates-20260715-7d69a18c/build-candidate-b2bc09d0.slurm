#!/usr/bin/env bash
#SBATCH --job-name=librpa-b2bc09d0-build
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-b2bc09d0-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-b2bc09d0-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-b2bc09d0-build-source.tar.gz
source_parent=$root/src-candidate-b2bc09d0-v1
src=$source_parent/source
build=$root/build-candidate-b2bc09d0-v1
fish_gate0=$root/external/fish-gate0-b2bc09d0
fish_bn=$root/external/fish-bn-headwing-b2bc09d0
expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
expected_archive_sha=2629705f202f09c74314efc12ed30441f18352e1b3e634bd4a63b47e992aed08
expected_archive_entries=1838
expected_fish_gate0_manifest_sha=b94e92356bbfb6d7e4ed544de2d2d4b4b509bcc6fbfdd02f714011de9fb1d17d
expected_fish_gate0_provenance_sha=57dedf7092e1e8c07cc97d7bd77652583d0a48a435b424d3668a9bb2d6681101
expected_fish_g0w0_comparison_sha=5006f8a3c93c10503561de2d2dcd2252606f9897bbefee785b52e2e9c0cf2cee
expected_fish_bn_manifest_sha=b547ef991e5c2aa924e770b31bda1e08b61ae857973b8de38a5f43fdf0a705a9
expected_fish_bn_provenance_sha=bbb343219fe17fc8ca6c6e7135236f0be335c581731ad0727fad48d7c69bdb1e

test -f "$archive"
test ! -e "$source_parent"
test ! -e "$build"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
test "$(tar -tzf "$archive" | wc -l)" -eq "$expected_archive_entries"
test "$(tar -tzf "$archive" | awk '/(^|\/)\.git(\/|$)/ {count++} END {print count+0}')" -eq 0
test "$(tar -tzf "$archive" | grep -Ec '^source/CMakeLists.txt$')" -eq 1
test "$(tar -tzf "$archive" | grep -Ec '^source/driver/tasks/qsgw.cpp$')" -eq 1
test "$(tar -tzf "$archive" | grep -Ec '^source/src/qsgw/occupation.cpp$')" -eq 1
test "$(sha256sum "$fish_gate0/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_fish_gate0_manifest_sha"
test "$(sha256sum "$fish_gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_fish_gate0_provenance_sha"
test "$(sha256sum "$fish_gate0/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_fish_g0w0_comparison_sha"
grep -Fqx "upstream_commit=$expected_upstream" \
  "$fish_gate0/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_commit" \
  "$fish_gate0/PROVENANCE.txt"
grep -Fq '"passed": true' "$fish_gate0/g0w0-comparison.json"
test "$(sha256sum "$fish_bn/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_fish_bn_manifest_sha"
test "$(sha256sum "$fish_bn/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_fish_bn_provenance_sha"
grep -Fqx "candidate_commit=$expected_commit" "$fish_bn/PROVENANCE.txt"
grep -Fqx 'result=PASS' "$fish_bn/PROVENANCE.txt"

mkdir -p "$source_parent" "$build"
tar -xzf "$archive" -C "$source_parent"
test -f "$src/CMakeLists.txt"
test "$(sha256sum "$src/driver/tasks/qsgw.cpp" | awk '{print $1}')" = \
  016a889a46aeeb4046b09d20a67416cee4231be99e8e7ef6cc69e6fe4d342dc0
test "$(sha256sum "$src/src/qsgw/occupation.cpp" | awk '{print $1}')" = \
  dfda28723b8a8790530a0a2a1bafc43c62de01fc2c327be8399d50c8739b37d4
test "$(sha256sum "$src/src/qsgw/occupation.h" | awk '{print $1}')" = \
  26274a473268ff58fdcf7f4fbded1f1b1a1cd5a990209c21c74844236e3afc6a
test "$(sha256sum "$src/src/test/test_qsgw_occupation.cpp" | awk '{print $1}')" = \
  a490e4d78f3f2546300a83a1a21f287996f5dc1530598f768f7b60cc244157f3
test "$(sha256sum "$src/src/core/gw.cpp" | awk '{print $1}')" = \
  bc4ae1af3883f8f976cd0d1526d703895b5581de6ab2e05b98f2af9efb9a9326
test "$(sha256sum "$src/src/core/gw.h" | awk '{print $1}')" = \
  beec940d42b851f4b0221124b6d8f0ce923cd1ec2a047cf3c1bb98239751c56b
test "$(sha256sum "$src/src/core/exx.cpp" | awk '{print $1}')" = \
  536a251dfb60ebe5c20f964b2c646514d12c82907c998f9ad4e8a4d212c4483b
test "$(sha256sum "$src/src/core/exx.h" | awk '{print $1}')" = \
  d02693ab6126613e48537467951c2f4cc3bef9b62c5eea1002044e402c4e4ee8
test "$(sha256sum "$src/src/core/dielecmodel.h" | awk '{print $1}')" = \
  ab864c7c3da809a7e466ad874943698e91626cc10b8c8e630528207130def768

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
cmake --build "$build" --parallel "${SLURM_CPUS_PER_TASK:-32}" \
  >"$build/build.log" 2>&1

exe=$build/chi0_main.exe
test -x "$exe"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
(
  cd "$source_parent"
  sha256sum --check --quiet "$build/source-files.sha256"
)

{
  printf 'side=candidate-clean-commit\n'
  printf 'upstream_base=%s\n' "$expected_upstream"
  printf 'source_commit=%s\n' "$expected_commit"
  printf 'source_status=clean_git_archive_from_commit\n'
  printf 'source_content_hash_mode=git_archive_blob_bytes\n'
  printf 'final_acceptance=true\n'
  printf 'source_archive=%s\n' "$archive"
  printf 'source_archive_sha256=%s\n' "$expected_archive_sha"
  printf 'source_archive_entry_count=%s\n' "$expected_archive_entries"
  printf 'source_archive_selection=%s\n' \
    'CMakeLists.txt cmake include driver src thirdparty'
  printf 'fish_gate0=%s\n' \
    '/home/bhj/ai-runs/librpa-qsgw-gate0-20260716T0500-b2bc09d0'
  printf 'fish_gate0_manifest_sha256=%s\n' \
    "$expected_fish_gate0_manifest_sha"
  printf 'fish_gate0_result=upstream_38_of_38_candidate_59_of_59_and_official_si_g0w0_exact\n'
  printf 'fish_bn_headwing_regression=%s\n' \
    '/home/bhj/ai-runs/librpa-qsgw-regression-bn-headwing-20260716T0515-b2bc09d0'
  printf 'fish_bn_headwing_manifest_sha256=%s\n' \
    "$expected_fish_bn_manifest_sha"
  printf 'fish_bn_headwing_result=one_of_one_pass\n'
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'slurm_node_list=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'build_type=RelWithDebInfo\n'
  printf 'compiler_environment=oneapi-2024.2\n'
  printf 'cxx_compiler=%s\n' "$(command -v mpiicpx)"
  printf 'fortran_compiler=%s\n' "$(command -v mpiifx)"
  printf 'cmake=%s\n' "$(cmake --version | head -n 1)"
  printf 'executable=%s\n' "$exe"
  printf 'executable_sha256=%s\n' \
    "$(sha256sum "$exe" | awk '{print $1}')"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$build/build-provenance.txt"

sha256sum \
  "$build/CMakeCache.txt" \
  "$build/configure.log" \
  "$build/build.log" \
  "$build/build-provenance.txt" \
  "$build/source-files.sha256" \
  "$exe" \
  "$archive" \
  "$fish_gate0/SHA256SUMS.txt" \
  "$fish_gate0/PROVENANCE.txt" \
  "$fish_gate0/g0w0-comparison.json" \
  "$fish_bn/OUTPUT_SHA256SUMS.txt" \
  "$fish_bn/PROVENANCE.txt" \
  >"$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
