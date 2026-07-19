#!/usr/bin/env bash
#SBATCH --job-name=librpa-a76fd826-b2
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-a76fd826-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-a76fd826-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-a76fd826-build-source.tar.gz
archive_binding=$root/external/archive-commit-a76fd826.txt
source_parent=$root/src-candidate-a76fd826-v2
src=$source_parent/source
build=$root/build-candidate-a76fd826-v2
fish_gate0=$root/external/fish-gate0-a76fd826
fish_tdd=$root/external/fish-headwing-unitary-green-a76fd826

expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_archive_sha=94d04bd1a4a7b142cccb2e6a2056b4c8d4ea2c53d5e6ae9e8a930389946cb10d
expected_archive_binding_sha=bd3d52e71ba3e5512d9774b22f5b71b44526cbfa1743e588f16e625a5170fc92
expected_archive_entries=1838
expected_fish_gate0_manifest_sha=13a148dbc04ac7a8887ed55f56ba479a1c42e668b7981475ef97aa94d9bcd755
expected_fish_gate0_provenance_sha=984e5a716a386afb3ff9e154bb45fd6a5c0c4d48b5c076ce5d5b0077e1ec3973
expected_fish_g0w0_comparison_sha=5270eae0d334b47a95b7a7dae92a405221e0df99216a25105aae65d777fc290a
expected_fish_tdd_manifest_sha=92934bf42050494134b560efeaa526b2e81b91b978f4eac703199aaf8f370557
expected_fish_tdd_provenance_sha=6ea2412b5b3979cfe85cff48f092dbea4d9874716388935e502eee5c8f91825d
expected_implementation_patch_sha=555f24a029ae77e0ecb45460a436e96d2c5a9ef15bfaf804e0a1842eafcfb3fa

test -f "$archive"
test -f "$archive_binding"
test ! -e "$source_parent"
test ! -e "$build"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
test "$(sha256sum "$archive_binding" | awk '{print $1}')" = \
  "$expected_archive_binding_sha"
grep -Fqx "archive_sha256=$expected_archive_sha" "$archive_binding"
grep -Fqx "embedded_commit=$expected_commit" "$archive_binding"
grep -Fqx 'verification_surface=dongfang_login' "$archive_binding"
test "$(tar -tzf "$archive" | wc -l)" -eq "$expected_archive_entries"
test "$(tar -tzf "$archive" | awk '/(^|\/)\.git(\/|$)/ {count++} END {print count+0}')" -eq 0
test "$(tar -tzf "$archive" | grep -Ec '^source/CMakeLists.txt$')" -eq 1
test "$(tar -tzf "$archive" | grep -Ec '^source/driver/tasks/qsgw.cpp$')" -eq 1
test "$(tar -tzf "$archive" | grep -Ec '^source/src/qsgw/fixed_basis.cpp$')" -eq 1

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
grep -Fqx 'protected_shared_implementation_diff=zero' \
  "$fish_gate0/PROVENANCE.txt"
grep -Fqx 'approved_head_getter_added_lines=5' \
  "$fish_gate0/PROVENANCE.txt"
grep -Fq '"passed": true' "$fish_gate0/g0w0-comparison.json"

test "$(sha256sum "$fish_tdd/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_fish_tdd_manifest_sha"
test "$(sha256sum "$fish_tdd/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_fish_tdd_provenance_sha"
test "$(sha256sum "$fish_tdd/implementation.diff" | awk '{print $1}')" = \
  "$expected_implementation_patch_sha"
grep -Fqx 'serial_test=pass' "$fish_tdd/PROVENANCE.txt"
grep -Fqx 'mpi_test_ranks=4' "$fish_tdd/PROVENANCE.txt"
grep -Fqx 'mpi_test=pass' "$fish_tdd/PROVENANCE.txt"

mkdir -p "$source_parent" "$build"
tar -xzf "$archive" -C "$source_parent"
test -f "$src/CMakeLists.txt"
test "$(sha256sum "$src/src/qsgw/fixed_basis.cpp" | awk '{print $1}')" = \
  1517f60c6d1229442b653ac48523ad9d259cf87d964965db8de6f0a68ef20412
test "$(sha256sum "$src/src/qsgw/fixed_basis.h" | awk '{print $1}')" = \
  3e3637e4e774b6c43c249de4fa7b6b0e2123331a9ff9dc17f8af542502578ff8
test "$(sha256sum "$src/src/test/test_qsgw_fixed_basis.cpp" | awk '{print $1}')" = \
  1f1e5837743ea641175e6bc32a49308a30c1757af90cfe71196695a1a8876be6
test "$(sha256sum "$src/src/test/test_qsgw_fixed_basis_mpi.cpp" | awk '{print $1}')" = \
  fc5a2680406313ac152c32b5fee569df2aefc72ad016e13f203d4cc8b1da5060
test "$(sha256sum "$src/driver/tasks/qsgw.cpp" | awk '{print $1}')" = \
  016a889a46aeeb4046b09d20a67416cee4231be99e8e7ef6cc69e6fe4d342dc0
test "$(sha256sum "$src/src/qsgw/occupation.cpp" | awk '{print $1}')" = \
  dfda28723b8a8790530a0a2a1bafc43c62de01fc2c327be8399d50c8739b37d4
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
  printf 'archive_commit_binding_sha256=%s\n' \
    "$expected_archive_binding_sha"
  printf 'final_acceptance=true\n'
  printf 'source_archive=%s\n' "$archive"
  printf 'source_archive_sha256=%s\n' "$expected_archive_sha"
  printf 'source_archive_entry_count=%s\n' "$expected_archive_entries"
  printf 'source_archive_selection=%s\n' \
    'CMakeLists.txt cmake include driver src thirdparty'
  printf 'fish_gate0=%s\n' \
    '/home/bhj/ai-runs/librpa-qsgw-gate0-20260716T0840-a76fd826'
  printf 'fish_gate0_manifest_sha256=%s\n' \
    "$expected_fish_gate0_manifest_sha"
  printf 'fish_gate0_result=upstream_38_of_38_candidate_59_of_59_and_official_si_g0w0_exact\n'
  printf 'fish_headwing_unitary_tdd_patch_sha256=%s\n' \
    "$expected_implementation_patch_sha"
  printf 'fish_headwing_unitary_tdd_result=serial_and_mpi4_pass\n'
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
  "$archive_binding" \
  "$fish_gate0/SHA256SUMS.txt" \
  "$fish_gate0/PROVENANCE.txt" \
  "$fish_gate0/g0w0-comparison.json" \
  "$fish_tdd/SHA256SUMS.txt" \
  "$fish_tdd/PROVENANCE.txt" \
  "$fish_tdd/implementation.diff" \
  >"$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
