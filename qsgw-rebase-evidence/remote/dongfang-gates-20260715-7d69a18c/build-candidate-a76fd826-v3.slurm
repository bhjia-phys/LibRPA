#!/usr/bin/env bash
#SBATCH --job-name=librpa-a76fd826-b3
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-a76fd826-v3-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-a76fd826-v3-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-a76fd826-linux-git-archive.tar.gz
source_evidence=$root/external/fish-source-archive-a76fd826-v3
source_parent=$root/src-candidate-a76fd826-v3
src=$source_parent/source
build=$root/build-candidate-a76fd826-v3
fish_gate0=$root/external/fish-gate0-a76fd826
fish_tdd=$root/external/fish-headwing-unitary-green-a76fd826

expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_commit=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_archive_sha=521936e27b18c325a79acedcecd7f0541d54347f3a0f831b89e4517d4104f641
expected_archive_entries=1838
expected_source_manifest_sha=e9f3ff302c0e4095d1b4c7f8e955890c511cab3e0ec378d24d11f114a97cd007
expected_source_provenance_sha=9bdf4a6d70d4b2db12189778f4f0ec813594da21f5ca3fdf7c5a3731540c89c1
expected_critical_blobs_sha=b7a174574d0dd7b2651c4b1fcc31b87b283e9eb021e02fc45b5e48705976365b
expected_fish_gate0_manifest_sha=13a148dbc04ac7a8887ed55f56ba479a1c42e668b7981475ef97aa94d9bcd755
expected_fish_gate0_provenance_sha=984e5a716a386afb3ff9e154bb45fd6a5c0c4d48b5c076ce5d5b0077e1ec3973
expected_fish_g0w0_comparison_sha=5270eae0d334b47a95b7a7dae92a405221e0df99216a25105aae65d777fc290a
expected_fish_tdd_manifest_sha=92934bf42050494134b560efeaa526b2e81b91b978f4eac703199aaf8f370557
expected_fish_tdd_provenance_sha=6ea2412b5b3979cfe85cff48f092dbea4d9874716388935e502eee5c8f91825d
expected_implementation_patch_sha=555f24a029ae77e0ecb45460a436e96d2c5a9ef15bfaf804e0a1842eafcfb3fa

test -f "$archive"
test -e "$source_evidence/COMPLETE"
test ! -e "$source_parent"
test ! -e "$build"
test "$(sha256sum "$archive" | awk '{print $1}')" = \
  "$expected_archive_sha"
test "$(tar -tzf "$archive" | wc -l)" -eq "$expected_archive_entries"
test "$(sha256sum "$source_evidence/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_manifest_sha"
test "$(sha256sum "$source_evidence/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_evidence/critical-blobs.sha256" | awk '{print $1}')" = \
  "$expected_critical_blobs_sha"
grep -Fqx "source_commit=$expected_commit" "$source_evidence/PROVENANCE.txt"
grep -Fqx "archive_sha256=$expected_archive_sha" "$source_evidence/PROVENANCE.txt"
grep -Fqx "archive_embedded_commit=$expected_commit" \
  "$source_evidence/PROVENANCE.txt"
grep -Fqx 'archive_line_endings=git_blob_lf_bytes' \
  "$source_evidence/PROVENANCE.txt"
cmp "$archive" \
  "$source_evidence/librpa-a76fd826-linux-git-archive.tar.gz"
(
  cd "$source_evidence"
  sha256sum --check --quiet SHA256SUMS.txt
)

test "$(sha256sum "$fish_gate0/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_fish_gate0_manifest_sha"
test "$(sha256sum "$fish_gate0/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_fish_gate0_provenance_sha"
test "$(sha256sum "$fish_gate0/g0w0-comparison.json" | awk '{print $1}')" = \
  "$expected_fish_g0w0_comparison_sha"
grep -Fqx "upstream_commit=$expected_upstream" "$fish_gate0/PROVENANCE.txt"
grep -Fqx "candidate_commit=$expected_commit" "$fish_gate0/PROVENANCE.txt"
grep -Fqx 'protected_shared_implementation_diff=zero' \
  "$fish_gate0/PROVENANCE.txt"
grep -Fq '"passed": true' "$fish_gate0/g0w0-comparison.json"

test "$(sha256sum "$fish_tdd/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_fish_tdd_manifest_sha"
test "$(sha256sum "$fish_tdd/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_fish_tdd_provenance_sha"
test "$(sha256sum "$fish_tdd/implementation.diff" | awk '{print $1}')" = \
  "$expected_implementation_patch_sha"
grep -Fqx 'serial_test=pass' "$fish_tdd/PROVENANCE.txt"
grep -Fqx 'mpi_test=pass' "$fish_tdd/PROVENANCE.txt"

mkdir -p "$source_parent" "$build"
tar -xzf "$archive" -C "$source_parent"
test -f "$src/CMakeLists.txt"
while read -r expected path; do
  test "$(sha256sum "$src/$path" | awk '{print $1}')" = "$expected"
done <"$source_evidence/critical-blobs.sha256"

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
  printf 'source_status=clean_linux_git_archive_from_commit\n'
  printf 'source_content_hash_mode=git_blob_lf_bytes\n'
  printf 'final_acceptance=true\n'
  printf 'source_archive=%s\n' "$archive"
  printf 'source_archive_sha256=%s\n' "$expected_archive_sha"
  printf 'source_archive_entry_count=%s\n' "$expected_archive_entries"
  printf 'source_archive_evidence_manifest_sha256=%s\n' \
    "$expected_source_manifest_sha"
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
  "$source_evidence/SHA256SUMS.txt" \
  "$source_evidence/PROVENANCE.txt" \
  "$source_evidence/critical-blobs.sha256" \
  "$fish_gate0/SHA256SUMS.txt" \
  "$fish_gate0/PROVENANCE.txt" \
  "$fish_gate0/g0w0-comparison.json" \
  "$fish_tdd/SHA256SUMS.txt" \
  "$fish_tdd/PROVENANCE.txt" \
  "$fish_tdd/implementation.diff" \
  >"$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
