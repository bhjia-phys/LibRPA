#!/usr/bin/env bash
#SBATCH --job-name=librpa-2e446af2-b2
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-2e446af2-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/build-2e446af2-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
archive=$root/source-archives/librpa-2e446af2-linux-git-archive.tar.gz
source_evidence=$root/external/fish-source-archive-2e446af2
fish_red=$root/external/fish-polar-red-oneapi-2e446af2
fish_green=$root/external/fish-polar-green-oneapi-2e446af2
source_parent=$root/src-candidate-2e446af2-v2
src=$source_parent/source
build=$root/build-candidate-2e446af2-v2

expected_upstream=95c4c08009aa6752a6d386289abe1fb4358489ca
expected_parent=a76fd826eed3929185ea47cfcb992075a72f2b77
expected_commit=2e446af200a0e5a5fd1ee05c8db99ebb7dde9418
expected_archive_sha=433d971ab428d221ce122d295b784f5bd7530c4dbd9643ad417a36948beb82ce
expected_archive_entries=1838
expected_source_manifest_sha=604b2fa4465f92bce840766d3cd7a0b3573369da9d1a22c7669a1b9fc4c341d6
expected_source_provenance_sha=ca8f093d5a5d4bb9fa2dcc52b7305dd5aaba790370104eaedec5b6b41a2cab41
expected_critical_blobs_sha=2b3497b9d15ace7f56be2e8a3729a0fbba27760d9fd77265fda7b443560120c9
expected_red_manifest_sha=92dae83f0bda1ea4cfec215a919fde544b02b7f6a095044404fbc131d1dc8130
expected_red_provenance_sha=58f0b77a3b9ebdeab0c5e595e75b13a3aeccb1cfad8740a03db702e85c018be1
expected_green_manifest_sha=5d857626f24ae69da5ba836fa4b6fc03c257b2b181b39979bb12653ff5a1b651
expected_green_provenance_sha=2598fe61c3ceaf927e69f4290e3f6221b63bc078fa1834eb0b8b65470c80d707

test -f "$archive"
test -e "$source_evidence/COMPLETE"
test -e "$fish_red/TDD_RED_CONFIRMED"
test -e "$fish_green/COMPLETE"
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
grep -Fqx "source_parent_commit=$expected_parent" "$source_evidence/PROVENANCE.txt"
grep -Fqx "archive_sha256=$expected_archive_sha" "$source_evidence/PROVENANCE.txt"
grep -Fqx "archive_embedded_commit=$expected_commit" "$source_evidence/PROVENANCE.txt"
grep -Fqx 'archive_line_endings=git_blob_lf_bytes' \
  "$source_evidence/PROVENANCE.txt"
cmp "$archive" "$source_evidence/librpa-2e446af2-linux-git-archive.tar.gz"
(
  cd "$source_evidence"
  sha256sum --check --quiet SHA256SUMS.txt
)

test "$(sha256sum "$fish_red/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_red_manifest_sha"
test "$(sha256sum "$fish_red/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_red_provenance_sha"
grep -Fqx 'expected_failure_confirmed=true' "$fish_red/PROVENANCE.txt"
grep -Fqx 'cmake_openmp_cxx_flags=-fiopenmp' "$fish_red/PROVENANCE.txt"
(
  cd "$fish_red"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

test "$(sha256sum "$fish_green/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_green_manifest_sha"
test "$(sha256sum "$fish_green/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_green_provenance_sha"
grep -Fqx 'serial_fixed_basis_test=PASS' "$fish_green/PROVENANCE.txt"
grep -Fqx 'mpi4_fixed_basis_test=PASS' "$fish_green/PROVENANCE.txt"
grep -Fqx 'candidate_ctest_result=PASS' "$fish_green/PROVENANCE.txt"
grep -Fqx 'candidate_ctest_count=59' "$fish_green/PROVENANCE.txt"
(
  cd "$fish_green"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$source_parent" "$build"
cp "$0" "$build/build-candidate-2e446af2-v2.slurm"
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

cat >"$build/build-provenance.txt" <<EOF
side=candidate-clean-commit
upstream_base=$expected_upstream
source_parent_commit=$expected_parent
source_commit=$expected_commit
source_status=clean_linux_git_archive_from_commit
source_content_hash_mode=git_blob_lf_bytes
source_archive=$archive
source_archive_sha256=$expected_archive_sha
source_archive_entry_count=$expected_archive_entries
source_archive_evidence_manifest_sha256=$expected_source_manifest_sha
source_archive_embedded_commit_validation=fish_source_evidence
fish_red_manifest_sha256=$expected_red_manifest_sha
fish_red_result=expected_failure_confirmed_oneapi
fish_green_manifest_sha256=$expected_green_manifest_sha
fish_green_result=serial_mpi4_and_59_of_59_pass_oneapi
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
slurm_job_gpus=${SLURM_JOB_GPUS:-unknown}
supersedes_failed_build_job=2346633
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
  "$build/build-candidate-2e446af2-v2.slurm" \
  "$build/source-files.sha256" \
  "$exe" \
  "$archive" \
  "$source_evidence/SHA256SUMS.txt" \
  "$source_evidence/PROVENANCE.txt" \
  "$source_evidence/critical-blobs.sha256" \
  "$fish_red/OUTPUT_SHA256SUMS.txt" \
  "$fish_red/PROVENANCE.txt" \
  "$fish_green/OUTPUT_SHA256SUMS.txt" \
  "$fish_green/PROVENANCE.txt" \
  >"$build/SHA256SUMS.txt"
sha256sum --check --quiet "$build/SHA256SUMS.txt"
touch "$build/COMPLETE"
cat "$build/build-provenance.txt"
