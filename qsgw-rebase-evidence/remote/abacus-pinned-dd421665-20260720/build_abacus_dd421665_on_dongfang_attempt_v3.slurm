#!/usr/bin/env bash
#SBATCH --job-name=abacus-dd421665-b3
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/slurm-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2
attempt="$root/build-attempt-v3"
source_dir="$root/source"
deps="$root/dependencies"
build="$attempt/build"
install="$attempt/install"

test -e "$root/SOURCE_PREPARED"
test -e "$root/DEPENDENCIES_FROZEN"
test ! -e "$attempt/BUILD_COMPLETE" || {
    echo "Refusing to mutate completed build attempt: $attempt" >&2
    exit 2
}

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
    > "$attempt/oneapi-setvars.stdout" \
    2> "$attempt/oneapi-setvars.stderr"
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
    printf 'analysis=abacus_pinned_build\n'
    printf 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a\n'
    printf 'source_tree=7284d42382b3555081badfdb4a15691d0bd1ec3d\n'
    printf 'submodule_commit=c211c0ab330adf3cc374f50ab3edee46b174e64c\n'
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
    printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unset}"
    printf 'slurm_cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK:-unset}"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'cmake=%s\n' "$(command -v cmake)"
    printf 'mpiicpx=%s\n' "$(command -v mpiicpx)"
    printf 'MKLROOT=%s\n' "${MKLROOT:-unset}"
} > "$attempt/BUILD_PROVENANCE.txt"

cmake --version > "$attempt/cmake-version.txt"
mpiicpx --version > "$attempt/mpiicpx-version.txt" 2>&1
mpiicpx -show > "$attempt/mpiicpx-show.txt" 2>&1
env | LC_ALL=C sort > "$attempt/build-environment.txt"

cmake -S "$source_dir" -B "$build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$(command -v mpiicpx)" \
    -DMPI_CXX_COMPILER="$(command -v mpiicpx)" \
    -DCMAKE_INSTALL_PREFIX="$install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGIT_SUBMODULE=OFF \
    -DCOMMIT_INFO=OFF \
    -DENABLE_MPI=ON \
    -DUSE_OPENMP=ON \
    -DENABLE_LCAO=ON \
    -DUSE_ELPA=ON \
    -DENABLE_LIBRI=ON \
    -DENABLE_LIBCOMM=ON \
    -DENABLE_LIBXC=OFF \
    -DENABLE_MLALGO=OFF \
    -DENABLE_PEXSI=OFF \
    -DENABLE_NATIVE_OPTIMIZATION=OFF \
    -DENABLE_FFT_TWO_CENTER=ON \
    -DBUILD_TESTING=OFF \
    -DLIBRI_DIR="$deps/LibRI" \
    -DLIBCOMM_DIR="$deps/LibComm" \
    -DCEREAL_INCLUDE_DIR="$deps/cereal-1.3.0/include" \
    -DELPA_INCLUDE_DIR="$deps/elpa/include/elpa_openmp-2025.01.001" \
    -DELPA_LIBRARIES="$deps/elpa/lib/libelpa_openmp.so" \
    -DMKLROOT="$MKLROOT" \
    > "$attempt/configure.stdout" 2> "$attempt/configure.stderr"

cmake --build "$build" --parallel "${SLURM_CPUS_PER_TASK:-48}" \
    > "$attempt/build.stdout" 2> "$attempt/build.stderr"
cmake --install "$build" > "$attempt/install.stdout" 2> "$attempt/install.stderr"

binary="$build/abacus_3p"
installed_binary="$install/bin/abacus_3p"
test -x "$binary"
test -x "$installed_binary"
sha256sum "$binary" "$installed_binary" > "$attempt/BINARY_SHA256SUMS.txt"
file "$binary" > "$attempt/binary-file.txt"
ldd "$binary" > "$attempt/binary-ldd.txt"
readelf -n "$binary" > "$attempt/binary-notes.txt"
"$binary" -v > "$attempt/binary-version.stdout" \
    2> "$attempt/binary-version.stderr" || true
cp "$build/CMakeCache.txt" "$attempt/CMakeCache.txt"

grep -v -E '  (CITATIONS\.md|tools/generate_orbital\.sh)$' \
    "$root/source_artifacts/TRACKED_SOURCE_SHA256SUMS.txt" \
    > "$attempt/TRACKED_REGULAR_SOURCE_SHA256SUMS.txt"
(
    cd "$source_dir"
    sha256sum -c "$attempt/TRACKED_REGULAR_SOURCE_SHA256SUMS.txt"
) > "$attempt/source-recheck.stdout" 2> "$attempt/source-recheck.stderr"
(
    cd "$source_dir/deps/libpaw_interface"
    sha256sum -c "$root/source_artifacts/LIBPAW_TRACKED_SOURCE_SHA256SUMS.txt"
) > "$attempt/libpaw-recheck.stdout" 2> "$attempt/libpaw-recheck.stderr"

cat >> "$attempt/BUILD_PROVENANCE.txt" <<EOF
binary=$binary
binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
installed_binary=$installed_binary
installed_binary_sha256=$(sha256sum "$installed_binary" | awk '{print $1}')
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

touch "$attempt/BUILD_COMPLETE"
find "$attempt" -maxdepth 1 -type f \
    ! -name BUILD_OUTPUT_SHA256SUMS.txt \
    ! -name BUILD_COMPLETE -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum \
    > "$attempt/BUILD_OUTPUT_SHA256SUMS.txt"

cat "$attempt/BUILD_PROVENANCE.txt"
cat "$attempt/BINARY_SHA256SUMS.txt"
