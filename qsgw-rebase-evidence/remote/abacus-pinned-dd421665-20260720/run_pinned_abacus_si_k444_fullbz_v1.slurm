#!/usr/bin/env bash
#SBATCH --job-name=si444-fullbz-dd421665
#SBATCH --partition=48cp3
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-20260721-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-20260721-v1/slurm-%j.err

set -Eeuo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-20260721-v1
contract="$root/producer-inputs-v2"
assets="$root/assets"
tools_dir="$root/tools"
run_dir="$root/run"
binary=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/build/abacus_3p
binary_sha=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17
runner_source=${BASH_SOURCE[0]}
runner_sha256=$(sha256sum "$runner_source" | awk '{print $1}')

on_exit()
{
    code=$?
    if [[ $code -ne 0 ]]; then
        {
            printf 'exit_code=%s\n' "$code"
            printf 'failed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
        } > "$root/FAILED"
    fi
}
trap on_exit EXIT

test -d "$contract"
test -d "$assets"
test -f "$tools_dir/validate_abacus_si_k444_fullbz_output_v1.py"
test -f "$tools_dir/validate_abacus_si_k444_symmetry_output_v1.py"
test ! -e "$root/COMPLETE"
test ! -e "$root/RUN_STARTED"
test ! -e "$run_dir"
test ! -e "$root/executed-runner.slurm"
grep -Eq '^symmetry[[:space:]]+-1$' "$contract/INPUT_scf_fullbz"
grep -Eq '^out_librpa_reader_version[[:space:]]+0$' \
    "$contract/INPUT_scf_fullbz"
touch "$root/RUN_STARTED"
mkdir "$run_dir"
cp "$runner_source" "$root/executed-runner.slurm"

python3 -B "$contract/validate_producer_inputs_v2.py" --assets-dir "$assets" \
    > "$root/input-contract-validation.out"

actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_binary_sha" != "$binary_sha" ]]; then
    printf 'Pinned binary hash mismatch: %s\n' "$actual_binary_sha" >&2
    exit 10
fi

cp "$contract/INPUT_scf_fullbz" "$run_dir/INPUT"
cp "$contract/KPT_k444" "$run_dir/KPT"
cp "$contract/STRU" "$run_dir/STRU"
while read -r digest name; do
    cp "$assets/$name" "$run_dir/$name"
done < "$contract/ASSET_SHA256SUMS"

(
    cd "$run_dir"
    sha256sum -c "$contract/ASSET_SHA256SUMS"
) > "$root/staged-asset-validation.out"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
    > "$root/oneapi-setvars.stdout" 2> "$root/oneapi-setvars.stderr"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
np="${SLURM_NTASKS:-8}"

{
    printf 'analysis=abacus_pinned_si_k444_fullbz_producer\n'
    printf 'producer_contract=producer-inputs-v2\n'
    printf 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a\n'
    printf 'source_tree=7284d42382b3555081badfdb4a15691d0bd1ec3d\n'
    printf 'submodule_commit=c211c0ab330adf3cc374f50ab3edee46b174e64c\n'
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$actual_binary_sha"
    printf 'runner_sha256=%s\n' "$runner_sha256"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
    printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unset}"
    printf 'slurm_nodes=%s\n' "${SLURM_JOB_NUM_NODES:-unset}"
    printf 'mpi_ranks=%s\n' "$np"
    printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'reader_version=0\n'
    printf 'symmetry=-1\n'
    printf 'k_grid=4x4x4\n'
    printf 'scf_kpoints=64\n'
    printf 'full_bz_kpoints=64\n'
} > "$root/RUN_PROVENANCE.txt"

env | LC_ALL=C sort > "$root/runtime-environment.txt"
file "$binary" > "$root/binary-file.txt"
ldd "$binary" > "$root/binary-ldd.txt"
mpirun -np "$np" hostname | LC_ALL=C sort > "$root/mpi-hosts.txt"

(
    cd "$run_dir"
    mpirun -np "$np" "$binary"
) > "$run_dir/abacus.stdout" 2> "$run_dir/abacus.stderr"

PYTHONPATH="$tools_dir" python3 -B \
    "$tools_dir/validate_abacus_si_k444_fullbz_output_v1.py" \
    "$run_dir" --mpi-ranks "$np" --report "$root/OUTPUT_VALIDATION.json" \
    > "$root/output-validation.stdout"

cat >> "$root/RUN_PROVENANCE.txt" <<EOF
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

find "$contract" "$assets" "$tools_dir" "$run_dir" \
    -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    > "$root/ARTIFACT_SHA256SUMS.txt"
sha256sum "$root/RUN_PROVENANCE.txt" "$root/OUTPUT_VALIDATION.json" \
    "$root/executed-runner.slurm" \
    >> "$root/ARTIFACT_SHA256SUMS.txt"
(
    cd "$root"
    sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)

touch "$root/COMPLETE"
trap - EXIT
cat "$root/OUTPUT_VALIDATION.json"
