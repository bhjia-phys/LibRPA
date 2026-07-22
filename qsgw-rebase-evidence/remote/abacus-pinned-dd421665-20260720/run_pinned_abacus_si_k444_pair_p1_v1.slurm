#!/usr/bin/env bash
#SBATCH --job-name=si444-pair-dd421665
#SBATCH --partition=p1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-p1-20260723-v1/submit/slurm-%A_%a.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-p1-20260723-v1/submit/slurm-%A_%a.err

set -Eeuo pipefail

pair_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-pair-p1-20260723-v1
common="$pair_root/common"
contract="$common/producer-inputs-v2"
assets="$common/assets"
tools_dir="$common/tools"
binary=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/build/abacus_3p
binary_sha=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17
runner_source=${BASH_SOURCE[0]}
runner_sha256=$(sha256sum "$runner_source" | awk '{print $1}')

case "${SLURM_ARRAY_TASK_ID:?array task id is required}" in
  0) mode=symmetry; input_name=INPUT_scf_symmetry; symmetry_value=1 ;;
  1) mode=fullbz; input_name=INPUT_scf_fullbz; symmetry_value=-1 ;;
  *) echo "invalid array task id" >&2; exit 2 ;;
esac
mode_root="$pair_root/$mode"
run_dir="$mode_root/run"

on_exit()
{
    code=$?
    if [[ $code -ne 0 && -d ${mode_root:-/nonexistent} ]]; then
        {
            printf 'exit_code=%s\n' "$code"
            printf 'failed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'slurm_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-unset}_${SLURM_ARRAY_TASK_ID:-unset}"
        } > "$mode_root/FAILED"
    fi
}
trap on_exit EXIT

test -e "$pair_root/STAGED"
test ! -e "$mode_root"
test -d "$contract"
test -d "$assets"
test -d "$tools_dir"
test -f "$tools_dir/validate_abacus_si_k444_symmetry_output_v1.py"
test -f "$tools_dir/validate_abacus_si_k444_fullbz_output_v1.py"
test -f "$pair_root/COMMON_SHA256SUMS.txt"
(
    cd "$pair_root"
    sha256sum --check --quiet COMMON_SHA256SUMS.txt
)
grep -Eq '^symmetry[[:space:]]+1$' "$contract/INPUT_scf_symmetry"
grep -Eq '^exx_symmetry_realspace[[:space:]]+1$' "$contract/INPUT_scf_symmetry"
grep -Eq '^symmetry[[:space:]]+-1$' "$contract/INPUT_scf_fullbz"
grep -Eq '^exx_symmetry_realspace[[:space:]]+0$' "$contract/INPUT_scf_fullbz"
grep -Eq '^out_librpa_reader_version[[:space:]]+0$' "$contract/$input_name"

mkdir -p "$run_dir"
cp "$runner_source" "$mode_root/executed-runner.slurm"
test "$(sha256sum "$mode_root/executed-runner.slurm" | awk '{print $1}')" = \
    "$runner_sha256"

python3 -B "$contract/validate_producer_inputs_v2.py" --assets-dir "$assets" \
    > "$mode_root/input-contract-validation.out"

actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_binary_sha" != "$binary_sha" ]]; then
    printf 'Pinned binary hash mismatch: %s\n' "$actual_binary_sha" >&2
    exit 10
fi

cp "$contract/$input_name" "$run_dir/INPUT"
cp "$contract/KPT_k444" "$run_dir/KPT"
cp "$contract/STRU" "$run_dir/STRU"
while read -r digest name; do
    cp "$assets/$name" "$run_dir/$name"
done < "$contract/ASSET_SHA256SUMS"

(
    cd "$run_dir"
    sha256sum --check --quiet "$contract/ASSET_SHA256SUMS"
) > "$mode_root/staged-asset-validation.out"

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
    > "$mode_root/oneapi-setvars.stdout" 2> "$mode_root/oneapi-setvars.stderr"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-40}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
np="${SLURM_NTASKS:-8}"

{
    printf 'analysis=abacus_pinned_si_k444_paired_producer\n'
    printf 'producer_mode=%s\n' "$mode"
    printf 'changed_factor=abacus_input_symmetry\n'
    printf 'changed_settings=symmetry,exx_symmetry_realspace\n'
    printf 'all_other_producer_inputs=identical\n'
    printf 'producer_contract=producer-inputs-v2\n'
    printf 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a\n'
    printf 'source_tree=7284d42382b3555081badfdb4a15691d0bd1ec3d\n'
    printf 'submodule_commit=c211c0ab330adf3cc374f50ab3edee46b174e64c\n'
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$actual_binary_sha"
    printf 'runner_sha256=%s\n' "$runner_sha256"
    printf 'common_manifest_sha256=%s\n' \
        "$(sha256sum "$pair_root/COMMON_SHA256SUMS.txt" | awk '{print $1}')"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-unset}_${SLURM_ARRAY_TASK_ID:-unset}"
    printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unset}"
    printf 'slurm_nodes=%s\n' "${SLURM_JOB_NUM_NODES:-unset}"
    printf 'mpi_ranks=%s\n' "$np"
    printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'reader_version=0\n'
    printf 'symmetry=%s\n' "$symmetry_value"
    printf 'k_grid=4x4x4\n'
    if [[ "$mode" == symmetry ]]; then
        printf 'scf_kpoints=8\n'
    else
        printf 'scf_kpoints=64\n'
    fi
    printf 'full_bz_kpoints=64\n'
} > "$mode_root/RUN_PROVENANCE.txt"

env | LC_ALL=C sort > "$mode_root/runtime-environment.txt"
file "$binary" > "$mode_root/binary-file.txt"
ldd "$binary" > "$mode_root/binary-ldd.txt"
mpirun -np "$np" hostname | LC_ALL=C sort > "$mode_root/mpi-hosts.txt"

(
    cd "$run_dir"
    mpirun -np "$np" "$binary"
) > "$run_dir/abacus.stdout" 2> "$run_dir/abacus.stderr"

if [[ "$mode" == symmetry ]]; then
    validator=$tools_dir/validate_abacus_si_k444_symmetry_output_v1.py
else
    validator=$tools_dir/validate_abacus_si_k444_fullbz_output_v1.py
fi
PYTHONPATH="$tools_dir" python3 -B "$validator" \
    "$run_dir" --mpi-ranks "$np" --report "$mode_root/OUTPUT_VALIDATION.json" \
    > "$mode_root/output-validation.stdout"

cat >> "$mode_root/RUN_PROVENANCE.txt" <<EOF
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

find "$contract" "$assets" "$tools_dir" "$run_dir" \
    -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    > "$mode_root/ARTIFACT_SHA256SUMS.txt"
sha256sum "$mode_root/RUN_PROVENANCE.txt" \
    "$mode_root/OUTPUT_VALIDATION.json" \
    "$mode_root/executed-runner.slurm" \
    >> "$mode_root/ARTIFACT_SHA256SUMS.txt"
(
    cd "$mode_root"
    sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)

touch "$mode_root/COMPLETE"
find "$mode_root" -type f -exec chmod a-w {} +
find "$mode_root" -depth -type d -exec chmod a-w {} +
trap - EXIT

test -e "$mode_root/COMPLETE"
test ! -e "$mode_root/FAILED"
test -z "$(find "$mode_root" -type l -print -quit)"
test -z "$(find "$mode_root" -perm /222 -print -quit)"
cat "$mode_root/OUTPUT_VALIDATION.json"
