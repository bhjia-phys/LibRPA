#!/usr/bin/env bash
#SBATCH --job-name=si-band-nscf-dd421665
#SBATCH --partition=48cp3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-band-nscf-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-band-nscf-submit-20260720-v1/slurm-%j.err

set -Eeuo pipefail

submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-band-nscf-submit-20260720-v1
inputs="$submit_root/producer-inputs-v2"
assets="$submit_root/assets"
tools_dir="$submit_root/tools"
scf_bundle=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v3
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-band-nscf-20260720-v1
run_dir="$root/run"
provenance="$root/provenance"
binary=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-build-20260720-v2/build-attempt-v3/build/abacus_3p
binary_sha=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17

on_exit()
{
    code=$?
    if [[ $code -ne 0 && -d ${root:-/nonexistent} ]]; then
        {
            printf 'exit_code=%s\n' "$code"
            printf 'failed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
        } > "$root/FAILED"
    fi
}
trap on_exit EXIT

test ! -e "$root"
test -d "$inputs"
test -d "$assets"
test -f "$inputs/validate_producer_inputs_v2.py"
test -f "$tools_dir/validate_abacus_si_band_nscf_output_v1.py"
test -e "$scf_bundle/COMPLETE"
test ! -e "$scf_bundle/FAILED"
test -f "$scf_bundle/OUTPUT_SHA256SUMS.txt"
test -z "$(find "$scf_bundle" -type l -print -quit)"
test -z "$(find "$scf_bundle" -perm /222 -print -quit)"
(
    cd "$scf_bundle"
    sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$run_dir" "$provenance/producer-inputs-v2" \
    "$provenance/assets" "$provenance/tools"
cp "$inputs/"* "$provenance/producer-inputs-v2/"
cp "$assets/"* "$provenance/assets/"
cp "$tools_dir/validate_abacus_si_band_nscf_output_v1.py" \
    "$provenance/tools/"
cp "$scf_bundle/PROVENANCE.txt" "$provenance/scf-bundle-PROVENANCE.txt"
cp "$scf_bundle/OUTPUT_SHA256SUMS.txt" \
    "$provenance/scf-bundle-OUTPUT_SHA256SUMS.txt"

python3 -B "$provenance/producer-inputs-v2/validate_producer_inputs_v2.py" \
    --assets-dir "$provenance/assets" \
    > "$root/input-contract-validation.out"

actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_binary_sha" != "$binary_sha" ]]; then
    printf 'Pinned binary hash mismatch: %s\n' "$actual_binary_sha" >&2
    exit 10
fi

cp "$provenance/producer-inputs-v2/INPUT_nscf_band" "$run_dir/INPUT"
cp "$provenance/producer-inputs-v2/KPT_band" "$run_dir/KPT"
cp "$provenance/producer-inputs-v2/STRU" "$run_dir/STRU"
while read -r digest name; do
    cp "$provenance/assets/$name" "$run_dir/$name"
done < "$provenance/producer-inputs-v2/ASSET_SHA256SUMS"
(
    cd "$run_dir"
    sha256sum --check "$provenance/producer-inputs-v2/ASSET_SHA256SUMS"
) > "$root/staged-asset-validation.out"

source_charge="$scf_bundle/dataset/ABACUS-CHARGE-DENSITY.restart"
test -f "$source_charge"
charge_sha256_before=$(sha256sum "$source_charge" | awk '{print $1}')
test ! -e "$run_dir/OUT.ABACUS"
mkdir "$run_dir/OUT.ABACUS"
staged_charge="$run_dir/OUT.ABACUS/ABACUS-CHARGE-DENSITY.restart"
cp "$source_charge" "$staged_charge"
test "$(sha256sum "$staged_charge" | awk '{print $1}')" = \
    "$charge_sha256_before"

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
np="${SLURM_NTASKS:-4}"

{
    printf 'analysis=abacus_pinned_si_band_nscf_producer\n'
    printf 'producer_contract=producer-inputs-v2/INPUT_nscf_band\n'
    printf 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a\n'
    printf 'source_tree=7284d42382b3555081badfdb4a15691d0bd1ec3d\n'
    printf 'submodule_commit=c211c0ab330adf3cc374f50ab3edee46b174e64c\n'
    printf 'binary=%s\n' "$binary"
    printf 'binary_sha256=%s\n' "$actual_binary_sha"
    printf 'scf_bundle=%s\n' "$scf_bundle"
    printf 'scf_charge_restart_sha256=%s\n' "$charge_sha256_before"
    printf 'charge_staging=physical_copy_into_fresh_OUT.ABACUS\n'
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
    printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-unset}"
    printf 'slurm_nodes=%s\n' "${SLURM_JOB_NUM_NODES:-unset}"
    printf 'mpi_ranks=%s\n' "$np"
    printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'calculation=nscf\n'
    printf 'symmetry=-1\n'
    printf 'band_kpoints_expected=143\n'
    printf 'n_bands=44\n'
    printf 'n_basis=44\n'
} > "$root/RUN_PROVENANCE.txt"

env | LC_ALL=C sort > "$root/runtime-environment.txt"
file "$binary" > "$root/binary-file.txt"
ldd "$binary" > "$root/binary-ldd.txt"
mpirun -np "$np" hostname | LC_ALL=C sort > "$root/mpi-hosts.txt"

(
    cd "$run_dir"
    mpirun -np "$np" "$binary"
) > "$run_dir/abacus.stdout" 2> "$run_dir/abacus.stderr"

charge_sha256_after=$(sha256sum "$staged_charge" | awk '{print $1}')
if [[ "$charge_sha256_after" != "$charge_sha256_before" ]]; then
    printf 'SCF charge restart changed during NSCF: %s -> %s\n' \
        "$charge_sha256_before" "$charge_sha256_after" >&2
    exit 11
fi

python3 -B "$provenance/tools/validate_abacus_si_band_nscf_output_v1.py" \
    "$run_dir" \
    --expected-charge-sha256 "$charge_sha256_before" \
    --expected-n-bands 44 --expected-n-basis 44 --expected-n-spins 1 \
    --output "$root/NSCF_OUTPUT_VALIDATION.json" \
    > "$root/output-validation.stdout"

python3 -B - "$root/NSCF_OUTPUT_VALIDATION.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="ascii") as stream:
    report = json.load(stream)
assert report["status"] == "PASS"
assert report["n_kpoints"] == 143
assert report["n_spins"] == 1
assert report["n_bands"] == 44
assert report["n_basis"] == 44
assert report["wfc_directory"] == "OUT.ABACUS/WFC"
assert report["vxc_matrix_basis"] == "ks_state"
PY

cat >> "$root/RUN_PROVENANCE.txt" <<EOF
charge_sha256_after=$charge_sha256_after
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

find "$root" -type f ! -name ARTIFACT_SHA256SUMS.txt \
    ! -name COMPLETE ! -name FAILED -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum \
    > "$root/ARTIFACT_SHA256SUMS.txt"
(
    cd "$root"
    sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)
test -z "$(find "$root" -type l -print -quit)"
touch "$root/COMPLETE"
find "$root" -type f -exec chmod a-w {} +
find "$root" -depth -type d -exec chmod a-w {} +
trap - EXIT

test -e "$root/COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -perm /222 -print -quit)"
echo PINNED_ABACUS_SI_BAND_NSCF_V1=PASS
cat "$root/NSCF_OUTPUT_VALIDATION.json"
