#!/usr/bin/env bash
#SBATCH --job-name=si444-sym-bundle
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v1/slurm-%j.err

set -Eeuo pipefail

producer=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-20260720-v1
source_run="$producer/run"
source_out="$source_run/OUT.ABACUS"
submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v1
tools_dir="$submit_root/tools"
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v1
dataset="$root/dataset"
provenance="$root/provenance"

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

test -e "$producer/COMPLETE"
test ! -e "$producer/FAILED"
test -f "$producer/ARTIFACT_SHA256SUMS.txt"
test -f "$producer/OUTPUT_VALIDATION.json"
test ! -e "$root"
test -f "$tools_dir/generate_abacus_basis_metadata_v1.py"
test -f "$tools_dir/prepare_abacus_qsgw_ibz_contract_v1.py"
(
    cd /
    sha256sum --check --quiet "$producer/ARTIFACT_SHA256SUMS.txt"
)

mkdir -p "$dataset" "$provenance"
cp --reflink=auto "$producer/RUN_PROVENANCE.txt" "$provenance/producer-RUN_PROVENANCE.txt"
cp --reflink=auto "$producer/OUTPUT_VALIDATION.json" "$provenance/producer-OUTPUT_VALIDATION.json"
cp --reflink=auto "$producer/ARTIFACT_SHA256SUMS.txt" "$provenance/producer-ARTIFACT_SHA256SUMS.txt"
cp --reflink=auto "$producer/producer-inputs-v2/ASSET_SHA256SUMS" "$provenance/"
cp --reflink=auto "$tools_dir/generate_abacus_basis_metadata_v1.py" "$provenance/"
cp --reflink=auto "$tools_dir/prepare_abacus_qsgw_ibz_contract_v1.py" "$provenance/"

for name in band_out stru_out bz_sampling_out; do
    cp --reflink=auto "$source_run/$name" "$dataset/$name"
done
for pattern in KS_eigenvector_\*.dat Cs_data_\*.txt coulomb_cut_\*.txt coulomb_mat_\*.txt; do
    for path in "$source_run"/$pattern; do
        test -f "$path"
        cp --reflink=auto "$path" "$dataset/$(basename "$path")"
    done
done
for name in vxc_out.dat Vxc_R_spin0.csr hrs1_nao.csr srs1_nao.csr rr.csr; do
    cp --reflink=auto "$source_out/$name" "$dataset/$name"
done
for index in $(seq 1 8); do
    source_vxc="$source_out/vxck${index}_nao.txt"
    producer_name="vxck${index}_nao.txt"
    legacy_name="vxcs1k${index}_nao.txt"
    test -f "$source_vxc"
    cp --reflink=auto "$source_vxc" "$dataset/$producer_name"
    cp --reflink=auto "$source_vxc" "$dataset/$legacy_name"
    cmp "$dataset/$producer_name" "$dataset/$legacy_name"
done

python3 -B "$tools_dir/generate_abacus_basis_metadata_v1.py" \
    --orb "$producer/assets/Si_gga_8au_100Ry_3s3p2d.orb" \
    --abfs "$producer/assets/Si_3s3p2d1f1g_pca1e-6.abfs" \
    --n-atoms 2 --output-dir "$dataset" \
    > "$provenance/basis-generator.stdout"

python3 -B "$tools_dir/prepare_abacus_qsgw_ibz_contract_v1.py" "$dataset" \
    > "$provenance/contract-generator.stdout"

test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxcs*k*_nao.txt' | wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxck*_nao.txt' | wc -l)" -eq 8
for pattern in Cs_data_\*.txt coulomb_cut_\*.txt coulomb_mat_\*.txt; do
    test "$(find "$dataset" -maxdepth 1 -type f -name "$pattern" | wc -l)" -eq 4
done
test ! -e "$dataset/basis_aux_shrink_out"
test ! -e "$dataset/Cs_shrinked_data_0.txt"
test ! -e "$dataset/shrink_sinvS_0.txt"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'n_scf_kpoints 8' "$dataset/qsgw_input.contract"
grep -Fq '"use_shrink_abfs": false' "$dataset/qsgw_input_contract.summary.json"
grep -Fq '"basis_per_atom": 22' "$dataset/basis_metadata_summary.json"
grep -Fq '"basis_per_atom": 397' "$dataset/basis_metadata_summary.json"
test -z "$(find "$root" -type l -print -quit)"

(
    cd "$root"
    find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        > DATASET_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

producer_manifest_sha=$(sha256sum "$producer/ARTIFACT_SHA256SUMS.txt" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
vxc_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
cat > "$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_symmetry_bundle_v1
acceptance_scope=shared_legacy_candidate_headwing_off_hartree_off_grid_input
producer=$producer
producer_slurm_job_id=2381071
producer_manifest_sha256=$producer_manifest_sha
producer_reader_version=0
dataset_basis_metadata=derived_once_from_frozen_orb_abfs_headers
vxc_producer_schema=vxck_one_based
vxc_legacy_compatibility=byte_identical_vxcs1k_copies
copy_mode=physical_copy_reflink_allowed_no_symlinks
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
use_shrink_abfs=false
symmetry=on
headwing=off
hartree=off
band_update=off
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_input_contract_sha256=$contract_sha
qsgw_vxc_manifest_sha256=$vxc_manifest_sha
slurm_job_id=${SLURM_JOB_ID:-unset}
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
    cd "$root"
    find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
        ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        > OUTPUT_SHA256SUMS.txt
    sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$root/COMPLETE"
find "$root" -type f -exec chmod a-w {} +
find "$root" -depth -type d -exec chmod a-w {} +
trap - EXIT

test -e "$root/COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -type l -print -quit)"
test -z "$(find "$root" -perm /222 -print -quit)"
echo PINNED_ABACUS_SI_K444_SYMMETRY_BUNDLE_V1=PASS
cat "$root/PROVENANCE.txt"
