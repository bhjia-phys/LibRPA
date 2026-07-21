#!/usr/bin/env bash
#SBATCH --job-name=si444-fullbz-hartree-bundle
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-hartree-bundle-submit-20260721-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-hartree-bundle-submit-20260721-v1/slurm-%j.err

set -Eeuo pipefail

producer=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-20260721-v1
source_run="$producer/run"
source_out="$source_run/OUT.ABACUS"
submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-hartree-bundle-submit-20260721-v1
tools_dir="$submit_root/tools"
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-fullbz-hartree-bundle-20260721-v1
dataset="$root/dataset"
provenance="$root/provenance"
runner_source=${BASH_SOURCE[0]}
runner_sha256=$(sha256sum "$runner_source" | awk '{print $1}')

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
test -f "$producer/RUN_PROVENANCE.txt"
test ! -e "$root"
for tool in generate_abacus_basis_metadata_v1.py \
    prepare_abacus_qsgw_fullbz_contract_v1.py \
    prepare_abacus_qsgw_ibz_contract_v3.py \
    extend_qsgw_contract_with_hartree_v1.py \
    audit_native_vxck_reader_compatibility_v1.py \
    native_vxck_reader_compatibility_v1.json \
    audit_native_vxck_state_basis_v1.py \
    native_vxck_state_basis_v1.json; do
    test -f "$tools_dir/$tool"
done
(
    cd "$producer"
    sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)
grep -Fqx 'source_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a' \
    "$producer/RUN_PROVENANCE.txt"
grep -Fqx \
    'binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17' \
    "$producer/RUN_PROVENANCE.txt"
grep -Fqx 'reader_version=0' "$producer/RUN_PROVENANCE.txt"
grep -Fqx 'symmetry=-1' "$producer/RUN_PROVENANCE.txt"
grep -Fqx 'scf_kpoints=64' "$producer/RUN_PROVENANCE.txt"
grep -Fqx 'full_bz_kpoints=64' "$producer/RUN_PROVENANCE.txt"

python3 -B - "$producer/OUTPUT_VALIDATION.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["status"] == "PASS"
assert report["reader_version"] == 0
assert report["sampling_mode"] == "full_bz"
assert report["bz_sampling"]["grid"] == [4, 4, 4]
assert report["bz_sampling"]["n_scf"] == 64
assert report["bz_sampling"]["n_ibz"] == 64
assert report["bz_sampling"]["full_bz"] is True
assert report["bz_sampling"]["periodic_unique_kpoints"] == 64
assert report["structure"]["n_symops"] == 0
assert report["structure"]["convention"] == "absent"
assert report["band"]["n_bands"] == 44
assert report["band"]["n_basis"] == 44
assert report["vxc_matrix_schema"]["validated_matrix_count"] == 64
assert report["vxc_matrix_schema"]["upper_triangle_entries_per_matrix"] == 990
PY

python3 -B - "$tools_dir/native_vxck_reader_compatibility_v1.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["source_contract_passed"] is True
assert report["legacy_parser_unchanged_by_scheme_a"] is True
assert report["both_readers_accept_native_comment_row_schema"] is True
assert report["both_readers_convert_rydberg_to_hartree"] is True
assert report["native_file_alias_required"] is False
PY
python3 -B - "$tools_dir/native_vxck_state_basis_v1.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["source_contract_passed"] is True
assert report["producer"]["matrix_basis"] == "ks_state"
assert report["producer"]["matrix_shape"] == "n_bands_x_n_bands"
assert report["legacy"]["basis_transform"] == "none"
assert report["candidate"]["basis_transform"] == "none"
assert report["generator"]["manifest_basis"] == "state"
assert report["generator"]["manifest_gauge"] == "mf0_state"
PY

mkdir -p "$dataset" "$provenance/producer-inputs-v2" "$provenance/assets"
cp "$runner_source" "$provenance/executed-runner.slurm"
cp --reflink=auto "$producer/RUN_PROVENANCE.txt" \
    "$provenance/producer-RUN_PROVENANCE.txt"
cp --reflink=auto "$producer/OUTPUT_VALIDATION.json" \
    "$provenance/producer-OUTPUT_VALIDATION.json"
cp --reflink=auto "$producer/ARTIFACT_SHA256SUMS.txt" \
    "$provenance/producer-ARTIFACT_SHA256SUMS.txt"
cp --reflink=auto "$producer/producer-inputs-v2/"* \
    "$provenance/producer-inputs-v2/"
cp --reflink=auto "$producer/assets/"* "$provenance/assets/"
for tool in generate_abacus_basis_metadata_v1.py \
    prepare_abacus_qsgw_fullbz_contract_v1.py \
    prepare_abacus_qsgw_ibz_contract_v3.py \
    extend_qsgw_contract_with_hartree_v1.py \
    audit_native_vxck_reader_compatibility_v1.py \
    native_vxck_reader_compatibility_v1.json \
    audit_native_vxck_state_basis_v1.py \
    native_vxck_state_basis_v1.json; do
    cp --reflink=auto "$tools_dir/$tool" "$provenance/$tool"
done
for path in "$source_run/abacus.stdout" "$source_run/abacus.stderr" \
    "$source_out/running_scf.log" "$source_out/INPUT.info"; do
    test -f "$path"
    cp --reflink=auto "$path" "$provenance/$(basename "$path")"
done

source_paths="$provenance/SOURCE_ARTIFACT_PATHS.txt"
{
    for name in band_out stru_out bz_sampling_out; do
        printf '%s/%s\n' "$source_run" "$name"
    done
    for index in $(seq 0 63); do
        printf '%s/KS_eigenvector_%s.dat\n' "$source_run" "$index"
    done
    for prefix in Cs_data coulomb_cut coulomb_mat; do
        for path in "$source_run"/${prefix}_*.txt; do
            test -f "$path"
            printf '%s\n' "$path"
        done
    done
    for name in ABACUS-CHARGE-DENSITY.restart vxc_out.dat Vxc_R_spin0.csr \
        hrs1_nao.csr srs1_nao.csr rr.csr; do
        printf '%s/%s\n' "$source_out" "$name"
    done
    for index in $(seq 1 64); do
        printf '%s/vxck%s_nao.txt\n' "$source_out" "$index"
    done
} > "$source_paths"
while IFS= read -r path; do
    test -f "$path"
done < "$source_paths"
while IFS= read -r path; do
    sha256sum "$path"
done < "$source_paths" > "$provenance/SOURCE_ARTIFACT_SHA256SUMS.txt"

while read -r digest name; do
    test "$(sha256sum "$producer/assets/$name" | awk '{print $1}')" = "$digest"
    test "$(sha256sum "$provenance/assets/$name" | awk '{print $1}')" = "$digest"
done < "$producer/producer-inputs-v2/ASSET_SHA256SUMS"

for name in band_out stru_out bz_sampling_out; do
    cp --reflink=auto "$source_run/$name" "$dataset/$name"
done
for pattern in KS_eigenvector_\*.dat Cs_data_\*.txt \
    coulomb_cut_\*.txt coulomb_mat_\*.txt; do
    for path in "$source_run"/$pattern; do
        test -f "$path"
        cp --reflink=auto "$path" "$dataset/$(basename "$path")"
    done
done
for name in ABACUS-CHARGE-DENSITY.restart vxc_out.dat Vxc_R_spin0.csr \
    hrs1_nao.csr srs1_nao.csr rr.csr; do
    cp --reflink=auto "$source_out/$name" "$dataset/$name"
done
for index in $(seq 1 64); do
    source_vxc="$source_out/vxck${index}_nao.txt"
    test -f "$source_vxc"
    cp --reflink=auto "$source_vxc" "$dataset/$(basename "$source_vxc")"
done

python3 -B "$tools_dir/generate_abacus_basis_metadata_v1.py" \
    --orb "$producer/assets/Si_gga_8au_100Ry_3s3p2d.orb" \
    --abfs "$producer/assets/Si_3s3p2d1f1g_pca1e-6.abfs" \
    --n-atoms 2 --output-dir "$dataset" \
    > "$provenance/basis-generator.stdout"
PYTHONPATH="$tools_dir" python3 -B \
    "$tools_dir/prepare_abacus_qsgw_fullbz_contract_v1.py" "$dataset" \
    > "$provenance/contract-generator.stdout"

python3 -B "$tools_dir/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_input.contract \
    --output-contract qsgw_input.hartree-full.contract \
    --output-summary qsgw_input.hartree-full.summary.json \
    --hartree-coulomb full \
    > "$provenance/hartree-full-contract.stdout"
python3 -B "$tools_dir/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_input.contract \
    --output-contract qsgw_input.hartree-truncated.contract \
    --output-summary qsgw_input.hartree-truncated.summary.json \
    --hartree-coulomb truncated \
    > "$provenance/hartree-truncated-contract.stdout"

test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | wc -l)" -eq 64
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxck*_nao.txt' | wc -l)" -eq 64
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxcs*k*_nao.txt' | wc -l)" -eq 0
rank_count=$(find "$dataset" -maxdepth 1 -type f -name 'Cs_data_*.txt' | wc -l)
test "$rank_count" -gt 0
test "$(find "$dataset" -maxdepth 1 -type f -name 'coulomb_cut_*.txt' | wc -l)" -eq "$rank_count"
test "$(find "$dataset" -maxdepth 1 -type f -name 'coulomb_mat_*.txt' | wc -l)" -eq "$rank_count"
test ! -e "$dataset/basis_aux_shrink_out"
test ! -e "$dataset/Cs_shrinked_data_0.txt"
test ! -e "$dataset/shrink_sinvS_0.txt"
grep -Fqx 'headwing_update none' "$dataset/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'n_scf_kpoints 64' "$dataset/qsgw_input.contract"
grep -Fqx 'basis state' "$dataset/qsgw_vxc_scf.manifest"
grep -Fqx 'gauge mf0_state' "$dataset/qsgw_vxc_scf.manifest"
test "$(grep -Ec '^1 ([1-9]|[1-5][0-9]|6[0-4]) ' "$dataset/qsgw_vxc_scf.manifest")" -eq 64
grep -Fq '"sampling_mode": "full_bz"' "$dataset/qsgw_input_contract.summary.json"
grep -Fq '"n_scf_kpoints": 64' "$dataset/qsgw_input_contract.summary.json"
grep -Fq '"vxc_matrix_basis": "ks_state"' "$dataset/qsgw_input_contract.summary.json"
grep -Fq '"vxc_matrix_gauge": "mf0_state"' "$dataset/qsgw_input_contract.summary.json"
grep -Fq '"vxc_basis_transform": "none"' "$dataset/qsgw_input_contract.summary.json"
for contract in qsgw_input.hartree-full.contract \
    qsgw_input.hartree-truncated.contract; do
    grep -Fqx 'hartree_update delta_density' "$dataset/$contract"
    test "$(grep -Ec '^hartree_ri_coefficients [0-9a-f]{64} Cs_data_[0-9]+\.txt$' "$dataset/$contract")" -eq "$rank_count"
done
test "$(grep -Ec '^hartree_coulomb [0-9a-f]{64} coulomb_mat_[0-9]+\.txt$' "$dataset/qsgw_input.hartree-full.contract")" -eq "$rank_count"
test "$(grep -Ec '^hartree_coulomb [0-9a-f]{64} coulomb_cut_[0-9]+\.txt$' "$dataset/qsgw_input.hartree-truncated.contract")" -eq "$rank_count"
test -z "$(find "$root" -type l -print -quit)"

(
    cd "$root"
    find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        > DATASET_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

python3 -B - "$provenance/SOURCE_ARTIFACT_SHA256SUMS.txt" \
    "$root/DATASET_SHA256SUMS.txt" \
    > "$provenance/SOURCE_COPY_VERIFICATION.txt" <<'PY'
import pathlib
import sys

def read_manifest(path):
    result = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[pathlib.PurePosixPath(name).name] = digest
    return result

source = read_manifest(sys.argv[1])
dataset = read_manifest(sys.argv[2])
missing = sorted(set(source) - set(dataset))
mismatched = sorted(name for name in source if dataset.get(name) != source[name])
assert not missing, missing
assert not mismatched, mismatched
print(f"PASS: {len(source)} copied source artifacts are SHA256-identical")
PY

producer_provenance_sha=$(sha256sum "$producer/RUN_PROVENANCE.txt" | awk '{print $1}')
producer_output_sha=$(sha256sum "$producer/OUTPUT_VALIDATION.json" | awk '{print $1}')
source_manifest_sha=$(sha256sum "$provenance/SOURCE_ARTIFACT_SHA256SUMS.txt" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
base_contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
full_contract_sha=$(sha256sum "$dataset/qsgw_input.hartree-full.contract" | awk '{print $1}')
truncated_contract_sha=$(sha256sum "$dataset/qsgw_input.hartree-truncated.contract" | awk '{print $1}')
vxc_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
producer_job_id=$(awk -F= '$1 == "slurm_job_id" {print $2}' "$producer/RUN_PROVENANCE.txt")
test -n "$producer_job_id"
cat > "$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_fullbz_hartree_bundle_v1
acceptance_scope=frozen_producer_native_headwing_off_fullbz_grid_hartree_inputs
runner_sha256=$runner_sha256
producer=$producer
producer_slurm_job_id=$producer_job_id
producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
producer_binary_sha256=a2676c36e318da831339cfb147c6027f3b7aed0b40925fa4a6086a947a403b17
producer_provenance_sha256=$producer_provenance_sha
producer_output_validation_sha256=$producer_output_sha
producer_reader_version=0
reader_version=0
dataset_basis_metadata=derived_once_from_frozen_orb_abfs_headers
scf_charge_restart=frozen_in_dataset
vxc_producer_schema=vxck_one_based
vxc_dataset_schema=producer_native_vxck_only
vxc_matrix_basis=ks_state
vxc_matrix_gauge=mf0_state
vxc_basis_transform=none
derived_vxcs_aliases=none
copy_mode=physical_copy_reflink_allowed_no_symlinks
grid=4x4x4
scf_kpoints=64
full_bz_kpoints=64
use_shrink_abfs=false
symmetry=-1_full_bz
headwing=off_fail_fast
hartree=contract_only_no_numerical_acceptance
hartree_contracts=full,truncated
band_update=off
source_artifact_manifest_sha256=$source_manifest_sha
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_input_contract_sha256=$base_contract_sha
qsgw_input_hartree_full_contract_sha256=$full_contract_sha
qsgw_input_hartree_truncated_contract_sha256=$truncated_contract_sha
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
echo PINNED_ABACUS_SI_K444_FULLBZ_HARTREE_BUNDLE_V1=PASS
cat "$root/PROVENANCE.txt"
