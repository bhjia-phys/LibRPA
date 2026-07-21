#!/usr/bin/env bash
#SBATCH --job-name=si444-sym-bundle
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v2/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v2/slurm-%j.err

set -Eeuo pipefail

producer=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-20260720-v1
postcheck=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-postcheck-20260720-v4
source_run="$producer/run"
source_out="$source_run/OUT.ABACUS"
submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-submit-20260720-v2
tools_dir="$submit_root/tools"
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v2
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

test -e "$producer/FAILED"
test ! -e "$producer/COMPLETE"
test -e "$postcheck/COMPLETE"
test ! -e "$postcheck/FAILED"
test -f "$postcheck/OUTPUT_VALIDATION.json"
test -f "$postcheck/POSTCHECK_ARTIFACT_SHA256SUMS.txt"
test -f "$postcheck/OBSERVER_SHA256SUMS.txt"
test -f "$postcheck/SOURCE_FAILED_SHA256SUMS.txt"
test -f "$postcheck/POSTCHECK_PROVENANCE.txt"
test ! -e "$root"
test -f "$tools_dir/generate_abacus_basis_metadata_v1.py"
test -f "$tools_dir/prepare_abacus_qsgw_ibz_contract_v2.py"
test -f "$tools_dir/audit_native_vxck_reader_compatibility_v1.py"
test -f "$tools_dir/native_vxck_reader_compatibility_v1.json"
(
    cd "$postcheck"
    sha256sum --check --quiet POSTCHECK_ARTIFACT_SHA256SUMS.txt
    sha256sum --check --quiet OBSERVER_SHA256SUMS.txt
    sha256sum --check --quiet SOURCE_FAILED_SHA256SUMS.txt
)
grep -Fqx 'source_numeric_stage=completed' "$postcheck/POSTCHECK_PROVENANCE.txt"
grep -Fqx 'source_terminal_marker=FAILED_observer_only' "$postcheck/POSTCHECK_PROVENANCE.txt"
python3 -B - "$postcheck/OUTPUT_VALIDATION.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
assert report["status"] == "PASS"
assert report["reader_version"] == 0
assert report["bz_sampling"]["grid"] == [4, 4, 4]
assert report["bz_sampling"]["n_scf"] == 8
assert report["structure"]["n_symops"] == 48
assert report["band"]["n_bands"] == 44
assert report["band"]["n_basis"] == 44
assert report["vxc_matrix_schema"]["validated_matrix_count"] == 8
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
assert report["numerical_runtime_parity_pending"] is True
PY

mkdir -p "$dataset" "$provenance/producer-inputs-v2" "$provenance/assets"
cp --reflink=auto "$producer/RUN_PROVENANCE.txt" "$provenance/producer-RUN_PROVENANCE.txt"
cp --reflink=auto "$producer/FAILED" "$provenance/producer-FAILED"
cp --reflink=auto "$postcheck/OUTPUT_VALIDATION.json" "$provenance/postcheck-OUTPUT_VALIDATION.json"
cp --reflink=auto "$postcheck/POSTCHECK_PROVENANCE.txt" "$provenance/"
cp --reflink=auto "$postcheck/OBSERVER_SHA256SUMS.txt" "$provenance/"
cp --reflink=auto "$postcheck/SOURCE_FAILED_SHA256SUMS.txt" "$provenance/"
cp --reflink=auto "$postcheck/POSTCHECK_ARTIFACT_SHA256SUMS.txt" "$provenance/"
cp --reflink=auto "$postcheck/POSTCHECK_REPORT_ASSERTIONS.txt" "$provenance/"
cp --reflink=auto "$postcheck/COMPLETE" "$provenance/postcheck-COMPLETE"
cp --reflink=auto "$producer/producer-inputs-v2/"* "$provenance/producer-inputs-v2/"
cp --reflink=auto "$producer/assets/"* "$provenance/assets/"
cp --reflink=auto "$tools_dir/generate_abacus_basis_metadata_v1.py" "$provenance/"
cp --reflink=auto "$tools_dir/prepare_abacus_qsgw_ibz_contract_v2.py" "$provenance/"
cp --reflink=auto "$tools_dir/audit_native_vxck_reader_compatibility_v1.py" "$provenance/"
cp --reflink=auto "$tools_dir/native_vxck_reader_compatibility_v1.json" "$provenance/"
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
    for index in $(seq 0 7); do
        printf '%s/KS_eigenvector_%s.dat\n' "$source_run" "$index"
    done
    for prefix in Cs_data coulomb_cut coulomb_mat; do
        for rank in $(seq 0 3); do
            printf '%s/%s_%s.txt\n' "$source_run" "$prefix" "$rank"
        done
    done
    for name in vxc_out.dat Vxc_R_spin0.csr hrs1_nao.csr srs1_nao.csr rr.csr; do
        printf '%s/%s\n' "$source_out" "$name"
    done
    for index in $(seq 1 8); do
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
    test -f "$source_vxc"
    cp --reflink=auto "$source_vxc" "$dataset/$producer_name"
done

python3 -B "$tools_dir/generate_abacus_basis_metadata_v1.py" \
    --orb "$producer/assets/Si_gga_8au_100Ry_3s3p2d.orb" \
    --abfs "$producer/assets/Si_3s3p2d1f1g_pca1e-6.abfs" \
    --n-atoms 2 --output-dir "$dataset" \
    > "$provenance/basis-generator.stdout"

python3 -B "$tools_dir/prepare_abacus_qsgw_ibz_contract_v2.py" "$dataset" \
    > "$provenance/contract-generator.stdout"

test "$(find "$dataset" -maxdepth 1 -type f -name 'KS_eigenvector_*.dat' | wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxck*_nao.txt' | wc -l)" -eq 8
test "$(find "$dataset" -maxdepth 1 -type f -name 'vxcs*k*_nao.txt' | wc -l)" -eq 0
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
grep -Fq '"vxc_filename_schema": "vxck<one-based-k-index>_nao.txt"' "$dataset/qsgw_input_contract.summary.json"
test "$(grep -Ec '^1 [1-8] .* vxck[1-8]_nao\.txt$' "$dataset/qsgw_vxc_scf.manifest")" -eq 8
grep -Fq '"basis_per_atom": 22' "$dataset/basis_metadata_summary.json"
grep -Fq '"basis_per_atom": 397' "$dataset/basis_metadata_summary.json"
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

postcheck_report_sha=$(sha256sum "$postcheck/OUTPUT_VALIDATION.json" | awk '{print $1}')
source_manifest_sha=$(sha256sum "$provenance/SOURCE_ARTIFACT_SHA256SUMS.txt" | awk '{print $1}')
reader_compatibility_sha=$(sha256sum "$provenance/native_vxck_reader_compatibility_v1.json" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
vxc_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_scf.manifest" | awk '{print $1}')
cat > "$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_symmetry_bundle_v2
acceptance_scope=frozen_producer_native_headwing_off_hartree_off_grid_input
producer=$producer
producer_slurm_job_id=2381071
producer_terminal_marker=FAILED_observer_only_preserved
producer_numeric_stage=completed
recovery_postcheck=$postcheck
recovery_postcheck_status=COMPLETE
recovery_postcheck_report_sha256=$postcheck_report_sha
producer_reader_version=0
dataset_basis_metadata=derived_once_from_frozen_orb_abfs_headers
vxc_producer_schema=vxck_one_based
vxc_dataset_schema=producer_native_vxck_only
vxc_candidate_manifest_binding=producer_native_vxck
vxc_legacy_compatibility=pending_same_bundle_runtime_gate
vxc_reader_source_compatibility=passed
vxc_reader_source_compatibility_report_sha256=$reader_compatibility_sha
derived_vxcs_aliases=none
copy_mode=physical_copy_reflink_allowed_no_symlinks
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
use_shrink_abfs=false
symmetry=on
headwing=off
hartree=off
band_update=off
source_artifact_manifest_sha256=$source_manifest_sha
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
echo PINNED_ABACUS_SI_K444_SYMMETRY_BUNDLE_V2=PASS
cat "$root/PROVENANCE.txt"
