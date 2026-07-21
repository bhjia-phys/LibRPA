#!/usr/bin/env bash
#SBATCH --job-name=si444-sym-band-bundle
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-bundle-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-bundle-submit-20260720-v1/slurm-%j.err

set -Eeuo pipefail

submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-bundle-submit-20260720-v1
tools_dir="$submit_root/tools"
scf_bundle=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-bundle-20260720-v3
nscf_bundle=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-band-nscf-20260720-v1
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-bundle-20260720-v1
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

test ! -e "$root"
test -e "$scf_bundle/COMPLETE"
test ! -e "$scf_bundle/FAILED"
test -f "$scf_bundle/OUTPUT_SHA256SUMS.txt"
test -f "$scf_bundle/DATASET_SHA256SUMS.txt"
test -z "$(find "$scf_bundle" -type l -print -quit)"
test -z "$(find "$scf_bundle" -perm /222 -print -quit)"
test -e "$nscf_bundle/COMPLETE"
test ! -e "$nscf_bundle/FAILED"
test -f "$nscf_bundle/ARTIFACT_SHA256SUMS.txt"
test -f "$nscf_bundle/NSCF_OUTPUT_VALIDATION.json"
test -z "$(find "$nscf_bundle" -type l -print -quit)"
test -z "$(find "$nscf_bundle" -perm /222 -print -quit)"
test -f "$tools_dir/preprocess_abacus_band_for_librpa_v2.py"
test -f "$tools_dir/prepare_abacus_qsgw_band_contract_v2.py"
test -f "$tools_dir/validate_abacus_si_band_nscf_output_v1.py"

(
    cd "$scf_bundle"
    sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
    cd "$nscf_bundle"
    sha256sum --check --quiet ARTIFACT_SHA256SUMS.txt
)

mkdir -p "$dataset" "$provenance/scf" "$provenance/nscf" \
    "$provenance/tools"
while IFS= read -r -d '' path; do
    cp --reflink=auto "$path" "$dataset/$(basename "$path")"
done < <(find "$scf_bundle/dataset" -maxdepth 1 -type f -print0)

cp --reflink=auto "$scf_bundle/PROVENANCE.txt" \
    "$provenance/scf/PROVENANCE.txt"
cp --reflink=auto "$scf_bundle/DATASET_SHA256SUMS.txt" \
    "$provenance/scf/DATASET_SHA256SUMS.txt"
cp --reflink=auto "$scf_bundle/OUTPUT_SHA256SUMS.txt" \
    "$provenance/scf/OUTPUT_SHA256SUMS.txt"
cp --reflink=auto "$nscf_bundle/RUN_PROVENANCE.txt" \
    "$provenance/nscf/RUN_PROVENANCE.txt"
cp --reflink=auto "$nscf_bundle/NSCF_OUTPUT_VALIDATION.json" \
    "$provenance/nscf/NSCF_OUTPUT_VALIDATION.json"
cp --reflink=auto "$nscf_bundle/ARTIFACT_SHA256SUMS.txt" \
    "$provenance/nscf/ARTIFACT_SHA256SUMS.txt"
for name in preprocess_abacus_band_for_librpa_v2.py \
    prepare_abacus_qsgw_band_contract_v2.py \
    validate_abacus_si_band_nscf_output_v1.py; do
    cp --reflink=auto "$tools_dir/$name" "$provenance/tools/$name"
done

(
    cd "$root"
    sha256sum --check --quiet "$scf_bundle/DATASET_SHA256SUMS.txt"
)

python3 -B "$provenance/tools/preprocess_abacus_band_for_librpa_v2.py" \
    --outdir "$nscf_bundle/run/OUT.ABACUS" \
    --output-dir "$dataset" \
    --validation-report "$nscf_bundle/NSCF_OUTPUT_VALIDATION.json" \
    > "$provenance/band-preprocess.stdout"
python3 -B "$provenance/tools/prepare_abacus_qsgw_band_contract_v2.py" \
    "$dataset" > "$provenance/band-contract.stdout"

test "$(find "$dataset" -maxdepth 1 -type f -name 'band_KS_eigenvalue_k_*.txt' | wc -l)" -eq 143
test "$(find "$dataset" -maxdepth 1 -type f -name 'band_KS_eigenvector_k_*.txt' | wc -l)" -eq 143
test "$(find "$dataset" -maxdepth 1 -type f -name 'band_vxc_k_*.txt' | wc -l)" -eq 143
test "$(find "$dataset" -maxdepth 1 -type f -name 'band_vxck*_nao.txt' | wc -l)" -eq 143
grep -Fq '"qsgw_full_vxc_copy": "byte_identical"' \
    "$dataset/band_preprocess.summary.json"
grep -Fq '"band_vxc_basis": "state"' \
    "$dataset/qsgw_band_input_contract.summary.json"
test -f "$dataset/qsgw_vxc_band.manifest"
test -f "$dataset/qsgw_band_input.contract"
test -f "$dataset/qsgw_input.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.contract"
grep -Fqx 'n_band_kpoints 0' "$dataset/qsgw_input.contract"
grep -Fqx 'band_update operator_fourier' "$dataset/qsgw_band_input.contract"
grep -Fqx 'n_band_kpoints 143' "$dataset/qsgw_band_input.contract"

cat > "$root/librpa-qsgw-grid.in" <<'EOF'
task = qsgw
input_dir = ./dataset
nfreq = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
replace_w_head = false
use_pyatb = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

cat > "$root/librpa-qsgw-band.in" <<'EOF'
task = qsgw_band
input_dir = ./dataset
nfreq = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
replace_w_head = false
use_pyatb = false
qsgw_input_contract = qsgw_band_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = 1
qsgw_max_iter = 1
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

test -z "$(find "$root" -type l -print -quit)"
(
    cd "$root"
    find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        > DATASET_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

python3 -B - "$provenance/scf/DATASET_SHA256SUMS.txt" \
    "$root/DATASET_SHA256SUMS.txt" \
    > "$provenance/SCF_SOURCE_COPY_VERIFICATION.txt" <<'PY'
import pathlib
import sys


def read_manifest(path):
    records = {}
    for line in pathlib.Path(path).read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        records[name] = digest
    return records


source = read_manifest(sys.argv[1])
combined = read_manifest(sys.argv[2])
missing = sorted(set(source) - set(combined))
mismatched = sorted(name for name in source if combined.get(name) != source[name])
assert not missing, missing
assert not mismatched, mismatched
print(f"PASS: {len(source)} frozen SCF dataset files are SHA256-identical")
PY

scf_manifest_sha=$(sha256sum "$scf_bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')
nscf_manifest_sha=$(sha256sum "$nscf_bundle/ARTIFACT_SHA256SUMS.txt" | awk '{print $1}')
nscf_validation_sha=$(sha256sum "$nscf_bundle/NSCF_OUTPUT_VALIDATION.json" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
scf_contract_sha=$(sha256sum "$dataset/qsgw_input.contract" | awk '{print $1}')
band_contract_sha=$(sha256sum "$dataset/qsgw_band_input.contract" | awk '{print $1}')
band_manifest_sha=$(sha256sum "$dataset/qsgw_vxc_band.manifest" | awk '{print $1}')
cat > "$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_symmetry_band_bundle_v1
acceptance_scope=frozen_scf_grid_plus_validated_nscf_band_input
scf_bundle=$scf_bundle
scf_bundle_output_manifest_sha256=$scf_manifest_sha
nscf_bundle=$nscf_bundle
nscf_bundle_artifact_manifest_sha256=$nscf_manifest_sha
nscf_output_validation_sha256=$nscf_validation_sha
copy_mode=physical_copy_reflink_allowed_no_symlinks
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
band_kpoints=143
n_bands=44
n_basis=44
symmetry=on_grid
headwing=off_fail_fast
hartree=off
scf_band_update=off
band_update=operator_fourier
vxc_matrix_basis=ks_state
vxc_matrix_gauge=mf0_state
vxc_basis_transform=none
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_scf_contract_sha256=$scf_contract_sha
qsgw_band_contract_sha256=$band_contract_sha
qsgw_band_vxc_manifest_sha256=$band_manifest_sha
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
echo PINNED_ABACUS_SI_K444_SYMMETRY_BAND_BUNDLE_V1=PASS
cat "$root/PROVENANCE.txt"
