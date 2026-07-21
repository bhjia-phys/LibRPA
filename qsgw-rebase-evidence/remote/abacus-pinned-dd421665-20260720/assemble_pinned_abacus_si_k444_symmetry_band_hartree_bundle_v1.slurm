#!/usr/bin/env bash
#SBATCH --job-name=si444-sym-band-hartree
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-hartree-bundle-submit-20260721-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-hartree-bundle-submit-20260721-v1/slurm-%j.err

set -Eeuo pipefail

submit_root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-hartree-bundle-submit-20260721-v1
tool_source=$submit_root/tools/extend_qsgw_contract_with_hartree_v1.py
tool_test_source=$submit_root/tools/test_extend_qsgw_contract_with_hartree_v1.py
source_bundle=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-bundle-20260720-v1
root=/data/home/df_iopcas_bhj/ai-runs/abacus-pinned-dd421665-si-k444-symmetry-band-hartree-bundle-20260721-v1
dataset=$root/dataset
provenance=$root/provenance

on_exit()
{
    code=$?
    if [[ $code -ne 0 && -d ${root:-/nonexistent} ]]; then
        {
            printf 'exit_code=%s\n' "$code"
            printf 'failed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
        } >"$root/FAILED"
    fi
}
trap on_exit EXIT

test ! -e "$root"
test -e "$source_bundle/COMPLETE"
test ! -e "$source_bundle/FAILED"
test -f "$source_bundle/PROVENANCE.txt"
test -f "$source_bundle/DATASET_SHA256SUMS.txt"
test -f "$source_bundle/OUTPUT_SHA256SUMS.txt"
test -z "$(find "$source_bundle" -type l -print -quit)"
test -f "$tool_source"
test -f "$tool_test_source"
(
    cd "$source_bundle"
    sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
grep -Fqx 'gate=pinned_abacus_si_k444_symmetry_band_bundle_v1' \
    "$source_bundle/PROVENANCE.txt"
grep -Fqx 'grid=4x4x4' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'scf_kpoints=8' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'full_bz_kpoints=64' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'band_kpoints=143' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'n_bands=44' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'n_basis=44' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'symmetry=on_grid' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'headwing=off_fail_fast' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'hartree=off' "$source_bundle/PROVENANCE.txt"
grep -Fqx 'band_update=operator_fourier' "$source_bundle/PROVENANCE.txt"

python3 -B "$tool_test_source" >"$submit_root/tool-test.stdout" \
    2>"$submit_root/tool-test.stderr"

mkdir -p "$dataset" "$provenance/tools"
while IFS= read -r -d '' path; do
    cp --reflink=auto "$path" "$dataset/$(basename "$path")"
done < <(find "$source_bundle/dataset" -maxdepth 1 -type f -print0)
cp --reflink=auto "$source_bundle/PROVENANCE.txt" \
    "$provenance/source-PROVENANCE.txt"
cp --reflink=auto "$source_bundle/DATASET_SHA256SUMS.txt" \
    "$provenance/source-DATASET_SHA256SUMS.txt"
cp --reflink=auto "$source_bundle/OUTPUT_SHA256SUMS.txt" \
    "$provenance/source-OUTPUT_SHA256SUMS.txt"
cp --reflink=auto "$tool_source" "$provenance/tools/"
cp --reflink=auto "$tool_test_source" "$provenance/tools/"

python3 -B "$provenance/tools/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_input.contract \
    --output-contract qsgw_input.hartree-full.contract \
    --output-summary qsgw_input.hartree-full.summary.json \
    --hartree-coulomb full \
    >"$provenance/grid-full-contract.stdout"
python3 -B "$provenance/tools/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_input.contract \
    --output-contract qsgw_input.hartree-truncated.contract \
    --output-summary qsgw_input.hartree-truncated.summary.json \
    --hartree-coulomb truncated \
    >"$provenance/grid-truncated-contract.stdout"
python3 -B "$provenance/tools/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_band_input.contract \
    --output-contract qsgw_band_input.hartree-full.contract \
    --output-summary qsgw_band_input.hartree-full.summary.json \
    --hartree-coulomb full \
    >"$provenance/band-full-contract.stdout"
python3 -B "$provenance/tools/extend_qsgw_contract_with_hartree_v1.py" \
    "$dataset" qsgw_band_input.contract \
    --output-contract qsgw_band_input.hartree-truncated.contract \
    --output-summary qsgw_band_input.hartree-truncated.summary.json \
    --hartree-coulomb truncated \
    >"$provenance/band-truncated-contract.stdout"

for contract in \
    qsgw_input.hartree-full.contract \
    qsgw_input.hartree-truncated.contract \
    qsgw_band_input.hartree-full.contract \
    qsgw_band_input.hartree-truncated.contract; do
    test -f "$dataset/$contract"
    grep -Fqx 'hartree_update delta_density' "$dataset/$contract"
    test "$(grep -Ec '^hartree_ri_coefficients [0-9a-f]{64} Cs_data_[0-9]+\.txt$' \
        "$dataset/$contract")" -eq 4
    test "$(grep -Ec '^hartree_aux_basis [0-9a-f]{64} basis_aux_out$' \
        "$dataset/$contract")" -eq 1
done
for contract in qsgw_input.hartree-full.contract \
    qsgw_band_input.hartree-full.contract; do
    test "$(grep -Ec '^hartree_coulomb [0-9a-f]{64} coulomb_mat_[0-9]+\.txt$' \
        "$dataset/$contract")" -eq 4
done
for contract in qsgw_input.hartree-truncated.contract \
    qsgw_band_input.hartree-truncated.contract; do
    test "$(grep -Ec '^hartree_coulomb [0-9a-f]{64} coulomb_cut_[0-9]+\.txt$' \
        "$dataset/$contract")" -eq 4
done
grep -Fqx 'band_update off' "$dataset/qsgw_input.hartree-full.contract"
grep -Fqx 'band_update off' "$dataset/qsgw_input.hartree-truncated.contract"
grep -Fqx 'band_update operator_fourier' \
    "$dataset/qsgw_band_input.hartree-full.contract"
grep -Fqx 'band_update operator_fourier' \
    "$dataset/qsgw_band_input.hartree-truncated.contract"

cat >"$root/librpa-qsgw-hartree-grid.in" <<'EOF'
task = qsgw
input_dir = ./dataset
constants_choice = internal
nfreq = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
replace_w_head = false
option_dielect_func = 0
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.hartree-full.contract
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = true
qsgw_hartree_coulomb = full
qsgw_hartree_normalization = weighted_occupations
EOF

cat >"$root/librpa-qsgw-hartree-band.in" <<'EOF'
task = qsgw_band
input_dir = ./dataset
constants_choice = internal
nfreq = 6
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
replace_w_head = false
option_dielect_func = 0
use_abacus_exx_symmetry = true
use_abacus_gw_symmetry = true
use_symmetry_exx = true
use_symmetry_gw = true
use_symmetry_rpa = true
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_band_input.hartree-full.contract
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
qsgw_min_iter = 2
qsgw_max_iter = 2
qsgw_write_iteration_matrices = true
qsgw_update_hartree = true
qsgw_hartree_coulomb = full
qsgw_hartree_normalization = weighted_occupations
qsgw_export_hamiltonian_for_pyatb = true
qsgw_band0_unoccupied_keep = 10
qsgw_band0_cut_mode = 0
qsgw_band0_cut_shift_ha = 20.0
EOF

test -z "$(find "$root" -type l -print -quit)"
(
    cd "$root"
    find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        >DATASET_SHA256SUMS.txt
    sha256sum --check --quiet DATASET_SHA256SUMS.txt
)
python3 -B - "$source_bundle/DATASET_SHA256SUMS.txt" \
    "$root/DATASET_SHA256SUMS.txt" \
    >"$provenance/SOURCE_COPY_VERIFICATION.txt" <<'PY'
import pathlib
import sys


def read_manifest(path):
    result = {}
    for line in pathlib.Path(path).read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


source = read_manifest(sys.argv[1])
extended = read_manifest(sys.argv[2])
missing = sorted(set(source) - set(extended))
mismatched = sorted(name for name in source if extended.get(name) != source[name])
assert not missing, missing
assert not mismatched, mismatched
print(f"PASS: {len(source)} source dataset files are SHA256-identical")
PY

source_provenance_sha=$(sha256sum "$source_bundle/PROVENANCE.txt" | awk '{print $1}')
source_output_sha=$(sha256sum "$source_bundle/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')
source_dataset_sha=$(sha256sum "$source_bundle/DATASET_SHA256SUMS.txt" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
grid_full_sha=$(sha256sum "$dataset/qsgw_input.hartree-full.contract" | awk '{print $1}')
grid_truncated_sha=$(sha256sum "$dataset/qsgw_input.hartree-truncated.contract" | awk '{print $1}')
band_full_sha=$(sha256sum "$dataset/qsgw_band_input.hartree-full.contract" | awk '{print $1}')
band_truncated_sha=$(sha256sum "$dataset/qsgw_band_input.hartree-truncated.contract" | awk '{print $1}')
tool_sha=$(sha256sum "$provenance/tools/extend_qsgw_contract_with_hartree_v1.py" | awk '{print $1}')
cat >"$root/PROVENANCE.txt" <<EOF
gate=pinned_abacus_si_k444_symmetry_band_hartree_bundle_v1
acceptance_scope=contract_extension_only_no_qsgw_numerical_acceptance
source_bundle=$source_bundle
source_bundle_provenance_sha256=$source_provenance_sha
source_bundle_output_manifest_sha256=$source_output_sha
source_dataset_manifest_sha256=$source_dataset_sha
copy_mode=physical_copy_reflink_allowed_no_symlinks
producer_commit=dd4216653386d32f79e3219f3ea5dd2d229c1c5a
grid=4x4x4
scf_kpoints=8
full_bz_kpoints=64
band_kpoints=143
n_bands=44
n_basis=44
symmetry=on_grid
headwing=off_fail_fast
hartree_contracts=full_and_truncated
hartree_normalizations_runtime_selectable=weighted_occupations,legacy_extra_inverse_nk
band_update=operator_fourier
dataset_manifest_sha256=$dataset_manifest_sha
qsgw_grid_hartree_full_contract_sha256=$grid_full_sha
qsgw_grid_hartree_truncated_contract_sha256=$grid_truncated_sha
qsgw_band_hartree_full_contract_sha256=$band_full_sha
qsgw_band_hartree_truncated_contract_sha256=$band_truncated_sha
contract_extension_tool_sha256=$tool_sha
slurm_job_id=${SLURM_JOB_ID:-unset}
created_host=$(hostname -f 2>/dev/null || hostname)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
    cd "$root"
    find . -type f ! -name OUTPUT_SHA256SUMS.txt ! -name COMPLETE \
        ! -name FAILED -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
        >OUTPUT_SHA256SUMS.txt
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
echo PINNED_ABACUS_SI_K444_SYMMETRY_BAND_HARTREE_BUNDLE_V1=PASS
cat "$root/PROVENANCE.txt"
