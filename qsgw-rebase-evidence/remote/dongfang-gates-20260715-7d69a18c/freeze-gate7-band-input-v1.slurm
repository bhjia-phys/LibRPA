#!/usr/bin/env bash
# Freeze the Gate-7 qsgw_band input: copy the frozen head-wing-synced Si
# k444 dataset (which carries all band assets) into a new immutable input
# directory, then generate the band contract and v2 band Vxc manifest with
# the TDD-tested generator. The source dataset is not modified. Only the
# two generated files are new; every other payload must stay byte-identical
# to the source manifest.
#SBATCH --job-name=qsgw-freeze-band-input
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/freeze-band-input-v1-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/freeze-band-input-v1-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_root=$root/inputs/si-k444-headwing-scf-synced-v2-2345960
source_dataset=$source_root/dataset
preparer=$root/jobs/prepare_qsgw_band_contract-v1-adf8b56b.py
preparer_test=$root/jobs/test_prepare_qsgw_band_contract-v1-1e4fab6d.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
target_root=$root/inputs/si-k444-band-v1-$SLURM_JOB_ID
target_dataset=$target_root/dataset

expected_source_provenance_sha=24d31390e092f55a2382c1e0ac5632c1e0c719df20d60986dc34754f8c7de995
expected_source_manifest_sha=0a07515a3a7eef1827ecc54b2471bf5a3afbbcfb92d9a920cbeab28a9e54bf11
expected_preparer_sha=adf8b56b57773cc0e2b57b740ea84080ed768a5439717fec1ae75ba2181502f8
expected_preparer_test_sha=1e4fab6dea4f3db0e80adda260a1ea900eed02a88427135799c3ffa97c182780
expected_n_band_kpoints=201

test ! -e "$target_root"
test -e "$source_root/COMPLETE"
test "$(sha256sum "$source_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$source_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_manifest_sha"
test "$(sha256sum "$preparer" | awk '{print $1}')" = "$expected_preparer_sha"
test "$(sha256sum "$preparer_test" | awk '{print $1}')" = \
  "$expected_preparer_test_sha"

mkdir -p "$target_root" "$target_root/tools"
cp "$0" "$target_root/freeze-gate7-band-input-v1.slurm"
cp "$preparer" "$target_root/tools/prepare_qsgw_band_contract_v1.py"
cp "$preparer_test" "$target_root/tools/test_prepare_qsgw_band_contract_v1.py"
cp "$source_root/PROVENANCE.txt" "$target_root/source-PROVENANCE.txt"
cp "$source_root/DATASET_SHA256SUMS.txt" \
  "$target_root/source-DATASET_SHA256SUMS.txt"

"$python" -B "$target_root/tools/test_prepare_qsgw_band_contract_v1.py" \
  >"$target_root/preparer-unit-test.stdout" \
  2>"$target_root/preparer-unit-test.stderr"

cp -a "$source_dataset" "$target_dataset"
test -f "$target_dataset/qsgw_input.disabled.contract"
test -f "$target_dataset/band_kpath_info"
test -d "$target_dataset/vxc_band"
test "$(find "$target_dataset" -type l | wc -l)" -eq 0
test ! -e "$target_dataset/qsgw_input.band.contract"
test ! -e "$target_dataset/qsgw_vxc_band.v2.manifest"

"$python" -B "$target_root/tools/prepare_qsgw_band_contract_v1.py" \
  --dataset "$target_dataset" \
  >"$target_root/generator-report.json" \
  2>"$target_root/generator.stderr"

test -s "$target_dataset/qsgw_input.band.contract"
test -s "$target_dataset/qsgw_vxc_band.v2.manifest"
grep -Fqx "n_band_kpoints $expected_n_band_kpoints" \
  "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'band_update operator_fourier' \
  "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'headwing_grid disabled' "$target_dataset/qsgw_input.band.contract"
grep -Fqx 'hartree_update off' "$target_dataset/qsgw_input.band.contract"
grep -Fqx '# librpa-qsgw-vxc-manifest-v2' \
  "$target_dataset/qsgw_vxc_band.v2.manifest"
grep -Fqx 'kind band' "$target_dataset/qsgw_vxc_band.v2.manifest"
test "$(grep -Ec '^band_mf0_eigenvalues ' \
  "$target_dataset/qsgw_input.band.contract")" -eq \
  "$expected_n_band_kpoints"
test "$(grep -Ec '^band_mf0_wavefunctions ' \
  "$target_dataset/qsgw_input.band.contract")" -eq \
  "$expected_n_band_kpoints"
test "$(grep -Ec '^vxc_band_manifest ' \
  "$target_dataset/qsgw_input.band.contract")" -eq 1
test "$(grep -Ec '^1 ' "$target_dataset/qsgw_vxc_band.v2.manifest")" -eq \
  "$expected_n_band_kpoints"

"$python" -B - "$target_dataset" "$expected_n_band_kpoints" \
  >"$target_root/contract-file-existence.txt" <<'PYEOF'
import hashlib, pathlib, sys
dataset = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
contract = (dataset / "qsgw_input.band.contract").read_text(encoding="ascii")
records = 0
for line in contract.splitlines():
    fields = line.split()
    if len(fields) != 3 or fields[0].startswith("#") or fields[0] == "role":
        continue
    role, digest, relative = fields
    path = dataset / relative
    assert path.is_file() and not path.is_symlink(), f"missing {relative}"
    assert sha(path) == digest, f"sha mismatch {relative}"
    records += 1
manifest = (dataset / "qsgw_vxc_band.v2.manifest").read_text(encoding="ascii")
entries = 0
for line in manifest.splitlines():
    fields = line.split()
    if len(fields) != 9 or fields[0] in {"#", "kind", "producer", "units",
                                         "basis", "gauge", "spin"}:
        continue
    _spin, _k, _x, _y, _z, _rows, _cols, digest, relative = fields
    path = dataset / relative
    assert path.is_file() and not path.is_symlink(), f"missing {relative}"
    assert sha(path) == digest, f"sha mismatch {relative}"
    entries += 1
assert entries == expected, f"manifest entries {entries} != {expected}"
print(f"contract_records_verified={records}")
print(f"manifest_entries_verified={entries}")
PYEOF

(
  cd "$target_dataset"
  find . -type f ! -name 'DATASET_SHA256SUMS.txt' -print0 | sort -z | \
    xargs -0 sha256sum >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

band_contract_sha=$(sha256sum "$target_dataset/qsgw_input.band.contract" \
  | awk '{print $1}')
band_manifest_sha=$(sha256sum "$target_dataset/qsgw_vxc_band.v2.manifest" \
  | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$target_dataset/DATASET_SHA256SUMS.txt" \
  | awk '{print $1}')

cat >"$target_root/PROVENANCE.txt" <<EOF
analysis=freeze_gate7_band_input
source_input_root=$source_root
source_dataset_manifest_sha256=$expected_source_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
preparer_sha256=$expected_preparer_sha
preparer_test_sha256=$expected_preparer_test_sha
n_band_kpoints=$expected_n_band_kpoints
band_contract_sha256=$band_contract_sha
band_vxc_manifest_sha256=$band_manifest_sha
dataset_manifest_sha256=$dataset_manifest_sha
headwing=off
hartree=off
band_update=operator_fourier
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$target_root"
  find . -maxdepth 1 -type f ! -name 'OUTPUT_SHA256SUMS.txt' \
    ! -name COMPLETE -print0 | sort -z | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target_root/COMPLETE"
echo GATE7_BAND_INPUT_FREEZE=COMPLETE
