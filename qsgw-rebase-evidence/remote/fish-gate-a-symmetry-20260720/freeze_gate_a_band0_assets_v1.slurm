#!/usr/bin/env bash
# Freeze only the producer-owned band-path and PyATB inputs used by the
# successful historical qsgw_band0 run. Grid inputs remain in the independent
# Gate-A symmetry bundle and are combined with these assets on fish.
#SBATCH --job-name=qsgw-a0-band0-assets
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-band0-assets-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-band0-assets-submit-20260720-v1/slurm-%j.err

set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

submit_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-band0-assets-submit-20260720-v1
source_root=/data/home/df_iopcas_bhj/ai-runs/si-qsgw-k444-headwing-kconv-20260517-203305/base_iter1
historical_run=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-band0-assets-20260720-v1-${SLURM_JOB_ID}
dataset=$root/dataset
provenance=$root/provenance
archive=$root.tar.gz
expected_band_kpoints=201
expected_band_basis=44
expected_band_states=44
expected_band_spins=1
expected_eigenvector_bytes=$((expected_band_basis * expected_band_states * 16))
expected_pyatb_files=67

record_failure() {
  local rc=$?
  trap - ERR
  if [[ -d ${root:-/nonexistent} ]]; then
    printf 'failed_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >"$root/FAILED"
  fi
  exit "$rc"
}
trap record_failure ERR

test -d "$submit_root"
test -d "$source_root"
test -d "$historical_run"
test ! -e "$root"
test ! -e "$archive"
test ! -e "$archive.sha256"
mkdir -p "$dataset/vxc_band" "$dataset/pyatb_librpa_df" "$provenance"
printf '%s\n' "$RUNNER_SHA256" >"$provenance/runner-sha256.txt"

read -r n_basis n_states n_spins n_kpoints <"$source_root/band_kpath_info"
test "$n_basis" -eq "$expected_band_basis"
test "$n_states" -eq "$expected_band_states"
test "$n_spins" -eq "$expected_band_spins"
test "$n_kpoints" -eq "$expected_band_kpoints"
test "$(awk 'NF == 3 {n++} END {print n + 0}' "$source_root/band_kpath_info")" \
  -eq "$expected_band_kpoints"

for pattern in \
  'band_KS_eigenvalue_k_*.txt' \
  'band_KS_eigenvector_k_*.txt' \
  'band_vxc_k_*.txt' \
  'band_vxck*_nao.txt' \
  'band_sk*_nao.txt'; do
  test "$(find "$source_root" -maxdepth 1 -type f -name "$pattern" | wc -l)" \
    -eq "$expected_band_kpoints"
  test "$(find "$historical_run" -maxdepth 1 -type l -name "$pattern" | wc -l)" \
    -eq "$expected_band_kpoints"
done

test -L "$historical_run/band_kpath_info"
test "$(readlink -f "$historical_run/band_kpath_info")" = \
  "$(readlink -f "$source_root/band_kpath_info")"
test -L "$historical_run/pyatb_librpa_df"
test "$(readlink -f "$historical_run/pyatb_librpa_df")" = \
  "$(readlink -f "$source_root/pyatb_librpa_df")"

for kpoint in $(seq 1 "$expected_band_kpoints"); do
  padded=$(printf '%05d' "$kpoint")
  for name in \
    "band_KS_eigenvalue_k_${padded}.txt" \
    "band_KS_eigenvector_k_${padded}.txt" \
    "band_vxc_k_${padded}.txt" \
    "band_vxck${kpoint}_nao.txt" \
    "band_sk${kpoint}_nao.txt"; do
    test -f "$source_root/$name"
    test -L "$historical_run/$name"
    test "$(readlink -f "$historical_run/$name")" = \
      "$(readlink -f "$source_root/$name")"
  done
  test "$(stat -c %s "$source_root/band_KS_eigenvector_k_${padded}.txt")" \
    -eq "$expected_eigenvector_bytes"
done

test "$(find -L "$source_root/pyatb_librpa_df" -type f | wc -l)" \
  -eq "$expected_pyatb_files"
test -z "$(find -L "$source_root/pyatb_librpa_df" ! -type d ! -type f -print -quit)"

(
  cd "$source_root"
  {
    printf '%s\0' ./band_kpath_info
    find . -maxdepth 1 -type f \
      \( -name 'band_KS_eigenvalue_k_*.txt' \
      -o -name 'band_KS_eigenvector_k_*.txt' \
      -o -name 'band_vxc_k_*.txt' \
      -o -name 'band_vxck*_nao.txt' \
      -o -name 'band_sk*_nao.txt' \) -print0
    find -L pyatb_librpa_df -type f -print0
  } | LC_ALL=C sort -z | xargs -0 sha256sum
) >"$provenance/SOURCE_SHA256SUMS.txt"
test "$(wc -l <"$provenance/SOURCE_SHA256SUMS.txt")" -eq 1073
(
  cd "$source_root"
  sha256sum --check --quiet "$provenance/SOURCE_SHA256SUMS.txt"
)

install -m 0644 "$source_root/band_kpath_info" "$dataset/band_kpath_info"
for kpoint in $(seq 1 "$expected_band_kpoints"); do
  padded=$(printf '%05d' "$kpoint")
  for name in \
    "band_KS_eigenvalue_k_${padded}.txt" \
    "band_KS_eigenvector_k_${padded}.txt" \
    "band_vxc_k_${padded}.txt" \
    "band_vxck${kpoint}_nao.txt" \
    "band_sk${kpoint}_nao.txt"; do
    install -m 0644 "$source_root/$name" "$dataset/$name"
  done
  install -m 0644 "$source_root/band_vxck${kpoint}_nao.txt" \
    "$dataset/vxc_band/vxck${kpoint}s1_nao.txt"
done
cp -aL "$source_root/pyatb_librpa_df/." "$dataset/pyatb_librpa_df/"
find "$dataset/pyatb_librpa_df" -type f -exec chmod 0644 {} +

test "$(find "$dataset" -type f | wc -l)" -eq 1274
test "$(find "$dataset" -maxdepth 1 -type f | wc -l)" -eq 1006
test "$(find "$dataset/vxc_band" -maxdepth 1 -type f | wc -l)" \
  -eq "$expected_band_kpoints"
test "$(find "$dataset/pyatb_librpa_df" -type f | wc -l)" \
  -eq "$expected_pyatb_files"
test -z "$(find "$root" -type l -print -quit)"

for kpoint in $(seq 1 "$expected_band_kpoints"); do
  padded=$(printf '%05d' "$kpoint")
  cmp "$source_root/band_KS_eigenvalue_k_${padded}.txt" \
    "$dataset/band_KS_eigenvalue_k_${padded}.txt"
  cmp "$source_root/band_KS_eigenvector_k_${padded}.txt" \
    "$dataset/band_KS_eigenvector_k_${padded}.txt"
  cmp "$source_root/band_vxc_k_${padded}.txt" \
    "$dataset/band_vxc_k_${padded}.txt"
  cmp "$source_root/band_vxck${kpoint}_nao.txt" \
    "$dataset/band_vxck${kpoint}_nao.txt"
  cmp "$source_root/band_vxck${kpoint}_nao.txt" \
    "$dataset/vxc_band/vxck${kpoint}s1_nao.txt"
  cmp "$source_root/band_sk${kpoint}_nao.txt" \
    "$dataset/band_sk${kpoint}_nao.txt"
done

(
  cd "$root"
  find dataset -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >DATASET_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
)

source_manifest_sha=$(sha256sum "$provenance/SOURCE_SHA256SUMS.txt" | awk '{print $1}')
dataset_manifest_sha=$(sha256sum "$root/DATASET_SHA256SUMS.txt" | awk '{print $1}')
cat >"$root/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_qsgw_band0_asset_freeze_v1
acceptance_scope=input_reproduction_only
source_host=$(hostname -f 2>/dev/null || hostname)
source_root=$source_root
historical_successful_run=$historical_run
historical_task=qsgw_band0
historical_job_id=2212829
copy_mode=selected_physical_copy_no_symlinks
band_basis=$expected_band_basis
band_states=$expected_band_states
band_spins=$expected_band_spins
band_kpoints=$expected_band_kpoints
pyatb_files=$expected_pyatb_files
source_file_count=1073
dataset_file_count=1274
vxc_band_aliases=201_exact_copies_of_band_vxck
source_manifest_sha256=$source_manifest_sha
dataset_manifest_sha256=$dataset_manifest_sha
runner_sha256=$RUNNER_SHA256
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
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

test -e "$root/COMPLETE"
test ! -e "$root/FAILED"
test -z "$(find "$root" -type l -print -quit)"
test -z "$(find "$root" -perm /222 -print -quit)"
(
  cd "$root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

tar -C "$(dirname "$root")" -czf "$archive" "$(basename "$root")"
sha256sum "$archive" >"$archive.sha256"

echo GATE_A_BAND0_ASSETS_FREEZE_V1=PASS
cat "$root/PROVENANCE.txt"
cat "$archive.sha256"
