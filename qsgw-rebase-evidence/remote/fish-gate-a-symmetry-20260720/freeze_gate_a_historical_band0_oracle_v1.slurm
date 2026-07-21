#!/usr/bin/env bash
# Freeze the native outputs of the successful July-4 qsgw_band0 run. This is
# output-oracle preservation only; it does not rerun LibRPA.
#SBATCH --job-name=qsgw-a0-old-oracle
#SBATCH --partition=48cp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-historical-band0-oracle-submit-20260720-v1/slurm-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-historical-band0-oracle-submit-20260720-v1/slurm-%j.err

set -euo pipefail

: "${RUNNER_SHA256:?RUNNER_SHA256 must identify this exact runner}"

submit_root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-historical-band0-oracle-submit-20260720-v1
source_run=/ssd/work/df_iopcas_bhj/qsgw-newarch-runs/si-k444-qsgw-old-exactinput-wcfq-20260704-014854/librpa
root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-gate-a-historical-band0-oracle-20260720-v1-${SLURM_JOB_ID}
oracle=$root/oracle
provenance=$root/provenance
archive=$root.tar.gz

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
test -d "$source_run/librpa.d/qsgw_checkpoints/iter_00001"
test -f "$source_run/logs/old-exactinput-wcfq-2212829.out"
test -f "$source_run/logs/old-exactinput-wcfq-2212829.err"
test ! -e "$root"
test ! -e "$archive"
test ! -e "$archive.sha256"
mkdir -p "$oracle/logs" "$provenance"
printf '%s\n' "$RUNNER_SHA256" >"$provenance/runner-sha256.txt"

flat_files=(
  librpa.in
  homo_lumo_vs_iterations.dat
  KS_band_spin_1_1.dat
  EXX_band_spin_1_1.dat
  QSGW_band_spin_1_1.dat
  hrs1_nao_qsgw_iter_0001.csr
  librpa_para_nprocs_1_myid_0.out
)
for file in "${flat_files[@]}"; do
  test -f "$source_run/$file"
  test ! -L "$source_run/$file"
done
test "$(find "$source_run/librpa.d" -type f | wc -l)" -eq 27
test "$(find "$source_run/librpa.d/qsgw_checkpoints/iter_00001" \
  -maxdepth 1 -name 'H0_GW_spin_01_k_*.bin' -type f | wc -l)" -eq 8
test "$(find "$source_run/librpa.d" -maxdepth 1 \
  -name 'SigcRF_ispin_00_s_00_iomega_*_myid_00000.dat' -type f | wc -l)" -eq 16
test -z "$(find "$source_run/librpa.d" -type l -print -quit)"

(
  cd "$source_run"
  {
    printf '%s\0' \
      librpa.in \
      homo_lumo_vs_iterations.dat \
      KS_band_spin_1_1.dat \
      EXX_band_spin_1_1.dat \
      QSGW_band_spin_1_1.dat \
      hrs1_nao_qsgw_iter_0001.csr \
      librpa_para_nprocs_1_myid_0.out \
      logs/old-exactinput-wcfq-2212829.out \
      logs/old-exactinput-wcfq-2212829.err
    find librpa.d -type f -print0
  } | LC_ALL=C sort -z | xargs -0 sha256sum
) >"$provenance/SOURCE_SHA256SUMS.txt"
test "$(wc -l <"$provenance/SOURCE_SHA256SUMS.txt")" -eq 36
(
  cd "$source_run"
  sha256sum --check --quiet "$provenance/SOURCE_SHA256SUMS.txt"
)

for file in "${flat_files[@]}"; do
  cp -a "$source_run/$file" "$oracle/$file"
done
cp -a "$source_run/librpa.d" "$oracle/librpa.d"
cp -a "$source_run/logs/old-exactinput-wcfq-2212829.out" "$oracle/logs/"
cp -a "$source_run/logs/old-exactinput-wcfq-2212829.err" "$oracle/logs/"
test "$(find "$oracle" -type f | wc -l)" -eq 36
test -z "$(find "$oracle" -type l -print -quit)"

(
  cd "$oracle"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$root/ORACLE_SHA256SUMS.txt"
  sha256sum --check --quiet "$root/ORACLE_SHA256SUMS.txt"
)

source_manifest_sha=$(sha256sum "$provenance/SOURCE_SHA256SUMS.txt" | awk '{print $1}')
oracle_manifest_sha=$(sha256sum "$root/ORACLE_SHA256SUMS.txt" | awk '{print $1}')
cat >"$root/PROVENANCE.txt" <<EOF
gate=gate_a0_historical_qsgw_band0_native_output_oracle_v1
acceptance_scope=native_output_preservation_only
source_host=$(hostname -f 2>/dev/null || hostname)
source_run=$source_run
historical_job_id=2212829
historical_job_state=successful
historical_task=qsgw_band0
historical_max_iter=1
historical_nfreq=16
historical_symmetry=on_ibz_8_to_full_bz_64
historical_headwing=on_option3_pyatb
historical_hartree=off
historical_h_qsgw_cut=mode2_keep10_shift20Ha
historical_initial_homo_ev=6.28506
historical_initial_lumo_ev=6.97617
historical_iteration1_homo_ev=5.9827
historical_iteration1_lumo_ev=7.43882
historical_iteration1_reported_gap_ev=1.27797
historical_source_checkout_commit=8476213f66c68efb43404713eacbd04966820f26
historical_source_worktree_state=dirty_not_frozen_at_execution
historical_executable_path=/data/home/df_iopcas_bhj/software-stack/src/LibRPA_codex/build/chi0_main.exe
historical_executable_sha256_at_execution=unknown_not_frozen_before_later_mutation
native_output_file_count=36
h0_gw_checkpoint_count=8
sigcrf_file_count=16
source_manifest_sha256=$source_manifest_sha
oracle_manifest_sha256=$oracle_manifest_sha
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

tar -C "$(dirname "$root")" -czf "$archive" "$(basename "$root")"
sha256sum "$archive" >"$archive.sha256"

echo GATE_A_HISTORICAL_BAND0_ORACLE_FREEZE_V1=PASS
cat "$root/PROVENANCE.txt"
cat "$archive.sha256"
