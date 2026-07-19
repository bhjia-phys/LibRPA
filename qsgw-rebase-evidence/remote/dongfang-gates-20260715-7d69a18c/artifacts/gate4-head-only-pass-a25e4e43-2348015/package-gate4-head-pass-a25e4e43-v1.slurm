#!/usr/bin/env bash
#SBATCH --job-name=qsgw-head-pass-pack
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-pass-pack-a25e4e43-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/head-pass-pack-a25e4e43-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
source_job=2347926
source_run=$root/runs/gate4-head-only-limit-live-a25e4e43-$source_job
target=$root/evidence/gate4-head-only-pass-a25e4e43-$SLURM_JOB_ID
tool_dir=$target/tools

expected_source_script_sha=3bf8531fe86f601c12281f4aa5c5cc64f3b49bdaf02f570181321c68d47c3a86
expected_source_manifest_sha=1d53d67c15786aae74e2f4bf190ca579827971ee81c5ea8395e5db92938a4334
expected_source_provenance_sha=952fa1e4d09c00b6f0a1fd5d1fb91da492c7891877440dbb1bbf49f1c64161fe
expected_executable_sha=2072fc09a44a5f246513be152a06832dccf777da04b3dd8911b70e0f9a31591e
expected_strict_sigc_sha=041b5312330c54ac101d3cfa44abaad6bae2bfac456d0b72d2205645842bbe2f
expected_pair_sha=8f4540c7c9d151e8bcd94d5abc152842d09551118d33cb106c4c74ddf8857298
expected_live_sha=8b63bff8dd3f06a05717635f124173682192033ea0480554e250542acfc6cef9

test ! -e "$target"
test -e "$source_run/COMPLETE"
test "$(sha256sum "$source_run/gate4-head-only-limit-live-a25e4e43-v1.slurm" | awk '{print $1}')" = \
  "$expected_source_script_sha"
test "$(sha256sum "$source_run/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_source_manifest_sha"
test "$(sha256sum "$source_run/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_source_provenance_sha"
test "$(sha256sum "$root/build-candidate-a25e4e43-v1/chi0_main.exe" | awk '{print $1}')" = \
  "$expected_executable_sha"
test "$(sha256sum "$source_run/head-only-iter1-vs-g0w0-sigc.json" | awk '{print $1}')" = \
  "$expected_strict_sigc_sha"
test "$(sha256sum "$source_run/head-mode-pair-comparison.json" | awk '{print $1}')" = \
  "$expected_pair_sha"
test "$(sha256sum "$source_run/head-live-validation.json" | awk '{print $1}')" = \
  "$expected_live_sha"
grep -Fq '"passed": true' "$source_run/head-only-iter1-vs-g0w0-sigc.json"
grep -Fq '"passed": true' "$source_run/head-mode-pair-comparison.json"
grep -Fq '"passed": true' "$source_run/head-live-validation.json"
(
  cd "$source_run"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

mkdir -p "$tool_dir" "$target/disabled" "$target/head-only" "$target/g0w0-head-only"
cp "$0" "$target/package-gate4-head-pass-a25e4e43-v1.slurm"
cp "$source_run/gate4-head-only-limit-live-a25e4e43-v1.slurm" "$target/"
cp "$source_run/OUTPUT_SHA256SUMS.txt" "$target/SOURCE_RAW_SHA256SUMS.txt"
cp "$source_run/PROVENANCE.txt" "$target/source-PROVENANCE.txt"
cp "$source_run/CONTROLLED_PAIR.json" "$target/"
cp "$source_run/CONTROLLED_PAIR.canonical.json" "$target/"
cp "$source_run/candidate-build-SHA256SUMS.txt" "$target/"
cp "$source_run/candidate-build-provenance.txt" "$target/"
cp "$source_run/DATASET_SHA256SUMS.txt" "$target/"
cp "$source_run/input-OUTPUT_SHA256SUMS.txt" "$target/"
cp "$source_run/input-PROVENANCE.txt" "$target/"
cp "$source_run/HEAD_MODE_VARIANTS.tsv" "$target/"
cp "$source_run/HEADWING_READER_SHA256SUMS.txt" "$target/"
cp "$source_run/qsgw_input.disabled.contract" "$target/"
cp "$source_run/qsgw_input.head-only.contract" "$target/"
cp "$source_run/input-source-band-out-comparison.json" "$target/"
cp "$source_run/input-synchronized-band-out-comparison.json" "$target/"
cp "$source_run"/*-initial-state.json "$target/"
cp "$source_run"/*-closure.json "$target/"
cp "$source_run"/*-fixed-basis.json "$target/"
cp "$source_run/head-only-iter1-vs-g0w0-sigc.json" "$target/"
cp "$source_run/head-mode-pair-comparison.json" "$target/"
cp "$source_run/head-live-validation.json" "$target/"
cp "$source_run"/*-comparator.stdout "$target/"
cp "$source_run/head-live-validator.stdout" "$target/"
cp "$source_run"/*-unit-test.stderr "$target/"
cp "$source_run"/*-unit-test.stdout "$target/"
cp "$source_run/tools"/*.py "$tool_dir/"

for mode in disabled head-only; do
  cp "$source_run/$mode/librpa.in" "$target/$mode/"
  cp "$source_run/$mode/librpa.stdout" "$target/$mode/"
  cp "$source_run/$mode/librpa.stderr" "$target/$mode/"
  cp "$source_run/$mode/qsgw_iterations.dat" "$target/$mode/"
  cp "$source_run/$mode/qsgw_eigenvalues.dat" "$target/$mode/"
done
cp "$source_run/g0w0-head-only/librpa.in" "$target/g0w0-head-only/"
cp "$source_run/g0w0-head-only/librpa.stdout" "$target/g0w0-head-only/"
cp "$source_run/g0w0-head-only/librpa.stderr" "$target/g0w0-head-only/"

sacct -j "$source_job" \
  --format=JobID,JobName%28,Partition,State,ExitCode,Elapsed,Start,End \
  -n -P >"$target/source-job.sacct"

cat >"$target/PROVENANCE.txt" <<EOF
analysis=gate4_head_only_pass_packet
acceptance_gate=true
gate_status=PASS
candidate_commit=a25e4e4398d96bccdf44df00bfaad31c5e6035d6
candidate_executable_sha256=$expected_executable_sha
source_gate4_job=$source_job
source_gate4_script_sha256=$expected_source_script_sha
source_raw_manifest_sha256=$expected_source_manifest_sha
source_provenance_sha256=$expected_source_provenance_sha
strict_sigc_comparison=PASS
strict_sigc_max_abs_difference_ha=7.017534962512885e-13
strict_sigc_max_abs_tolerance_ha=1e-10
strict_sigc_relative_frobenius_difference=5.407528165878914e-14
strict_sigc_relative_frobenius_tolerance=1e-10
head_initial_replay=PASS
head_live_update=PASS
velocity_rotation_formula=U_dagger_v0_U
velocity_rotation_relative_residual=2.0965105601348375e-16
fixed_basis_validation=PASS
closure_validation=PASS
product_source_changed=false
execution_surface=dongfang_sbatch_compute_node
slurm_job_id=$SLURM_JOB_ID
slurm_node_list=$SLURM_JOB_NODELIST
slurm_partition=${SLURM_JOB_PARTITION:-unknown}
slurm_job_gpus=${SLURM_JOB_GPUS:-unknown}
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cat >"$target/GATE_STATUS.json" <<EOF
{
  "gate": "gate4a-head-only-g0w0-limit-live-update",
  "passed": true,
  "max_abs_difference_ha": 7.017534962512885e-13,
  "max_abs_tolerance_ha": 1e-10,
  "relative_frobenius_difference": 5.407528165878914e-14,
  "relative_frobenius_tolerance": 1e-10,
  "head_initial_replay_passed": true,
  "head_live_update_passed": true
}
EOF

(
  cd "$target"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$target/COMPLETE"
printf 'QSGW_GATE4_HEAD_PASS_PACKET=PASS\n'
