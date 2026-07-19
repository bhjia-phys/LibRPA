#!/usr/bin/env bash
#SBATCH --job-name=qsgw-gate3-frozen
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-frozen-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-frozen-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
frozen_root=$root/runs/gate3-old-new-direct-miniter10-ce1e1859-2342954
current_root=$root/runs/gate3-old-new-direct-miniter10-b2bc09d0-2343109
tool_source=$root/jobs/compare_qsgw_component_traces-v3-9f9ec9be.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/evidence/gate3-frozen-old-vs-b2bc09d0-$SLURM_JOB_ID
tool_dir=$run_root/tools

expected_frozen_output_sha=6b3c4cbe496cf0f4d2ca38c539444d3f56d48ee31ad84987cc3dbcdb51db3e4b
expected_frozen_provenance_sha=ddcb456b20a5d6dc4c48811e7421947d375d28807ef978e5f778b08efae9896c
expected_frozen_matrix_sha=eece0f666871076a61607a8c5c4c17993148c2a932c6ebdf00bd53984090f074
expected_frozen_gap_sha=0d5471827f4ec76161bc1a4af79b46eee746350401fa81d1a26b44940fa31b38
expected_current_output_sha=REPLACE_WITH_CURRENT_OUTPUT_MANIFEST_SHA
expected_current_provenance_sha=REPLACE_WITH_CURRENT_PROVENANCE_SHA
expected_tool_sha=9f9ec9beb0d08754d1f9a39cb528df9aa4000df7f3443479984ec2df6144bf89

test ! -e "$run_root"
test -e "$frozen_root/COMPLETE"
test -e "$current_root/COMPLETE"
test "$(sha256sum "$frozen_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_frozen_output_sha"
test "$(sha256sum "$frozen_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_frozen_provenance_sha"
test "$(sha256sum "$frozen_root/old/qsgw_oracle_matrices.dat" | awk '{print $1}')" = \
  "$expected_frozen_matrix_sha"
test "$(sha256sum "$frozen_root/old/homo_lumo_vs_iterations.dat" | awk '{print $1}')" = \
  "$expected_frozen_gap_sha"
test "$(sha256sum "$current_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_current_output_sha"
test "$(sha256sum "$current_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_current_provenance_sha"
test "$(sha256sum "$tool_source" | awk '{print $1}')" = \
  "$expected_tool_sha"
grep -Fq 'GATE3_OLD_NEW_DIRECT_MINITER10=PASS' \
  "$root/logs/gate3-min10-2343109.out"

mkdir -p "$tool_dir"
cp "$0" "$run_root/gate3-frozen-old-vs-b2bc09d0.slurm"
cp "$tool_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$frozen_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/frozen-OUTPUT_SHA256SUMS.txt"
cp "$frozen_root/PROVENANCE.txt" "$run_root/frozen-PROVENANCE.txt"
cp "$current_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/current-OUTPUT_SHA256SUMS.txt"
cp "$current_root/PROVENANCE.txt" "$run_root/current-PROVENANCE.txt"

(
  cd "$frozen_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
(
  cd "$current_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$frozen_root/old/qsgw_oracle_matrices.dat" \
  "$current_root/current/qsgw_matrices.dat" \
  "$run_root/frozen-old-vs-current-miniter10.json" \
  --iterations 0:10 \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_root/current/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_root/current/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/miniter10-comparator.stdout" \
  2>"$run_root/miniter10-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/frozen-old-vs-current-miniter10.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$frozen_root/old/qsgw_oracle_matrices.dat" \
  "$current_root/current/qsgw_matrices.dat" \
  "$run_root/frozen-old-vs-current-miniter5.json" \
  --iterations 0:5 \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_root/current/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_root/current/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/miniter5-comparator.stdout" \
  2>"$run_root/miniter5-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/frozen-old-vs-current-miniter5.json"

cat >"$run_root/PROVENANCE.txt" <<EOF
gate=gate3-frozen-old-vs-current
final_acceptance=true
frozen_old_job=2342954
frozen_old_root=$frozen_root
frozen_old_output_manifest_sha256=$expected_frozen_output_sha
frozen_old_provenance_sha256=$expected_frozen_provenance_sha
frozen_old_matrix_sha256=$expected_frozen_matrix_sha
frozen_old_homo_lumo_sha256=$expected_frozen_gap_sha
current_job=2343109
current_root=$current_root
current_output_manifest_sha256=$expected_current_output_sha
current_provenance_sha256=$expected_current_provenance_sha
current_matrix_sha256=$(sha256sum "$current_root/current/qsgw_matrices.dat" | awk '{print $1}')
current_eigenvalue_sha256=$(sha256sum "$current_root/current/qsgw_eigenvalues.dat" | awk '{print $1}')
current_iteration_sha256=$(sha256sum "$current_root/current/qsgw_iterations.dat" | awk '{print $1}')
comparator_sha256=$expected_tool_sha
iterations=0:5,0:10
matrix_max_abs_tolerance_ha=1e-8
matrix_relative_tolerance=1e-8
eigenvalue_tolerance_ha=1e-6
gap_tolerance_ev=1e-5
slurm_job_id=$SLURM_JOB_ID
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
  cd "$run_root"
  find . -type f ! -name OUTPUT_SHA256SUMS.txt -print0 | sort -z | \
    xargs -0 sha256sum >OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
)
touch "$run_root/COMPLETE"
echo GATE3_FROZEN_OLD_VS_CURRENT=PASS
