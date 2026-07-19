#!/usr/bin/env bash
# Diagnostic postmortem for rejected Gate 6B7 (job 2350992).
# The formal old/new comparator rejected at
#   "frequency differs for matrix block (2, 'sigma_c_iw', 0, 0, 0)".
# This observer does NOT rerun any physics and does NOT accept the gate.
# It records:
#   1. strict old/new component comparison restricted to iterations 0:1
#      (before the first spectrum-dependent Sigma frequency regeneration)
#   2. per-iteration sigma_c_iw frequency-grid differences for iterations 1-2
#   3. diagnostic 0:2 comparison with the frequency check disabled, so
#      iteration-2 static components (raw_h/mixed_h/delta_vh/exx/vc) are
#      still quantified against the frozen oracle
#   4. the single-side validators the formal job never reached
#      (old closure, current closure, current fixed basis, current initial
#      state, hartree old/new validator, linear residual summary)
# benchmark_acceptance=false: COMPLETE only means the diagnostics executed.
#SBATCH --job-name=qsgw-g6b7-postmortem-v2
#SBATCH --partition=v100m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-postmortem-v2-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate6b7-postmortem-v2-%j.err

set -euo pipefail

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
failed_run=$root/runs/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-2350992
oracle_run=$root/runs/gate6b6-legacy-full-reader-hartree-on-mpi4-omp12-miniter2-2348987
old_trace=$oracle_run/old/qsgw_oracle_matrices.dat
current_trace=$failed_run/current/qsgw_matrices.dat
current_eigenvalues=$failed_run/current/qsgw_eigenvalues.dat
current_iterations=$failed_run/current/qsgw_iterations.dat
input_dir=$root/inputs/si-k444-gate6b1-mpi1-full-abfs-hartree-contracts-2348832/dataset
comparator=$root/jobs/compare_qsgw_component_traces-v6-hartree-1715ae0a.py
closure=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
fixed_basis=$root/jobs/validate_qsgw_fixed_basis.py
initial_state=$root/jobs/validate_qsgw_initial_state-v1.py
hartree_validator=$root/jobs/validate_qsgw_hartree_trace-v1-c0512c35.py
residual_tool=$root/jobs/summarize_qsgw_linear_residuals-638f1195.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
evidence=$root/evidence/gate6b7-miniter2-postmortem-v2-$SLURM_JOB_ID

expected_failed_job=2350992
expected_oracle_job=2348987
expected_gate_script_sha=3706b087d15238117d7d0fc72b4ff4dda8158251d708b6a3c0feac21ff1ecf31
expected_failed_comparison_sha=85a3f70570592f955f55a44f37b6622ddfdc8c8d6e3e50cd8b5325091c62da43
expected_comparator_sha=1715ae0a7029bde53599c646c35259381bd1430a847678bd13c8f9600778718f
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_fixed_basis_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_state_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_hartree_validator_sha=c0512c356d8a681cf2aa1ee5114c0b13a24e13b435a758610056b7c34536206a
expected_residual_sha=638f1195ba6a639f5750eb7c6f78911daffb76a42222f4a1c0f215768d163bba
expected_hartree_contract_sha=ea2d80953e1e03524f2c63f484a0cf333cbb59a2ed5c00f78eed141c8988edf6

test ! -e "$evidence"
test -e "$failed_run/FAILED"
test ! -e "$failed_run/COMPLETE"
test -e "$oracle_run/COMPLETE"
test ! -e "$oracle_run/FAILED"
test -s "$old_trace"
test -s "$current_trace"
test -s "$current_eigenvalues"
test -s "$current_iterations"
test -s "$failed_run/old-new-comparison.json"
test "$(sha256sum \
  "$failed_run/gate6b7-hartree-on-oracle-v40-vs-a25e4e43-mpi4-omp12-miniter2-v1.slurm" \
  | awk '{print $1}')" = "$expected_gate_script_sha"
test "$(sha256sum "$failed_run/old-new-comparison.json" | awk '{print $1}')" = \
  "$expected_failed_comparison_sha"
test "$(sha256sum "$comparator" | awk '{print $1}')" = "$expected_comparator_sha"
test "$(sha256sum "$closure" | awk '{print $1}')" = "$expected_closure_sha"
test "$(sha256sum "$fixed_basis" | awk '{print $1}')" = "$expected_fixed_basis_sha"
test "$(sha256sum "$initial_state" | awk '{print $1}')" = "$expected_initial_state_sha"
test "$(sha256sum "$hartree_validator" | awk '{print $1}')" = \
  "$expected_hartree_validator_sha"
test "$(sha256sum "$residual_tool" | awk '{print $1}')" = "$expected_residual_sha"
test "$(sha256sum "$input_dir/qsgw_input.hartree.contract" | awk '{print $1}')" = \
  "$expected_hartree_contract_sha"
grep -Fq '"passed": false' "$failed_run/old-new-comparison.json"
grep -Fq "frequency differs for matrix block (2, 'sigma_c_iw', 0, 0, 0)" \
  "$failed_run/old-new-comparison.json"
# The formal job stopped at the first comparator; the later validators must
# not exist, otherwise this postmortem would be auditing a mutated run.
test ! -e "$failed_run/hartree-validation.json"
test ! -e "$failed_run/current-closure-validation.json"

mkdir -p "$evidence/tools" "$evidence/trimmed"
cp "$0" "$evidence/postmortem-gate6b7-miniter2-v2.slurm"
cp "$comparator" "$evidence/tools/compare_qsgw_component_traces.py"
cp "$closure" "$evidence/tools/validate_qsgw_trace_closure.py"
cp "$fixed_basis" "$evidence/tools/validate_qsgw_fixed_basis.py"
cp "$initial_state" "$evidence/tools/validate_qsgw_initial_state.py"
cp "$hartree_validator" "$evidence/tools/validate_qsgw_hartree_trace_v1.py"
cp "$residual_tool" "$evidence/tools/summarize_qsgw_linear_residuals.py"
cp "$failed_run/old-new-comparison.json" \
  "$evidence/formal-old-new-comparison.json"
cp "$failed_run/old-new-comparator.stdout" "$evidence/" 2>/dev/null || true
cp "$failed_run/old-new-comparator.stderr" "$evidence/" 2>/dev/null || true
cp "$failed_run/FAILED" "$evidence/formal-FAILED.txt"
cp "$failed_run/PARAMETER_MAPPING.txt" "$evidence/formal-PARAMETER_MAPPING.txt"
cp "$oracle_run/old/homo_lumo_vs_iterations.dat" \
  "$evidence/old-homo-lumo-vs-iterations.dat"
cp "$current_iterations" "$evidence/current-qsgw-iterations.dat"

sha256sum \
  "$old_trace" \
  "$current_trace" \
  "$current_eigenvalues" \
  "$current_iterations" \
  "$oracle_run/old/librpa.in" \
  "$failed_run/current/librpa.in" \
  "$oracle_run/old/librpa.stdout" \
  "$oracle_run/old/librpa.stderr" \
  "$failed_run/current/librpa.stdout" \
  "$failed_run/current/librpa.stderr" \
  >"$evidence/SOURCE_SHA256SUMS.txt"

# ---- 1. strict comparison on the pre-regeneration subrange 0:1 ----
awk '/^#/ || ($1 ~ /^[0-9]+$/ && $1 <= 1)' "$old_trace" \
  >"$evidence/trimmed/old-iterations-0-1.dat"
awk '/^#/ || ($1 ~ /^[0-9]+$/ && $1 <= 1)' "$current_trace" \
  >"$evidence/trimmed/current-iterations-0-1.dat"
awk '/^#/ || ($1 ~ /^[0-9]+$/ && $1 <= 1)' "$current_eigenvalues" \
  >"$evidence/trimmed/current-eigenvalues-0-1.dat"
awk '/^#/ || ($1 ~ /^[0-9]+$/ && $1 <= 1)' "$current_iterations" \
  >"$evidence/trimmed/current-iterations-0-1.summary"

sha256sum "$evidence/trimmed/"* \
  >"$evidence/TRIMMED_BEFORE_HEADER_TRANSFORM_SHA256SUMS.txt"
legacy_trimmed=$evidence/trimmed/old-iterations-0-1.dat
sed -i 's/^# qsgw_min_iter 2$/# qsgw_min_iter 1/' "$legacy_trimmed"
sed -i 's/^# qsgw_max_iter 2$/# qsgw_max_iter 1/' "$legacy_trimmed"
grep -Fqx '# qsgw_min_iter 1' "$legacy_trimmed"
grep -Fqx '# qsgw_max_iter 1' "$legacy_trimmed"
for current_trimmed in \
  "$evidence/trimmed/current-iterations-0-1.dat" \
  "$evidence/trimmed/current-eigenvalues-0-1.dat" \
  "$evidence/trimmed/current-iterations-0-1.summary"
do
  ! grep -Eq '^# qsgw_(min|max)_iter ' "$current_trimmed"
done
sha256sum "$evidence/trimmed/"* \
  >"$evidence/TRIMMED_AFTER_HEADER_TRANSFORM_SHA256SUMS.txt"
cat >"$evidence/DIAGNOSTIC_HEADER_TRANSFORM.txt" <<EOF
purpose=enable_fail_closed_comparator_on_continuous_iteration_subrange_0_1
source_physical_rows=unchanged
legacy_source_headers=qsgw_min_iter_2,qsgw_max_iter_2
legacy_diagnostic_headers=qsgw_min_iter_1,qsgw_max_iter_1
current_v5_headers=unchanged_no_qsgw_minmax_fields
source_trace_hashes=$evidence/SOURCE_SHA256SUMS.txt
before_hashes=$evidence/TRIMMED_BEFORE_HEADER_TRANSFORM_SHA256SUMS.txt
after_hashes=$evidence/TRIMMED_AFTER_HEADER_TRANSFORM_SHA256SUMS.txt
EOF

set +e
"$python" -B "$evidence/tools/compare_qsgw_component_traces.py" \
  "$evidence/trimmed/old-iterations-0-1.dat" \
  "$evidence/trimmed/current-iterations-0-1.dat" \
  "$evidence/iterations0-1-comparison.json" \
  --iterations 0:1 \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace \
    "$evidence/trimmed/current-eigenvalues-0-1.dat" \
  --current-iteration-trace \
    "$evidence/trimmed/current-iterations-0-1.summary" \
  --expected-legacy-use-fullcoul-exx 0 \
  --expected-use-shrink-abfs 0 \
  --expected-hartree 1 \
  --expected-mixing-beta 0.2 \
  --expected-current-mixer linear \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$evidence/iterations0-1-comparator.stdout" \
  2>"$evidence/iterations0-1-comparator.stderr"
iterations0_1_rc=$?
set -e
test -s "$evidence/iterations0-1-comparison.json"

# ---- 2. sigma_c_iw frequency grid quantification (iterations 1-2) ----
awk '$3 == "sigma_c_iw" {key=$1 FS $6; if (!(key in seen)) {print $1, $6, $7; seen[key]=1}}' \
  "$old_trace" | sort -n -k1,1 -k2,2 \
  >"$evidence/old-sigma-iw-frequency-grid.dat"
awk '$3 == "sigma_c_iw" {key=$1 FS $6; if (!(key in seen)) {print $1, $6, $7; seen[key]=1}}' \
  "$current_trace" | sort -n -k1,1 -k2,2 \
  >"$evidence/current-sigma-iw-frequency-grid.dat"
test "$(wc -l <"$evidence/old-sigma-iw-frequency-grid.dat")" -eq 12
test "$(wc -l <"$evidence/current-sigma-iw-frequency-grid.dat")" -eq 12
paste \
  "$evidence/old-sigma-iw-frequency-grid.dat" \
  "$evidence/current-sigma-iw-frequency-grid.dat" | \
awk '
  BEGIN { first_iter = -1 }
  {
    if ($1 != $4 || $2 != $5) exit 2
    difference = $3 - $6
    if (difference < 0) difference = -difference
    if (difference > maximum[$1]) maximum[$1] = difference
    if (difference > 1.0e-10 && first_iter < 0) {
      first_iter = $1
      first_index = $2
      first_difference = difference
    }
  }
  END {
    print "# iteration max_abs_frequency_difference_ha"
    for (iteration = 1; iteration <= 2; iteration++)
      printf "%d %.17e\n", iteration, maximum[iteration] + 0.0
    printf "# first_difference iteration=%d frequency_index=%d abs_ha=%.17e\n", \
      first_iter, first_index, first_difference
  }
' >"$evidence/sigma-iw-frequency-grid-comparison.dat"

# ---- 3. diagnostic 0:2 comparison with frequency check disabled ----
set +e
"$python" -B "$evidence/tools/compare_qsgw_component_traces.py" \
  "$old_trace" \
  "$current_trace" \
  "$evidence/iterations0-2-nofreqcheck-comparison.json" \
  --iterations 0:2 \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_eigenvalues" \
  --current-iteration-trace "$current_iterations" \
  --expected-legacy-use-fullcoul-exx 0 \
  --expected-use-shrink-abfs 0 \
  --expected-hartree 1 \
  --expected-mixing-beta 0.2 \
  --expected-current-mixer linear \
  --frequency-tolerance 1e30 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$evidence/iterations0-2-nofreqcheck-comparator.stdout" \
  2>"$evidence/iterations0-2-nofreqcheck-comparator.stderr"
iterations0_2_nofreq_rc=$?
set -e
test -s "$evidence/iterations0-2-nofreqcheck-comparison.json"

# ---- 4. deferred formal observers (diagnostic, recorded not asserted) ----
set +e
"$python" -B "$evidence/tools/validate_qsgw_hartree_trace_v1.py" \
  "$old_trace" "$current_trace" "$current_iterations" \
  "$input_dir/bz_sampling_out" \
  "$evidence/hartree-validation.json" \
  --iterations 0:2 --channel 0 \
  --initial-delta-tolerance-ha 1e-10 \
  --charge-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$evidence/hartree-validator.stdout" \
  2>"$evidence/hartree-validator.stderr"
hartree_rc=$?

"$python" -B "$evidence/tools/summarize_qsgw_linear_residuals.py" \
  "$old_trace" "$current_trace" "$current_iterations" \
  "$evidence/linear-residual-comparison.json" \
  "$evidence/linear-residual-trajectory.csv" \
  --iterations 0:2 \
  --absolute-tolerance-ha 1e-8 --relative-tolerance 1e-8 \
  >"$evidence/residual-observer.stdout" \
  2>"$evidence/residual-observer.stderr"
residual_rc=$?

"$python" -B "$evidence/tools/validate_qsgw_initial_state.py" \
  "$current_trace" "$current_iterations" \
  "$input_dir/band_out" "$evidence/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 --occupation-tolerance 1e-12 \
  >"$evidence/current-initial-state-validator.stdout" \
  2>"$evidence/current-initial-state-validator.stderr"
initial_state_rc=$?

"$python" -B "$evidence/tools/validate_qsgw_trace_closure.py" \
  "$old_trace" "$evidence/old-closure-validation.json" \
  --iterations 0:2 --channel 0 --legacy-contract \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$evidence/old-closure-validator.stdout" \
  2>"$evidence/old-closure-validator.stderr"
old_closure_rc=$?

"$python" -B "$evidence/tools/validate_qsgw_trace_closure.py" \
  "$current_trace" "$evidence/current-closure-validation.json" \
  --iterations 0:2 --channel 0 \
  --closure-tolerance-ha 1e-10 --hermiticity-tolerance-ha 1e-10 \
  --hartree-reference-tolerance-ha 1e-10 \
  --hartree-response-minimum-ha 1e-12 \
  >"$evidence/current-closure-validator.stdout" \
  2>"$evidence/current-closure-validator.stderr"
current_closure_rc=$?

"$python" -B "$evidence/tools/validate_qsgw_fixed_basis.py" \
  "$current_trace" "$current_eigenvalues" "$input_dir/band_out" \
  "$evidence/current-fixed-basis-validation.json" \
  --iterations 0:2 --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 --invariant-tolerance 1e-10 \
  >"$evidence/current-fixed-basis-validator.stdout" \
  2>"$evidence/current-fixed-basis-validator.stderr"
current_fixed_basis_rc=$?
set -e

{
  printf 'analysis=gate6b7_miniter2_negative_result_postmortem_v2\n'
  printf 'benchmark_acceptance=false\n'
  printf 'diagnostic_only=true\n'
  printf 'formal_job=%s\n' "$expected_failed_job"
  printf 'formal_result=rejected\n'
  printf 'formal_failure=iteration2_sigma_c_iw_frequency_grid_mismatch\n'
  printf 'oracle_job=%s\n' "$expected_oracle_job"
  printf 'iterations0_1_comparison_rc=%s\n' "$iterations0_1_rc"
  printf 'iterations0_2_nofreqcheck_comparison_rc=%s\n' "$iterations0_2_nofreq_rc"
  printf 'hartree_validator_rc=%s\n' "$hartree_rc"
  printf 'residual_observer_rc=%s\n' "$residual_rc"
  printf 'initial_state_rc=%s\n' "$initial_state_rc"
  printf 'old_closure_rc=%s\n' "$old_closure_rc"
  printf 'current_closure_rc=%s\n' "$current_closure_rc"
  printf 'current_fixed_basis_rc=%s\n' "$current_fixed_basis_rc"
  printf 'diagnostic_header_transform=legacy_minmax_2_to_1_physical_rows_unchanged\n'
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'slurm_node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$evidence/PROVENANCE.txt"

(
  cd "$evidence"
  find . -type f ! -name 'SHA256SUMS.txt' -print0 | sort -z | \
    xargs -0 sha256sum >SHA256SUMS.txt
  sha256sum --check --quiet SHA256SUMS.txt
)
touch "$evidence/COMPLETE"
echo GATE6B7_POSTMORTEM_V2=DONE
