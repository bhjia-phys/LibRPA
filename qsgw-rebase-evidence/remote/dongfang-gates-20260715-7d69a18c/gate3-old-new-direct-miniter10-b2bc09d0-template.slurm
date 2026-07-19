#!/usr/bin/env bash
#SBATCH --job-name=qsgw-gate3-min10
#SBATCH --partition=48cp2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-min10-%j.out
#SBATCH --error=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c/logs/gate3-min10-%j.err

set -euo pipefail

target_iter=10

root=/data/home/df_iopcas_bhj/ai-runs/librpa-qsgw-rebase-gates-20260715T1910-7d69a18c
common_root=$root/inputs/si-k444-gate3-common-v3-2342408
input_dir=$common_root/dataset
input_verify_root=$root/evidence/gate3-common-v3-independent-verification-2342438
old_reader_preflight_root=$root/runs/gate3-old-reader-iter0-v3-2342440
gate2_root=$root/runs/gate2-qsgw-iter1-b2bc09d0-2343098
triangle_replay_root=$root/evidence/gate3-exx-truncated-upper-triangle-replay-2342685
miniter5_root=$root/runs/gate3-old-new-direct-miniter5-ce1e1859-2342693
miniter5_verification_root=$root/evidence/gate3-miniter5-artifact-verification-2342944
miniter5_v3_replay_root=$root/evidence/gate3-miniter5-comparator-v3-replay-2342945
old_build=/data/home/df_iopcas_bhj/qsgw-builds/old-new-a03a4c0-majorfix-v14-20260713
old_exe=$old_build/old/build/chi0_main.exe
candidate_build=$root/build-candidate-b2bc09d0-v1
candidate_exe=$candidate_build/chi0_main.exe
comparator_source=$root/jobs/compare_qsgw_component_traces-v3-9f9ec9be.py
comparator_test_source=$root/jobs/test_compare_qsgw_legacy_v4_current_v5-v3-72a79b04.py
closure_source=$root/jobs/validate_qsgw_trace_closure-v3-4a5de94e.py
closure_test_source=$root/jobs/test_validate_qsgw_trace_closure-v3-38de02fa.py
fixed_source=$root/jobs/validate_qsgw_fixed_basis.py
initial_source=$root/jobs/validate_qsgw_initial_state-v1.py
initial_test_source=$root/jobs/test_validate_qsgw_initial_state-v1.py
python=/data/home/df_iopcas_bhj/software-stack/miniconda3/bin/python3
run_root=$root/runs/gate3-old-new-direct-miniter$target_iter-b2bc09d0-$SLURM_JOB_ID
old_run=$run_root/old
current_run=$run_root/current
tool_dir=$run_root/tools

expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01
expected_old_exe_sha=a1292eff5364565d5b7463596882580a9a758b6e0e3e600ac5dfe67113bef788
expected_old_build_provenance_sha=8dfa0742e5618be14c27a96764ddf240af2ed8fc19662bac4d506bd44e77c96b
expected_candidate_commit=b2bc09d00ab49a2ff39c46c72d83ab89bc5ddac6
expected_candidate_exe_sha=a898ef66390d0207158c0f24d1b311fabde1bb11f4a1cd89b6082aee75e54998
expected_candidate_build_manifest_sha=e647822680e91a4597ff2908ed48dcb0280ae2e081329bf623dc1ba3cf9a3761
expected_candidate_build_provenance_sha=3489aeae4e1632ab24483b1241252fbe43187b63946df75e15e9e5be4d77cb8c
expected_common_dataset_manifest_sha=df793f964077ca03de7ad4723975736d0d0fb85d7e548241f3a7a04677cc05f7
expected_common_output_manifest_sha=3b7811c12df23a563e597cae2fb80f592c045d27c803a1afc7ead31d7d7439d4
expected_common_provenance_sha=54e2a4b0f07883f4341d46e41a80fd8ae1ebac7e25d263699eda8819b00ce0d8
expected_common_contract_sha=0dd83763717853fb64960f56354a2d7b7acd9c6759adcdff49dac9936635f110
expected_common_validation_sha=f37f26fa347e50da9064dc1b9dfd57df25d6f6843e61b7baa44e2b4797e76d25
expected_common_preparation_sha=14e3c624d3606972d08da6a6adf27628ae6f7a03cb802658c2077c6575b585f7
expected_common_compatibility_sha=c3f2226a6d05dde8e7d2819a789f8db33e62526ee82a80fbf0917245930491b7
expected_input_verify_manifest_sha=dbc92ba649a399881dc14227194743d8166d70203ec56297aaaa8a8971cf9379
expected_input_verify_provenance_sha=3ddeccf3f6cc06765aa8957f693fe3c0f5198547a703a388346c86487dd8f699
expected_old_reader_preflight_manifest_sha=2f5dc6fd2c3f16ae34423898b6e16c57c908e40c11dd59608e6e53d5a1e94941
expected_old_reader_preflight_provenance_sha=1195e9c010d2c8ae52744bd3bce0b95a1012175aa44085f719ae23d6b5aade35
expected_old_reader_preflight_comparison_sha=1c3ccf5a7e4382e23d401c421647d471258d2cfa020cd8c210fbfed75456a5da
expected_gate2_output_manifest_sha=REPLACE_WITH_GATE2_OUTPUT_MANIFEST_SHA
expected_gate2_provenance_sha=REPLACE_WITH_GATE2_PROVENANCE_SHA
expected_triangle_replay_manifest_sha=1a138f4bcff1d19e250564c4dbb07b6d7a768be5bba49c0b192cb9eff4f2017b
expected_triangle_replay_provenance_sha=b73eef9989ac2b0a27b6f5268f1d74f7754359c389822cb193f311fdd351cd20
expected_triangle_replay_comparison_sha=307f55150264c59c05523925d909dde1a80b377b30eebeb6deb31f316563c14b
expected_triangle_replay_old_closure_sha=e59ea0f24ec0977b5a013a1abf861ae755c768a5336f6e3d715c128746ddf314
expected_triangle_replay_current_closure_sha=533e5be7ea7ccece86ffbba43b7d850fe4d0e4f103fce9ec0b3f2ef2d6f3b3f4
expected_triangle_replay_fixed_sha=e520d0a3c73b18fae834bc659f2fbde3674b65396856f19653671fc543f29e05
expected_miniter5_output_manifest_sha=819d298e217a873d449597a3164955a3b646a2d4aa9e2c0dc9555b806a9ea57a
expected_miniter5_provenance_sha=fc2960bdac7e405b7a314addf8bb350ec023d7ea65b98b7dfeeead76db55de96
expected_miniter5_comparison_sha=e9bcd2c03a4d3c292629b427cfc5d3b3981f8089d293654b7cdf1f969da8ef08
expected_miniter5_verification_manifest_sha=802223a3d9f58521993bc2d9729e38589a854daa85e2173289ed2858eefcab73
expected_miniter5_verification_provenance_sha=345863963df8f4fffc01d9b96d935a9dff1a8aea432ccdaa4e590ee194e9cda2
expected_miniter5_v3_replay_manifest_sha=6ed92f082611ad170cffe61daa26abb98b200846d03706cc058f249e62a77422
expected_miniter5_v3_replay_provenance_sha=248756a30e52ec522f4f7b22e4cff507687bea88160c5db6bd8c05a6d2b25dcc
expected_comparator_sha=9f9ec9beb0d08754d1f9a39cb528df9aa4000df7f3443479984ec2df6144bf89
expected_comparator_test_sha=72a79b04274c4c1dd13ef9ee3dcfffea7d0d972d8ad7cfe928ad0c23dc5cc9f8
expected_closure_sha=4a5de94e6dbf590dded4a6ecd140aa4227fa61ffa0e73af17ec0f388610cffaf
expected_closure_test_sha=38de02fabc0dde41911e5152b0b08a9987b312b0cea29c69396bb9e392e9c745
expected_fixed_sha=569ecb1366bc6dd4584216bd42ac55905a4848b1a0b1bdf96c6236c782de7cb2
expected_initial_sha=6bbade9eaeb207b6cea9fa2f80d8cbcd0baeb8cbd760a2ffab5a0fe6f6d4868a
expected_initial_test_sha=82634e292a8fc1eb5ed454360ea2367e06687f0547da6503291c850a00e5b339
expected_efermi_ha=0.2564169101593575

test ! -e "$run_root"
test -e "$common_root/COMPLETE"
test -e "$input_verify_root/COMPLETE"
test -e "$old_reader_preflight_root/COMPLETE"
test -e "$gate2_root/COMPLETE"
test -e "$triangle_replay_root/COMPLETE"
test -e "$miniter5_root/COMPLETE"
test -e "$miniter5_verification_root/COMPLETE"
test -e "$miniter5_v3_replay_root/COMPLETE"
test -e "$candidate_build/COMPLETE"
test -x "$old_exe"
test -x "$candidate_exe"
test "$(sha256sum "$old_exe" | awk '{print $1}')" = \
  "$expected_old_exe_sha"
test "$(sha256sum "$old_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_old_build_provenance_sha"
grep -Fqx "old_oracle_commit=$expected_old_commit" \
  "$old_build/build-provenance.txt"
test "$(sha256sum "$candidate_exe" | awk '{print $1}')" = \
  "$expected_candidate_exe_sha"
test "$(sha256sum "$candidate_build/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_manifest_sha"
test "$(sha256sum "$candidate_build/build-provenance.txt" | awk '{print $1}')" = \
  "$expected_candidate_build_provenance_sha"
grep -Fqx "source_commit=$expected_candidate_commit" \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'source_status=clean_git_archive_from_commit' \
  "$candidate_build/build-provenance.txt"
grep -Fqx 'final_acceptance=true' \
  "$candidate_build/build-provenance.txt"
test "$(sha256sum "$common_root/DATASET_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_dataset_manifest_sha"
test "$(sha256sum "$common_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_common_output_manifest_sha"
test "$(sha256sum "$common_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_common_provenance_sha"
test "$(sha256sum "$input_dir/qsgw_input.contract" | awk '{print $1}')" = \
  "$expected_common_contract_sha"
test "$(sha256sum "$common_root/contract-validation.json" | awk '{print $1}')" = \
  "$expected_common_validation_sha"
test "$(sha256sum "$common_root/evidence/source-input-preparation.json" | awk '{print $1}')" = \
  "$expected_common_preparation_sha"
test "$(sha256sum "$common_root/legacy-root-compatibility.txt" | awk '{print $1}')" = \
  "$expected_common_compatibility_sha"
test "$(sha256sum "$input_verify_root/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_input_verify_manifest_sha"
test "$(sha256sum "$input_verify_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_input_verify_provenance_sha"
test "$(sha256sum "$old_reader_preflight_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_old_reader_preflight_manifest_sha"
test "$(sha256sum "$old_reader_preflight_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_old_reader_preflight_provenance_sha"
test "$(sha256sum "$old_reader_preflight_root/ITERATION0_COMPARISON.txt" | awk '{print $1}')" = \
  "$expected_old_reader_preflight_comparison_sha"
test "$(sha256sum "$gate2_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_gate2_output_manifest_sha"
test "$(sha256sum "$gate2_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_gate2_provenance_sha"
test "$(sha256sum "$triangle_replay_root/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_triangle_replay_manifest_sha"
test "$(sha256sum "$triangle_replay_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_triangle_replay_provenance_sha"
test "$(sha256sum "$triangle_replay_root/old-current-comparison.json" | awk '{print $1}')" = \
  "$expected_triangle_replay_comparison_sha"
test "$(sha256sum "$triangle_replay_root/old-closure-validation.json" | awk '{print $1}')" = \
  "$expected_triangle_replay_old_closure_sha"
test "$(sha256sum "$triangle_replay_root/current-closure-validation.json" | awk '{print $1}')" = \
  "$expected_triangle_replay_current_closure_sha"
test "$(sha256sum "$triangle_replay_root/current-fixed-basis-validation.json" | awk '{print $1}')" = \
  "$expected_triangle_replay_fixed_sha"
test "$(sha256sum "$miniter5_root/OUTPUT_SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_miniter5_output_manifest_sha"
test "$(sha256sum "$miniter5_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_miniter5_provenance_sha"
test "$(sha256sum "$miniter5_root/old-new-comparison.json" | awk '{print $1}')" = \
  "$expected_miniter5_comparison_sha"
test "$(sha256sum "$miniter5_verification_root/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_miniter5_verification_manifest_sha"
test "$(sha256sum "$miniter5_verification_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_miniter5_verification_provenance_sha"
test "$(sha256sum "$miniter5_v3_replay_root/SHA256SUMS.txt" | awk '{print $1}')" = \
  "$expected_miniter5_v3_replay_manifest_sha"
test "$(sha256sum "$miniter5_v3_replay_root/PROVENANCE.txt" | awk '{print $1}')" = \
  "$expected_miniter5_v3_replay_provenance_sha"
test "$(sha256sum "$miniter5_v3_replay_root/old-new-comparison-v3.json" | awk '{print $1}')" = \
  "$expected_miniter5_comparison_sha"
test "$(sha256sum "$comparator_source" | awk '{print $1}')" = \
  "$expected_comparator_sha"
test "$(sha256sum "$comparator_test_source" | awk '{print $1}')" = \
  "$expected_comparator_test_sha"
test "$(sha256sum "$closure_source" | awk '{print $1}')" = \
  "$expected_closure_sha"
test "$(sha256sum "$closure_test_source" | awk '{print $1}')" = \
  "$expected_closure_test_sha"
test "$(sha256sum "$fixed_source" | awk '{print $1}')" = \
  "$expected_fixed_sha"
test "$(sha256sum "$initial_source" | awk '{print $1}')" = \
  "$expected_initial_sha"
test "$(sha256sum "$initial_test_source" | awk '{print $1}')" = \
  "$expected_initial_test_sha"
grep -Fq '"passed": true' \
  "$common_root/evidence/source-input-preparation.json"
grep -Fq '"passed": true' "$common_root/contract-validation.json"
grep -Fq "\"target_efermi_ha\": $expected_efermi_ha" \
  "$common_root/evidence/source-input-preparation.json"
grep -Fqx 'bytewise_cmp=PASS' \
  "$old_reader_preflight_root/ITERATION0_COMPARISON.txt"
grep -Fqx 'first_vxc_cmp=PASS' \
  "$old_reader_preflight_root/ITERATION0_COMPARISON.txt"
grep -Fq '"passed": true' "$gate2_root/initial-state-validation.json"
grep -Fq '"passed": true' \
  "$gate2_root/qsgw-iter1-vs-upstream-g0w0-sigc.json"
grep -Fq '"passed": true' "$gate2_root/closure-validation.json"
grep -Fq '"passed": true' "$gate2_root/fixed-basis-validation.json"
grep -Fq '"passed": true' \
  "$triangle_replay_root/old-current-comparison.json"
grep -Fq '"passed": true' \
  "$triangle_replay_root/old-closure-validation.json"
grep -Fq '"passed": true' \
  "$triangle_replay_root/current-closure-validation.json"
grep -Fq '"passed": true' \
  "$triangle_replay_root/current-fixed-basis-validation.json"
grep -Fq '"legacy_effective_hamiltonian_semantics": "upper_triangle_authoritative"' \
  "$triangle_replay_root/old-current-comparison.json"
grep -Fq '"passed": true' "$miniter5_root/old-new-comparison.json"
grep -Fqx 'outer_manifest_check=PASS' \
  "$miniter5_verification_root/PROVENANCE.txt"
grep -Fqx 'nested_old_manifest_check=PASS' \
  "$miniter5_verification_root/PROVENANCE.txt"
grep -Fqx 'nested_current_manifest_check=PASS' \
  "$miniter5_verification_root/PROVENANCE.txt"
grep -Fqx 'v2_v3_report_bytewise_cmp=PASS' \
  "$miniter5_v3_replay_root/PROVENANCE.txt"
grep -Fqx 'n_scf_kpoints 64' "$input_dir/qsgw_input.contract"
grep -Fqx 'n_headwing_kpoints 0' "$input_dir/qsgw_input.contract"
grep -Fqx 'headwing_grid disabled' "$input_dir/qsgw_input.contract"
grep -Fqx 'hartree_update off' "$input_dir/qsgw_input.contract"
grep -Fqx 'band_update off' "$input_dir/qsgw_input.contract"
test "$(awk 'NR == 2 {print $1, $2}' "$input_dir/bz_sampling_out")" = \
  '64 64'
test "$(find "$input_dir" -maxdepth 1 -name 'KS_eigenvector_*.dat' | wc -l)" \
  -eq 64
test "$(find "$input_dir" -maxdepth 1 -name 'Cs_data_*.txt' | wc -l)" -eq 4
test "$(find "$input_dir" -type l | wc -l)" -eq 0
test "$(find "$input_dir" -maxdepth 1 -type f \
  -name 'vxcs1k*_nao.txt' | wc -l)" -eq 64
test "$(find "$input_dir" -maxdepth 1 -type f \
  -name 'sks1k*_nao.txt' | wc -l)" -eq 64
test "$(wc -l <"$common_root/legacy-root-compatibility.txt")" -eq 129
test "$(wc -l <"$common_root/DATASET_SHA256SUMS.txt")" -eq 1367

mkdir -p "$old_run" "$current_run" "$tool_dir"
cp "$0" "$run_root/gate3-old-new-direct-miniter10-b2bc09d0-v4.slurm"
cp "$old_build/build-provenance.txt" \
  "$run_root/old-build-provenance.txt"
cp "$candidate_build/build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt"
cp "$candidate_build/SHA256SUMS.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt"
cp "$common_root/DATASET_SHA256SUMS.txt" "$run_root/"
cp "$common_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/common-input-OUTPUT_SHA256SUMS.txt"
cp "$common_root/PROVENANCE.txt" \
  "$run_root/common-input-PROVENANCE.txt"
cp "$common_root/evidence/source-input-preparation.json" \
  "$run_root/input-preparation.json"
cp "$common_root/contract-validation.json" "$run_root/"
cp "$common_root/legacy-root-compatibility.txt" "$run_root/"
cp "$input_dir/qsgw_input.contract" "$run_root/"
cp "$input_verify_root/SHA256SUMS.txt" \
  "$run_root/input-verification-SHA256SUMS.txt"
cp "$input_verify_root/PROVENANCE.txt" \
  "$run_root/input-verification-PROVENANCE.txt"
cp "$old_reader_preflight_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/old-reader-preflight-OUTPUT_SHA256SUMS.txt"
cp "$old_reader_preflight_root/PROVENANCE.txt" \
  "$run_root/old-reader-preflight-PROVENANCE.txt"
cp "$old_reader_preflight_root/ITERATION0_COMPARISON.txt" \
  "$run_root/old-reader-preflight-ITERATION0_COMPARISON.txt"
cp "$gate2_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate2-OUTPUT_SHA256SUMS.txt"
cp "$gate2_root/PROVENANCE.txt" "$run_root/gate2-PROVENANCE.txt"
cp "$gate2_root/initial-state-validation.json" \
  "$run_root/gate2-initial-state-validation.json"
cp "$gate2_root/qsgw-iter1-vs-upstream-g0w0-sigc.json" \
  "$run_root/gate2-qsgw-vs-g0w0.json"
cp "$gate2_root/closure-validation.json" \
  "$run_root/gate2-closure-validation.json"
cp "$gate2_root/fixed-basis-validation.json" \
  "$run_root/gate2-fixed-basis-validation.json"
cp "$triangle_replay_root/SHA256SUMS.txt" \
  "$run_root/triangle-replay-SHA256SUMS.txt"
cp "$triangle_replay_root/PROVENANCE.txt" \
  "$run_root/triangle-replay-PROVENANCE.txt"
cp "$triangle_replay_root/old-current-comparison.json" \
  "$run_root/triangle-replay-old-current-comparison.json"
cp "$triangle_replay_root/old-closure-validation.json" \
  "$run_root/triangle-replay-old-closure-validation.json"
cp "$triangle_replay_root/current-closure-validation.json" \
  "$run_root/triangle-replay-current-closure-validation.json"
cp "$triangle_replay_root/current-fixed-basis-validation.json" \
  "$run_root/triangle-replay-current-fixed-basis-validation.json"
cp "$miniter5_root/OUTPUT_SHA256SUMS.txt" \
  "$run_root/miniter5-OUTPUT_SHA256SUMS.txt"
cp "$miniter5_root/PROVENANCE.txt" \
  "$run_root/miniter5-PROVENANCE.txt"
cp "$miniter5_root/old-new-comparison.json" \
  "$run_root/miniter5-old-new-comparison.json"
cp "$miniter5_verification_root/SHA256SUMS.txt" \
  "$run_root/miniter5-verification-SHA256SUMS.txt"
cp "$miniter5_verification_root/PROVENANCE.txt" \
  "$run_root/miniter5-verification-PROVENANCE.txt"
cp "$miniter5_v3_replay_root/SHA256SUMS.txt" \
  "$run_root/miniter5-v3-replay-SHA256SUMS.txt"
cp "$miniter5_v3_replay_root/PROVENANCE.txt" \
  "$run_root/miniter5-v3-replay-PROVENANCE.txt"
cp "$miniter5_v3_replay_root/old-new-comparison-v3.json" \
  "$run_root/miniter5-old-new-comparison-v3.json"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces.py"
cp "$comparator_source" "$tool_dir/compare_qsgw_component_traces_v3.py"
cp "$comparator_test_source" \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure.py"
cp "$closure_source" "$tool_dir/validate_qsgw_trace_closure_v3.py"
cp "$closure_test_source" \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py"
cp "$fixed_source" "$tool_dir/validate_qsgw_fixed_basis.py"
cp "$initial_source" "$tool_dir/validate_qsgw_initial_state.py"
cp "$initial_test_source" \
  "$tool_dir/test_validate_qsgw_initial_state.py"

(
  cd "$common_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet DATASET_SHA256SUMS.txt
  printf 'common 1367-file input manifests verified\n'
  cd "$input_verify_root"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'independent common-input verification manifest verified\n'
  cd "$old_reader_preflight_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  printf 'old-reader iteration-zero preflight manifest verified\n'
  cd "$candidate_build"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'clean candidate build manifest verified\n'
  cd "$gate2_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  printf 'accepted Gate 2 output manifest verified\n'
  cd "$triangle_replay_root"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'accepted EXX/upper-triangle diagnostic replay manifest verified\n'
  cd "$miniter5_root"
  sha256sum --check --quiet OUTPUT_SHA256SUMS.txt
  printf 'accepted Gate 3 miniter5 output manifest verified\n'
  cd "$miniter5_verification_root"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'independent Gate 3 miniter5 artifact verification manifest verified\n'
  cd "$miniter5_v3_replay_root"
  sha256sum --check --quiet SHA256SUMS.txt
  printf 'Gate 3 miniter5 comparator-v3 replay manifest verified\n'
) >"$run_root/prerequisite-hash-check.log"

cat >"$old_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_wc = false
use_abacus_exx_symmetry = false
use_abacus_gw_symmetry = false
qsgw_iterative_headwing = false
max_iter = $target_iter
EOF

cat >"$current_run/librpa.in" <<EOF
task = qsgw
input_dir = $input_dir
output_dir = .
constants_choice = internal
nfreq = 6
tfgrid_type = minimax
n_params_anacon = -1
n_params_anacon_resample = -1
anacon_nfreq = -1
parallel_routing = libri
vq_threshold = 0
sqrt_coulomb_threshold = 0
use_scalapack_gw_wc = true
output_gw_sigc_ks_mat_kf = false
use_shrink_abfs = false
use_shrink_chi = false
use_pyatb = false
replace_w_head = false
option_dielect_func = 0
use_fullcoul_exx = false
use_fullcoul_eps = true
use_fullcoul_wc = false
use_symmetry_exx = false
use_symmetry_gw = false
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
qsgw_input_contract = qsgw_input.contract
qsgw_mixer = none
qsgw_mixing_beta = 0.2
qsgw_min_iter = $target_iter
qsgw_max_iter = $target_iter
qsgw_write_iteration_matrices = true
qsgw_update_hartree = false
EOF

cat >"$run_root/PARAMETER_MAPPING.txt" <<EOF
comparison=legacy_v4_to_current_v5
iterations=0:$target_iter
refreshed_miniter5_observer=iterations_0:5_from_same_miniter10_trajectory
old_max_iter=$target_iter
current_qsgw_min_iter=$target_iter
current_qsgw_max_iter=$target_iter
old_update=linear_beta_1_direct
current_update=mixer_none_direct
old_symmetry_switches=use_abacus_exx_symmetry_false,use_abacus_gw_symmetry_false
current_symmetry_switches=use_symmetry_exx_false,use_symmetry_gw_false,use_symmetry_rpa_false
old_headwing=replace_w_head_false,option_dielect_func_0,qsgw_iterative_headwing_false
current_headwing=replace_w_head_false,option_dielect_func_0,contract_headwing_disabled
old_hartree=QSGW_ORACLE_UPDATE_HARTREE_0
current_hartree=qsgw_update_hartree_false
old_fullcoul_exx=false
current_fullcoul_exx=false
exx_coulomb=truncated_both_sides
legacy_exx_actual_behavior=hardcoded_Vq_cut
legacy_effective_hamiltonian_semantics=eigsh_UPLO_U
current_effective_hamiltonian_semantics=explicit_full_hermitian_from_legacy_upper
common_input=$input_dir
common_input_manifest_sha256=$expected_common_dataset_manifest_sha
common_contract_sha256=$expected_common_contract_sha
common_input_freeze_job=2342408
common_input_verification_job=2342438
old_reader_iteration0_preflight_job=2342440
exx_upper_triangle_replay_job=2342685
exx_upper_triangle_replay_manifest_sha256=$expected_triangle_replay_manifest_sha
accepted_miniter5_job=2342693
accepted_miniter5_output_manifest_sha256=$expected_miniter5_output_manifest_sha
accepted_miniter5_comparison_sha256=$expected_miniter5_comparison_sha
miniter5_artifact_verification_job=2342944
miniter5_artifact_verification_manifest_sha256=$expected_miniter5_verification_manifest_sha
miniter5_comparator_v3_replay_job=2342945
miniter5_comparator_v3_replay_manifest_sha256=$expected_miniter5_v3_replay_manifest_sha
EOF

set +u
source /data/app/intel/oneapi-2024.2/setvars.sh --force \
  >"$run_root/oneapi-setvars.log" 2>&1
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export LIBRI_DETERMINISTIC_REDUCTION=1
base_ld_library_path=${LD_LIBRARY_PATH:-}

{
  printf 'gate=old-new-direct-update-miniter%s\n' "$target_iter"
  printf 'gate2_dependency_job=2343098\n'
  printf 'gate2_output_manifest_sha256=%s\n' \
    "$expected_gate2_output_manifest_sha"
  printf 'triangle_replay_job=2342685\n'
  printf 'triangle_replay_manifest_sha256=%s\n' \
    "$expected_triangle_replay_manifest_sha"
  printf 'triangle_replay_comparison_sha256=%s\n' \
    "$expected_triangle_replay_comparison_sha"
  printf 'accepted_miniter5_job=2342693\n'
  printf 'accepted_miniter5_output_manifest_sha256=%s\n' \
    "$expected_miniter5_output_manifest_sha"
  printf 'accepted_miniter5_comparison_sha256=%s\n' \
    "$expected_miniter5_comparison_sha"
  printf 'miniter5_artifact_verification_job=2342944\n'
  printf 'miniter5_artifact_verification_manifest_sha256=%s\n' \
    "$expected_miniter5_verification_manifest_sha"
  printf 'miniter5_comparator_v3_replay_job=2342945\n'
  printf 'miniter5_comparator_v3_replay_manifest_sha256=%s\n' \
    "$expected_miniter5_v3_replay_manifest_sha"
  printf 'common_input_freeze_job=2342408\n'
  printf 'common_input=%s\n' "$input_dir"
  printf 'common_input_manifest_sha256=%s\n' \
    "$expected_common_dataset_manifest_sha"
  printf 'common_input_contract_sha256=%s\n' \
    "$expected_common_contract_sha"
  printf 'common_input_verification_job=2342438\n'
  printf 'common_input_verification_manifest_sha256=%s\n' \
    "$expected_input_verify_manifest_sha"
  printf 'old_reader_iteration0_preflight_job=2342440\n'
  printf 'old_reader_iteration0_preflight_manifest_sha256=%s\n' \
    "$expected_old_reader_preflight_manifest_sha"
  printf 'old_commit=%s\n' "$expected_old_commit"
  printf 'old_executable=%s\n' "$old_exe"
  printf 'old_executable_sha256=%s\n' "$expected_old_exe_sha"
  printf 'old_build_provenance_sha256=%s\n' \
    "$expected_old_build_provenance_sha"
  printf 'current_commit=%s\n' "$expected_candidate_commit"
  printf 'current_source_status=clean_git_archive_from_commit\n'
  printf 'current_executable=%s\n' "$candidate_exe"
  printf 'current_executable_sha256=%s\n' "$expected_candidate_exe_sha"
  printf 'current_build_manifest_sha256=%s\n' \
    "$expected_candidate_build_manifest_sha"
  printf 'comparator_sha256=%s\n' "$expected_comparator_sha"
  printf 'closure_validator_sha256=%s\n' "$expected_closure_sha"
  printf 'target_iteration=%s\n' "$target_iter"
  printf 'refreshed_miniter5_observer=%s\n' \
    'iterations_0:5_from_same_miniter10_trajectory'
  printf 'crystal_symmetry=off\n'
  printf 'time_reversal_reduction=off_full_64_kpoint_input\n'
  printf 'headwing=off\n'
  printf 'hartree=off\n'
  printf 'band=off\n'
  printf 'old_mixing=linear_beta_1_direct\n'
  printf 'current_mixing=none_direct\n'
  printf 'exx_coulomb=truncated_both_sides\n'
  printf 'legacy_exx_source=old/src/driver/task_qsgw.cpp:1638-1639\n'
  printf 'legacy_diagonalization=old/src/math/matrix_m.h:889-914_UPLO_U\n'
  printf 'current_materialization=src/qsgw/effective_hamiltonian.cpp:164-176\n'
  printf 'mpi_ranks=%s\n' "${SLURM_NTASKS:-unknown}"
  printf 'omp_threads=%s\n' "$OMP_NUM_THREADS"
  printf 'LIBRI_DETERMINISTIC_REDUCTION=%s\n' \
    "$LIBRI_DETERMINISTIC_REDUCTION"
  printf 'python=%s\n' "$("$python" --version 2>&1)"
  printf 'numpy=%s\n' \
    "$("$python" -c 'import numpy; print(numpy.__version__)')"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'slurm_node_list=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mpirun -np "$SLURM_NTASKS" hostname | sort | uniq -c
  sha256sum "$old_exe" "$candidate_exe" \
    "$old_run/librpa.in" "$current_run/librpa.in" \
    "$input_dir/band_out" "$input_dir/stru_out" \
    "$input_dir/vxc_out.dat" "$input_dir/qsgw_input.contract"
} >"$run_root/PROVENANCE.txt"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  >"$run_root/comparator-unit-test.stdout" \
  2>"$run_root/comparator-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  >"$run_root/closure-unit-test.stdout" \
  2>"$run_root/closure-unit-test.stderr"
PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/initial-state-unit-test.stdout" \
  2>"$run_root/initial-state-unit-test.stderr"

(
  cd "$old_run"
  export QSGW_ORACLE_TRACE=1
  export QSGW_ORACLE_UPDATE_HARTREE=0
  export QSGW_HROUND_SCALE=0
  export LIBRPA_QSGW_MIXING_BETA=1
  export LD_LIBRARY_PATH="$old_build/old/build/qsgw:$old_build/old/build/src:$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$old_exe" \
    >librpa.stdout 2>librpa.stderr
)

test -s "$old_run/qsgw_oracle_matrices.dat"
test -s "$old_run/homo_lumo_vs_iterations.dat"
grep -Fqx '# qsgw_contract_version 4' \
  "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixer linear' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_mixing_beta 1' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx "# qsgw_min_iter $target_iter" \
  "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx "# qsgw_max_iter $target_iter" \
  "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# qsgw_update_hartree 0' \
  "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_gw 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_symmetry_exx 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# replace_w_head 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# option_dielect_func 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_fullcoul_exx 0' "$old_run/qsgw_oracle_matrices.dat"
grep -Fq 'use_fullcoul_exx = F' "$old_run/librpa.stdout"
grep -Fqx '# use_fullcoul_eps 1' "$old_run/qsgw_oracle_matrices.dat"
grep -Fqx '# use_fullcoul_wc 0' "$old_run/qsgw_oracle_matrices.dat"
test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
  "$old_run/qsgw_oracle_matrices.dat")" = "$target_iter"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$old_run/homo_lumo_vs_iterations.dat")" = "$target_iter"

(
  cd "$current_run"
  unset QSGW_ORACLE_TRACE
  unset QSGW_ORACLE_UPDATE_HARTREE
  unset QSGW_HROUND_SCALE
  unset LIBRPA_QSGW_MIXING_BETA
  export LD_LIBRARY_PATH="$base_ld_library_path"
  mpirun -np "$SLURM_NTASKS" "$candidate_exe" \
    >librpa.stdout 2>librpa.stderr
)

test -s "$current_run/qsgw_matrices.dat"
test -s "$current_run/qsgw_eigenvalues.dat"
test -s "$current_run/qsgw_iterations.dat"
grep -Fq 'use_fullcoul_exx = false' "$current_run/librpa.stdout"
for trace in \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_eigenvalues.dat" \
  "$current_run/qsgw_iterations.dat"
do
  grep -Fqx '# qsgw_contract_version 5' "$trace"
  grep -Fqx '# fixed_basis immutable_mf0' "$trace"
  grep -Fqx '# live_update eigenvalues_wfc' "$trace"
  grep -Fqx '# symmetry unsupported_full_bz_only' "$trace"
  grep -Fqx '# headwing disabled_stage1' "$trace"
  grep -Fqx '# hartree disabled_stage1' "$trace"
  grep -Fqx '# band disabled_stage1' "$trace"
  grep -Fqx '# qsgw_mixer none' "$trace"
  grep -Fqx \
    "# qsgw_input_contract_sha256 $expected_common_contract_sha" "$trace"
done
test "$(awk '$1 !~ /^#/ && NF {if ($1 > max) max=$1} END {print max+0}' \
  "$current_run/qsgw_matrices.dat")" = "$target_iter"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$current_run/qsgw_eigenvalues.dat")" = "$target_iter"
test "$(awk 'NF && $1 !~ /^#/ {last=$1} END {print last}' \
  "$current_run/qsgw_iterations.dat")" = "$target_iter"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_iterations.dat" \
  "$input_dir/band_out" \
  "$run_root/current-initial-state-validation.json" \
  --efermi-tolerance-ha 1e-12 \
  --occupation-tolerance 1e-12 \
  >"$run_root/current-initial-state-validator.stdout" \
  2>"$run_root/current-initial-state-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/current-initial-state-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/old-new-comparison.json" \
  --iterations "0:$target_iter" \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_run/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_run/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/old-new-comparator.stdout" \
  2>"$run_root/old-new-comparator.stderr"
grep -Fq '"passed": true' "$run_root/old-new-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/old-new-miniter5-comparison.json" \
  --iterations "0:5" \
  --channel 0 \
  --contract-mode legacy_v4_to_current_v5 \
  --current-eigenvalue-trace "$current_run/qsgw_eigenvalues.dat" \
  --current-iteration-trace "$current_run/qsgw_iterations.dat" \
  --expected-legacy-use-fullcoul-exx 0 \
  --frequency-tolerance 1e-10 \
  --matrix-max-abs-tolerance-ha 1e-8 \
  --matrix-relative-tolerance 1e-8 \
  --eigenvalue-tolerance 1e-6 \
  --gap-tolerance-ev 1e-5 \
  --degeneracy-tolerance 1e-8 \
  --state-tolerance 1e-10 \
  >"$run_root/old-new-miniter5-comparator.stdout" \
  2>"$run_root/old-new-miniter5-comparator.stderr"
grep -Fq '"passed": true' \
  "$run_root/old-new-miniter5-comparison.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$old_run/qsgw_oracle_matrices.dat" \
  "$run_root/old-closure-validation.json" \
  --iterations "0:$target_iter" \
  --channel 0 \
  --legacy-contract \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/old-closure-validator.stdout" \
  2>"$run_root/old-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/old-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$current_run/qsgw_matrices.dat" \
  "$run_root/current-closure-validation.json" \
  --iterations "0:$target_iter" \
  --channel 0 \
  --closure-tolerance-ha 1e-10 \
  --hermiticity-tolerance-ha 1e-10 \
  >"$run_root/current-closure-validator.stdout" \
  2>"$run_root/current-closure-validator.stderr"
grep -Fq '"passed": true' "$run_root/current-closure-validation.json"

PYTHONPATH="$tool_dir" "$python" -B \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$current_run/qsgw_matrices.dat" \
  "$current_run/qsgw_eigenvalues.dat" \
  "$input_dir/band_out" \
  "$run_root/current-fixed-basis-validation.json" \
  --iterations "0:$target_iter" \
  --channel 0 \
  --eigenvalue-tolerance-ha 1e-10 \
  --invariant-tolerance 1e-10 \
  >"$run_root/current-fixed-basis-validator.stdout" \
  2>"$run_root/current-fixed-basis-validator.stderr"
grep -Fq '"passed": true' \
  "$run_root/current-fixed-basis-validation.json"

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"$run_root/PROVENANCE.txt"
(
  cd "$run_root"
  find old -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >OLD_OUTPUT_SHA256SUMS.txt
  find current -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    >CURRENT_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet OLD_OUTPUT_SHA256SUMS.txt
  sha256sum --check --quiet CURRENT_OUTPUT_SHA256SUMS.txt
)
sha256sum \
  "$run_root/PROVENANCE.txt" \
  "$run_root/PARAMETER_MAPPING.txt" \
  "$run_root/OLD_OUTPUT_SHA256SUMS.txt" \
  "$run_root/CURRENT_OUTPUT_SHA256SUMS.txt" \
  "$run_root/old-new-comparison.json" \
  "$run_root/old-new-miniter5-comparison.json" \
  "$run_root/old-closure-validation.json" \
  "$run_root/current-closure-validation.json" \
  "$run_root/current-fixed-basis-validation.json" \
  "$run_root/current-initial-state-validation.json" \
  "$run_root/old-new-comparator.stdout" \
  "$run_root/old-new-comparator.stderr" \
  "$run_root/old-new-miniter5-comparator.stdout" \
  "$run_root/old-new-miniter5-comparator.stderr" \
  "$run_root/old-closure-validator.stdout" \
  "$run_root/old-closure-validator.stderr" \
  "$run_root/current-closure-validator.stdout" \
  "$run_root/current-closure-validator.stderr" \
  "$run_root/current-fixed-basis-validator.stdout" \
  "$run_root/current-fixed-basis-validator.stderr" \
  "$run_root/current-initial-state-validator.stdout" \
  "$run_root/current-initial-state-validator.stderr" \
  "$run_root/comparator-unit-test.stdout" \
  "$run_root/comparator-unit-test.stderr" \
  "$run_root/closure-unit-test.stdout" \
  "$run_root/closure-unit-test.stderr" \
  "$run_root/initial-state-unit-test.stdout" \
  "$run_root/initial-state-unit-test.stderr" \
  "$run_root/prerequisite-hash-check.log" \
  "$run_root/gate3-old-new-direct-miniter10-b2bc09d0-v4.slurm" \
  "$run_root/old-build-provenance.txt" \
  "$run_root/candidate-build-provenance.txt" \
  "$run_root/candidate-build-SHA256SUMS.txt" \
  "$run_root/DATASET_SHA256SUMS.txt" \
  "$run_root/common-input-OUTPUT_SHA256SUMS.txt" \
  "$run_root/common-input-PROVENANCE.txt" \
  "$run_root/input-preparation.json" \
  "$run_root/contract-validation.json" \
  "$run_root/legacy-root-compatibility.txt" \
  "$run_root/qsgw_input.contract" \
  "$run_root/input-verification-SHA256SUMS.txt" \
  "$run_root/input-verification-PROVENANCE.txt" \
  "$run_root/old-reader-preflight-OUTPUT_SHA256SUMS.txt" \
  "$run_root/old-reader-preflight-PROVENANCE.txt" \
  "$run_root/old-reader-preflight-ITERATION0_COMPARISON.txt" \
  "$run_root/gate2-OUTPUT_SHA256SUMS.txt" \
  "$run_root/gate2-PROVENANCE.txt" \
  "$run_root/gate2-initial-state-validation.json" \
  "$run_root/gate2-qsgw-vs-g0w0.json" \
  "$run_root/gate2-closure-validation.json" \
  "$run_root/gate2-fixed-basis-validation.json" \
  "$run_root/triangle-replay-SHA256SUMS.txt" \
  "$run_root/triangle-replay-PROVENANCE.txt" \
  "$run_root/triangle-replay-old-current-comparison.json" \
  "$run_root/triangle-replay-old-closure-validation.json" \
  "$run_root/triangle-replay-current-closure-validation.json" \
  "$run_root/triangle-replay-current-fixed-basis-validation.json" \
  "$run_root/miniter5-OUTPUT_SHA256SUMS.txt" \
  "$run_root/miniter5-PROVENANCE.txt" \
  "$run_root/miniter5-old-new-comparison.json" \
  "$run_root/miniter5-verification-SHA256SUMS.txt" \
  "$run_root/miniter5-verification-PROVENANCE.txt" \
  "$run_root/miniter5-v3-replay-SHA256SUMS.txt" \
  "$run_root/miniter5-v3-replay-PROVENANCE.txt" \
  "$run_root/miniter5-old-new-comparison-v3.json" \
  "$tool_dir/compare_qsgw_component_traces.py" \
  "$tool_dir/compare_qsgw_component_traces_v3.py" \
  "$tool_dir/test_compare_qsgw_legacy_v4_current_v5_v3.py" \
  "$tool_dir/validate_qsgw_trace_closure.py" \
  "$tool_dir/validate_qsgw_trace_closure_v3.py" \
  "$tool_dir/test_validate_qsgw_trace_closure_v3.py" \
  "$tool_dir/validate_qsgw_fixed_basis.py" \
  "$tool_dir/validate_qsgw_initial_state.py" \
  "$tool_dir/test_validate_qsgw_initial_state.py" \
  >"$run_root/OUTPUT_SHA256SUMS.txt"
touch "$run_root/COMPLETE"
printf 'GATE3_OLD_NEW_DIRECT_MINITER%s=PASS\n' "$target_iter"
