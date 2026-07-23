from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("gate-a1-fullbz-old-current-v5.sh")


class FishFormalGateA1RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_binds_only_the_valid_legacy_full_bz_oracle(self):
        self.assertIn(
            "expected_old_commit=e08f4a130df7661e9ac355b9be45fb2bf9c3ed01",
            self.source,
        )
        self.assertIn("legacy_symmetry=off_full_bz", self.source)
        self.assertIn("--expected-legacy-symmetry off", self.source)
        self.assertNotIn("exact847", self.source)
        self.assertNotIn("legacy_symmetry=on", self.source)

    def test_binds_rebased_candidate_and_accepted_gate0(self):
        self.assertIn(
            "expected_upstream_commit=67b9888dac0d09870361398165d0b3c1acc931ff",
            self.source,
        )
        self.assertIn(
            "expected_candidate_commit=4f9ab0cfc90f54910158ab01a877581b080f136e",
            self.source,
        )
        self.assertIn("GATE0_PROVENANCE_SHA256", self.source)
        self.assertIn("CANDIDATE_EXE_SHA256", self.source)
        self.assertIn("gate=fish_gate0_current_v2", self.source)

    def test_requires_accepted_gate1_before_numerics(self):
        self.assertIn("GATE1_PROVENANCE_SHA256", self.source)
        self.assertIn(
            "librpa-qsgw-gate1-current-postcheck-20260723-36d74369-v1",
            self.source,
        )
        self.assertIn(
            "gate=fish_gate1_current_g0w0_ab_recovery_v2", self.source
        )
        self.assertIn('test -e "$candidate_gate1/GREEN_CONFIRMED"', self.source)
        self.assertIn('test ! -e "$candidate_gate1/FAILED"', self.source)
        self.assertIn(
            'sha256sum --check --quiet OUTPUT_SHA256SUMS.txt', self.source
        )

    def test_requires_accepted_gate2_before_numerics(self):
        self.assertIn("GATE2_PROVENANCE_SHA256", self.source)
        self.assertIn(
            "librpa-qsgw-gate2-current-20260723-dd7a75f2-v1",
            self.source,
        )
        self.assertIn(
            "gate=fish_gate2_current_qsgw_first_self_energy_v2",
            self.source,
        )
        self.assertIn('test -e "$candidate_gate2/GREEN_CONFIRMED"', self.source)
        self.assertIn('test ! -e "$candidate_gate2/FAILED"', self.source)
        self.assertIn("expected_gate2_output_manifest_sha", self.source)
        self.assertIn("sigc_block_count=48", self.source)
        self.assertIn("symmetry=exx_on_gw_on_rpa_on", self.source)
        self.assertIn("mixing=none", self.source)
        self.assertIn("candidate-gate2-PROVENANCE.txt", self.source)
        self.assertIn("candidate-gate2-OUTPUT_SHA256SUMS.txt", self.source)

    def test_binds_runner_checkout_and_writes_failure_marker(self):
        self.assertIn("RUNNER_SOURCE", self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" rev-parse HEAD', self.source)
        self.assertIn('git -C "$RUNNER_SOURCE" status --porcelain', self.source)
        self.assertIn('>"$run_root/FAILED"', self.source)
        self.assertIn('touch "$run_root/GREEN_CONFIRMED"', self.source)
        self.assertIn("run_succeeded=1", self.source)

    def test_runs_both_required_iteration_and_mixing_modes(self):
        self.assertIn("run_mode no-mix-miniter2 2 1 none 0.2", self.source)
        self.assertIn(
            "run_mode linear-beta-0.2-miniter5 5 0.2 linear 0.2",
            self.source,
        )
        self.assertIn('--iterations "0:$target_iter"', self.source)
        self.assertIn("qsgw_update_hartree = false", self.source)
        self.assertIn("replace_w_head = false", self.source)

    def test_uses_each_implementation_native_gap_trace(self):
        self.assertIn(
            'test -s "$legacy_run/homo_lumo_vs_iterations.dat"', self.source
        )
        self.assertNotIn(
            'test -s "$candidate_run/homo_lumo_vs_iterations.dat"',
            self.source,
        )
        self.assertIn('test -s "$candidate_run/qsgw_eigenvalues.dat"', self.source)
        self.assertIn('test -s "$candidate_run/qsgw_iterations.dat"', self.source)

    def test_runs_directly_on_fish_with_fixed_parallelism(self):
        self.assertIn("mpi_ranks=4", self.source)
        self.assertIn("omp_threads=12", self.source)
        self.assertIn('mpirun -np "$mpi_ranks"', self.source)
        self.assertIn("execution_surface=fish_direct", self.source)
        self.assertNotIn("#SBATCH", self.source)
        self.assertNotIn("SLURM_", self.source)

    def test_preserves_and_relocates_original_bundle_manifest(self):
        self.assertIn(
            "abacus-pinned-dd421665-si-k444-pair-physical-bundles-20260723-v2",
            self.source,
        )
        self.assertIn("expected_pair_output_manifest_sha", self.source)
        self.assertIn("expected_pair_validation_sha", self.source)
        self.assertIn('test -e "$pair_root/PAIR_COMPLETE"', self.source)
        self.assertIn("expected_common_output_manifest_sha", self.source)
        self.assertIn("common-input-OUTPUT_SHA256SUMS.txt", self.source)
        self.assertIn("common-input-OUTPUT_SHA256SUMS.relocated.txt", self.source)
        self.assertIn("sha256sum --check --quiet DATASET_SHA256SUMS.txt", self.source)

    def test_builds_one_validated_legacy_input_overlay_for_both_sides(self):
        self.assertIn("build_legacy_fullbz_stru_overlay_v2.py", self.source)
        self.assertIn("test_build_legacy_fullbz_stru_overlay_v2.py", self.source)
        self.assertIn("expected_overlay_builder_sha", self.source)
        self.assertIn("expected_overlay_builder_test_sha", self.source)
        self.assertIn("overlay_builder_source=$RUNNER_SOURCE/", self.source)
        self.assertIn("overlay_builder_test_source=$RUNNER_SOURCE/", self.source)
        self.assertNotIn("overlay_builder_source=$candidate_source/", self.source)
        self.assertNotIn("overlay_builder_test_source=$candidate_source/", self.source)
        self.assertIn("expected_overlay_dataset_manifest_sha", self.source)
        self.assertIn("expected_overlay_report_sha", self.source)
        self.assertIn("expected_overlay_stru_sha", self.source)
        self.assertIn("expected_overlay_vxc1_sha", self.source)
        self.assertIn("legacy_fullbz_same_input_overlay", self.source)
        self.assertIn("legacy_fullbz_input_overlay_v2", self.source)
        self.assertIn("same_input_for_legacy_and_candidate=true", self.source)
        self.assertIn('input_dir=$overlay_root/dataset', self.source)
        self.assertIn('source_input_dir=$common_root/dataset', self.source)
        self.assertIn('test "$input_dir/band_out" -ef "$source_input_dir/band_out"',
                      self.source)
        self.assertIn('test ! "$input_dir/stru_out" -ef "$source_input_dir/stru_out"',
                      self.source)
        self.assertIn('test "$(wc -l <"$input_dir/stru_out")" = 138', self.source)
        self.assertIn('"legacy_vxc_files_generated": 64', self.source)
        self.assertIn('"legacy_vxc_values_equal": true', self.source)
        self.assertIn('"dataset_file_count": 226', self.source)
        self.assertIn("-name 'vxcs1k*_nao.txt'", self.source)
        self.assertIn(
            'test "$input_dir/vxck1_nao.txt" -ef "$source_input_dir/vxck1_nao.txt"',
            self.source,
        )
        self.assertIn(
            'test ! "$input_dir/vxcs1k1_nao.txt" -ef '
            '"$source_input_dir/vxck1_nao.txt"',
            self.source,
        )
        self.assertIn(
            "! grep -Fq 'Both HF and VXC files not found'",
            self.source,
        )
        self.assertIn("find \"$overlay_root\" -type f -exec chmod a-w", self.source)
        self.assertNotIn("LEGACY_FULLBZ_STRU_OVERLAY.json", self.source)
        self.assertNotIn("TO_BE_FILLED", self.source)

    def test_orders_source_overlay_numerics_and_postflight(self):
        common_preflight = self.source.index("preflight=common_input_bundle")
        overlay_build = self.source.index("preflight=legacy_fullbz_same_input_overlay")
        numerics = self.source.index("numerics=begin")
        postflight = self.source.index("postflight=input_integrity")
        self.assertLess(common_preflight, overlay_build)
        self.assertLess(overlay_build, numerics)
        self.assertLess(numerics, postflight)

    def test_requires_full_matrix_and_state_observers(self):
        for observer in (
            "compare_qsgw_legacy_v4_current_v6.py",
            "validate_qsgw_trace_closure.py",
            "validate_qsgw_fixed_basis.py",
            "validate_qsgw_initial_state.py",
            "current-electron-count.txt",
        ):
            self.assertIn(observer, self.source)
        self.assertIn("matrix_state_and_invariants=passed", self.source)


if __name__ == "__main__":
    unittest.main()
