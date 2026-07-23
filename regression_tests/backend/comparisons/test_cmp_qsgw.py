import unittest

import cmp_qsgw


CONTRACT_HEADER = (
    "# qsgw_contract_version 6\n"
    "# fixed_basis immutable_mf0\n"
    "# live_update eigenvalues_wfc\n"
    "# velocity disabled_stage1\n"
    "# head disabled_stage1\n"
    "# wing disabled_stage1\n"
    "# symmetry unsupported_full_bz_only\n"
    "# hartree disabled_stage1\n"
    "# band disabled_stage1\n"
    "# h_qsgw_cut disabled_non_band\n"
    "# qsgw_input_contract qsgw_input.contract\n"
    "# qsgw_input_contract_sha256 " + "a" * 64 + "\n"
    "# qsgw_mixer linear\n"
    "# qsgw_mixing_beta 0.2\n"
)
MATRIX_HEADER = CONTRACT_HEADER + (
    "# iter channel component spin kpoint frequency_index frequency_Ha "
    "row column real_value imag_value\n"
)
EIGENVALUE_HEADER = CONTRACT_HEADER + (
    "# iter channel spin kpoint kx ky kz band energy_eV\n"
)
SUMMARY_HEADER = CONTRACT_HEADER + (
    "# iter max_delta_eV residual_l2_Ha residual_max_Ha "
    "efermi_eV gap_eV electron_count requested_mode applied_mode beta "
    "fallback rcond coefficient_l1 coefficient_count converged "
    "coefficients fallback_reason\n"
)


def matrix_rows(component, values, iteration=1, channel=0,
                frequency_index=-1, frequency=0.0):
    rows = []
    for row, entries in enumerate(values):
        for column, value in enumerate(entries):
            rows.append(
                "{} {} {} 0 0 {} {:.17e} {} {} {:.17e} {:.17e}\n".format(
                    iteration,
                    channel,
                    component,
                    frequency_index,
                    frequency,
                    row,
                    column,
                    complex(value).real,
                    complex(value).imag,
                )
            )
    return "".join(rows)


def band_table(conduction_shift=0.0, valence_shift=0.0):
    return (
        "1 0.0 0.0 0.0 "
        "2.0 {:.8f} 0.0 {:.8f} 0.0 2.00000000\n"
        "2 0.5 0.0 0.0 "
        "2.0 {:.8f} 0.0 {:.8f} 0.0 2.20000000\n"
    ).format(
        -1.0 + valence_shift,
        0.5 + conduction_shift,
        -0.8 + valence_shift,
        0.4 + conduction_shift,
    )


class TestQsgwBandIterations(unittest.TestCase):

    def _files(self, shifts=(0.0, 0.0)):
        return {
            "QSGW_band_spin_1_{}.dat".format(iteration):
                band_table(conduction_shift=shift)
            for iteration, shift in enumerate(shifts, 1)
        }

    def test_all_band_energies_and_indirect_gaps_pass_within_tolerance(self):
        compare = cmp_qsgw.band_iterations(
            occupied_bands="1",
            energy_tolerance_ev="1e-4",
            gap_tolerance_ev="2e-4",
        )

        passed, msg = compare(
            self._files((5.0e-5, -5.0e-5)),
            self._files(),
        )

        self.assertTrue(passed, msg)
        self.assertIn("max abs band-energy diff", msg)
        self.assertIn("max gap diff", msg)
        self.assertIn("2 iterations", msg)

    def test_any_band_energy_above_tolerance_fails(self):
        compare = cmp_qsgw.band_iterations(
            occupied_bands="1",
            energy_tolerance_ev="1e-4",
            gap_tolerance_ev="1e-3",
        )

        passed, msg = compare(
            self._files((2.0e-4,)),
            self._files((0.0,)),
        )

        self.assertFalse(passed)
        self.assertIn("band-energy", msg)

    def test_gap_above_tolerance_fails_without_mixing_diagnostics(self):
        compare = cmp_qsgw.band_iterations(
            occupied_bands="1",
            energy_tolerance_ev="1e-3",
            gap_tolerance_ev="1e-4",
        )

        passed, msg = compare(
            self._files((2.0e-4,)),
            self._files((0.0,)),
        )

        self.assertFalse(passed)
        self.assertIn("gap", msg)

    def test_iteration_files_must_start_at_one_and_be_continuous(self):
        files = {
            "QSGW_band_spin_1_1.dat": band_table(),
            "QSGW_band_spin_1_3.dat": band_table(),
        }
        compare = cmp_qsgw.band_iterations(occupied_bands="1")

        passed, msg = compare(files, files)

        self.assertFalse(passed)
        self.assertIn("continuous", msg)


class TestQsgwMatrixTrace(unittest.TestCase):

    def _compare(self, test, reference, **kwargs):
        kwargs.setdefault("require_complete_trajectory", "false")
        compare = cmp_qsgw.matrix_trace(**kwargs)
        return compare({"qsgw_matrices.dat": test},
                       {"qsgw_matrices.dat": reference})

    def test_matching_blocks_and_invariants_pass(self):
        reference = (
            MATRIX_HEADER
            + matrix_rows("h0", [[1.0, 0.1j], [-0.1j, 2.0]])
            + matrix_rows("rotation_u", [[1.0, 0.0], [0.0, 1.0]])
        )
        test = (
            MATRIX_HEADER
            + matrix_rows("h0", [[1.0 + 1.0e-10, 0.1j],
                                 [-0.1j, 2.0]])
            + matrix_rows("rotation_u", [[1.0, 0.0], [0.0, 1.0]])
        )

        passed, msg = self._compare(
            test, reference,
            relative_tolerance="1e-8",
            absolute_tolerance="1e-12",
            hermiticity_tolerance="1e-10",
            unitarity_tolerance="1e-10",
        )

        self.assertTrue(passed, msg)
        self.assertIn("max relative Frobenius", msg)

    def test_missing_matrix_element_fails(self):
        reference = MATRIX_HEADER + matrix_rows("h0", [[1.0, 0.0], [0.0, 2.0]])
        test = reference.rsplit("\n", 2)[0] + "\n"

        passed, msg = self._compare(test, reference)

        self.assertFalse(passed)
        self.assertIn("incomplete matrix block", msg)

    def test_duplicate_matrix_element_fails(self):
        row = matrix_rows("h0", [[1.0]])
        trace = MATRIX_HEADER + row + row

        passed, msg = self._compare(trace, MATRIX_HEADER + row)

        self.assertFalse(passed)
        self.assertIn("duplicate matrix row", msg)

    def test_nonhermitian_static_component_fails_even_if_equal(self):
        trace = MATRIX_HEADER + matrix_rows(
            "raw_h", [[1.0, 0.2j], [0.1j, 2.0]]
        )

        passed, msg = self._compare(
            trace, trace, hermiticity_tolerance="1e-12"
        )

        self.assertFalse(passed)
        self.assertIn("Hermiticity", msg)

    def test_nonunitary_rotation_fails_even_if_equal(self):
        trace = MATRIX_HEADER + matrix_rows(
            "rotation_u", [[1.0, 0.0], [0.0, 1.01]]
        )

        passed, msg = self._compare(
            trace, trace, unitarity_tolerance="1e-10"
        )

        self.assertFalse(passed)
        self.assertIn("unitarity", msg)

    def test_zero_reference_uses_absolute_tolerance(self):
        reference = MATRIX_HEADER + matrix_rows("vc", [[0.0]])
        test = MATRIX_HEADER + matrix_rows("vc", [[2.0e-11]])

        passed, msg = self._compare(
            test, reference,
            relative_tolerance="1e-8",
            absolute_tolerance="1e-12",
        )

        self.assertFalse(passed)
        self.assertIn("Frobenius tolerance", msg)

    def test_nonzero_reference_strictly_uses_relative_tolerance(self):
        reference = MATRIX_HEADER + matrix_rows("vc", [[1.0e-9]])
        test = MATRIX_HEADER + matrix_rows("vc", [[1.0005e-9]])

        passed, msg = self._compare(
            test, reference,
            relative_tolerance="1e-8",
            absolute_tolerance="1e-12",
        )

        self.assertFalse(passed)
        self.assertIn("Frobenius tolerance", msg)

    def test_complete_stage_one_trajectory_is_required_by_default(self):
        rows = [
            matrix_rows("h0", [[1.0]], iteration=0),
            matrix_rows("vxc_dft", [[-0.2]], iteration=0),
            matrix_rows("wfc_spinor0", [[1.0]], iteration=0),
            matrix_rows("occupation", [[1.0]], iteration=0),
            matrix_rows("sigma_c_iw", [[0.1 + 0.02j]],
                        frequency_index=0, frequency=0.25),
            matrix_rows("exx", [[-0.3]]),
            matrix_rows("vc", [[0.1]]),
            matrix_rows("raw_h", [[0.6]]),
            matrix_rows("mixed_h", [[0.52]]),
            matrix_rows("rotation_u", [[1.0]]),
            matrix_rows("wfc_spinor0", [[1.0]]),
            matrix_rows("occupation", [[1.0]]),
        ]
        trace = MATRIX_HEADER + "".join(rows)
        compare = cmp_qsgw.matrix_trace()

        passed, msg = compare(
            {"qsgw_matrices.dat": trace},
            {"qsgw_matrices.dat": trace},
        )
        self.assertTrue(passed, msg)

        missing_exx = MATRIX_HEADER + "".join(
            row for index, row in enumerate(rows) if index != 5
        )
        passed, msg = compare(
            {"qsgw_matrices.dat": missing_exx},
            {"qsgw_matrices.dat": missing_exx},
        )
        self.assertFalse(passed)
        self.assertIn("missing required components", msg)

    def test_same_grid_head_contract_uses_the_emitted_grid_trace_schema(self):
        header = MATRIX_HEADER.replace(
            "# velocity disabled_stage1",
            "# velocity fixed_basis_rotation",
        ).replace(
            "# head disabled_stage1",
            "# head scf_grid_analytic_live",
        )
        rows = [
            matrix_rows("h0", [[1.0]], iteration=0),
            matrix_rows("vxc_dft", [[-0.2]], iteration=0),
            matrix_rows("wfc_spinor0", [[1.0]], iteration=0),
            matrix_rows("occupation", [[1.0]], iteration=0),
            matrix_rows("sigma_c_iw", [[0.1 + 0.02j]],
                        frequency_index=0, frequency=0.25),
            matrix_rows("exx", [[-0.3]]),
            matrix_rows("vc", [[0.1]]),
            matrix_rows("raw_h", [[0.6]]),
            matrix_rows("mixed_h", [[0.52]]),
            matrix_rows("rotation_u", [[1.0]]),
            matrix_rows("wfc_spinor0", [[1.0]]),
            matrix_rows("occupation", [[1.0]]),
        ]
        trace = header + "".join(rows)

        passed, msg = self._compare(trace, trace)

        self.assertTrue(passed, msg)

    def test_band_trajectory_accepts_legacy_direct_rotation_components(self):
        header = MATRIX_HEADER.replace(
            "# band disabled_stage1",
            "# band fixed_reference_rotation_live",
        ).replace(
            "# h_qsgw_cut disabled_non_band",
            "# h_qsgw_cut band_postprocess\n"
            "# qsgw_band0_unoccupied_keep 10\n"
            "# qsgw_band0_cut_mode 2\n"
            "# qsgw_band0_cut_shift_ha 20",
        )
        grid_rows = [
            matrix_rows("h0", [[1.0]], iteration=0),
            matrix_rows("vxc_dft", [[-0.2]], iteration=0),
            matrix_rows("wfc_spinor0", [[1.0]], iteration=0),
            matrix_rows("occupation", [[1.0]], iteration=0),
            matrix_rows("sigma_c_iw", [[0.1 + 0.02j]],
                        frequency_index=0, frequency=0.25),
            matrix_rows("exx", [[-0.3]]),
            matrix_rows("vc", [[0.1]]),
            matrix_rows("raw_h", [[0.6]]),
            matrix_rows("mixed_h", [[0.52]]),
            matrix_rows("rotation_u", [[1.0]]),
            matrix_rows("wfc_spinor0", [[1.0]]),
            matrix_rows("occupation", [[1.0]]),
        ]
        band_rows = [
            matrix_rows("h0", [[1.1]], iteration=0, channel=1),
            matrix_rows("vxc_dft", [[-0.25]], iteration=0, channel=1),
            matrix_rows("wfc_spinor0", [[1.0]], iteration=0, channel=1),
            matrix_rows("occupation", [[1.0]], iteration=0, channel=1),
            matrix_rows("exx", [[-0.28]], channel=1),
            matrix_rows("vc", [[0.12]], channel=1),
            matrix_rows("raw_h", [[0.69]], channel=1),
            matrix_rows("mixed_h", [[0.608]], channel=1),
            matrix_rows("rotation_u", [[1.0]], channel=1),
            matrix_rows("wfc_spinor0", [[1.0]], channel=1),
            matrix_rows("occupation", [[1.0]], channel=1),
        ]
        trace = header + "".join(grid_rows + band_rows)
        compare = cmp_qsgw.matrix_trace()

        passed, msg = compare(
            {"qsgw_matrices.dat": trace},
            {"qsgw_matrices.dat": trace},
        )
        self.assertTrue(passed, msg)


class TestQsgwEigenvalueTrace(unittest.TestCase):

    def _trace(self, energy_ev, kx=0.0):
        return EIGENVALUE_HEADER + (
            "0 0 0 0 {:.17e} 0.0 0.0 0 5.00000000000000000e-1\n"
            "1 0 0 0 {:.17e} 0.0 0.0 0 {:.17e}\n"
        ).format(kx, kx, energy_ev)

    def _compare(self, test, reference, **kwargs):
        compare = cmp_qsgw.eigenvalue_trace(**kwargs)
        return compare({"qsgw_eigenvalues.dat": test},
                       {"qsgw_eigenvalues.dat": reference})

    def test_energy_difference_within_hartree_tolerance_passes(self):
        reference = self._trace(1.0)
        test = self._trace(1.0 + 0.5 * cmp_qsgw.HA2EV * 1.0e-6)

        passed, msg = self._compare(test, reference, tolerance_ha="1e-6")

        self.assertTrue(passed, msg)
        self.assertIn("max abs eigenvalue diff", msg)

    def test_energy_difference_above_hartree_tolerance_fails(self):
        reference = self._trace(1.0)
        test = self._trace(1.0 + 2.0 * cmp_qsgw.HA2EV * 1.0e-6)

        passed, msg = self._compare(test, reference, tolerance_ha="1e-6")

        self.assertFalse(passed)
        self.assertIn("max abs eigenvalue diff", msg)

    def test_kpoint_coordinate_mismatch_fails(self):
        passed, msg = self._compare(
            self._trace(1.0, kx=1.0e-6),
            self._trace(1.0),
            coordinate_tolerance="1e-12",
        )

        self.assertFalse(passed)
        self.assertIn("k-point coordinate", msg)


class TestQsgwIterationSummary(unittest.TestCase):

    def _trace(self, gap=1.0, applied_mode=0):
        return SUMMARY_HEADER + (
            "0 0.0 0.0 0.0 -1.0 {:.17e} 8.0 "
            "-1 -1 2.0e-1 0 1.0 0.0 0 0 none none\n"
            "1 1.0e-3 2.0e-4 1.0e-4 -1.0 {:.17e} 8.0 "
            "0 {} 2.0e-1 0 1.0 1.0 1 0 1.0 none\n"
        ).format(gap, gap, applied_mode)

    def _compare(self, test, reference, **kwargs):
        compare = cmp_qsgw.iteration_summary(**kwargs)
        return compare({"qsgw_iterations.dat": test},
                       {"qsgw_iterations.dat": reference})

    def test_summary_within_field_tolerances_passes(self):
        passed, msg = self._compare(
            self._trace(gap=1.0 + 5.0e-6),
            self._trace(gap=1.0),
            energy_tolerance_ev="1e-5",
        )

        self.assertTrue(passed, msg)
        self.assertIn("max gap diff", msg)

    def test_gap_above_tolerance_fails(self):
        passed, msg = self._compare(
            self._trace(gap=1.0 + 2.0e-5),
            self._trace(gap=1.0),
            energy_tolerance_ev="1e-5",
        )

        self.assertFalse(passed)
        self.assertIn("gap", msg)

    def test_discrete_mixing_metadata_mismatch_fails(self):
        passed, msg = self._compare(
            self._trace(applied_mode=1), self._trace(applied_mode=0)
        )

        self.assertFalse(passed)
        self.assertIn("applied_mode", msg)

    def test_contract_mismatch_fails(self):
        test = self._trace().replace(
            "# qsgw_mixing_beta 0.2", "# qsgw_mixing_beta 0.3"
        )

        passed, msg = self._compare(test, self._trace())

        self.assertFalse(passed)
        self.assertIn("contract differs", msg)

    def test_missing_contract_fails(self):
        trace = self._trace().replace(CONTRACT_HEADER, "")

        passed, msg = self._compare(trace, trace)

        self.assertFalse(passed)
        self.assertIn("missing QSGW contract", msg)

    def test_iteration_zero_is_required(self):
        trace = self._trace()
        trace = trace.replace(
            "0 0.0 0.0 0.0 -1.0 1.00000000000000000e+00 8.0 "
            "-1 -1 2.0e-1 0 1.0 0.0 0 0 none none\n",
            "",
        )

        passed, msg = self._compare(trace, trace)

        self.assertFalse(passed)
        self.assertIn("iteration zero", msg)

    def test_contract_accepts_explicit_symmetry_and_rejects_invalid_modes(self):
        symmetric = self._trace().replace(
            "# symmetry unsupported_full_bz_only",
            "# symmetry exx_on_gw_on_rpa_on",
        )
        passed, msg = self._compare(symmetric, symmetric)
        self.assertTrue(passed, msg)

        invalid_symmetry = self._trace().replace(
            "# symmetry unsupported_full_bz_only",
            "# symmetry crystal_reduction",
        )
        passed, msg = self._compare(invalid_symmetry, invalid_symmetry)
        self.assertFalse(passed)
        self.assertIn("symmetry", msg)

        inconsistent_headwing = self._trace().replace(
            "# head disabled_stage1",
            "# head scf_grid_analytic_live",
        )
        passed, msg = self._compare(
            inconsistent_headwing, inconsistent_headwing
        )
        self.assertFalse(passed)
        self.assertIn("headwing/velocity", msg)

        incomplete_hartree = self._trace().replace(
            "# hartree disabled_stage1", "# hartree delta_density"
        )
        passed, msg = self._compare(incomplete_hartree, incomplete_hartree)
        self.assertFalse(passed)
        self.assertIn("enabled Hartree", msg)

    def test_contract_accepts_v5_headwing_metadata(self):
        legacy = (
            self._trace()
            .replace("# qsgw_contract_version 6", "# qsgw_contract_version 5")
            .replace(
                "# head disabled_stage1\n# wing disabled_stage1",
                "# headwing disabled_stage1",
            )
            .replace("# h_qsgw_cut disabled_non_band\n", "")
        )

        passed, msg = self._compare(legacy, legacy)

        self.assertTrue(passed, msg)

    def test_contract_v6_rejects_legacy_headwing_or_enabled_wing(self):
        legacy_field = self._trace().replace(
            "# head disabled_stage1\n# wing disabled_stage1",
            "# headwing disabled_stage1",
        )
        passed, msg = self._compare(legacy_field, legacy_field)
        self.assertFalse(passed)
        self.assertIn("split head/wing", msg)

        enabled_wing = self._trace().replace(
            "# wing disabled_stage1", "# wing scf_grid_analytic_live"
        )
        passed, msg = self._compare(enabled_wing, enabled_wing)
        self.assertFalse(passed)
        self.assertIn("unsupported iterative wing", msg)


if __name__ == "__main__":
    unittest.main()
