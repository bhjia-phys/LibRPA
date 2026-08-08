import unittest

import cmp_qp_sym


BANNER = "-" * 124


def _qp_text(blocks):
    """Build an `energy_qp`-style text.

    blocks: list of (kfrac, states, spin) with kfrac a 3-tuple of floats,
    states a list of (state, e_qp) floats, spin a 1-based spin label.
    """
    lines = ["  state     occ_num        e_gs(Ha)        e_qp(Ha)", BANNER]
    for index, (k, states, spin) in enumerate(blocks, 1):
        lines.append(" K_point {:3d} :{:10.7f}{:10.7f}{:10.7f}     Spin {:d}".format(
            index, k[0], k[1], k[2], spin))
        lines.append(BANNER)
        for state, eqp in states:
            lines.append("{:7d}{:10.5f}{:18.10E}{:18.10E}".format(
                state, 2.0, -0.5, eqp))
        lines.append(BANNER)
        lines.append("")
    return "\n".join(lines)


FULL = [
    ((0.0, 0.0, 0.0), [(1, -7.0737178992E-01), (2, -1.6557924365E-01), (3, 1.2345678901E-01)], 1),
    ((0.0, 0.0, 0.3333333), [(1, -6.4192085739E-01), (2, -4.0736291391E-01), (3, 2.2745510334E-01)], 1),
]


class TestCmpQpSym(unittest.TestCase):

    def _compare(self, test_blocks, ref_blocks, **kwargs):
        compare = cmp_qp_sym.sym_on_off(1e-4, **kwargs)
        test = {"librpa/energy_qp": _qp_text(test_blocks)}
        ref = {"librpa/energy_qp": _qp_text(ref_blocks)}
        return compare(test, ref)

    def test_identical_full_grid_pass(self):
        passed, msg = self._compare(FULL, FULL)
        self.assertTrue(passed)
        self.assertIn("2 common k", msg)
        self.assertIn("0 test-only k", msg)
        self.assertIn("0 ref-only k", msg)

    def test_common_k_mismatch_fails(self):
        broken = [[((0.0, 0.0, 0.0), [(1, -7.0737178992E-01), (2, -1.6557924365E-01),
                                       (3, 1.2345678901E-01)], 1)],
                  [((0.0, 0.0, 0.0), [(1, -7.0737178992E-01), (2, -1.6557924365E-01),
                                       (3, 1.3345678901E-01)], 1)]]
        passed, msg = self._compare(broken[0], broken[1])
        self.assertFalse(passed)
        self.assertIn("max abs diff", msg)
        self.assertIn("state 3", msg)

    def test_matching_nan_pairs_pass(self):
        test = [((0.0, 0.0, 0.0), [(1, float("nan"))], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, float("nan"))], 1)]
        passed, _ = self._compare(test, ref)
        self.assertTrue(passed)

    def test_nan_in_test_only_fails(self):
        test = [((0.0, 0.0, 0.0), [(1, float("nan"))], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.5)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertFalse(passed)
        self.assertIn("nan mismatch", msg)

    def test_k_point_order_insensitive(self):
        rev = list(reversed(FULL))
        passed, msg = self._compare(FULL, rev)
        self.assertTrue(passed)
        self.assertIn("2 common k", msg)

    def test_reciprocal_lattice_reduction(self):
        test = [((1.0, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)

    def test_spin_column_tolerance(self):
        import re as _re
        test_text = _qp_text([((0.0, 0.0, 0.0), [(1, -0.7)], 1)])
        ref_text = _re.sub(r"\s+Spin\s+\d+", "", test_text)
        compare = cmp_qp_sym.sym_on_off(1e-4)
        passed, msg = compare({"librpa/energy_qp": test_text},
                              {"librpa/energy_qp": ref_text})
        self.assertTrue(passed)

    def test_spin_channels_are_separate_buckets(self):
        test = [((0.0, 0.0, 0.0), [(1, -0.7)], 1),
                ((0.0, 0.0, 0.0), [(1, -0.8)], 2)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1),
               ((0.0, 0.0, 0.0), [(1, -0.8)], 2)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("2 common k", msg)

    def test_ref_only_nonibz_degenerate_pass(self):
        # test (symmetry-on) run holds only the IBZ k; ref (symmetry-off) run
        # adds a non-IBZ star member whose e_qp matches its representative.
        test = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1),
               ((0.5, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)
        self.assertIn("1 ref-only k", msg)

    def test_ref_only_nonibz_non_degenerate_fails(self):
        test = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1),
               ((0.5, 0.0, 0.0), [(1, -0.9), (2, 0.3)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertFalse(passed)
        self.assertIn("sanity check", msg)
        self.assertIn("state 1", msg)

    def test_test_only_nonibz_degenerate_pass(self):
        test = [((0.0, 0.0, 0.0), [(1, -0.7)], 1),
                ((0.5, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 test-only k", msg)

    def test_state_count_mismatch_fails(self):
        test = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertFalse(passed)
        self.assertIn("state count mismatch", msg)

    def test_missing_file_fails(self):
        compare = cmp_qp_sym.sym_on_off(1e-4)
        passed, msg = compare({}, {"librpa/energy_qp": _qp_text(FULL)})
        self.assertFalse(passed)
        self.assertIn("missing file", msg)

    def test_empty_output_fails(self):
        passed, msg = self._compare([], [])
        self.assertFalse(passed)
        self.assertIn("no QP blocks found", msg)

    def test_rounded_k_coordinates_match_within_ktol(self):
        test = [((0.0, 0.0, 0.3333333), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.3333334), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)

    def test_ambiguous_kpoint_match_fails(self):
        compare = cmp_qp_sym.sym_on_off(1e-4, ktol=0.5)
        test = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1),
               ((0.1, 0.0, 0.0), [(1, -0.7)], 1)]
        passed, msg = compare({"librpa/energy_qp": _qp_text(test)},
                              {"librpa/energy_qp": _qp_text(ref)})
        self.assertFalse(passed)
        self.assertIn("ambiguous k-point match", msg)

    def test_fortran_style_exponent_parsed(self):
        text = ("  state     occ_num        e_gs(Ha)        e_qp(Ha)\n"
                + BANNER + "\n"
                " K_point    1 :           0.0000          0.0000          0.0000\n"
                + BANNER + "\n"
                "      11    2.0000   -6.5993965145D-01   -7.0737178992D-01\n"
                + BANNER + "\n")
        blocks, errors = cmp_qp_sym.parse_energy_qp(text)
        self.assertEqual(errors, [])
        self.assertEqual(blocks[(0, (0.0, 0.0, 0.0))], {11: -7.0737178992E-01})

    def test_torus_distance_helper(self):
        self.assertAlmostEqual(cmp_qp_sym._torus_distance(0.0, 0.9999999), 1e-7)
        self.assertAlmostEqual(cmp_qp_sym._torus_distance(0.3333333, 0.3333334), 1e-7)
        self.assertAlmostEqual(cmp_qp_sym._torus_distance(0.0, 0.25), 0.25)
        self.assertAlmostEqual(cmp_qp_sym._torus_distance(0.0, 0.75), 0.25)

    def test_torus_matching_across_periodic_boundary(self):
        # k=(0,0,0) and k=(0,0,1-1e-7) are the same point on the [0,1) torus:
        # they must land in the common bucket instead of failing with
        # "no common k-points found".
        test = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.9999999), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)

    def test_negative_coordinate_reduces_across_boundary(self):
        # -1e-7 reduces mod 1 to 0.9999999 and must match 0.
        test = [((0.0, 0.0, -1e-7), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)

    def test_ref_only_uses_own_common_coordinates(self):
        # The common pair matches only within ktol (0 vs 1-1e-7), so its two
        # runs disagree on the printed coordinates. The ref-only k must be
        # checked against the ref run's own common coordinate; before the fix
        # this raised KeyError on the test coordinate.
        test = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        ref = [((0.0, 0.0, 0.9999999), [(1, -0.7), (2, 0.3)], 1),
               ((0.5, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)
        self.assertIn("1 ref-only k", msg)

    def test_test_only_uses_own_common_coordinates(self):
        # Mirror of test_ref_only_uses_own_common_coordinates for the test run.
        test = [((0.0, 0.0, 0.9999999), [(1, -0.7)], 1),
                ((0.5, 0.0, 0.0), [(1, -0.7)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertTrue(passed)
        self.assertIn("1 common k", msg)
        self.assertIn("1 test-only k", msg)

    def test_non_common_violation_count_uses_states(self):
        # The violation report counts states, not k-points: one non-common k
        # with two violating states must read "2/2 states across 1 k".
        test = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1)]
        ref = [((0.0, 0.0, 0.0), [(1, -0.7), (2, 0.3)], 1),
               ((0.5, 0.0, 0.0), [(1, -0.9), (2, 0.5)], 1)]
        passed, msg = self._compare(test, ref)
        self.assertFalse(passed)
        self.assertIn("ref-only: 2/2 states across 1 k", msg)

    def test_duplicate_kpoint_block_reports_error(self):
        text = _qp_text([((0.0, 0.0, 0.0), [(1, -0.7)], 1),
                         ((0.0, 0.0, 0.0), [(1, -0.7)], 1)])
        blocks, errors = cmp_qp_sym.parse_energy_qp(text)
        self.assertEqual(len(errors), 1)
        self.assertIn("duplicate k-point block", errors[0])


if __name__ == "__main__":
    unittest.main()
