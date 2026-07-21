#!/usr/bin/env python3
"""Check that QSGW band cut-mode runs form a controlled numerical trio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cmp_qsgw


SCHEMA = "librpa-qsgw-band-cut-mode-comparison-v1"


class CutModeComparisonError(ValueError):
    pass


def _load_run(root, expected_mode):
    paths = {
        "matrix": root / "qsgw_matrices.dat",
        "eigenvalue": root / "qsgw_eigenvalues.dat",
        "summary": root / "qsgw_iterations.dat",
        "validation": root / "band-validation.json",
    }
    texts = {
        key: path.read_text(encoding="ascii")
        for key, path in paths.items()
        if key != "validation"
    }
    contracts = {
        key: cmp_qsgw._parse_contract(text, "{} {}".format(root, key))
        for key, text in texts.items()
    }
    if any(value != contracts["matrix"] for value in contracts.values()):
        raise CutModeComparisonError("{} trace contracts differ".format(root))
    contract = contracts["matrix"]
    if contract.get("qsgw_band0_cut_mode") != expected_mode:
        raise CutModeComparisonError("{} has the wrong cut mode".format(root))
    validation = json.loads(paths["validation"].read_text(encoding="ascii"))
    if validation.get("passed") is not True:
        raise CutModeComparisonError("{} did not pass its absolute validation".format(root))
    return {
        "root": root,
        "contract": contract,
        "blocks": cmp_qsgw._parse_matrix_trace(texts["matrix"], str(root)),
        "validation": validation,
    }


def _normalize_contract(contract):
    result = dict(contract)
    result["qsgw_band0_cut_mode"] = "CONTROLLED_FACTOR"
    return result


def _matrix(blocks, key):
    return np.asarray(blocks[key][1], dtype=np.complex128)


def _metrics(actual, expected):
    delta = actual - expected
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    numerator = float(np.linalg.norm(delta))
    denominator = float(np.linalg.norm(expected))
    relative = numerator / denominator if denominator else numerator
    return maximum, relative


def _require_close(actual, expected, label, absolute_tolerance,
                   relative_tolerance):
    if actual.shape != expected.shape:
        raise CutModeComparisonError("{} shape differs".format(label))
    maximum, relative = _metrics(actual, expected)
    if maximum > absolute_tolerance or relative > relative_tolerance:
        raise CutModeComparisonError(
            "{} differs: max_abs={:.6e}, relative_frobenius={:.6e}".format(
                label, maximum, relative
            )
        )
    return maximum, relative


def compare_cut_modes(mode0_root, mode1_root, mode2_root,
                      absolute_tolerance=1.0e-10,
                      relative_tolerance=1.0e-8):
    runs = {
        0: _load_run(mode0_root, 0),
        1: _load_run(mode1_root, 1),
        2: _load_run(mode2_root, 2),
    }
    normalized = [_normalize_contract(run["contract"]) for run in runs.values()]
    if any(item != normalized[0] for item in normalized[1:]):
        differing = sorted(
            key
            for key in set().union(*(item.keys() for item in normalized))
            if len({repr(item.get(key)) for item in normalized}) != 1
        )
        raise CutModeComparisonError(
            "cut-mode contracts differ outside the controlled factor: {}".format(differing)
        )

    baseline_keys = set(runs[0]["blocks"])
    if any(set(run["blocks"]) != baseline_keys for run in runs.values()):
        raise CutModeComparisonError("cut-mode matrix layouts differ")
    invariant_keys = {
        key
        for key in baseline_keys
        if key[0] == 0
        or (
            key[0] == 1
            and key[2]
            in {
                "sigma_c_iw",
                "exx",
                "vc",
                "basis_inverse_residual",
                "basis_condition_estimate",
                "fourier_orthogonality_residual",
                "source_roundtrip_relative_error",
                "target_hermiticity_error",
                "target_relative_hermiticity_error",
                "repaired_target_hermiticity_error",
            }
        )
    }
    maximum = 0.0
    relative = 0.0
    for mode in (1, 2):
        for key in invariant_keys:
            reference_frequency = runs[0]["blocks"][key][0]
            observed_frequency = runs[mode]["blocks"][key][0]
            if abs(reference_frequency - observed_frequency) > absolute_tolerance:
                raise CutModeComparisonError("frequency differs for {}".format(key))
            block_maximum, block_relative = _require_close(
                _matrix(runs[mode]["blocks"], key),
                _matrix(runs[0]["blocks"], key),
                "mode 0 vs {} invariant {}".format(mode, key),
                absolute_tolerance,
                relative_tolerance,
            )
            maximum = max(maximum, block_maximum)
            relative = max(relative, block_relative)

    active_maximum = 0.0
    active_relative = 0.0
    for component in ("raw_h", "mixed_h"):
        keys = sorted(
            key
            for key in baseline_keys
            if key[0] == 1 and key[2] == component and key[1] in (0, 1)
        )
        for key in keys:
            iteration, channel, _component, spin, kpoint, _frequency = key
            active_key = "{}:{}:{}:{}".format(iteration, channel, spin, kpoint)
            limits = {
                mode: runs[mode]["validation"]["active_limits"][active_key]
                for mode in (1, 2)
            }
            if limits[1] != limits[2]:
                raise CutModeComparisonError("mode 1/2 active limits differ")
            active_limit = limits[1]
            mode1 = _matrix(runs[1]["blocks"], key)[:active_limit, :active_limit]
            mode2 = _matrix(runs[2]["blocks"], key)[:active_limit, :active_limit]
            block_maximum, block_relative = _require_close(
                mode2,
                mode1,
                "mode 1/2 active block {}".format(key),
                absolute_tolerance,
                relative_tolerance,
            )
            active_maximum = max(active_maximum, block_maximum)
            active_relative = max(active_relative, block_relative)

    byte_equal_tables = {}
    for prefix in ("KS", "EXX"):
        filename = "{}_band_spin_1_1.dat".format(prefix)
        payloads = [(runs[mode]["root"] / filename).read_bytes() for mode in (0, 1, 2)]
        byte_equal_tables[prefix.lower()] = payloads[1:] == payloads[:1] * 2
        if not byte_equal_tables[prefix.lower()]:
            raise CutModeComparisonError("{} tables differ across cut modes".format(prefix))

    return {
        "schema": SCHEMA,
        "passed": True,
        "controlled_factor": "qsgw_band0_cut_mode",
        "configured_unoccupied_keep": runs[0]["contract"]["qsgw_band0_unoccupied_keep"],
        "configured_shift_ha": runs[0]["contract"]["qsgw_band0_cut_shift_ha"],
        "invariant_matrix_max_abs_ha": maximum,
        "invariant_matrix_relative_frobenius": relative,
        "mode1_mode2_active_block_max_abs_ha": active_maximum,
        "mode1_mode2_active_block_relative_frobenius": active_relative,
        "byte_equal_cut_independent_tables": byte_equal_tables,
        "thresholds": {
            "absolute_ha": absolute_tolerance,
            "relative_frobenius": relative_tolerance,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode0", type=Path)
    parser.add_argument("mode1", type=Path)
    parser.add_argument("mode2", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = compare_cut_modes(args.mode0, args.mode1, args.mode2)
    except (CutModeComparisonError, ValueError, OSError) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        raise SystemExit(str(error))
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
