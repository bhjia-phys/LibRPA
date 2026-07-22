#!/usr/bin/env python3
"""Compare legacy and current LibRPA SigcRF checkpoint directories."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


SCHEMA = "librpa-exact847-sigcrf-comparison-v1"
FILENAME = re.compile(
    r"^SigcRF_ispin_(\d+)_s_(\d+)_iomega_(\d+)_myid_(\d+)\.dat$"
)


class ComparisonError(ValueError):
    pass


def read_file(path: Path) -> dict[tuple[int, int, int], np.ndarray]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ComparisonError(f"truncated SigcRF header: {path}")
    block_count = struct.unpack_from("<Q", payload, 0)[0]
    offset = 8
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    for block_index in range(block_count):
        if offset + 40 > len(payload):
            raise ComparisonError(
                f"truncated SigcRF dimensions in {path} block {block_index}"
            )
        r_index, atom_i, atom_j, rows, columns = struct.unpack_from(
            "<5Q", payload, offset
        )
        offset += 40
        element_count = rows * columns
        byte_count = element_count * 16
        if offset + byte_count > len(payload):
            raise ComparisonError(
                f"truncated SigcRF matrix in {path} block {block_index}"
            )
        key = (int(r_index), int(atom_i), int(atom_j))
        if key in blocks:
            raise ComparisonError(f"duplicate SigcRF block {key} in {path}")
        blocks[key] = np.frombuffer(
            payload, dtype="<c16", count=element_count, offset=offset
        ).copy().reshape(int(rows), int(columns))
        offset += byte_count
    if offset != len(payload):
        raise ComparisonError(f"trailing bytes in SigcRF file: {path}")
    return blocks


def read_directory(
    directory: Path,
) -> tuple[dict[tuple[int, int, int, int, int, int], np.ndarray], int]:
    matrices: dict[tuple[int, int, int, int, int, int], np.ndarray] = {}
    file_count = 0
    for path in sorted(directory.glob("SigcRF_*.dat")):
        match = FILENAME.match(path.name)
        if match is None:
            continue
        spin, spin_pair, frequency_index, _rank = map(int, match.groups())
        file_count += 1
        for (r_index, atom_i, atom_j), matrix in read_file(path).items():
            key = (
                spin,
                spin_pair,
                frequency_index,
                r_index,
                atom_i,
                atom_j,
            )
            if key in matrices:
                raise ComparisonError(
                    f"duplicate distributed SigcRF block {key} in {directory}"
                )
            matrices[key] = matrix
    if file_count == 0:
        raise ComparisonError(f"no SigcRF files found in {directory}")
    if not matrices:
        raise ComparisonError(f"no SigcRF blocks found in {directory}")
    return matrices, file_count


def metrics(
    reference: dict[tuple[int, int, int, int, int, int], np.ndarray],
    observed: dict[tuple[int, int, int, int, int, int], np.ndarray],
) -> dict[str, object]:
    if set(reference) != set(observed):
        missing = sorted(set(reference) - set(observed))
        extra = sorted(set(observed) - set(reference))
        raise ComparisonError(
            f"SigcRF block-key mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    max_abs = -1.0
    maximum_key: tuple[int, int, int, int, int, int] | None = None
    maximum_index: tuple[int, int] | None = None
    maximum_difference = 0.0j
    difference_norm_squared = 0.0
    reference_norm_squared = 0.0
    observed_norm_squared = 0.0
    element_count = 0
    for key in sorted(reference):
        reference_matrix = reference[key]
        observed_matrix = observed[key]
        if reference_matrix.shape != observed_matrix.shape:
            raise ComparisonError(
                f"SigcRF block shape mismatch for {key}: "
                f"{reference_matrix.shape} != {observed_matrix.shape}"
            )
        difference = observed_matrix - reference_matrix
        flat_index = int(np.argmax(np.abs(difference)))
        block_index = np.unravel_index(flat_index, difference.shape)
        block_difference = difference[block_index]
        if abs(block_difference) > max_abs:
            max_abs = float(abs(block_difference))
            maximum_key = key
            maximum_index = (int(block_index[0]), int(block_index[1]))
            maximum_difference = complex(block_difference)
        difference_norm_squared += float(np.vdot(difference, difference).real)
        reference_norm_squared += float(
            np.vdot(reference_matrix, reference_matrix).real
        )
        observed_norm_squared += float(
            np.vdot(observed_matrix, observed_matrix).real
        )
        element_count += int(reference_matrix.size)

    scale = max(
        math.sqrt(reference_norm_squared),
        math.sqrt(observed_norm_squared),
        1.0e-30,
    )
    assert maximum_key is not None and maximum_index is not None
    return {
        "block_count": len(reference),
        "element_count": element_count,
        "max_abs_ha": max_abs,
        "relative_frobenius": math.sqrt(difference_norm_squared) / scale,
        "maximum_difference_key": list(maximum_key),
        "maximum_difference_matrix_index": list(maximum_index),
        "maximum_difference_real_ha": maximum_difference.real,
        "maximum_difference_imag_ha": maximum_difference.imag,
    }


def analyze(
    reference: dict[tuple[int, int, int, int, int, int], np.ndarray],
    observed: dict[tuple[int, int, int, int, int, int], np.ndarray],
    reference_file_count: int,
    observed_file_count: int,
    max_abs_tolerance_ha: float,
    relative_frobenius_tolerance: float,
) -> dict[str, object]:
    overall = metrics(reference, observed)
    frequency_indices = sorted({key[2] for key in reference})
    by_frequency = []
    for frequency_index in frequency_indices:
        reference_frequency = {
            key: value for key, value in reference.items() if key[2] == frequency_index
        }
        observed_frequency = {
            key: value for key, value in observed.items() if key[2] == frequency_index
        }
        by_frequency.append(
            {
                "frequency_index": frequency_index,
                **metrics(reference_frequency, observed_frequency),
            }
        )
    passed = (
        overall["max_abs_ha"] <= max_abs_tolerance_ha
        and overall["relative_frobenius"] <= relative_frobenius_tolerance
    )
    return {
        "schema": SCHEMA,
        "diagnostic_complete": True,
        "reference_file_count": reference_file_count,
        "observed_file_count": observed_file_count,
        "frequency_count": len(frequency_indices),
        "overall": overall,
        "by_frequency": by_frequency,
        "thresholds": {
            "max_abs_ha": max_abs_tolerance_ha,
            "relative_frobenius": relative_frobenius_tolerance,
        },
        "numerical_parity_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("observed_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-abs-tolerance-ha", type=float, default=1.0e-6)
    parser.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1.0e-8
    )
    args = parser.parse_args()
    try:
        reference, reference_file_count = read_directory(args.reference_dir)
        observed, observed_file_count = read_directory(args.observed_dir)
        report = analyze(
            reference,
            observed,
            reference_file_count,
            observed_file_count,
            args.max_abs_tolerance_ha,
            args.relative_frobenius_tolerance,
        )
        report["inputs"] = {
            "reference_dir": str(args.reference_dir.resolve()),
            "observed_dir": str(args.observed_dir.resolve()),
        }
    except (ComparisonError, OSError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "diagnostic_complete": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("diagnostic_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
