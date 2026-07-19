#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

import numpy as np


FILE_PATTERN = re.compile(
    r"^Sigc_fk_mn_(?P<source>.+)_ispin_(?P<spin>\d+)_ik_(?P<kpoint>\d+)"
    r"_ifreq_(?P<frequency>\d+)\.bin$"
)


class DumpComparisonError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _read_sigc_binary(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise DumpComparisonError(f"{path} is too short for a SigC header")
    dimension, scalar_bytes = struct.unpack("=ii", raw[:8])
    if dimension <= 0 or scalar_bytes != 8:
        raise DumpComparisonError(
            f"{path} has invalid header dimension={dimension}, "
            f"scalar_bytes={scalar_bytes}"
        )
    expected = 8 + dimension * dimension * 16
    if len(raw) != expected:
        raise DumpComparisonError(
            f"{path} has {len(raw)} bytes, expected {expected}"
        )
    values = np.frombuffer(raw, dtype=np.dtype("=f8"), offset=8)
    pairs = values.reshape(dimension, dimension, 2)
    return pairs[:, :, 0] + 1j * pairs[:, :, 1]


def _load_directory(
    directory: Path, source: str
) -> tuple[dict[tuple[int, int, int], np.ndarray], str]:
    directory = Path(directory).resolve()
    matrices = {}
    manifest_lines = []
    for path in sorted(directory.glob("Sigc_fk_mn_*.bin")):
        match = FILE_PATTERN.match(path.name)
        if match is None or match.group("source") != source:
            continue
        key = tuple(
            int(match.group(field))
            for field in ("spin", "kpoint", "frequency")
        )
        if key in matrices:
            raise DumpComparisonError(f"duplicate SigC block {key}")
        matrices[key] = _read_sigc_binary(path)
        manifest_lines.append(f"{path.name} {_sha256(path)}\n")
    if not matrices:
        raise DumpComparisonError(
            f"no SigC binaries for source={source} in {directory}"
        )
    manifest_sha = hashlib.sha256(
        "".join(manifest_lines).encode("ascii")
    ).hexdigest()
    return matrices, manifest_sha


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = {
        "q000": 0.0,
        "q050": 0.5,
        "q090": 0.9,
        "q099": 0.99,
        "q100": 1.0,
    }
    return {
        label: float(np.quantile(values, probability))
        for label, probability in probabilities.items()
    }


def compare_directories(
    reference_directory: Path,
    candidate_directory: Path,
    *,
    source: str = "kgrid",
    max_abs_tolerance_ha: float = 1.0e-10,
    relative_frobenius_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    if min(max_abs_tolerance_ha, relative_frobenius_tolerance) < 0.0:
        raise DumpComparisonError("comparison tolerances must be nonnegative")
    reference, reference_manifest = _load_directory(
        reference_directory, source
    )
    candidate, candidate_manifest = _load_directory(
        candidate_directory, source
    )
    if set(reference) != set(candidate):
        raise DumpComparisonError(
            "SigC block keys differ: "
            f"reference_only={sorted(set(reference) - set(candidate))}, "
            f"candidate_only={sorted(set(candidate) - set(reference))}"
        )

    maximum = -1.0
    maximum_location = None
    reference_at_max = 0.0j
    candidate_at_max = 0.0j
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    exact_blocks = 0
    exact_elements = 0
    element_count = 0
    dimensions = set()
    all_absolute_differences = []
    block_records = []
    frequency_maxima = {}

    for key in sorted(reference):
        reference_matrix = reference[key]
        candidate_matrix = candidate[key]
        if reference_matrix.shape != candidate_matrix.shape:
            raise DumpComparisonError(f"SigC dimensions differ for block {key}")
        dimensions.add(reference_matrix.shape[0])
        difference = candidate_matrix - reference_matrix
        absolute = np.abs(difference)
        flat_index = int(np.argmax(absolute))
        location = np.unravel_index(flat_index, absolute.shape)
        block_maximum = float(absolute[location])
        if block_maximum > maximum:
            maximum = block_maximum
            maximum_location = (*key, int(location[0]), int(location[1]))
            reference_at_max = complex(reference_matrix[location])
            candidate_at_max = complex(candidate_matrix[location])
        difference_square_sum += float(np.vdot(difference, difference).real)
        reference_square_sum += float(
            np.vdot(reference_matrix, reference_matrix).real
        )
        exact_blocks += int(np.array_equal(reference_matrix, candidate_matrix))
        exact_elements += int(np.count_nonzero(difference == 0.0))
        element_count += difference.size
        all_absolute_differences.append(absolute.reshape(-1))
        frequency_maxima[key[2]] = max(
            frequency_maxima.get(key[2], 0.0), block_maximum
        )
        block_records.append(
            {
                "spin": key[0],
                "kpoint": key[1],
                "frequency": key[2],
                "max_abs_difference_ha": block_maximum,
                "relative_frobenius_difference": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(reference_matrix), np.finfo(float).tiny)
                ),
            }
        )

    differences = np.concatenate(all_absolute_differences)
    relative_frobenius = math.sqrt(difference_square_sum) / max(
        math.sqrt(reference_square_sum), np.finfo(float).tiny
    )
    block_maxima = np.array(
        [record["max_abs_difference_ha"] for record in block_records],
        dtype=float,
    )
    worst_blocks = sorted(
        block_records,
        key=lambda record: record["max_abs_difference_ha"],
        reverse=True,
    )[:10]
    checks = {
        "max_abs_within_tolerance": maximum <= max_abs_tolerance_ha,
        "relative_frobenius_within_tolerance": (
            relative_frobenius <= relative_frobenius_tolerance
        ),
    }
    return {
        "schema": "g0w0-sigc-directory-comparison-v1",
        "passed": all(checks.values()),
        "source": source,
        "block_count": len(reference),
        "exact_block_count": exact_blocks,
        "element_count": element_count,
        "exact_element_count": exact_elements,
        "spin_count": len({key[0] for key in reference}),
        "kpoint_count": len({key[1] for key in reference}),
        "frequency_count": len({key[2] for key in reference}),
        "matrix_dimensions": sorted(dimensions),
        "reference_directory": str(Path(reference_directory).resolve()),
        "candidate_directory": str(Path(candidate_directory).resolve()),
        "reference_manifest_sha256": reference_manifest,
        "candidate_manifest_sha256": candidate_manifest,
        "byte_identical_manifest": reference_manifest == candidate_manifest,
        "max_abs_difference_ha": maximum,
        "max_abs_location": list(maximum_location) if maximum_location else None,
        "reference_value_at_max": _complex_json(reference_at_max),
        "candidate_value_at_max": _complex_json(candidate_at_max),
        "relative_frobenius_difference": relative_frobenius,
        "element_abs_difference_quantiles_ha": _quantiles(differences),
        "block_max_abs_difference_quantiles_ha": _quantiles(block_maxima),
        "frequency_max_abs_difference_ha": {
            str(index): frequency_maxima[index]
            for index in sorted(frequency_maxima)
        },
        "worst_blocks": worst_blocks,
        "thresholds": {
            "max_abs_tolerance_ha": max_abs_tolerance_ha,
            "relative_frobenius_tolerance": relative_frobenius_tolerance,
        },
        "checks": checks,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two complete LibRPA G0W0 SigC dump directories."
    )
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--source", default="kgrid")
    parser.add_argument("--max-abs-tolerance-ha", type=float, default=1.0e-10)
    parser.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1.0e-10
    )
    args = parser.parse_args()
    try:
        report = compare_directories(
            args.reference_directory,
            args.candidate_directory,
            source=args.source,
            max_abs_tolerance_ha=args.max_abs_tolerance_ha,
            relative_frobenius_tolerance=args.relative_frobenius_tolerance,
        )
        status = 0 if report["passed"] else 1
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 1
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    sys.exit(main())
