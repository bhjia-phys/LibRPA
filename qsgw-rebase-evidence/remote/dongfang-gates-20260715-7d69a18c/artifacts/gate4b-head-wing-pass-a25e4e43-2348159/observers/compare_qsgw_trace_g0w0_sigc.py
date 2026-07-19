#!/usr/bin/env python3
"""Compare QSGW iteration-one SigC trace blocks with upstream G0W0 dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from summarize_librpa_sigc_matrix_dumps import read_sigc_ks_binary
from validate_qsgw_trace_closure import _parse_closure_contract


FILE_PATTERN = re.compile(
    r"^Sigc_fk_mn_(?P<source>.+)_ispin_(?P<spin>\d+)_ik_(?P<kpoint>\d+)"
    r"_ifreq_(?P<frequency>\d+)\.bin$"
)


class SigcComparisonError(ValueError):
    """Raised when the paired QSGW/G0W0 SigC layouts are incompatible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_trace_sigc(
    path: Path, iteration: int, channel: int
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[int, float], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    contract = _parse_closure_contract(text, require_current_contract=True)
    entries: dict[
        tuple[int, int, int], dict[tuple[int, int], complex]
    ] = {}
    frequencies: dict[int, float] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 11:
            raise SigcComparisonError(
                f"{path}:{line_number}: expected 11 trace columns"
            )
        if (
            int(fields[0]) != iteration
            or int(fields[1]) != channel
            or fields[2] != "sigma_c_iw"
        ):
            continue
        spin = int(fields[3])
        kpoint = int(fields[4])
        ifrequency = int(fields[5])
        frequency = float(fields[6])
        row = int(fields[7])
        column = int(fields[8])
        value = complex(float(fields[9]), float(fields[10]))
        if min(spin, kpoint, ifrequency, row, column) < 0:
            raise SigcComparisonError(
                f"{path}:{line_number}: negative SigC trace index"
            )
        if not math.isfinite(frequency) or not (
            math.isfinite(value.real) and math.isfinite(value.imag)
        ):
            raise SigcComparisonError(
                f"{path}:{line_number}: non-finite SigC trace value"
            )
        previous_frequency = frequencies.setdefault(ifrequency, frequency)
        if not math.isclose(
            previous_frequency, frequency, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise SigcComparisonError(
                f"trace frequency index {ifrequency} has inconsistent values"
            )
        key = (spin, kpoint, ifrequency)
        block = entries.setdefault(key, {})
        matrix_key = (row, column)
        if matrix_key in block:
            raise SigcComparisonError(
                f"{path}:{line_number}: duplicate SigC matrix element"
            )
        block[matrix_key] = value
    if not entries:
        raise SigcComparisonError("QSGW trace has no selected sigma_c_iw rows")

    matrices: dict[tuple[int, int, int], np.ndarray] = {}
    for key, block in entries.items():
        nrows = max(row for row, _ in block) + 1
        ncolumns = max(column for _, column in block) + 1
        if nrows != ncolumns or len(block) != nrows * ncolumns:
            raise SigcComparisonError(f"incomplete square QSGW SigC block {key}")
        matrix = np.empty((nrows, ncolumns), dtype=np.complex128)
        for (row, column), value in block.items():
            matrix[row, column] = value
        matrices[key] = matrix
    return matrices, frequencies, contract


def _read_g0w0_sigc(
    directory: Path, source: str
) -> tuple[dict[tuple[int, int, int], np.ndarray], str]:
    matrices: dict[tuple[int, int, int], np.ndarray] = {}
    manifest_lines: list[str] = []
    for path in sorted(directory.glob("Sigc_fk_mn_*.bin")):
        match = FILE_PATTERN.match(path.name)
        if match is None or match.group("source") != source:
            continue
        key = tuple(
            int(match.group(name))
            for name in ("spin", "kpoint", "frequency")
        )
        if key in matrices:
            raise SigcComparisonError(f"duplicate G0W0 SigC block {key}")
        matrices[key] = np.asarray(
            read_sigc_ks_binary(path), dtype=np.complex128
        )
        manifest_lines.append(f"{path.name} {_sha256(path)}\n")
    if not matrices:
        raise SigcComparisonError(
            f"no G0W0 SigC binaries for source={source} in {directory}"
        )
    manifest_sha256 = hashlib.sha256(
        "".join(manifest_lines).encode("ascii")
    ).hexdigest()
    return matrices, manifest_sha256


def compare_sigc(
    trace: Path,
    g0w0_directory: Path,
    *,
    iteration: int = 1,
    channel: int = 0,
    source: str = "kgrid",
    max_abs_tolerance_ha: float = 1.0e-10,
    relative_frobenius_tolerance: float = 1.0e-10,
    input_contract: Path | None = None,
) -> dict[str, object]:
    if min(max_abs_tolerance_ha, relative_frobenius_tolerance) < 0.0:
        raise SigcComparisonError("comparison tolerances must be nonnegative")
    traced, frequencies, contract = _read_trace_sigc(
        trace, iteration, channel
    )
    dumped, dump_manifest_sha256 = _read_g0w0_sigc(
        g0w0_directory, source
    )
    if set(traced) != set(dumped):
        raise SigcComparisonError(
            "QSGW/G0W0 SigC block keys differ: "
            f"missing_in_g0w0={sorted(set(traced) - set(dumped))}, "
            f"missing_in_qsgw={sorted(set(dumped) - set(traced))}"
        )

    expected_contract_sha256 = contract.get("qsgw_input_contract_sha256")
    actual_contract_sha256 = None
    if input_contract is not None:
        actual_contract_sha256 = _sha256(input_contract)
        if expected_contract_sha256 != actual_contract_sha256:
            raise SigcComparisonError(
                "QSGW input contract SHA256 differs from trace header"
            )

    maximum = 0.0
    maximum_key: tuple[int, int, int, int, int] | None = None
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    exact_blocks = 0
    dimensions: set[int] = set()
    for key in sorted(traced):
        qsgw = traced[key]
        g0w0 = dumped[key]
        if qsgw.shape != g0w0.shape:
            raise SigcComparisonError(
                f"QSGW/G0W0 SigC dimensions differ for block {key}"
            )
        dimensions.add(qsgw.shape[0])
        difference = qsgw - g0w0
        location = np.unravel_index(
            int(np.argmax(np.abs(difference))), difference.shape
        )
        block_maximum = float(np.max(np.abs(difference)))
        if block_maximum > maximum:
            maximum = block_maximum
            maximum_key = (*key, int(location[0]), int(location[1]))
        difference_square_sum += float(np.vdot(difference, difference).real)
        reference_square_sum += float(np.vdot(g0w0, g0w0).real)
        exact_blocks += int(np.array_equal(qsgw, g0w0))
    relative_frobenius = math.sqrt(difference_square_sum) / max(
        math.sqrt(reference_square_sum), 1.0e-300
    )
    passed = (
        maximum <= max_abs_tolerance_ha
        and relative_frobenius <= relative_frobenius_tolerance
    )
    return {
        "passed": passed,
        "iteration": iteration,
        "channel": channel,
        "source": source,
        "block_count": len(traced),
        "exact_block_count": exact_blocks,
        "spin_count": len({key[0] for key in traced}),
        "kpoint_count": len({key[1] for key in traced}),
        "frequency_count": len({key[2] for key in traced}),
        "frequency_ha": [frequencies[index] for index in sorted(frequencies)],
        "matrix_dimensions": sorted(dimensions),
        "max_abs_difference_ha": maximum,
        "max_abs_location": list(maximum_key) if maximum_key else None,
        "relative_frobenius_difference": relative_frobenius,
        "max_abs_tolerance_ha": max_abs_tolerance_ha,
        "relative_frobenius_tolerance": relative_frobenius_tolerance,
        "qsgw_trace_sha256": _sha256(trace),
        "g0w0_sigc_manifest_sha256": dump_manifest_sha256,
        "qsgw_input_contract_expected_sha256": expected_contract_sha256,
        "qsgw_input_contract_actual_sha256": actual_contract_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("g0w0_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--source", default="kgrid")
    parser.add_argument("--max-abs-tolerance-ha", type=float, default=1e-10)
    parser.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--input-contract", type=Path)
    args = parser.parse_args()
    try:
        report = compare_sigc(
            args.trace,
            args.g0w0_directory,
            iteration=args.iteration,
            channel=args.channel,
            source=args.source,
            max_abs_tolerance_ha=args.max_abs_tolerance_ha,
            relative_frobenius_tolerance=args.relative_frobenius_tolerance,
            input_contract=args.input_contract,
        )
        status = 0 if report["passed"] else 1
    except (OSError, ValueError) as error:
        report = {"passed": False, "error": str(error)}
        status = 1
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
