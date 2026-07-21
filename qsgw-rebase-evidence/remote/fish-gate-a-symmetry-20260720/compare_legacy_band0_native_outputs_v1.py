#!/usr/bin/env python3
"""Compare a reproduced legacy qsgw_band0 iteration with native outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


H0_MAX_ABS_HA = 1.0e-6
H0_REL_FROBENIUS = 1.0e-8
SIGC_MAX_ABS_HA = 1.0e-6
SIGC_REL_FROBENIUS = 1.0e-8
HERMITICITY_HA = 1.0e-10
BAND_MAX_ABS_EV = 1.0e-5
FLOAT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


class ComparisonError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ComparisonError(f"missing file: {path}")
    return path


def read_matz_binary(path: Path) -> tuple[int, int, list[complex]]:
    data = require_file(path).read_bytes()
    if len(data) < 8:
        raise ComparisonError(f"short Matz file: {path}")
    rows, columns = struct.unpack_from("<ii", data, 0)
    if rows <= 0 or columns <= 0:
        raise ComparisonError(f"invalid Matz dimensions in {path}")
    expected = 8 + rows * columns * 16
    if len(data) != expected:
        raise ComparisonError(
            f"Matz size mismatch for {path}: {len(data)} != {expected}"
        )
    values = [
        complex(*struct.unpack_from("<dd", data, 8 + index * 16))
        for index in range(rows * columns)
    ]
    return rows, columns, values


def read_sigcrf_binary(
    path: Path,
) -> dict[tuple[int, int, int, int, int], list[complex]]:
    data = require_file(path).read_bytes()
    offset = 0

    def unpack_size_t() -> int:
        nonlocal offset
        if offset + 8 > len(data):
            raise ComparisonError(f"truncated size_t in {path} at {offset}")
        value = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        return value

    n_records = unpack_size_t()
    records: dict[tuple[int, int, int, int, int], list[complex]] = {}
    for _ in range(n_records):
        dims = tuple(unpack_size_t() for _index in range(5))
        count = dims[3] * dims[4]
        end = offset + count * 16
        if end > len(data):
            raise ComparisonError(f"truncated SigcRF matrix in {path}")
        values = [
            complex(*struct.unpack_from("<dd", data, offset + index * 16))
            for index in range(count)
        ]
        offset = end
        if dims in records:
            raise ComparisonError(f"duplicate SigcRF key {dims} in {path}")
        records[dims] = values
    if offset != len(data):
        raise ComparisonError(
            f"trailing bytes in SigcRF file {path}: {len(data) - offset}"
        )
    return records


def vector_metrics(reference: list[complex], observed: list[complex]) -> dict[str, float]:
    if len(reference) != len(observed):
        raise ComparisonError(
            f"vector length mismatch: {len(reference)} != {len(observed)}"
        )
    max_abs = 0.0
    difference_sq = 0.0
    reference_sq = 0.0
    for expected, actual in zip(reference, observed):
        if not all(
            math.isfinite(value)
            for value in (expected.real, expected.imag, actual.real, actual.imag)
        ):
            raise ComparisonError("non-finite matrix value")
        delta = actual - expected
        magnitude = abs(delta)
        max_abs = max(max_abs, magnitude)
        difference_sq += magnitude * magnitude
        reference_sq += abs(expected) ** 2
    frobenius = math.sqrt(difference_sq)
    reference_frobenius = math.sqrt(reference_sq)
    relative = frobenius / reference_frobenius if reference_frobenius else frobenius
    return {
        "max_abs": max_abs,
        "frobenius": frobenius,
        "reference_frobenius": reference_frobenius,
        "relative_frobenius": relative,
    }


def hermiticity_max_abs(rows: int, columns: int, values: list[complex]) -> float:
    if rows != columns:
        return math.inf
    maximum = 0.0
    for row in range(rows):
        for column in range(columns):
            maximum = max(
                maximum,
                abs(values[row * columns + column] - values[column * columns + row].conjugate()),
            )
    return maximum


def strict_numeric_file(path: Path) -> list[complex]:
    values: list[complex] = []
    for line_number, line in enumerate(require_file(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        for token in line.split():
            try:
                value = float(token)
            except ValueError as error:
                raise ComparisonError(
                    f"non-numeric token in {path}:{line_number}: {token}"
                ) from error
            if not math.isfinite(value):
                raise ComparisonError(f"non-finite value in {path}:{line_number}")
            values.append(complex(value, 0.0))
    return values


def loose_numeric_file(path: Path) -> list[complex]:
    text = require_file(path).read_text()
    return [complex(float(token), 0.0) for token in FLOAT_PATTERN.findall(text)]


def compare_h0(oracle: Path, reproduced: Path) -> dict[str, object]:
    oracle_dir = oracle / "librpa.d/qsgw_checkpoints/iter_00001"
    reproduced_dir = reproduced / "librpa.d/qsgw_checkpoints/iter_00001"
    reports = []
    for kpoint in range(1, 9):
        name = f"H0_GW_spin_01_k_{kpoint:06d}.bin"
        oracle_path = oracle_dir / name
        reproduced_path = reproduced_dir / name
        rows, columns, expected = read_matz_binary(oracle_path)
        rows_actual, columns_actual, actual = read_matz_binary(reproduced_path)
        if (rows, columns) != (rows_actual, columns_actual):
            raise ComparisonError(f"H0 dimensions differ for {name}")
        if (rows, columns) != (44, 44):
            raise ComparisonError(f"unexpected H0 dimensions for {name}: {rows}x{columns}")
        metrics = vector_metrics(expected, actual)
        metrics.update(
            {
                "file": name,
                "byte_equal": sha256_file(oracle_path) == sha256_file(reproduced_path),
                "oracle_hermiticity_max_abs_ha": hermiticity_max_abs(
                    rows, columns, expected
                ),
                "reproduced_hermiticity_max_abs_ha": hermiticity_max_abs(
                    rows, columns, actual
                ),
            }
        )
        reports.append(metrics)
    maximum = max(report["max_abs"] for report in reports)
    relative = max(report["relative_frobenius"] for report in reports)
    hermiticity = max(
        max(
            report["oracle_hermiticity_max_abs_ha"],
            report["reproduced_hermiticity_max_abs_ha"],
        )
        for report in reports
    )
    return {
        "files": reports,
        "max_abs_ha": maximum,
        "max_relative_frobenius": relative,
        "max_hermiticity_abs_ha": hermiticity,
        "all_byte_equal": all(report["byte_equal"] for report in reports),
        "passed": (
            maximum <= H0_MAX_ABS_HA
            and relative <= H0_REL_FROBENIUS
            and hermiticity <= HERMITICITY_HA
        ),
    }


def compare_sigcrf(oracle: Path, reproduced: Path) -> dict[str, object]:
    oracle_dir = oracle / "librpa.d"
    reproduced_dir = reproduced / "librpa.d"
    names = sorted(path.name for path in oracle_dir.glob("SigcRF_*.dat"))
    if len(names) != 16:
        raise ComparisonError(f"expected 16 oracle SigcRF files, found {len(names)}")
    reports = []
    for name in names:
        oracle_path = oracle_dir / name
        reproduced_path = reproduced_dir / name
        expected = read_sigcrf_binary(oracle_path)
        actual = read_sigcrf_binary(reproduced_path)
        if expected.keys() != actual.keys():
            raise ComparisonError(f"SigcRF record keys differ for {name}")
        expected_values: list[complex] = []
        actual_values: list[complex] = []
        for key in sorted(expected):
            expected_values.extend(expected[key])
            actual_values.extend(actual[key])
        metrics = vector_metrics(expected_values, actual_values)
        metrics.update(
            {
                "file": name,
                "record_count": len(expected),
                "value_count": len(expected_values),
                "byte_equal": sha256_file(oracle_path) == sha256_file(reproduced_path),
            }
        )
        reports.append(metrics)
    maximum = max(report["max_abs"] for report in reports)
    relative = max(report["relative_frobenius"] for report in reports)
    return {
        "files": reports,
        "max_abs_ha": maximum,
        "max_relative_frobenius": relative,
        "all_byte_equal": all(report["byte_equal"] for report in reports),
        "passed": maximum <= SIGC_MAX_ABS_HA and relative <= SIGC_REL_FROBENIUS,
    }


def compare_text_outputs(oracle: Path, reproduced: Path) -> dict[str, object]:
    reports: dict[str, dict[str, object]] = {}
    for name in (
        "KS_band_spin_1_1.dat",
        "EXX_band_spin_1_1.dat",
        "QSGW_band_spin_1_1.dat",
    ):
        oracle_path = oracle / name
        reproduced_path = reproduced / name
        metrics = vector_metrics(
            strict_numeric_file(oracle_path), strict_numeric_file(reproduced_path)
        )
        reports[name] = {
            **metrics,
            "byte_equal": sha256_file(oracle_path) == sha256_file(reproduced_path),
            "passed": metrics["max_abs"] <= BAND_MAX_ABS_EV,
        }
    history_name = "homo_lumo_vs_iterations.dat"
    history_oracle = oracle / history_name
    history_reproduced = reproduced / history_name
    history_metrics = vector_metrics(
        loose_numeric_file(history_oracle), loose_numeric_file(history_reproduced)
    )
    reports[history_name] = {
        **history_metrics,
        "byte_equal": sha256_file(history_oracle) == sha256_file(history_reproduced),
        "passed": history_metrics["max_abs"] <= BAND_MAX_ABS_EV,
    }
    return {
        "files": reports,
        "max_abs_ev": max(report["max_abs"] for report in reports.values()),
        "all_byte_equal": all(report["byte_equal"] for report in reports.values()),
        "passed": all(report["passed"] for report in reports.values()),
    }


def compare_outputs(oracle: Path, reproduced: Path) -> dict[str, object]:
    oracle = oracle.resolve(strict=True)
    reproduced = reproduced.resolve(strict=True)
    h0 = compare_h0(oracle, reproduced)
    sigcrf = compare_sigcrf(oracle, reproduced)
    text_outputs = compare_text_outputs(oracle, reproduced)
    artifacts = {}
    for name in ("librpa.in", "hrs1_nao_qsgw_iter_0001.csr"):
        oracle_path = require_file(oracle / name)
        reproduced_path = require_file(reproduced / name)
        artifacts[name] = {
            "oracle_sha256": sha256_file(oracle_path),
            "reproduced_sha256": sha256_file(reproduced_path),
            "byte_equal": sha256_file(oracle_path) == sha256_file(reproduced_path),
            "oracle_size": oracle_path.stat().st_size,
            "reproduced_size": reproduced_path.stat().st_size,
        }
    passed = h0["passed"] and sigcrf["passed"] and text_outputs["passed"]
    return {
        "schema": "librpa-legacy-qsgw-band0-native-comparison-v1",
        "oracle": str(oracle),
        "reproduced": str(reproduced),
        "thresholds": {
            "h0_max_abs_ha": H0_MAX_ABS_HA,
            "h0_relative_frobenius": H0_REL_FROBENIUS,
            "sigcrf_max_abs_ha": SIGC_MAX_ABS_HA,
            "sigcrf_relative_frobenius": SIGC_REL_FROBENIUS,
            "hermiticity_ha": HERMITICITY_HA,
            "band_max_abs_ev": BAND_MAX_ABS_EV,
        },
        "h0": h0,
        "sigcrf": sigcrf,
        "text_outputs": text_outputs,
        "artifacts": artifacts,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("reproduced", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_outputs(args.oracle, args.reproduced)
    except (ComparisonError, OSError, ValueError, struct.error) as error:
        report = {
            "schema": "librpa-legacy-qsgw-band0-native-comparison-v1",
            "passed": False,
            "error": str(error),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
