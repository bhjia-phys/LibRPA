#!/usr/bin/env python3
"""Compare the first current-v6 QSGW SigmaC with an upstream G0W0 dump."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


FILE_PATTERN = re.compile(
    r"^Sigc_fk_mn_(?P<source>.+)_ispin_(?P<spin>\d+)_ik_(?P<kpoint>\d+)"
    r"_ifreq_(?P<frequency>\d+)\.bin$"
)


class ComparisonError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ComparisonError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_contract(contract: dict[str, object]) -> None:
    expected = {
        "qsgw_contract_version": 6,
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
        "velocity": "disabled_stage1",
        "headwing": "disabled_stage1",
        "symmetry": "exx_on_gw_on_rpa_on",
        "hartree": "disabled_stage1",
        "band": "disabled_stage1",
        "h_qsgw_cut": "disabled_non_band",
        "qsgw_mixer": "none",
    }
    differences = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if differences:
        raise ComparisonError(f"QSGW v6 contract mismatch: {differences}")
    if not math.isclose(
        float(contract["qsgw_mixing_beta"]), 0.2, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ComparisonError("QSGW configured mixing beta differs from 0.2")


def _read_sigc_binary(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ComparisonError(f"{path}: truncated SigC header")
        nstates, scalar_bytes = struct.unpack("=ii", header)
        payload = handle.read()
    if nstates <= 0 or scalar_bytes != 8:
        raise ComparisonError(f"{path}: unsupported SigC header")
    expected_bytes = nstates * nstates * 16
    if len(payload) != expected_bytes:
        raise ComparisonError(
            f"{path}: payload has {len(payload)} bytes, expected {expected_bytes}"
        )
    values = np.frombuffer(payload, dtype="=f8")
    matrix = values[0::2] + 1j * values[1::2]
    return matrix.reshape((nstates, nstates))


def _trace_sigc(
    cmp_qsgw: ModuleType, text: str, iteration: int, channel: int
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[int, float], dict]:
    contract = cmp_qsgw._parse_contract(text, "QSGW matrix trace")
    _require_contract(contract)
    blocks = cmp_qsgw._parse_matrix_trace(text, "QSGW matrix trace")
    cmp_qsgw._validate_matrix_trajectory(
        blocks, contract, "QSGW matrix trace"
    )
    iterations = sorted({key[0] for key in blocks})
    if iterations != [0, 1]:
        raise ComparisonError(f"QSGW trace iterations differ: {iterations}")

    matrices: dict[tuple[int, int, int], np.ndarray] = {}
    frequencies: dict[int, float] = {}
    for key, (frequency, matrix) in blocks.items():
        row_iteration, row_channel, component, spin, kpoint, ifrequency = key
        if (
            row_iteration != iteration
            or row_channel != channel
            or component != "sigma_c_iw"
        ):
            continue
        block_key = (spin, kpoint, ifrequency)
        matrices[block_key] = np.asarray(matrix, dtype=np.complex128)
        previous = frequencies.setdefault(ifrequency, frequency)
        if not math.isclose(previous, frequency, rel_tol=0.0, abs_tol=1.0e-15):
            raise ComparisonError("trace frequency index is inconsistent")
    if not matrices:
        raise ComparisonError("QSGW trace has no selected SigmaC blocks")
    return matrices, frequencies, contract


def _g0w0_sigc(
    directory: Path, source: str
) -> tuple[dict[tuple[int, int, int], np.ndarray], str]:
    matrices: dict[tuple[int, int, int], np.ndarray] = {}
    manifest: list[str] = []
    for path in sorted(directory.glob("Sigc_fk_mn_*.bin")):
        match = FILE_PATTERN.match(path.name)
        if match is None or match.group("source") != source:
            continue
        key = tuple(
            int(match.group(name))
            for name in ("spin", "kpoint", "frequency")
        )
        if key in matrices:
            raise ComparisonError(f"duplicate upstream G0W0 block {key}")
        matrices[key] = _read_sigc_binary(path)
        manifest.append(f"{path.name} {_sha256(path)}\n")
    if not matrices:
        raise ComparisonError(
            f"no upstream G0W0 source={source} SigmaC blocks in {directory}"
        )
    digest = hashlib.sha256("".join(manifest).encode("ascii")).hexdigest()
    return matrices, digest


def compare(
    *,
    trace: Path,
    g0w0_directory: Path,
    input_contract: Path,
    cmp_qsgw_path: Path,
    iteration: int = 1,
    channel: int = 0,
    source: str = "kgrid",
    max_abs_tolerance_ha: float = 1.0e-10,
    relative_frobenius_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    if min(max_abs_tolerance_ha, relative_frobenius_tolerance) < 0.0:
        raise ComparisonError("comparison tolerances must be nonnegative")
    cmp_qsgw = _load_module(cmp_qsgw_path, "gate2_cmp_qsgw")
    traced, frequencies, contract = _trace_sigc(
        cmp_qsgw, trace.read_text(encoding="utf-8"), iteration, channel
    )
    dumped, dump_manifest_sha = _g0w0_sigc(g0w0_directory, source)
    if set(traced) != set(dumped):
        raise ComparisonError(
            "QSGW/G0W0 SigmaC block keys differ: "
            f"missing_in_g0w0={sorted(set(traced) - set(dumped))}, "
            f"missing_in_qsgw={sorted(set(dumped) - set(traced))}"
        )
    actual_contract_sha = _sha256(input_contract)
    if contract["qsgw_input_contract_sha256"] != actual_contract_sha:
        raise ComparisonError("QSGW input-contract SHA256 differs from trace")

    maximum = 0.0
    maximum_key = None
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    exact_blocks = 0
    dimensions: set[int] = set()
    for key in sorted(traced):
        qsgw = traced[key]
        g0w0 = dumped[key]
        if qsgw.shape != g0w0.shape:
            raise ComparisonError(f"SigmaC shape differs at block {key}")
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
    relative = math.sqrt(difference_square_sum) / max(
        math.sqrt(reference_square_sum), 1.0e-300
    )
    passed = (
        maximum <= max_abs_tolerance_ha
        and relative <= relative_frobenius_tolerance
    )
    return {
        "schema": "librpa-qsgw-iter1-vs-upstream-g0w0-sigc-v1",
        "passed": passed,
        "semantic_iteration_zero": "immutable_initial_state",
        "semantic_first_self_energy": "trace_iteration_1_channel_0",
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
        "relative_frobenius_difference": relative,
        "max_abs_tolerance_ha": max_abs_tolerance_ha,
        "relative_frobenius_tolerance": relative_frobenius_tolerance,
        "qsgw_contract_version": contract["qsgw_contract_version"],
        "qsgw_trace_sha256": _sha256(trace),
        "g0w0_sigc_manifest_sha256": dump_manifest_sha,
        "qsgw_input_contract_sha256": actual_contract_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("g0w0_directory", type=Path)
    parser.add_argument("input_contract", type=Path)
    parser.add_argument("cmp_qsgw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0)
    parser.add_argument("--source", default="kgrid")
    parser.add_argument("--max-abs-tolerance-ha", type=float, default=1e-10)
    parser.add_argument(
        "--relative-frobenius-tolerance", type=float, default=1e-10
    )
    args = parser.parse_args()
    try:
        report = compare(
            trace=args.trace,
            g0w0_directory=args.g0w0_directory,
            input_contract=args.input_contract,
            cmp_qsgw_path=args.cmp_qsgw,
            iteration=args.iteration,
            channel=args.channel,
            source=args.source,
            max_abs_tolerance_ha=args.max_abs_tolerance_ha,
            relative_frobenius_tolerance=args.relative_frobenius_tolerance,
        )
        status = 0 if report["passed"] else 2
    except (OSError, ValueError) as error:
        report = {
            "schema": "librpa-qsgw-iter1-vs-upstream-g0w0-sigc-v1",
            "passed": False,
            "error": str(error),
        }
        status = 1
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
