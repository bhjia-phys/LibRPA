#!/usr/bin/env python3
"""Compare exact847 H0 checkpoints with current contract-v6 QSGW traces."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

tool_dir = Path(__file__).resolve().parent
if not (tool_dir / "compare_legacy_h0_candidate_trace_v1.py").is_file():
    sibling_tool_dir = tool_dir.parent / "fish-gate-a-symmetry-20260720"
    if not (sibling_tool_dir / "compare_legacy_h0_candidate_trace_v1.py").is_file():
        raise ImportError("compare_legacy_h0_candidate_trace_v1.py is unavailable")
    sys.path.insert(0, str(sibling_tool_dir))

import compare_legacy_h0_candidate_trace_v1 as legacy


SCHEMA = "librpa-legacy-h0-candidate-trace-comparison-v2"
CANDIDATE_CONTRACT_VERSION = 6


def require_candidate_contract(path: Path) -> None:
    expected = f"# qsgw_contract_version {CANDIDATE_CONTRACT_VERSION}"
    if not any(line.strip() == expected for line in path.read_text().splitlines()):
        raise legacy.ComparisonError(
            f"missing contract-v{CANDIDATE_CONTRACT_VERSION} header: {path}"
        )


# Reuse the fully tested matrix/checkpoint parser while making the accepted
# candidate contract explicit. The v1 tool remains byte-stable for old runs.
legacy.require_contract_v5 = require_candidate_contract


def compare_outputs(
    checkpoint_root: Path,
    matrix_trace: Path,
    eigenvalue_trace: Path,
    iterations: list[int],
    n_spins: int,
    n_kpoints: int,
    n_bands: int,
    occupied_bands: int,
) -> dict[str, object]:
    report = legacy.compare_outputs(
        checkpoint_root,
        matrix_trace,
        eigenvalue_trace,
        iterations,
        n_spins,
        n_kpoints,
        n_bands,
        occupied_bands,
    )
    report["schema"] = SCHEMA
    report["candidate_contract_version"] = CANDIDATE_CONTRACT_VERSION
    report["comparison_semantics"] = {
        "legacy_hamiltonian": "upper_triangle_hermitized",
        "candidate_hamiltonian": "mixed_h_grid_channel",
        "candidate_rotation": "rotation_u_grid_channel",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_checkpoint_root", type=Path)
    parser.add_argument("candidate_matrix_trace", type=Path)
    parser.add_argument("candidate_eigenvalue_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", default="1:2")
    parser.add_argument("--n-spins", type=int, default=1)
    parser.add_argument("--n-kpoints", type=int, default=8)
    parser.add_argument("--n-bands", type=int, default=44)
    parser.add_argument("--occupied-bands", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.n_spins < 1 or args.n_kpoints < 1 or args.n_bands < 2:
            raise legacy.ComparisonError("invalid matrix layout")
        report = compare_outputs(
            args.legacy_checkpoint_root,
            args.candidate_matrix_trace,
            args.candidate_eigenvalue_trace,
            legacy.parse_iterations(args.iterations),
            args.n_spins,
            args.n_kpoints,
            args.n_bands,
            args.occupied_bands,
        )
    except (
        legacy.ComparisonError,
        legacy.native.ComparisonError,
        OSError,
        ValueError,
        KeyError,
        struct.error,
    ) as error:
        report = {"schema": SCHEMA, "passed": False, "error": str(error)}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
