#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class BandState:
    kpoint: int
    spin: int
    band: int
    occupation: float
    eigenvalue_ha: float
    eigenvalue_ev: float


@dataclass(frozen=True)
class BandOut:
    n_kpoints: int
    n_spins: int
    n_basis: int
    n_bands: int
    fermi_ha: float
    states: Tuple[BandState, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_lines(path: Path) -> List[Tuple[int, str]]:
    lines = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line:
                lines.append((line_number, line))
    return lines


def _single_value(
    lines: List[Tuple[int, str]], cursor: int, converter, label: str
):
    if cursor >= len(lines):
        raise ValueError(f"missing {label}")
    line_number, line = lines[cursor]
    fields = line.split()
    if len(fields) != 1:
        raise ValueError(f"line {line_number}: expected one value for {label}")
    try:
        value = converter(fields[0])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid {label}: {fields[0]}") from exc
    return value, cursor + 1


def read_band_out(path: Path) -> BandOut:
    path = Path(path)
    lines = _nonempty_lines(path)
    cursor = 0
    n_kpoints, cursor = _single_value(lines, cursor, int, "n_kpoints")
    n_spins, cursor = _single_value(lines, cursor, int, "n_spins")
    n_basis, cursor = _single_value(lines, cursor, int, "n_basis")
    n_bands, cursor = _single_value(lines, cursor, int, "n_bands")
    fermi_ha, cursor = _single_value(lines, cursor, float, "fermi_ha")

    if min(n_kpoints, n_spins, n_basis, n_bands) <= 0:
        raise ValueError("band_out dimensions must all be positive")

    states: List[BandState] = []
    seen_blocks = set()
    for _ in range(n_kpoints * n_spins):
        if cursor >= len(lines):
            raise ValueError("missing k-point/spin block header")
        line_number, line = lines[cursor]
        cursor += 1
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"line {line_number}: expected k-point and spin indices"
            )
        try:
            kpoint, spin = (int(value) for value in fields)
        except ValueError as exc:
            raise ValueError(
                f"line {line_number}: invalid k-point/spin block header"
            ) from exc
        block = (kpoint, spin)
        if block in seen_blocks:
            raise ValueError(f"line {line_number}: duplicate block {block}")
        if not (1 <= kpoint <= n_kpoints and 1 <= spin <= n_spins):
            raise ValueError(f"line {line_number}: block index out of range: {block}")
        seen_blocks.add(block)

        for expected_band in range(1, n_bands + 1):
            if cursor >= len(lines):
                raise ValueError(f"missing band row for block {block}")
            row_number, row = lines[cursor]
            cursor += 1
            values = row.split()
            if len(values) < 4:
                raise ValueError(f"line {row_number}: expected four band columns")
            try:
                band = int(values[0])
                occupation = float(values[1])
                eigenvalue_ha = float(values[2])
                eigenvalue_ev = float(values[3])
            except ValueError as exc:
                raise ValueError(f"line {row_number}: invalid numeric band row") from exc
            if band != expected_band:
                raise ValueError(
                    f"line {row_number}: expected band {expected_band}, found {band}"
                )
            states.append(
                BandState(
                    kpoint=kpoint,
                    spin=spin,
                    band=band,
                    occupation=occupation,
                    eigenvalue_ha=eigenvalue_ha,
                    eigenvalue_ev=eigenvalue_ev,
                )
            )

    if cursor != len(lines):
        line_number, _ = lines[cursor]
        raise ValueError(f"line {line_number}: unexpected trailing data")
    if len(seen_blocks) != n_kpoints * n_spins:
        raise ValueError("incomplete k-point/spin block set")

    return BandOut(
        n_kpoints=n_kpoints,
        n_spins=n_spins,
        n_basis=n_basis,
        n_bands=n_bands,
        fermi_ha=fermi_ha,
        states=tuple(states),
    )


def _metadata(data: BandOut) -> Dict[str, int]:
    return {
        "n_kpoints": data.n_kpoints,
        "n_spins": data.n_spins,
        "n_basis": data.n_basis,
        "n_bands": data.n_bands,
        "state_count": len(data.states),
    }


def _state_map(data: BandOut) -> Dict[Tuple[int, int, int], BandState]:
    return {
        (state.kpoint, state.spin, state.band): state for state in data.states
    }


def _location(key: Tuple[int, int, int]) -> Dict[str, int]:
    return {"kpoint": key[0], "spin": key[1], "band": key[2]}


def _maximum_difference(
    keys: Iterable[Tuple[int, int, int]],
    reference: Dict[Tuple[int, int, int], BandState],
    candidate: Dict[Tuple[int, int, int], BandState],
    attribute: str,
) -> Tuple[float, Dict[str, int]]:
    maximum = -1.0
    maximum_key = None
    for key in keys:
        difference = abs(
            getattr(reference[key], attribute) - getattr(candidate[key], attribute)
        )
        if difference > maximum:
            maximum = difference
            maximum_key = key
    if maximum_key is None:
        return 0.0, {}
    return maximum, _location(maximum_key)


def compare_band_out_files(
    reference_path: Path,
    candidate_path: Path,
    *,
    eigenvalue_tolerance_ha: float = 0.0,
    occupation_tolerance: float = 0.0,
    fermi_tolerance_ha: float = 0.0,
    require_byte_identical: bool = False,
) -> Dict[str, object]:
    reference_path = Path(reference_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    reference = read_band_out(reference_path)
    candidate = read_band_out(candidate_path)
    reference_sha = _sha256(reference_path)
    candidate_sha = _sha256(candidate_path)
    byte_identical = reference_sha == candidate_sha

    reference_metadata = _metadata(reference)
    candidate_metadata = _metadata(candidate)
    metadata_equal = reference_metadata == candidate_metadata
    reference_states = _state_map(reference)
    candidate_states = _state_map(candidate)
    reference_keys = set(reference_states)
    candidate_keys = set(candidate_states)
    state_keys_equal = reference_keys == candidate_keys
    common_keys = sorted(reference_keys & candidate_keys)

    eigenvalue_max_abs_ha, eigenvalue_location = _maximum_difference(
        common_keys, reference_states, candidate_states, "eigenvalue_ha"
    )
    occupation_max_abs, occupation_location = _maximum_difference(
        common_keys, reference_states, candidate_states, "occupation"
    )
    fermi_max_abs_ha = abs(reference.fermi_ha - candidate.fermi_ha)

    checks = {
        "metadata_equal": metadata_equal,
        "state_keys_equal": state_keys_equal,
        "eigenvalues_within_tolerance": (
            eigenvalue_max_abs_ha <= eigenvalue_tolerance_ha
        ),
        "occupations_within_tolerance": occupation_max_abs <= occupation_tolerance,
        "fermi_within_tolerance": fermi_max_abs_ha <= fermi_tolerance_ha,
        "byte_identical_if_required": (
            byte_identical if require_byte_identical else True
        ),
    }
    return {
        "schema": "qsgw-band-out-comparison-v1",
        "passed": all(checks.values()),
        "byte_identical": byte_identical,
        "reference": {
            "path": str(reference_path),
            "sha256": reference_sha,
            "metadata": reference_metadata,
            "fermi_ha": reference.fermi_ha,
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
            "metadata": candidate_metadata,
            "fermi_ha": candidate.fermi_ha,
        },
        "thresholds": {
            "eigenvalue_tolerance_ha": eigenvalue_tolerance_ha,
            "occupation_tolerance": occupation_tolerance,
            "fermi_tolerance_ha": fermi_tolerance_ha,
            "require_byte_identical": require_byte_identical,
        },
        "differences": {
            "eigenvalue_max_abs_ha": eigenvalue_max_abs_ha,
            "eigenvalue_max_location": eigenvalue_location,
            "occupation_max_abs": occupation_max_abs,
            "occupation_max_location": occupation_location,
            "fermi_max_abs_ha": fermi_max_abs_ha,
            "reference_only_state_count": len(reference_keys - candidate_keys),
            "candidate_only_state_count": len(candidate_keys - reference_keys),
        },
        "checks": checks,
    }


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare QSGW band_out initial-state data."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--eigenvalue-tolerance-ha", type=float, default=0.0)
    parser.add_argument("--occupation-tolerance", type=float, default=0.0)
    parser.add_argument("--fermi-tolerance-ha", type=float, default=0.0)
    parser.add_argument("--require-byte-identical", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = compare_band_out_files(
        args.reference,
        args.candidate,
        eigenvalue_tolerance_ha=args.eigenvalue_tolerance_ha,
        occupation_tolerance=args.occupation_tolerance,
        fermi_tolerance_ha=args.fermi_tolerance_ha,
        require_byte_identical=args.require_byte_identical,
    )
    _write_json(args.output_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
