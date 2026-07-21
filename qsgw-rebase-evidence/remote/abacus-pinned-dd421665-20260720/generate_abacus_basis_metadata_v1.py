#!/usr/bin/env python3
"""Generate LibRPA basis metadata from frozen ABACUS ORB/ABFS headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_ASSET_SHA256 = {
    "Si_gga_8au_100Ry_3s3p2d.orb": (
        "3d6e4fb7014b51cca13fb0a505d7556a6e7ca2c2c4f1e16c0a66788cf86f3279"
    ),
    "Si_3s3p2d1f1g_pca1e-6.abfs": (
        "1ea35d9b063c061c614600f27b20599c77c371a0b10ad0fa4551c2c99f333d6e"
    ),
}
EXPECTED_WFC_SHELL_COUNTS = [3, 3, 2]
EXPECTED_AUX_SHELL_COUNTS = [12, 12, 11, 9, 9, 6, 4, 1, 1]
ORBITAL_LABELS = "SPDFGHIJKLMNOPQRSTUVWXYZ"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path) -> str:
    expected = EXPECTED_ASSET_SHA256.get(path.name)
    if expected is None:
        fail(f"unexpected frozen asset name: {path.name}")
    if not path.is_file():
        fail(f"missing frozen asset: {path}")
    actual = sha256(path)
    if actual != expected:
        fail(f"asset hash mismatch for {path.name}: {actual}")
    return actual


def parse_shell_counts(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    lmax_match = re.search(r"^Lmax\s+(\d+)\s*$", text, flags=re.MULTILINE)
    if not lmax_match:
        fail(f"{path.name}: missing Lmax")
    lmax = int(lmax_match.group(1))
    counts: dict[int, int] = {}
    pattern = re.compile(
        r"^Number of ([A-Za-z])orbital-->\s+(\d+)\s*$", flags=re.MULTILINE
    )
    for label, raw_count in pattern.findall(text):
        upper = label.upper()
        if upper not in ORBITAL_LABELS:
            fail(f"{path.name}: unsupported orbital label {label!r}")
        angular_momentum = ORBITAL_LABELS.index(upper)
        if angular_momentum in counts:
            fail(f"{path.name}: duplicate shell-count entry for l={angular_momentum}")
        counts[angular_momentum] = int(raw_count)
    expected_keys = set(range(lmax + 1))
    if set(counts) != expected_keys:
        fail(
            f"{path.name}: shell-count keys {sorted(counts)}, "
            f"expected {sorted(expected_keys)}"
        )
    result = [counts[index] for index in range(lmax + 1)]
    if any(count <= 0 for count in result):
        fail(f"{path.name}: every frozen shell count must be positive")
    return result


def basis_size(shell_counts: list[int]) -> int:
    return sum(count * (2 * angular_momentum + 1) for angular_momentum, count in enumerate(shell_counts))


def write_split_basis(path: Path, n_atoms: int, shell_counts: list[int]) -> None:
    per_atom = basis_size(shell_counts)
    lines = [
        f"{1:10d}{per_atom * n_atoms:10d}    abacus",
        f"{1:10d}{per_atom:10d}",
        f"{1:10d}{sum(shell_counts):10d}",
    ]
    for angular_momentum, count in enumerate(shell_counts):
        lines.extend(f"{angular_momentum:10d}" for _ in range(count))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_combined_basis(
    path: Path,
    n_atoms: int,
    wfc_shell_counts: list[int],
    aux_shell_counts: list[int],
) -> None:
    wfc_per_atom = basis_size(wfc_shell_counts)
    aux_per_atom = basis_size(aux_shell_counts)
    lines = [
        f"{1:10d}{wfc_per_atom * n_atoms:10d}{aux_per_atom * n_atoms:10d}    fallback",
        f"{1:10d}{wfc_per_atom:10d}{aux_per_atom:10d}",
    ]
    for shell_counts in (wfc_shell_counts, aux_shell_counts):
        lines.append(f"{1:10d}{sum(shell_counts):10d}")
        for angular_momentum, count in enumerate(shell_counts):
            lines.extend(f"{angular_momentum:10d}" for _ in range(count))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb", type=Path, required=True)
    parser.add_argument("--abfs", type=Path, required=True)
    parser.add_argument("--n-atoms", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_atoms <= 0:
        fail("--n-atoms must be positive")
    if not args.output_dir.is_dir():
        fail(f"output directory does not exist: {args.output_dir}")

    outputs = {
        "wfc": args.output_dir / "basis_wfc_out",
        "aux": args.output_dir / "basis_aux_out",
        "combined": args.output_dir / "basis_out",
        "summary": args.output_dir / "basis_metadata_summary.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        fail(f"refusing to overwrite basis metadata: {existing}")

    orb_sha = verify_asset(args.orb)
    abfs_sha = verify_asset(args.abfs)
    wfc_shell_counts = parse_shell_counts(args.orb)
    aux_shell_counts = parse_shell_counts(args.abfs)
    if wfc_shell_counts != EXPECTED_WFC_SHELL_COUNTS:
        fail(f"unexpected WFC shell layout: {wfc_shell_counts}")
    if aux_shell_counts != EXPECTED_AUX_SHELL_COUNTS:
        fail(f"unexpected auxiliary shell layout: {aux_shell_counts}")

    write_split_basis(outputs["wfc"], args.n_atoms, wfc_shell_counts)
    write_split_basis(outputs["aux"], args.n_atoms, aux_shell_counts)
    write_combined_basis(
        outputs["combined"], args.n_atoms, wfc_shell_counts, aux_shell_counts
    )

    report = {
        "status": "PASS",
        "n_atom_types": 1,
        "n_atoms": args.n_atoms,
        "orb": {
            "file": args.orb.name,
            "sha256": orb_sha,
            "shell_counts": wfc_shell_counts,
            "basis_per_atom": basis_size(wfc_shell_counts),
            "basis_total": basis_size(wfc_shell_counts) * args.n_atoms,
        },
        "abfs": {
            "file": args.abfs.name,
            "sha256": abfs_sha,
            "shell_counts": aux_shell_counts,
            "basis_per_atom": basis_size(aux_shell_counts),
            "basis_total": basis_size(aux_shell_counts) * args.n_atoms,
        },
    }
    outputs["summary"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
