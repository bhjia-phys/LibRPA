#!/usr/bin/env python3
"""Deterministically convert validated ABACUS NSCF output for LibRPA band readers."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from validate_abacus_si_band_nscf_output_v1 import (
    DiagonalVxcData,
    WfcData,
    indexed_names,
    parse_kpt_info_text,
    parse_native_vxc_text,
    parse_vxc_out_text,
    parse_wfc_text,
    require_file,
    sha256_file,
)


HA2EV = 27.211396
RY_TO_HA = 0.5
VXC_MAGIC = "# librpa-qsgw-vxc-manifest-v2"


@dataclass(frozen=True)
class SourcePaths:
    kpoints: Path
    diagonal_vxc: Path
    wfc: list[Path]
    vxc: list[Path]


def resolve_source_paths(outdir: str | Path, n_kpoints: int) -> SourcePaths:
    if n_kpoints <= 0:
        raise ValueError("band preprocessing needs a positive k-point count")
    root = Path(outdir)
    return SourcePaths(
        kpoints=root / "KPT.info",
        diagonal_vxc=root / "vxc_out.dat",
        wfc=[root / "WFC" / f"wfk{index}_nao.txt" for index in range(1, n_kpoints + 1)],
        vxc=[root / f"vxck{index}_nao.txt" for index in range(1, n_kpoints + 1)],
    )


def ensure_native_complex_compatibility() -> None:
    if sys.byteorder != "little" or struct.calcsize("dd") != 16:
        raise ValueError(
            "LibRPA band eigenvector output requires a little-endian host "
            "with 64-bit IEEE doubles"
        )


def encode_eigenvectors(wfc: WfcData) -> bytes:
    ensure_native_complex_compatibility()
    expected = wfc.n_bands * wfc.n_basis
    if len(wfc.coefficients) != expected:
        raise ValueError(
            f"WFC coefficient count mismatch: {len(wfc.coefficients)} != {expected}"
        )
    result = bytearray(expected * 16)
    offset = 0
    for value in wfc.coefficients:
        struct.pack_into("<dd", result, offset, value.real, value.imag)
        offset += 16
    return bytes(result)


def render_eigenvalue_text(wfc: WfcData, spin_index: int) -> str:
    if spin_index <= 0:
        raise ValueError("spin index must be one-based and positive")
    if len(wfc.eigenvalues_ry) != wfc.n_bands or len(wfc.occupations) != wfc.n_bands:
        raise ValueError("WFC eigenvalue or occupation count mismatch")
    rows = []
    for band, (occupation, eigenvalue_ry) in enumerate(
        zip(wfc.occupations, wfc.eigenvalues_ry), start=1
    ):
        eigenvalue_ha = RY_TO_HA * eigenvalue_ry
        rows.append(
            f"{spin_index:8d} {band:7d} {occupation:27.16E} "
            f"{eigenvalue_ha:27.16E} {eigenvalue_ha * HA2EV:27.16E}"
        )
    return "\n".join(rows) + "\n"


def render_band_kpath_info(
    n_basis: int,
    n_states: int,
    n_spins: int,
    kpoints: list[tuple[float, float, float]],
) -> str:
    if min(n_basis, n_states, n_spins) <= 0 or not kpoints:
        raise ValueError("invalid band k-path dimensions")
    lines = [f"{n_basis:6d} {n_states:5d} {n_spins:5d} {len(kpoints):5d}"]
    lines.extend(
        f"{kx:18.12f} {ky:17.12f} {kz:17.12f}"
        for kx, ky, kz in kpoints
    )
    return "\n".join(lines) + "\n"


def render_diagonal_vxc_text(
    diagonal: DiagonalVxcData, kpoint_index: int
) -> str:
    if not 1 <= kpoint_index <= diagonal.n_kpoints:
        raise ValueError("diagonal Vxc k-point index is out of range")
    block = diagonal.n_spins * diagonal.n_bands
    start = (kpoint_index - 1) * block
    rows = []
    for spin in range(diagonal.n_spins):
        for band in range(diagonal.n_bands):
            value = diagonal.values_ha[start + spin * diagonal.n_bands + band]
            rows.append(f"{spin + 1:8d} {band + 1:7d} {value:27.16E}")
    return "\n".join(rows) + "\n"


def render_band_vxc_manifest(
    kpoints: list[tuple[float, float, float]],
    n_spins: int,
    n_bands: int,
    records: list[tuple[str, str]],
) -> str:
    if n_spins != 1:
        raise ValueError("ABACUS band Vxc manifest currently requires one spin")
    if n_bands <= 0 or len(records) != len(kpoints) or not records:
        raise ValueError("band Vxc manifest dimensions are inconsistent")
    lines = [
        VXC_MAGIC,
        "kind band",
        "producer abacus",
        "units Ry",
        "basis state",
        "gauge mf0_state",
        "spin k_index kx ky kz rows columns sha256 file",
    ]
    for index, (kpoint, record) in enumerate(zip(kpoints, records), start=1):
        name, digest = record
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid Vxc SHA256 for {name}")
        lines.append(
            f"1 {index} {kpoint[0]:.17g} {kpoint[1]:.17g} {kpoint[2]:.17g} "
            f"{n_bands} {n_bands} {digest} {name}"
        )
    return "\n".join(lines) + "\n"


def require_fresh_outputs(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite band outputs: " + ", ".join(existing))


def load_validation_report(path: Path) -> dict[str, object]:
    with require_file(path).open(encoding="ascii") as stream:
        report = json.load(stream)
    if (
        report.get("schema") != "abacus-si-band-nscf-output-validation-v1"
        or report.get("status") != "PASS"
    ):
        raise ValueError("NSCF validation report is not an accepted PASS report")
    return report


def preprocess(
    outdir: Path,
    output_dir: Path,
    validation_report_path: Path,
    manifest_name: str,
    summary_name: str,
) -> dict[str, object]:
    outdir = outdir.resolve(strict=True)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_dir = output_dir.resolve(strict=True)
    if not output_dir.is_dir():
        raise ValueError(f"band output target is not a directory: {output_dir}")
    validation = load_validation_report(validation_report_path)
    n_kpoints = int(validation["n_kpoints"])
    n_spins = int(validation["n_spins"])
    n_bands = int(validation["n_bands"])
    n_basis = int(validation["n_basis"])
    if n_spins != 1:
        raise ValueError("ABACUS band preprocessing currently requires one spin")

    sources = resolve_source_paths(outdir, n_kpoints)
    if sha256_file(require_file(sources.kpoints)) != validation["kpt_info_sha256"]:
        raise ValueError("KPT.info changed after NSCF validation")
    if sha256_file(require_file(sources.diagonal_vxc)) != validation["diagonal_vxc_sha256"]:
        raise ValueError("vxc_out.dat changed after NSCF validation")
    kpoints = parse_kpt_info_text(
        sources.kpoints.read_text(encoding="utf-8"), str(sources.kpoints)
    )
    if len(kpoints) != n_kpoints:
        raise ValueError("KPT.info count changed after NSCF validation")
    diagonal = parse_vxc_out_text(
        sources.diagonal_vxc.read_text(encoding="utf-8"),
        str(sources.diagonal_vxc),
    )
    if (diagonal.n_kpoints, diagonal.n_spins, diagonal.n_bands) != (
        n_kpoints,
        n_spins,
        n_bands,
    ):
        raise ValueError("diagonal Vxc dimensions changed after validation")

    wfc_data: list[WfcData] = []
    source_wfc_hashes = validation["wfc_hashes"]
    source_vxc_hashes = validation["vxc_hashes"]
    if not isinstance(source_wfc_hashes, dict) or not isinstance(source_vxc_hashes, dict):
        raise ValueError("NSCF validation report has malformed file hashes")
    for index, path in enumerate(sources.wfc, start=1):
        require_file(path)
        if sha256_file(path) != source_wfc_hashes.get(path.name):
            raise ValueError(f"WFC changed after NSCF validation: {path}")
        parsed = parse_wfc_text(path.read_text(encoding="utf-8"), str(path))
        if (
            parsed.k_index != index
            or parsed.n_bands != n_bands
            or parsed.n_basis != n_basis
        ):
            raise ValueError(f"WFC index or dimensions changed: {path}")
        wfc_data.append(parsed)
    for path in sources.vxc:
        require_file(path)
        if sha256_file(path) != source_vxc_hashes.get(path.name):
            raise ValueError(f"Vxc changed after NSCF validation: {path}")
        dimension, _ = parse_native_vxc_text(
            path.read_text(encoding="utf-8"), str(path)
        )
        if dimension != n_bands:
            raise ValueError(f"native Vxc state dimension changed: {path}")

    kpath_target = output_dir / "band_kpath_info"
    eigenvalue_targets = [
        output_dir / f"band_KS_eigenvalue_k_{index:05d}.txt"
        for index in range(1, n_kpoints + 1)
    ]
    eigenvector_targets = [
        output_dir / f"band_KS_eigenvector_k_{index:05d}.txt"
        for index in range(1, n_kpoints + 1)
    ]
    diagonal_targets = [
        output_dir / f"band_vxc_k_{index:05d}.txt"
        for index in range(1, n_kpoints + 1)
    ]
    matrix_targets = [
        output_dir / f"band_vxck{index}_nao.txt"
        for index in range(1, n_kpoints + 1)
    ]
    manifest_target = output_dir / manifest_name
    summary_target = output_dir / summary_name
    targets = [
        kpath_target,
        *eigenvalue_targets,
        *eigenvector_targets,
        *diagonal_targets,
        *matrix_targets,
        manifest_target,
        summary_target,
    ]
    require_fresh_outputs(targets)

    kpath_target.write_text(
        render_band_kpath_info(n_basis, n_bands, n_spins, kpoints),
        encoding="ascii",
    )
    for index, wfc in enumerate(wfc_data):
        eigenvalue_targets[index].write_text(
            render_eigenvalue_text(wfc, 1), encoding="ascii"
        )
        eigenvector_targets[index].write_bytes(encode_eigenvectors(wfc))
        diagonal_targets[index].write_text(
            render_diagonal_vxc_text(diagonal, index + 1), encoding="ascii"
        )

    manifest_records: list[tuple[str, str]] = []
    for source, target in zip(sources.vxc, matrix_targets):
        shutil.copyfile(source, target)
        source_hash = sha256_file(source)
        target_hash = sha256_file(target)
        if target_hash != source_hash:
            raise ValueError(f"copied band Vxc is not byte-identical: {target}")
        manifest_records.append((target.name, target_hash))
    manifest_target.write_text(
        render_band_vxc_manifest(kpoints, n_spins, n_bands, manifest_records),
        encoding="ascii",
    )

    generated_hashes = {
        path.name: sha256_file(path)
        for path in targets
        if path != summary_target
    }
    summary = {
        "schema": "abacus-librpa-band-preprocess-v2",
        "status": "PASS",
        "n_kpoints": n_kpoints,
        "n_spins": n_spins,
        "n_states": n_bands,
        "n_basis": n_basis,
        "band_kpath_header_order": [
            "n_basis",
            "n_states",
            "n_spins",
            "n_kpoints",
        ],
        "wfc_source_directory": "OUT.ABACUS/WFC",
        "wfc_source_schema": "wfk<one-based-k-index>_nao.txt",
        "eigenvector_binary_layout": "little_endian_band_major_complex128",
        "eigenvalue_source_units": "Ry",
        "eigenvalue_internal_units": "Ha",
        "rydberg_to_hartree": RY_TO_HA,
        "hartree_to_ev": HA2EV,
        "g0w0_diagonal_vxc_units": "Ha",
        "qsgw_full_vxc_units": "Ry",
        "qsgw_full_vxc_basis": "state",
        "qsgw_full_vxc_gauge": "mf0_state",
        "qsgw_full_vxc_copy": "byte_identical",
        "validation_report_sha256": sha256_file(validation_report_path),
        "generated_hashes": generated_hashes,
    }
    summary_target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--manifest-name", default="qsgw_vxc_band.manifest"
    )
    parser.add_argument(
        "--summary-name", default="band_preprocess.summary.json"
    )
    args = parser.parse_args()
    for name in (args.manifest_name, args.summary_name):
        if not name or Path(name).name != name:
            raise ValueError(f"output name must be a basename: {name}")
    summary = preprocess(
        args.outdir,
        args.output_dir,
        args.validation_report,
        args.manifest_name,
        args.summary_name,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
