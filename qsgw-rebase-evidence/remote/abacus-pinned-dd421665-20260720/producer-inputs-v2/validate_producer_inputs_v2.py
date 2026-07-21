#!/usr/bin/env python3
"""Validate the frozen ABACUS Si producer input contract."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXPECTED_ASSETS = {
    "Si_ONCV_PBE-1.0.upf": (
        "e3f420b7057c50912907ef801c19df9e4760d029b6b56788a0153fce78f6a08a"
    ),
    "Si_gga_8au_100Ry_3s3p2d.orb": (
        "3d6e4fb7014b51cca13fb0a505d7556a6e7ca2c2c4f1e16c0a66788cf86f3279"
    ),
    "Si_3s3p2d1f1g_pca1e-6.abfs": (
        "1ea35d9b063c061c614600f27b20599c77c371a0b10ad0fa4551c2c99f333d6e"
    ),
}

COMMON_SCF = {
    "calculation": "scf",
    "nbands": "44",
    "basis_type": "lcao",
    "ecutwfc": "120",
    "rpa": "1",
    "out_mat_xc": "1",
    "out_mat_xc2": "1",
    "out_app_flag": "0",
    "out_librpa_reader_version": "0",
    "out_ri_cv": "0",
    "shrink_abfs_pca_thr": "-1",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def parse_input(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "INPUT_PARAMETERS":
        fail(f"{path.name}: missing INPUT_PARAMETERS header")
    for number, raw in enumerate(lines[1:], start=2):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        parts = content.split()
        if len(parts) < 2:
            fail(f"{path.name}:{number}: malformed setting")
        key = parts[0].lower()
        if key in values:
            fail(f"{path.name}:{number}: duplicate key {key}")
        values[key] = " ".join(parts[1:])
    return values


def require_values(name: str, values: dict[str, str], expected: dict[str, str]) -> None:
    for key, wanted in expected.items():
        actual = values.get(key)
        if actual != wanted:
            fail(f"{name}: {key}={actual!r}, expected {wanted!r}")


def validate_inputs() -> None:
    cases = {
        "INPUT_scf_symmetry": {
            **COMMON_SCF,
            "symmetry": "1",
            "exx_symmetry_realspace": "1",
        },
        "INPUT_scf_fullbz": {
            **COMMON_SCF,
            "symmetry": "-1",
            "exx_symmetry_realspace": "0",
        },
        "INPUT_nscf_band": {
            "calculation": "nscf",
            "nbands": "44",
            "basis_type": "lcao",
            "symmetry": "-1",
            "init_chg": "file",
            "ecutwfc": "120",
            "out_mat_xc": "1",
            "out_mat_xc2": "1",
            "out_wfc_lcao": "1",
            "out_app_flag": "0",
            "exx_symmetry_realspace": "0",
            "shrink_abfs_pca_thr": "-1",
        },
    }
    forbidden = {"fold_c", "n_params_anacon"}
    for name, expected in cases.items():
        values = parse_input(ROOT / name)
        require_values(name, values, expected)
        present = forbidden.intersection(values)
        if present:
            fail(f"{name}: forbidden keys present: {sorted(present)}")


def significant_lines(path: Path) -> list[str]:
    result = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        content = raw.split("#", 1)[0].strip()
        if content:
            result.append(content)
    return result


def validate_kpt_grid(name: str, grid: int) -> None:
    lines = significant_lines(ROOT / name)
    expected = ["K_POINTS", "0", "Gamma", f"{grid} {grid} {grid} 0 0 0"]
    if lines != expected:
        fail(f"{name}: expected exact Gamma {grid}x{grid}x{grid} card")


def validate_kpt_band() -> None:
    lines = significant_lines(ROOT / "KPT_band")
    if len(lines) != 13 or lines[:3] != ["K_POINTS", "10", "Line"]:
        fail("KPT_band: expected a ten-point Line-mode card")
    for number, line in enumerate(lines[3:], start=4):
        fields = line.split()
        if len(fields) != 4:
            fail(f"KPT_band:{number}: expected kx ky kz segment_count")
        try:
            tuple(float(value) for value in fields[:3])
            if int(fields[3]) <= 0:
                raise ValueError
        except ValueError:
            fail(f"KPT_band:{number}: invalid numeric field")


def validate_stru() -> None:
    text = (ROOT / "STRU").read_text(encoding="utf-8")
    for asset in EXPECTED_ASSETS:
        matches = re.findall(rf"(?<!\S){re.escape(asset)}(?!\S)", text)
        if len(matches) != 1:
            fail(f"STRU: expected exactly one reference to {asset}")


def parse_asset_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for number, raw in enumerate(
        (ROOT / "ASSET_SHA256SUMS").read_text(encoding="ascii").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            fail(f"ASSET_SHA256SUMS:{number}: malformed entry")
        digest, name = fields
        if name in manifest:
            fail(f"ASSET_SHA256SUMS:{number}: duplicate asset {name}")
        manifest[name] = digest
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(assets_dir: Path | None) -> None:
    manifest = parse_asset_manifest()
    if manifest != EXPECTED_ASSETS:
        fail("ASSET_SHA256SUMS does not match the frozen asset contract")
    if assets_dir is None:
        return
    for name, expected in EXPECTED_ASSETS.items():
        path = assets_dir / name
        if not path.is_file():
            fail(f"asset missing: {path}")
        actual = sha256(path)
        if actual != expected:
            fail(f"asset hash mismatch: {name}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="also verify the three copied producer assets in this directory",
    )
    args = parser.parse_args()
    validate_inputs()
    validate_kpt_grid("KPT_k444", 4)
    validate_kpt_grid("KPT_k888", 8)
    validate_kpt_band()
    validate_stru()
    validate_assets(args.assets_dir)
    print("PASS: pinned ABACUS Si producer inputs v2")


if __name__ == "__main__":
    main()
