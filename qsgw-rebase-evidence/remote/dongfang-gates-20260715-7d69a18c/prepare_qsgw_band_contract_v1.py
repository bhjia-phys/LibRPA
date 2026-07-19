#!/usr/bin/env python3
"""Generate the qsgw_band input contract and band Vxc manifest (v2).

Reads a dataset directory containing a disabled baseline contract
(``qsgw_input.disabled.contract``), ``band_kpath_info``,
``band_KS_eigenvalue_k_NNNNN.txt`` / ``band_KS_eigenvector_k_NNNNN.txt``
and ``vxc_band/vxckKs1_nao.txt`` files, and writes two new files:

* ``qsgw_input.band.contract`` — the baseline contract with exactly two
  metadata lines changed (``n_band_kpoints 0 -> N`` and
  ``band_update off -> operator_fourier``) and the band role records
  appended (``band_kpoints`` x1, ``band_mf0_eigenvalues`` xN,
  ``band_mf0_wavefunctions`` xN, ``vxc_band_manifest`` x1), matching the
  roles enforced by ``src/qsgw/input_contract.cpp``.
* ``qsgw_vxc_band.v2.manifest`` — the v2 manifest parsed by
  ``src/qsgw/vxc_io.cpp`` (magic line, exactly five metadata keys,
  ``spin k_index kx ky kz rows columns sha256 file`` table with
  one-based spin/k_index and verbatim k coordinates).

The generator is fail-closed: missing/symlinked inputs, wrong
eigenvector byte sizes, duplicate roles, malformed baselines, or
pre-existing outputs abort the run before anything is written. The
baseline contract is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CONTRACT_MAGIC = "# librpa-qsgw-input-contract-v1"
MANIFEST_MAGIC = "# librpa-qsgw-vxc-manifest-v2"
BASELINE_CONTRACT_NAME = "qsgw_input.disabled.contract"
BAND_CONTRACT_NAME = "qsgw_input.band.contract"
BAND_MANIFEST_NAME = "qsgw_vxc_band.v2.manifest"
BAND_KPATH_NAME = "band_kpath_info"
COMPLEX_DOUBLE_BYTES = 16
METADATA_EDITS = (
    ("n_band_kpoints", "0"),
    ("band_update", "off"),
)
BAND_UPDATE_ENABLED = "operator_fourier"


class ContractGenerationError(ValueError):
    """Raised when the dataset or baseline contract is unusable."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_text(path: Path, label: str) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ContractGenerationError(f"{label}: unreadable: {error}") from error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractGenerationError(f"{label}: not UTF-8") from error
    if "\r" in text:
        raise ContractGenerationError(f"{label}: non-LF line endings")
    return text


def parse_band_kpath_info(text: str) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Parse band_kpath_info; return (n_bands, n_basis, kpoint tokens)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ContractGenerationError("band_kpath_info: empty file")
    header = lines[0].split()
    if len(header) != 4:
        raise ContractGenerationError(
            "band_kpath_info: header must have 4 integers"
        )
    try:
        n_bands, n_basis, n_spins, n_kpath = (int(field) for field in header)
    except ValueError as error:
        raise ContractGenerationError(
            "band_kpath_info: header must have 4 integers"
        ) from error
    if n_bands <= 0 or n_basis <= 0 or n_kpath <= 0:
        raise ContractGenerationError(
            "band_kpath_info: non-positive dimension or k-point count"
        )
    if n_spins != 1:
        raise ContractGenerationError(
            f"band_kpath_info: unsupported n_spins {n_spins} (only 1)"
        )
    kpoint_lines = lines[1:]
    if len(kpoint_lines) != n_kpath:
        raise ContractGenerationError(
            f"band_kpath_info: expected {n_kpath} k-point lines, "
            f"found {len(kpoint_lines)}"
        )
    kpoints: list[tuple[str, str, str]] = []
    for index, line in enumerate(kpoint_lines, 1):
        fields = line.split()
        if len(fields) != 3:
            raise ContractGenerationError(
                f"band_kpath_info:{index + 1}: expected 3 coordinates"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ContractGenerationError(
                f"band_kpath_info:{index + 1}: non-numeric coordinate"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise ContractGenerationError(
                f"band_kpath_info:{index + 1}: non-finite coordinate"
            )
        kpoints.append((fields[0], fields[1], fields[2]))
    return n_bands, n_basis, kpoints


def parse_baseline_contract(text: str) -> tuple[list[str], dict[str, str], set[tuple[str, str]]]:
    """Split the baseline into lines, metadata, and existing (role, file)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0].strip() != CONTRACT_MAGIC:
        raise ContractGenerationError(
            "baseline contract: missing librpa-qsgw-input-contract-v1 header"
        )
    metadata: dict[str, str] = {}
    roles: set[tuple[str, str]] = set()
    in_table = False
    for line_number, line in enumerate(lines[1:], 2):
        content = line.strip()
        if not content:
            raise ContractGenerationError(
                f"baseline contract:{line_number}: unexpected blank line"
            )
        fields = content.split()
        if not in_table:
            if fields == ["role", "sha256", "file"]:
                in_table = True
                continue
            if len(fields) != 2 or fields[0].startswith("#"):
                raise ContractGenerationError(
                    f"baseline contract:{line_number}: malformed metadata line"
                )
            key, value = fields
            if key in metadata:
                raise ContractGenerationError(
                    f"baseline contract:{line_number}: duplicate metadata {key}"
                )
            metadata[key] = value
        else:
            if len(fields) != 3 or fields[0].startswith("#"):
                raise ContractGenerationError(
                    f"baseline contract:{line_number}: malformed role line"
                )
            role, _sha256, filename = fields
            if (role, filename) in roles:
                raise ContractGenerationError(
                    f"baseline contract:{line_number}: duplicate role record "
                    f"{role} {filename}"
                )
            roles.add((role, filename))
    if not in_table:
        raise ContractGenerationError(
            "baseline contract: missing role sha256 file table header"
        )
    for key, expected in METADATA_EDITS:
        actual = metadata.get(key)
        if actual != expected:
            raise ContractGenerationError(
                f"baseline contract: expected {key} {expected}, "
                f"found {actual!r} (not a disabled contract)"
            )
    for key in ("n_bands", "n_aos"):
        value = metadata.get(key)
        try:
            parsed = int(value) if value is not None else 0
        except ValueError:
            parsed = 0
        if parsed <= 0:
            raise ContractGenerationError(
                f"baseline contract: invalid {key} metadata"
            )
    return lines, metadata, roles


def build_band_manifest(
    kpoints: list[tuple[str, str, str]],
    n_aos: int,
    vxc_records: list[tuple[str, str]],
) -> str:
    lines = [
        MANIFEST_MAGIC,
        "kind band",
        "producer abacus",
        "units Ry",
        "basis nao",
        "gauge ao_bloch",
        "spin k_index kx ky kz rows columns sha256 file",
    ]
    for index, ((kx, ky, kz), (sha256, filename)) in enumerate(
        zip(kpoints, vxc_records), 1
    ):
        lines.append(
            f"1 {index} {kx} {ky} {kz} {n_aos} {n_aos} {sha256} {filename}"
        )
    return "\n".join(lines) + "\n"


def build_band_contract(
    baseline_lines: list[str],
    n_band_kpoints: int,
    appended_roles: list[tuple[str, str, str]],
) -> tuple[str, list[dict[str, str]]]:
    replacements = {
        "n_band_kpoints": str(n_band_kpoints),
        "band_update": BAND_UPDATE_ENABLED,
    }
    changes: list[dict[str, str]] = []
    changed_indices: list[int] = []
    new_lines: list[str] = []
    for index, line in enumerate(baseline_lines):
        fields = line.split()
        if len(fields) == 2 and fields[0] in replacements:
            new_line = f"{fields[0]} {replacements[fields[0]]}"
            changes.append(
                {"key": fields[0], "old": fields[1], "new": replacements[fields[0]]}
            )
            changed_indices.append(index)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    if sorted(changed_indices) != changed_indices or len(changed_indices) != 2:
        raise ContractGenerationError(
            "internal error: metadata edit did not touch exactly 2 lines"
        )
    appended_lines = [
        f"{role} {sha256} {filename}"
        for role, sha256, filename in appended_roles
    ]
    new_lines.extend(appended_lines)

    # Fail-closed self-check: exactly 2 changed lines plus pure appends.
    head = new_lines[: len(baseline_lines)]
    actual_changes = [
        index
        for index, (old, new) in enumerate(zip(baseline_lines, head))
        if old != new
    ]
    if actual_changes != changed_indices:
        raise ContractGenerationError(
            "internal error: contract diff is not the 2 metadata lines"
        )
    if new_lines[len(baseline_lines):] != appended_lines:
        raise ContractGenerationError(
            "internal error: contract tail is not a pure append"
        )
    return "\n".join(new_lines) + "\n", changes


def _check_input_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ContractGenerationError(f"{label}: symlink not allowed: {path}")
    if not path.is_file():
        raise ContractGenerationError(f"{label}: missing file: {path}")


def generate_report(dataset: Path, dry_run: bool = False) -> dict[str, object]:
    dataset = Path(dataset)
    if not dataset.is_dir():
        raise ContractGenerationError(f"dataset directory missing: {dataset}")

    baseline_path = dataset / BASELINE_CONTRACT_NAME
    _check_input_file(baseline_path, "baseline contract")
    baseline_text = _read_text(baseline_path, "baseline contract")
    baseline_lines, metadata, existing_roles = parse_baseline_contract(
        baseline_text
    )
    n_aos = int(metadata["n_aos"])
    n_bands_contract = int(metadata["n_bands"])

    kpath_path = dataset / BAND_KPATH_NAME
    _check_input_file(kpath_path, "band_kpath_info")
    n_bands, n_basis, kpoints = parse_band_kpath_info(
        _read_text(kpath_path, "band_kpath_info")
    )
    if n_bands != n_bands_contract or n_basis != n_aos:
        raise ContractGenerationError(
            f"band_kpath_info dimensions ({n_bands}, {n_basis}) do not match "
            f"baseline contract n_bands/n_aos ({n_bands_contract}, {n_aos})"
        )
    n_band_kpoints = len(kpoints)
    expected_eigenvector_bytes = (
        n_basis * n_bands * COMPLEX_DOUBLE_BYTES
    )

    eigenvalue_records: list[tuple[str, str, str]] = []
    wavefunction_records: list[tuple[str, str, str]] = []
    vxc_records: list[tuple[str, str]] = []
    for kpoint in range(1, n_band_kpoints + 1):
        eigenvalue_name = f"band_KS_eigenvalue_k_{kpoint:05d}.txt"
        eigenvector_name = f"band_KS_eigenvector_k_{kpoint:05d}.txt"
        vxc_name = f"vxc_band/vxck{kpoint}s1_nao.txt"

        eigenvalue_path = dataset / eigenvalue_name
        eigenvector_path = dataset / eigenvector_name
        vxc_path = dataset / vxc_name
        _check_input_file(eigenvalue_path, "band eigenvalue")
        _check_input_file(eigenvector_path, "band eigenvector")
        _check_input_file(vxc_path, "band vxc")
        eigenvector_size = eigenvector_path.stat().st_size
        if eigenvector_size != expected_eigenvector_bytes:
            raise ContractGenerationError(
                f"band eigenvector size mismatch: {eigenvector_name} is "
                f"{eigenvector_size} bytes, expected "
                f"{expected_eigenvector_bytes} "
                f"({n_basis}x{n_bands} complex<double>)"
            )
        eigenvalue_records.append(
            ("band_mf0_eigenvalues", _sha256_file(eigenvalue_path), eigenvalue_name)
        )
        wavefunction_records.append(
            (
                "band_mf0_wavefunctions",
                _sha256_file(eigenvector_path),
                eigenvector_name,
            )
        )
        vxc_records.append((_sha256_file(vxc_path), vxc_name))

    manifest_text = build_band_manifest(kpoints, n_aos, vxc_records)
    manifest_sha256 = _sha256_bytes(manifest_text.encode("utf-8"))

    appended_roles = (
        [("band_kpoints", _sha256_file(kpath_path), BAND_KPATH_NAME)]
        + eigenvalue_records
        + wavefunction_records
        + [("vxc_band_manifest", manifest_sha256, BAND_MANIFEST_NAME)]
    )
    duplicates = sorted(
        {(role, filename) for role, _sha, filename in appended_roles}
        & existing_roles
    )
    if duplicates:
        raise ContractGenerationError(
            f"duplicate role records already in baseline: {duplicates[:5]}"
        )

    contract_text, metadata_changes = build_band_contract(
        baseline_lines, n_band_kpoints, appended_roles
    )
    contract_sha256 = _sha256_bytes(contract_text.encode("utf-8"))

    band_contract_path = dataset / BAND_CONTRACT_NAME
    manifest_path = dataset / BAND_MANIFEST_NAME
    if not dry_run:
        for output in (band_contract_path, manifest_path):
            if output.exists():
                raise ContractGenerationError(
                    f"refusing to overwrite existing output: {output}"
                )
        manifest_path.write_text(
            manifest_text, encoding="utf-8", newline="\n"
        )
        band_contract_path.write_text(
            contract_text, encoding="utf-8", newline="\n"
        )

    return {
        "band_contract": str(band_contract_path),
        "vxc_manifest": str(manifest_path),
        "n_band_kpoints": n_band_kpoints,
        "band_roles_added": len(appended_roles),
        "metadata_changes": metadata_changes,
        "diff_summary": {
            "metadata_lines_changed": len(metadata_changes),
            "role_lines_appended": len(appended_roles),
        },
        "sha256": {
            "band_contract": contract_sha256,
            "vxc_manifest": manifest_sha256,
        },
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qsgw_input.band.contract and "
            "qsgw_vxc_band.v2.manifest from a disabled baseline contract "
            "and band assets."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="dataset directory holding the baseline contract and band files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the report without writing files",
    )
    args = parser.parse_args(argv)

    try:
        report = generate_report(args.dataset, dry_run=args.dry_run)
    except (ContractGenerationError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
