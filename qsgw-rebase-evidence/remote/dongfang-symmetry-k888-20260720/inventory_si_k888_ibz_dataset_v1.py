#!/usr/bin/env python3
"""Inventory and hash the historical Si k888 IBZ QSGW dataset.

The source tree is read-only. The output directory must not exist. This tool
records the exact 29-point SCF, 512-point head/wing, and 143-point band input
surfaces before a QSGW input contract is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Iterable


EXPECTED_SCF_KPOINTS = 29
EXPECTED_FULL_KPOINTS = 512
EXPECTED_BAND_KPOINTS = 143
EXPECTED_SPINS = 1
EXPECTED_BANDS = 44
EXPECTED_AOS = 44
EXPECTED_GRID = (8, 8, 8)


GROUP_GLOBS: dict[str, tuple[str, ...]] = {
    "scf_wavefunctions_root": ("KS_eigenvector_*.dat",),
    "scf_wavefunctions_scf_librpa_root": (
        "scf_librpa_root/KS_eigenvector_*.dat",
    ),
    "headwing_wavefunctions": (
        "pyatb_librpa_df/KS_eigenvector_*.dat",
    ),
    "band_eigenvalues": ("band_KS_eigenvalue_k_*.txt",),
    "band_wavefunctions": ("band_KS_eigenvector_k_*.txt",),
    "scf_vxc": (
        "vxcs*k*_nao.txt",
        "vxck*s*_nao.txt",
        "scf_librpa_root/vxcs*k*_nao.txt",
        "scf_librpa_root/vxck*s*_nao.txt",
    ),
    "band_vxc": (
        "band_vxck*_nao.txt",
        "vxc_band/vxck*_nao.txt",
    ),
    "reader_cs_full": (
        "Cs_data_*.txt",
        "scf_librpa_root/Cs_data_*.txt",
    ),
    "reader_cs_shrink": (
        "Cs_shrinked_data_*.txt",
        "scf_librpa_root/Cs_shrinked_data_*.txt",
    ),
    "reader_shrink_transform": (
        "shrink_sinvS_*.txt",
        "scf_librpa_root/shrink_sinvS_*.txt",
    ),
    "reader_coulomb_full": (
        "coulomb_mat_*.txt",
        "scf_librpa_root/coulomb_mat_*.txt",
    ),
    "reader_coulomb_cut": (
        "coulomb_cut_*.txt",
        "scf_librpa_root/coulomb_cut_*.txt",
    ),
}


FIXED_CANDIDATES = (
    "band_out",
    "stru_out",
    "basis_out",
    "basis_wfc_out",
    "basis_aux_out",
    "basis_aux_shrink_out",
    "basis_out_shrink",
    "basis_out.shrink_backup",
    "band_kpath_info",
    "librpa.in",
    "scf_librpa_root/band_out",
    "scf_librpa_root/stru_out",
    "scf_librpa_root/basis_out",
    "scf_librpa_root/basis_wfc_out",
    "scf_librpa_root/basis_aux_out",
    "pyatb_librpa_df/k_path_info",
    "pyatb_librpa_df/band_out",
    "pyatb_librpa_df/velocity_matrix",
    "OUT.ABACUS_scf/INPUT",
    "OUT.ABACUS_scf/KPT",
    "OUT.ABACUS_scf/STRU",
    "OUT.ABACUS_scf/KPT.info",
)


INDEX_PATTERNS = {
    "scf_wavefunctions_root": re.compile(r"KS_eigenvector_(\d+)\.dat"),
    "scf_wavefunctions_scf_librpa_root": re.compile(
        r"KS_eigenvector_(\d+)\.dat"
    ),
    "headwing_wavefunctions": re.compile(r"KS_eigenvector_(\d+)\.dat"),
    "band_eigenvalues": re.compile(r"band_KS_eigenvalue_k_(\d+)\.txt"),
    "band_wavefunctions": re.compile(r"band_KS_eigenvector_k_(\d+)\.txt"),
    "scf_vxc": re.compile(r"(?:vxcs\d+k|vxck)(\d+)(?:s\d+)?_nao\.txt"),
    "band_vxc": re.compile(r"(?:band_)?vxck(\d+)(?:s\d+)?_nao\.txt"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(source: Path, path: Path) -> str:
    return path.relative_to(source).as_posix()


def discover(source: Path, patterns: Iterable[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in source.glob(pattern):
            if path.is_file():
                found[relative(source, path)] = path
    return [found[key] for key in sorted(found)]


def file_record(source: Path, path: Path) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {
        "path": relative(source, path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        record["symlink_target"] = os.readlink(path)
    return record


def summarize_indices(paths: list[Path], pattern: re.Pattern[str]) -> dict[str, object]:
    indexed: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    unmatched: list[str] = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            duplicates.setdefault(index, [indexed[index]]).append(path.as_posix())
        else:
            indexed[index] = path.as_posix()
    indices = sorted(indexed)
    return {
        "count": len(paths),
        "unique_index_count": len(indices),
        "index_min": indices[0] if indices else None,
        "index_max": indices[-1] if indices else None,
        "indices": indices,
        "duplicates": duplicates,
        "unmatched": unmatched,
    }


def read_integer_header(path: Path, count: int = 4) -> list[int]:
    fields = path.read_text(encoding="utf-8", errors="strict").split()
    if len(fields) < count:
        raise ValueError(f"incomplete integer header: {path}")
    return [int(value) for value in fields[:count]]


def read_text_head(path: Path, max_bytes: int = 16384) -> str:
    with path.open("rb") as stream:
        data = stream.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _is_symmetry_convention(token: str) -> bool:
    return token.lower() in {"row", "col"}


def _legacy_tail_ends_at(tokens: list[str], position: int) -> bool:
    return position == len(tokens) or (
        position + 1 < len(tokens)
        and _is_symmetry_convention(tokens[position + 1])
    )


def parse_legacy_stru_sampling(path: Path, n_reduced: int) -> dict[str, object]:
    tokens = path.read_text(encoding="utf-8", errors="strict").split()
    position = 18
    if len(tokens) <= position:
        raise ValueError(f"incomplete stru_out: {path}")
    n_atoms = int(tokens[position])
    position += 1 + 4 * n_atoms
    if len(tokens) < position + 3:
        raise ValueError(f"stru_out lacks k-grid: {path}")
    grid = tuple(int(value) for value in tokens[position : position + 3])
    position += 3
    n_full = math.prod(grid)
    selected_mapping = False
    end = position + 3 * n_reduced
    if _legacy_tail_ends_at(tokens, end + n_full):
        selected_mapping = True
    elif not _legacy_tail_ends_at(tokens, end):
        raise ValueError(
            f"cannot locate {n_reduced}-point legacy k-grid payload: {path}"
        )
    kpoint_tokens = tokens[position:end]
    mapping_tokens = tokens[end : end + n_full] if selected_mapping else []
    mapping = [int(value) for value in mapping_tokens]
    return {
        "grid": list(grid),
        "n_full": n_full,
        "n_reduced": n_reduced,
        "has_full_mapping": selected_mapping,
        "mapping_count": len(mapping),
        "mapping_min": min(mapping) if mapping else None,
        "mapping_max": max(mapping) if mapping else None,
        "mapping_unique_count": len(set(mapping)),
        "kpoint_payload_sha256": hashlib.sha256(
            " ".join(kpoint_tokens).encode("ascii")
        ).hexdigest(),
        "mapping_payload_sha256": hashlib.sha256(
            " ".join(mapping_tokens).encode("ascii")
        ).hexdigest(),
    }


def _check_index_summary(
    summary: dict[str, object], expected_start: int, expected_count: int
) -> bool:
    expected = list(range(expected_start, expected_start + expected_count))
    return (
        summary["indices"] == expected
        and not summary["duplicates"]
        and not summary["unmatched"]
    )


def collect_inventory(source: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")

    groups = {
        name: discover(source, patterns)
        for name, patterns in GROUP_GLOBS.items()
    }
    fixed = [source / name for name in FIXED_CANDIDATES if (source / name).is_file()]
    kpt_info = sorted(
        (path for path in source.rglob("KPT.info") if path.is_file()),
        key=lambda path: relative(source, path),
    )

    selected_by_name: dict[str, Path] = {}
    for path in fixed + kpt_info:
        selected_by_name[relative(source, path)] = path
    for paths in groups.values():
        for path in paths:
            selected_by_name[relative(source, path)] = path
    selected = [selected_by_name[key] for key in sorted(selected_by_name)]

    all_files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: relative(source, path),
    )
    group_records: dict[str, object] = {}
    for name, paths in groups.items():
        item: dict[str, object] = {
            "count": len(paths),
            "total_bytes": sum(path.stat().st_size for path in paths),
            "files": [relative(source, path) for path in paths],
        }
        if name in INDEX_PATTERNS:
            item["index_summary"] = summarize_indices(paths, INDEX_PATTERNS[name])
        group_records[name] = item

    headers: dict[str, object] = {}
    for name in (
        "band_out",
        "scf_librpa_root/band_out",
        "pyatb_librpa_df/band_out",
        "pyatb_librpa_df/k_path_info",
        "band_kpath_info",
    ):
        path = source / name
        if path.is_file():
            headers[name] = {
                "integers": read_integer_header(path),
                "text_head": read_text_head(path),
            }

    scf_band_path = next(
        (
            source / name
            for name in ("band_out", "scf_librpa_root/band_out")
            if (source / name).is_file()
        ),
        None,
    )
    stru_sampling: dict[str, object] = {}
    if scf_band_path is not None:
        n_reduced = read_integer_header(scf_band_path)[0]
        for name in ("stru_out", "scf_librpa_root/stru_out"):
            path = source / name
            if path.is_file():
                stru_sampling[name] = parse_legacy_stru_sampling(path, n_reduced)

    assertions: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        assertions.append({"name": name, "passed": bool(passed), "detail": detail})

    scf_headers = [
        headers[name]["integers"]
        for name in ("band_out", "scf_librpa_root/band_out")
        if name in headers
    ]
    add(
        "scf_band_dimensions",
        [EXPECTED_SCF_KPOINTS, EXPECTED_SPINS, EXPECTED_BANDS, EXPECTED_AOS]
        in scf_headers,
        scf_headers,
    )
    add(
        "headwing_band_dimensions",
        headers.get("pyatb_librpa_df/band_out", {}).get("integers")
        == [EXPECTED_FULL_KPOINTS, EXPECTED_SPINS, EXPECTED_BANDS, EXPECTED_AOS],
        headers.get("pyatb_librpa_df/band_out"),
    )
    add(
        "headwing_kpath_dimensions",
        headers.get("pyatb_librpa_df/k_path_info", {}).get("integers")
        == [EXPECTED_AOS, EXPECTED_BANDS, EXPECTED_SPINS, EXPECTED_FULL_KPOINTS],
        headers.get("pyatb_librpa_df/k_path_info"),
    )
    add(
        "band_kpath_dimensions",
        headers.get("band_kpath_info", {}).get("integers")
        == [EXPECTED_BANDS, EXPECTED_AOS, EXPECTED_SPINS, EXPECTED_BAND_KPOINTS],
        headers.get("band_kpath_info"),
    )

    root_scf = group_records["scf_wavefunctions_root"]["index_summary"]
    nested_scf = group_records["scf_wavefunctions_scf_librpa_root"]["index_summary"]
    add(
        "scf_wavefunction_indices",
        _check_index_summary(root_scf, 0, EXPECTED_SCF_KPOINTS)
        or _check_index_summary(nested_scf, 0, EXPECTED_SCF_KPOINTS),
        {"root": root_scf, "scf_librpa_root": nested_scf},
    )
    for name, start, count in (
        ("headwing_wavefunctions", 0, EXPECTED_FULL_KPOINTS),
        ("band_eigenvalues", 1, EXPECTED_BAND_KPOINTS),
        ("band_wavefunctions", 1, EXPECTED_BAND_KPOINTS),
        ("scf_vxc", 1, EXPECTED_SCF_KPOINTS),
        ("band_vxc", 1, EXPECTED_BAND_KPOINTS),
    ):
        summary = group_records[name]["index_summary"]
        add(f"{name}_indices", _check_index_summary(summary, start, count), summary)

    add(
        "reader_static_basis",
        all((source / name).is_file() for name in ("stru_out", "basis_wfc_out", "basis_aux_out")),
        [name for name in ("stru_out", "basis_wfc_out", "basis_aux_out") if (source / name).is_file()],
    )
    for name in (
        "reader_cs_full",
        "reader_cs_shrink",
        "reader_shrink_transform",
        "reader_coulomb_full",
        "reader_coulomb_cut",
    ):
        add(f"{name}_nonempty", group_records[name]["count"] > 0, group_records[name]["count"])

    valid_stru = [
        record
        for record in stru_sampling.values()
        if record["grid"] == list(EXPECTED_GRID)
        and record["n_full"] == EXPECTED_FULL_KPOINTS
        and record["n_reduced"] == EXPECTED_SCF_KPOINTS
        and record["has_full_mapping"]
        and record["mapping_count"] == EXPECTED_FULL_KPOINTS
        and record["mapping_min"] == 1
        and record["mapping_max"] == EXPECTED_SCF_KPOINTS
        and record["mapping_unique_count"] == EXPECTED_SCF_KPOINTS
    ]
    add("stru_k888_ibz_mapping", bool(valid_stru), stru_sampling)

    return {
        "schema": "librpa-qsgw-si-k888-ibz-inventory-v1",
        "source_root": str(source),
        "expected": {
            "grid": list(EXPECTED_GRID),
            "n_scf_kpoints": EXPECTED_SCF_KPOINTS,
            "n_full_kpoints": EXPECTED_FULL_KPOINTS,
            "n_band_kpoints": EXPECTED_BAND_KPOINTS,
            "n_spins": EXPECTED_SPINS,
            "n_bands": EXPECTED_BANDS,
            "n_aos": EXPECTED_AOS,
        },
        "source_file_count": len(all_files),
        "source_total_bytes": sum(path.stat().st_size for path in all_files),
        "selected_file_count": len(selected),
        "selected_total_bytes": sum(path.stat().st_size for path in selected),
        "selected_files": [relative(source, path) for path in selected],
        "all_file_records": [file_record(source, path) for path in all_files],
        "groups": group_records,
        "headers": headers,
        "stru_sampling": stru_sampling,
        "assertions": assertions,
        "passed": all(item["passed"] for item in assertions),
    }


def write_inventory(source: Path, output: Path, inventory: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "SOURCE_ROOT.txt").write_text(str(source.resolve()) + "\n", encoding="ascii")
    (output / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    with (output / "FILE_LIST.tsv").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("path\tsize\tmtime_ns\tis_symlink\tsymlink_target\n")
        for record in inventory["all_file_records"]:
            stream.write(
                f"{record['path']}\t{record['size']}\t{record['mtime_ns']}\t"
                f"{str(record['is_symlink']).lower()}\t{record.get('symlink_target', '')}\n"
            )
    with (output / "SELECTED_SHA256SUMS.txt").open(
        "w", encoding="ascii", newline="\n"
    ) as stream:
        for name in inventory["selected_files"]:
            path = source / name
            stream.write(f"{sha256_file(path)}  {name}\n")
    (output / "VALIDATION.json").write_text(
        json.dumps(
            {"passed": inventory["passed"], "assertions": inventory["assertions"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    inventory: dict[str, object] | None = None
    try:
        inventory = collect_inventory(args.source)
        write_inventory(args.source, args.output, inventory)
        if not inventory["passed"]:
            (args.output / "FAILED").write_text(
                "dataset inventory validation failed\n", encoding="ascii"
            )
            return 2
    except Exception as error:  # preserve a machine-readable failed evidence packet
        if args.output.exists():
            (args.output / "FAILED.json").write_text(
                json.dumps(
                    {"error": str(error), "traceback": traceback.format_exc()},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
        raise
    print(json.dumps({"passed": True, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
