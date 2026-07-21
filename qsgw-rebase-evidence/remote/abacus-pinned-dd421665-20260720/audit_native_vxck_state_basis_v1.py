#!/usr/bin/env python3
"""Prove the basis contract of native ABACUS out_mat_xc matrices from source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_ABACUS_COMMIT = "dd4216653386d32f79e3219f3ea5dd2d229c1c5a"
EXPECTED_ABACUS_WRITE_VXC_SHA256 = (
    "f5fd558c74d634bca0f6952fc88df3b49bbf0f49a69a5389b131c5d24193935f"
)
EXPECTED_LEGACY_SOURCE_SHA256 = (
    "a932e60fa1b96e10a44eaa2f07da23bf57bbbb1c36600b1d4f9861a5d843e68e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required source file is missing: {path}")
    return path


def require_fragments(source: str, fragments: list[str], label: str) -> None:
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        raise ValueError(f"{label}: missing required source fragments: {missing}")


def require_ordered_fragments(
    source: str, fragments: list[str], label: str
) -> None:
    offset = 0
    for fragment in fragments:
        position = source.find(fragment, offset)
        if position < 0:
            raise ValueError(
                f"{label}: required ordered source fragment is missing: {fragment}"
            )
        offset = position + len(fragment)


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def provenance_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_report(
    repo: Path,
    abacus_repo: Path,
    legacy_source: Path,
    generator: Path,
) -> dict[str, object]:
    repo = repo.resolve(strict=True)
    abacus_repo = abacus_repo.resolve(strict=True)
    legacy_source = require_file(legacy_source.resolve(strict=True))
    generator = require_file(generator.resolve(strict=True))
    producer_source = require_file(
        abacus_repo / "source/source_io/module_hs/write_vxc.hpp"
    )
    candidate_vxc = require_file(repo / "src/qsgw/vxc_io.cpp")
    candidate_driver = require_file(repo / "driver/tasks/qsgw.cpp")
    candidate_test = require_file(repo / "src/test/test_qsgw_vxc_io.cpp")

    producer_commit = git_output(abacus_repo, "rev-parse", "HEAD")
    if producer_commit != EXPECTED_ABACUS_COMMIT:
        raise ValueError(f"unexpected ABACUS source commit: {producer_commit}")
    producer_dirty = git_output(abacus_repo, "status", "--porcelain")
    if producer_dirty:
        raise ValueError("pinned ABACUS source checkout is dirty")

    producer_sha = sha256_file(producer_source)
    if producer_sha != EXPECTED_ABACUS_WRITE_VXC_SHA256:
        raise ValueError(f"ABACUS write_vxc.hpp SHA256 mismatch: {producer_sha}")
    legacy_sha = sha256_file(legacy_source)
    if legacy_sha != EXPECTED_LEGACY_SOURCE_SHA256:
        raise ValueError(f"exact-847 source SHA256 mismatch: {legacy_sha}")

    producer_text = producer_source.read_text(encoding="utf-8")
    require_ordered_fragments(
        producer_text,
        [
            "const std::vector<TK>& vxc_tot_k_mo = cVc(",
            '"vxc","nao"',
            "ModuleIO::save_mat(",
            "vxc_tot_k_mo.data()",
            "nbands,",
            "true /*triangle*/",
        ],
        "pinned ABACUS out_mat_xc writer",
    )
    require_fragments(
        producer_text,
        [
            "nbands, nbands, nbasis",
            "transa = (std::is_same<T, double>::value ? 'T' : 'C')",
        ],
        "pinned ABACUS cVc transform",
    )

    legacy_text = legacy_source.read_text(encoding="utf-8", errors="replace")
    require_fragments(
        legacy_text,
        [
            'oss_vxc_text_k << "vxck"',
            "read_abacus_upper_triangle_matrix(vxc_text_k_path, vxc0[ispin][ikpt]",
            "read_abacus_upper_triangle_matrix(vxc_band_text_k_path",
            "band_vxck*_nao.txt is also in KS-orbital representation despite the suffix",
            "construct_H0_GW_cut(",
            "H_KS0_band, vxc_band",
        ],
        "exact-847 native Vxc consumer",
    )

    candidate_vxc_text = candidate_vxc.read_text(
        encoding="utf-8", errors="replace"
    )
    candidate_driver_text = candidate_driver.read_text(
        encoding="utf-8", errors="replace"
    )
    candidate_test_text = candidate_test.read_text(
        encoding="utf-8", errors="replace"
    )
    require_fragments(
        candidate_vxc_text,
        [
            "const bool valid_abacus_state",
            "result.basis_ == VxcBasis::State",
            "result.gauge_ == VxcGauge::Mf0State",
            "const bool valid_abacus_nao",
            "if (basis == VxcBasis::State)",
            "return input.copy();",
        ],
        "candidate QSGW Vxc contract",
    )
    require_fragments(
        candidate_driver_text,
        [
            "manifest.basis() == VxcBasis::Nao",
            "? reference.get_n_aos() * reference.get_n_spinor()",
            ": reference.get_n_bands();",
            "prepare_vxc_in_fixed_state_basis(",
        ],
        "candidate QSGW Vxc caller",
    )
    require_fragments(
        candidate_test_text,
        [
            "test_abacus_out_mat_xc_manifest_is_state_basis_without_projection",
            '"basis\\tstate\\n"',
            '"gauge\\tmf0_state\\n"',
        ],
        "candidate QSGW Vxc unit test",
    )

    generator_text = generator.read_text(encoding="utf-8")
    require_fragments(
        generator_text,
        [
            '"basis state"',
            '"gauge mf0_state"',
            "read_abacus_native_vxc_dimension(matrix) != n_bands",
            "dim=n_bands",
            '"vxc_basis_transform": "none"',
        ],
        "ABACUS QSGW contract generator v3",
    )

    return {
        "schema": "librpa-native-vxck-state-basis-audit-v1",
        "source_contract_passed": True,
        "conclusion": (
            "ABACUS out_mat_xc writes C-dagger Vxc_AO C in the producer KS "
            "state basis; the _nao filename suffix is not a basis label"
        ),
        "producer": {
            "commit": producer_commit,
            "source": provenance_path(producer_source, abacus_repo),
            "source_sha256": producer_sha,
            "producer_transform": "C_dagger_Vxc_AO_C",
            "matrix_basis": "ks_state",
            "matrix_shape": "n_bands_x_n_bands",
            "energy_units": "Ry",
            "nao_suffix_is_not_basis_label": True,
        },
        "legacy": {
            "commit": "8476213f66c68efb43404713eacbd04966820f26",
            "source": provenance_path(legacy_source, repo),
            "source_sha256": legacy_sha,
            "basis_interpretation": "producer_ks_state",
            "basis_transform": "none",
            "energy_conversion": "Ry_to_Ha_scale_0.5",
        },
        "candidate": {
            "vxc_io_source": provenance_path(candidate_vxc, repo),
            "vxc_io_source_sha256": sha256_file(candidate_vxc),
            "driver_source": provenance_path(candidate_driver, repo),
            "driver_source_sha256": sha256_file(candidate_driver),
            "unit_test_source": provenance_path(candidate_test, repo),
            "unit_test_source_sha256": sha256_file(candidate_test),
            "manifest_contract": "abacus_Ry_state_mf0_state",
            "basis_transform": "none",
            "true_nao_projection_path_retained": True,
            "energy_conversion": "Ry_to_Ha_scale_0.5",
        },
        "generator": {
            "source": provenance_path(generator, repo),
            "source_sha256": sha256_file(generator),
            "manifest_basis": "state",
            "manifest_gauge": "mf0_state",
            "dimension_binding": "n_bands",
        },
        "numerical_runtime_parity_pending": True,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_repo = script_dir.parents[2]
    workspace = default_repo.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument(
        "--abacus-repo",
        type=Path,
        default=(
            workspace
            / "research/librpa/analysis/si8_newlat_origin_k999_g0w0_20260625"
            / "upstream_input_contract_study_20260720/abacus-develop-master_ghj"
        ),
    )
    parser.add_argument(
        "--legacy-source",
        type=Path,
        default=(
            default_repo
            / "qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720"
            / "oracle-source-audit-v1/exact847/task_qsgw_band_0.cpp"
        ),
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=script_dir / "prepare_abacus_qsgw_ibz_contract_v3.py",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.repo,
        args.abacus_repo,
        args.legacy_source,
        args.generator,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="ascii")
    print(rendered, end="")


if __name__ == "__main__":
    main()
