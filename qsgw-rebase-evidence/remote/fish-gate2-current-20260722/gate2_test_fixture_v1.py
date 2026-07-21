from __future__ import annotations

import hashlib
import shutil
import struct
import uuid
from contextlib import contextmanager
from pathlib import Path


HA2EV = 27.211386245988


@contextmanager
def scratch_directory(parent: Path):
    path = parent / f"unit-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def contract_header(contract_sha: str, *, symmetry: str = "exx_on_gw_on_rpa_on") -> str:
    return (
        "# qsgw_contract_version 6\n"
        "# fixed_basis immutable_mf0\n"
        "# live_update eigenvalues_wfc\n"
        "# velocity disabled_stage1\n"
        "# headwing disabled_stage1\n"
        f"# symmetry {symmetry}\n"
        "# hartree disabled_stage1\n"
        "# band disabled_stage1\n"
        "# h_qsgw_cut disabled_non_band\n"
        "# qsgw_input_contract qsgw_input.contract\n"
        f"# qsgw_input_contract_sha256 {contract_sha}\n"
        "# qsgw_mixer none\n"
        "# qsgw_mixing_beta 0.20000000000000001\n"
    )


def _matrix_rows(
    iteration: int,
    component: str,
    matrix: list[list[complex]],
    *,
    ifrequency: int = -1,
    frequency: float = 0.0,
) -> list[str]:
    rows = []
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            rows.append(
                f"{iteration} 0 {component} 0 0 {ifrequency} "
                f"{frequency:.17g} {row} {column} "
                f"{value.real:.17g} {value.imag:.17g}"
            )
    return rows


def matrix_trace(
    contract_sha: str,
    *,
    raw_shift: float = 0.0,
    exx_lower_shift: float = 0.0,
    symmetry: str = "exx_on_gw_on_rpa_on",
) -> str:
    h0 = [[-1.0 + 0j, 0j], [0j, 1.0 + 0j]]
    vxc = [[0.1 + 0j, 0j], [0j, 0.2 + 0j]]
    exx = [[0.3 + 0j, 0j], [exx_lower_shift + 0j, 0.4 + 0j]]
    vc = [[0.01 + 0j, 0j], [0j, 0.02 + 0j]]
    raw = [[-0.79 + raw_shift + 0j, 0j], [0j, 1.22 + 0j]]
    identity = [[1.0 + 0j, 0j], [0j, 1.0 + 0j]]
    occupations = [[2.0 + 0j, 0j]]
    sigma0 = [[0.01 + 0.02j, 0j], [0j, 0.03 + 0.04j]]
    sigma1 = [[0.02 + 0.01j, 0j], [0j, 0.04 + 0.03j]]

    rows = [
        contract_header(contract_sha, symmetry=symmetry).rstrip(),
        "# iter channel component spin kpoint frequency_index frequency_Ha row column real_value imag_value",
    ]
    rows += _matrix_rows(0, "h0", h0)
    rows += _matrix_rows(0, "vxc_dft", vxc)
    rows += _matrix_rows(0, "occupation", occupations)
    rows += _matrix_rows(0, "wfc_spinor0", identity)
    rows += _matrix_rows(1, "sigma_c_iw", sigma0, ifrequency=0, frequency=0.1)
    rows += _matrix_rows(1, "sigma_c_iw", sigma1, ifrequency=1, frequency=0.2)
    rows += _matrix_rows(1, "exx", exx)
    rows += _matrix_rows(1, "vc", vc)
    rows += _matrix_rows(1, "raw_h", raw)
    rows += _matrix_rows(1, "mixed_h", raw)
    rows += _matrix_rows(1, "rotation_u", identity)
    rows += _matrix_rows(1, "occupation", occupations)
    rows += _matrix_rows(1, "wfc_spinor0", identity)
    return "\n".join(rows) + "\n"


def eigenvalue_trace(contract_sha: str, *, symmetry: str = "exx_on_gw_on_rpa_on") -> str:
    rows = [
        contract_header(contract_sha, symmetry=symmetry).rstrip(),
        "# iter channel spin kpoint kx ky kz band energy_eV",
        f"0 0 0 0 0 0 0 0 {-1.0 * HA2EV:.17g}",
        f"0 0 0 0 0 0 0 1 {1.0 * HA2EV:.17g}",
        f"1 0 0 0 0 0 0 0 {-0.79 * HA2EV:.17g}",
        f"1 0 0 0 0 0 0 1 {1.22 * HA2EV:.17g}",
    ]
    return "\n".join(rows) + "\n"


def iteration_trace(contract_sha: str, *, symmetry: str = "exx_on_gw_on_rpa_on") -> str:
    rows = [
        contract_header(contract_sha, symmetry=symmetry).rstrip(),
        "# iter max_delta_eV residual_l2_Ha residual_max_Ha efermi_eV gap_eV electron_count requested_mode applied_mode beta fallback rcond coefficient_l1 coefficient_count converged coefficients fallback_reason",
        "0 0 0 0 0 54.422772491976 2 0 0 0.2 0 1 0 0 0 none initial",
        "1 5.71439111165748 0.3 0.22 0 54.696886354714 2 0 0 0.2 0 1 0 0 0 none not_converged",
    ]
    return "\n".join(rows) + "\n"


def band_out() -> str:
    return (
        "1\n"
        "1\n"
        "2\n"
        "2\n"
        "0.0\n"
        "1 1\n"
        "1 2.0 -1.0 -27.211386245988\n"
        "2 0.0 1.0 27.211386245988\n"
    )


def write_sigc(path: Path, matrix: list[list[complex]]) -> None:
    nstates = len(matrix)
    values: list[float] = []
    for row in matrix:
        for value in row:
            values.extend((value.real, value.imag))
    path.write_bytes(
        struct.pack("=ii", nstates, 8)
        + struct.pack(f"={len(values)}d", *values)
    )


def write_fixture(
    root: Path,
    *,
    raw_shift: float = 0.0,
    exx_lower_shift: float = 0.0,
    symmetry: str = "exx_on_gw_on_rpa_on",
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "qsgw_input.contract"
    contract.write_text("fixture contract\n", encoding="utf-8", newline="\n")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    matrix = root / "qsgw_matrices.dat"
    eigenvalues = root / "qsgw_eigenvalues.dat"
    iterations = root / "qsgw_iterations.dat"
    bands = root / "band_out"
    matrix.write_text(
        matrix_trace(
            contract_sha,
            raw_shift=raw_shift,
            exx_lower_shift=exx_lower_shift,
            symmetry=symmetry,
        ),
        encoding="utf-8",
        newline="\n",
    )
    eigenvalues.write_text(
        eigenvalue_trace(contract_sha, symmetry=symmetry),
        encoding="utf-8",
        newline="\n",
    )
    iterations.write_text(
        iteration_trace(contract_sha, symmetry=symmetry),
        encoding="utf-8",
        newline="\n",
    )
    bands.write_text(band_out(), encoding="utf-8", newline="\n")
    g0w0 = root / "g0w0"
    g0w0.mkdir()
    write_sigc(
        g0w0 / "Sigc_fk_mn_kgrid_ispin_0_ik_0_ifreq_0.bin",
        [[0.01 + 0.02j, 0j], [0j, 0.03 + 0.04j]],
    )
    write_sigc(
        g0w0 / "Sigc_fk_mn_kgrid_ispin_0_ik_0_ifreq_1.bin",
        [[0.02 + 0.01j, 0j], [0j, 0.04 + 0.03j]],
    )
    return {
        "contract": contract,
        "matrix": matrix,
        "eigenvalues": eigenvalues,
        "iterations": iterations,
        "band_out": bands,
        "g0w0": g0w0,
    }
