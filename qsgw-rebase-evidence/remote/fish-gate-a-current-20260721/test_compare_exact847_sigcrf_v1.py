import importlib.util
import struct
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_exact847_sigcrf_v1.py")
SPEC = importlib.util.spec_from_file_location("sigcrf", MODULE_PATH)
sigcrf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sigcrf)


def write_sigcrf(directory, values):
    directory.mkdir()
    path = directory / "SigcRF_ispin_00_s_00_iomega_000_myid_00000.dat"
    payload = bytearray(struct.pack("<Q", len(values)))
    for r_index, value in enumerate(values):
        payload.extend(struct.pack("<5Q", r_index, 0, 0, 1, 1))
        payload.extend(np.asarray([value], dtype="<c16").tobytes())
    path.write_bytes(payload)


def test_identical_sigcrf_passes(tmp_path):
    reference_dir = tmp_path / "reference"
    observed_dir = tmp_path / "observed"
    write_sigcrf(reference_dir, [1.0 + 2.0j, -3.0 + 0.5j])
    write_sigcrf(observed_dir, [1.0 + 2.0j, -3.0 + 0.5j])
    reference, reference_files = sigcrf.read_directory(reference_dir)
    observed, observed_files = sigcrf.read_directory(observed_dir)
    report = sigcrf.analyze(
        reference, observed, reference_files, observed_files, 1.0e-6, 1.0e-8
    )
    assert report["numerical_parity_passed"] is True
    assert report["overall"]["max_abs_ha"] == 0.0
    assert report["frequency_count"] == 1


def test_sigcrf_difference_is_localized(tmp_path):
    reference_dir = tmp_path / "reference"
    observed_dir = tmp_path / "observed"
    write_sigcrf(reference_dir, [1.0 + 2.0j, -3.0 + 0.5j])
    write_sigcrf(observed_dir, [1.0 + 2.0j, -2.75 + 0.5j])
    reference, reference_files = sigcrf.read_directory(reference_dir)
    observed, observed_files = sigcrf.read_directory(observed_dir)
    report = sigcrf.analyze(
        reference, observed, reference_files, observed_files, 1.0e-6, 1.0e-8
    )
    assert report["numerical_parity_passed"] is False
    assert report["overall"]["max_abs_ha"] == 0.25
    assert report["overall"]["maximum_difference_key"] == [0, 0, 0, 1, 0, 0]


def test_rejects_block_key_mismatch(tmp_path):
    reference_dir = tmp_path / "reference"
    observed_dir = tmp_path / "observed"
    write_sigcrf(reference_dir, [1.0 + 0.0j])
    write_sigcrf(observed_dir, [1.0 + 0.0j, 2.0 + 0.0j])
    reference, _ = sigcrf.read_directory(reference_dir)
    observed, _ = sigcrf.read_directory(observed_dir)
    try:
        sigcrf.metrics(reference, observed)
    except sigcrf.ComparisonError as error:
        assert "block-key mismatch" in str(error)
    else:
        raise AssertionError("block mismatch was accepted")
