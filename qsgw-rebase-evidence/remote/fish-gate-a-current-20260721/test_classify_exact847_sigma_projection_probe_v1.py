import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "classify_exact847_sigma_projection_probe_v1.py"
)
SPEC = importlib.util.spec_from_file_location("projection_probe", MODULE_PATH)
projection_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(projection_probe)


def component_report(max_abs_ha=2.0e-7, relative_frobenius=3.0e-9):
    return {
        "diagnostic_complete": True,
        "frequency_grid_max_abs_ha": 0.0,
        "sigma_c_iw": {
            "overall": {
                "max_abs_ha": max_abs_ha,
                "relative_frobenius": relative_frobenius,
            }
        },
    }


def test_classifies_projection_parity():
    report = projection_probe.classify(component_report(), 1.0e-6, 1.0e-8)
    assert report["sigma_projection_parity_passed"] is True
    assert report["inference"] == (
        "candidate_projection_matches_given_legacy_sigcrf"
    )


def test_classifies_projection_or_reader_difference():
    report = projection_probe.classify(
        component_report(max_abs_ha=2.0e-4, relative_frobenius=4.0e-5),
        1.0e-6,
        1.0e-8,
    )
    assert report["sigma_projection_parity_passed"] is False
    assert report["inference"] == (
        "candidate_projection_or_legacy_sigcrf_reader_differs"
    )


def test_rejects_incomplete_component_report():
    try:
        projection_probe.classify(
            {"diagnostic_complete": False}, 1.0e-6, 1.0e-8
        )
    except projection_probe.ClassificationError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete report was accepted")
