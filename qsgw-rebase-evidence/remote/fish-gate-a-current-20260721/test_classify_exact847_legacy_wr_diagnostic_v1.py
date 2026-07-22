import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("classify_exact847_legacy_wr_diagnostic_v1.py")
SPEC = importlib.util.spec_from_file_location("legacy_wr_classifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payloads(sigcrf_passed=True, sigma_max=1.0e-10, sigma_rel=1.0e-11):
    sigcrf = {
        "diagnostic_complete": True,
        "numerical_parity_passed": sigcrf_passed,
        "overall": {"max_abs_ha": 1.0e-10, "relative_frobenius": 1.0e-11},
    }
    components = {
        "diagnostic_complete": True,
        "frequency_grid_max_abs_ha": 0.0,
        "sigma_c_iw": {
            "overall": {
                "max_abs_ha": sigma_max,
                "relative_frobenius": sigma_rel,
            }
        },
        "static_components": {
            "exx": {
                "upper_triangle_hermitized": {
                    "max_abs_ha": 1.2e-4,
                    "relative_frobenius": 1.3e-5,
                }
            }
        },
    }
    return sigcrf, components


def test_classifies_causal_recovery_without_requiring_exx_parity():
    sigcrf, components = payloads()
    result = MODULE.classify(sigcrf, components)
    assert result["legacy_wr_causal_hypothesis_passed"] is True
    assert result["inference"] == (
        "legacy_wr_route_recovers_exact847_real_space_and_projected_sigma"
    )
    assert result["remaining_static_differences"]["exx"]["max_abs_ha"] == 1.2e-4


def test_rejects_causal_recovery_when_both_sigma_levels_differ():
    sigcrf, components = payloads(False, sigma_max=1.0, sigma_rel=0.5)
    result = MODULE.classify(sigcrf, components)
    assert result["legacy_wr_causal_hypothesis_passed"] is False
    assert result["inference"] == (
        "legacy_wr_route_does_not_explain_exact847_sigma_difference"
    )


def test_requires_complete_comparisons():
    sigcrf, components = payloads()
    components["diagnostic_complete"] = False
    try:
        MODULE.classify(sigcrf, components)
    except ValueError as error:
        assert "component comparison is incomplete" in str(error)
    else:
        raise AssertionError("incomplete component comparison was accepted")
