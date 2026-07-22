#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SIGMA_MAX_ABS_HA = 1.0e-6
SIGMA_RELATIVE_FROBENIUS = 1.0e-8
FREQUENCY_MAX_ABS_HA = 1.0e-12


def classify(sigcrf: dict, components: dict) -> dict:
    if not sigcrf.get("diagnostic_complete"):
        raise ValueError("SigcRF comparison is incomplete")
    if not components.get("diagnostic_complete"):
        raise ValueError("component comparison is incomplete")

    sigcrf_overall = sigcrf["overall"]
    sigma_overall = components["sigma_c_iw"]["overall"]
    frequency_max_abs = float(components["frequency_grid_max_abs_ha"])

    sigcrf_passed = bool(sigcrf.get("numerical_parity_passed"))
    sigma_passed = (
        float(sigma_overall["max_abs_ha"]) <= SIGMA_MAX_ABS_HA
        and float(sigma_overall["relative_frobenius"])
        <= SIGMA_RELATIVE_FROBENIUS
        and frequency_max_abs <= FREQUENCY_MAX_ABS_HA
    )

    if sigcrf_passed and sigma_passed:
        inference = "legacy_wr_route_recovers_exact847_real_space_and_projected_sigma"
    elif sigcrf_passed:
        inference = "legacy_wr_route_recovers_real_space_sigma_but_not_projection"
    elif sigma_passed:
        inference = "projected_sigma_matches_despite_real_space_sigma_difference"
    else:
        inference = "legacy_wr_route_does_not_explain_exact847_sigma_difference"

    static = components.get("static_components", {})
    return {
        "schema": "librpa-exact847-legacy-wr-diagnostic-classification-v1",
        "diagnostic_complete": True,
        "acceptance": "false_diagnostic_only",
        "inference": inference,
        "legacy_wr_causal_hypothesis_passed": sigcrf_passed and sigma_passed,
        "sigcrf": {
            "parity_passed": sigcrf_passed,
            "max_abs_ha": float(sigcrf_overall["max_abs_ha"]),
            "relative_frobenius": float(sigcrf_overall["relative_frobenius"]),
        },
        "sigma_c_iw": {
            "parity_passed": sigma_passed,
            "max_abs_ha": float(sigma_overall["max_abs_ha"]),
            "relative_frobenius": float(sigma_overall["relative_frobenius"]),
            "frequency_grid_max_abs_ha": frequency_max_abs,
        },
        "remaining_static_differences": {
            name: values.get("upper_triangle_hermitized", values.get("raw", {}))
            for name, values in static.items()
        },
        "thresholds": {
            "sigma_max_abs_ha": SIGMA_MAX_ABS_HA,
            "sigma_relative_frobenius": SIGMA_RELATIVE_FROBENIUS,
            "frequency_max_abs_ha": FREQUENCY_MAX_ABS_HA,
        },
        "claim_boundary": (
            "This diagnostic isolates the upstream W(q)-to-W(R) route. "
            "It does not accept Gate A1 or authorize a shared GW source change."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sigcrf_json", type=Path)
    parser.add_argument("component_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    sigcrf = json.loads(args.sigcrf_json.read_text(encoding="utf-8"))
    components = json.loads(args.component_json.read_text(encoding="utf-8"))
    result = classify(sigcrf, components)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
