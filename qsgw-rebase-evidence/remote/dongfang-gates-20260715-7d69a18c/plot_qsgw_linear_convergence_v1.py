#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def load_rows(text: str) -> list[dict[str, float]]:
    reader = csv.DictReader(text.splitlines())
    required = {
        "iteration", "gap_ev", "residual_l2_ha", "old_residual_l2_ha",
        "reported_residual_l2_ha", "beta",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("convergence CSV lacks required columns")
    rows = []
    for source in reader:
        row = {key: float(source[key]) for key in required}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("convergence CSV contains non-finite values")
        rows.append(row)
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(len(rows))) or len(rows) < 3:
        raise ValueError("convergence CSV must contain continuous iterations 0..N")
    if any(row["residual_l2_ha"] <= 0.0 or row["old_residual_l2_ha"] <= 0.0
           for row in rows[1:]):
        raise ValueError("positive-iteration residuals must be positive")
    beta = rows[0]["beta"]
    if beta <= 0.0 or beta > 1.0 or any(
        not math.isclose(row["beta"], beta, abs_tol=1.0e-15)
        for row in rows
    ):
        raise ValueError("convergence CSV has inconsistent beta")
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    residuals = [row["residual_l2_ha"] for row in rows[1:]]
    old_residuals = [row["old_residual_l2_ha"] for row in rows[1:]]
    ratios = [
        residuals[index] / residuals[index - 1]
        for index in range(1, len(residuals))
    ]
    geometric_ratio = (
        math.prod(ratios) ** (1.0 / len(ratios)) if ratios else 0.0
    )
    return {
        "passed": True,
        "iteration_count": len(rows) - 1,
        "beta": rows[0]["beta"],
        "initial_residual_l2_ha": residuals[0],
        "final_residual_l2_ha": residuals[-1],
        "residual_ratios": ratios,
        "geometric_mean_residual_ratio": geometric_ratio,
        "residual_monotonic_decrease": all(
            residuals[index] < residuals[index - 1]
            for index in range(1, len(residuals))
        ),
        "max_old_current_residual_abs_diff_ha": max(
            abs(old - current)
            for old, current in zip(old_residuals, residuals)
        ),
        "initial_gap_ev": rows[0]["gap_ev"],
        "final_gap_ev": rows[-1]["gap_ev"],
    }


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def render_svg(rows: list[dict[str, float]], summary: dict[str, object]) -> str:
    width, height = 1000, 700
    left, right = 90.0, 955.0
    residual_top, residual_bottom = 90.0, 390.0
    gap_top, gap_bottom = 475.0, 640.0
    maximum_iteration = int(rows[-1]["iteration"])

    def x_position(iteration: float) -> float:
        return left + (right - left) * iteration / maximum_iteration

    positive = rows[1:]
    residual_values = [
        value
        for row in positive
        for value in (row["residual_l2_ha"], row["old_residual_l2_ha"])
    ]
    log_min = math.floor(math.log10(min(residual_values)))
    log_max = math.ceil(math.log10(max(residual_values)))
    if log_max == log_min:
        log_max += 1

    def residual_y(value: float) -> float:
        fraction = (math.log10(value) - log_min) / (log_max - log_min)
        return residual_bottom - fraction * (residual_bottom - residual_top)

    gap_values = [row["gap_ev"] for row in rows]
    gap_min, gap_max = min(gap_values), max(gap_values)
    padding = max(0.05 * (gap_max - gap_min), 0.05)
    gap_min -= padding
    gap_max += padding

    def gap_y(value: float) -> float:
        fraction = (value - gap_min) / (gap_max - gap_min)
        return gap_bottom - fraction * (gap_bottom - gap_top)

    old_points = [
        (x_position(row["iteration"]), residual_y(row["old_residual_l2_ha"]))
        for row in positive
    ]
    current_points = [
        (x_position(row["iteration"]), residual_y(row["residual_l2_ha"]))
        for row in positive
    ]
    gap_points = [
        (x_position(row["iteration"]), gap_y(row["gap_ev"]))
        for row in rows
    ]
    beta = float(summary["beta"])
    ratio = float(summary["geometric_mean_residual_ratio"])
    title = html.escape(f"Si k444 QSGW linear mixing: beta={beta:g}")
    subtitle = html.escape(
        f"old/new miniter{maximum_iteration}; geometric residual ratio={ratio:.4f}"
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#171717;letter-spacing:0} .axis{stroke:#333;stroke-width:1.2} .grid{stroke:#d8d8d8;stroke-width:1} .label{font-size:14px} .small{font-size:12px}</style>',
        f'<text x="{left:.0f}" y="35" font-size="24" font-weight="700">{title}</text>',
        f'<text x="{left:.0f}" y="60" font-size="14" fill="#555">{subtitle}</text>',
    ]
    for exponent in range(log_min, log_max + 1):
        y = residual_y(10.0 ** exponent)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        elements.append(f'<text class="small" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">10^{exponent}</text>')
    elements.extend((
        f'<line class="axis" x1="{left}" y1="{residual_top}" x2="{left}" y2="{residual_bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{residual_bottom}" x2="{right}" y2="{residual_bottom}"/>',
        f'<text class="label" x="25" y="{(residual_top + residual_bottom) / 2}" transform="rotate(-90 25 {(residual_top + residual_bottom) / 2})" text-anchor="middle">Hamiltonian residual L2 (Ha)</text>',
        f'<polyline points="{polyline(old_points)}" fill="none" stroke="#9a9a9a" stroke-width="5" stroke-dasharray="8 5"/>',
        f'<polyline points="{polyline(current_points)}" fill="none" stroke="#005a9c" stroke-width="2.5"/>',
    ))
    for x, y in current_points:
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#005a9c"/>')
    elements.extend((
        f'<line x1="{right - 205}" y1="{residual_top + 20}" x2="{right - 160}" y2="{residual_top + 20}" stroke="#9a9a9a" stroke-width="5" stroke-dasharray="8 5"/>',
        f'<text class="small" x="{right - 150}" y="{residual_top + 24}">legacy</text>',
        f'<line x1="{right - 205}" y1="{residual_top + 43}" x2="{right - 160}" y2="{residual_top + 43}" stroke="#005a9c" stroke-width="2.5"/>',
        f'<text class="small" x="{right - 150}" y="{residual_top + 47}">current</text>',
    ))

    for index in range(5):
        value = gap_min + index * (gap_max - gap_min) / 4.0
        y = gap_y(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        elements.append(f'<text class="small" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.2f}</text>')
    elements.extend((
        f'<line class="axis" x1="{left}" y1="{gap_top}" x2="{left}" y2="{gap_bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{gap_bottom}" x2="{right}" y2="{gap_bottom}"/>',
        f'<text class="label" x="25" y="{(gap_top + gap_bottom) / 2}" transform="rotate(-90 25 {(gap_top + gap_bottom) / 2})" text-anchor="middle">Gap (eV)</text>',
        f'<polyline points="{polyline(gap_points)}" fill="none" stroke="#c14924" stroke-width="2.5"/>',
    ))
    for x, y in gap_points:
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#c14924"/>')
    for iteration in range(maximum_iteration + 1):
        x = x_position(iteration)
        elements.append(f'<line class="axis" x1="{x:.2f}" y1="{gap_bottom}" x2="{x:.2f}" y2="{gap_bottom + 6}"/>')
        elements.append(f'<text class="small" x="{x:.2f}" y="{gap_bottom + 23}" text-anchor="middle">{iteration}</text>')
    elements.append(f'<text class="label" x="{(left + right) / 2}" y="685" text-anchor="middle">QSGW iteration</text>')
    elements.append('</svg>')
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    try:
        rows = load_rows(args.input_csv.read_text(encoding="utf-8"))
        report = summarize(rows)
        args.output_svg.write_text(render_svg(rows, report), encoding="utf-8")
    except Exception as error:
        report = {"passed": False, "error": str(error)}
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
