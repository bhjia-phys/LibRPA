#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    n_params: int
    ridge_lambda: float = 0.0
    den_weight: float = 1.0
    den_floor: float = 1.0e-12
    guard_den_cut: float = 0.0
    avg_n_min: int = 0
    avg_n_max: int = 0
    avg_den_cut: float = 1.0e-4
    avg_trim: float = 0.2
    avg_max_abs: float = 1.0e6


def parse_methods(text: str) -> list[Method]:
    methods: list[Method] = []
    for item in text.split(","):
        parts = item.split(":")
        if parts[0] == "thiele":
            n_params = int(parts[1]) if len(parts) > 1 else 32
            methods.append(Method(f"thiele{n_params}", "thiele", n_params))
        elif parts[0] == "ridge":
            n_params = int(parts[1])
            lam = float(parts[2])
            den_weight = float(parts[3]) if len(parts) > 3 else 1.0
            methods.append(Method(f"ridge{n_params}_l{lam:g}_dw{den_weight:g}", "ridge", n_params, lam, den_weight))
        elif parts[0] == "guard":
            n_params = int(parts[1])
            lam = float(parts[2])
            den_weight = float(parts[3]) if len(parts) > 3 else 1.0
            den_cut = float(parts[4]) if len(parts) > 4 else 1.0e-3
            methods.append(Method(
                f"guard{n_params}_l{lam:g}_dw{den_weight:g}_dc{den_cut:g}",
                "guard", n_params, lam, den_weight, guard_den_cut=den_cut))
        elif parts[0] == "avg":
            n_min = int(parts[1])
            n_max = int(parts[2])
            den_cut = float(parts[3]) if len(parts) > 3 else 1.0e-4
            trim = float(parts[4]) if len(parts) > 4 else 0.2
            max_abs = float(parts[5]) if len(parts) > 5 else 1.0e6
            methods.append(Method(
                f"avg{n_min}_{n_max}_dc{den_cut:g}_tr{trim:g}",
                "avg", n_max,
                avg_n_min=n_min,
                avg_n_max=n_max,
                avg_den_cut=den_cut,
                avg_trim=trim,
                avg_max_abs=max_abs))
        else:
            raise ValueError(f"unknown method spec: {item}")
    return methods


def read_ac_input(path: Path) -> tuple[np.ndarray, dict[tuple[int, int, int, int, int], np.ndarray]]:
    data = np.loadtxt(path, comments="#")
    n_omega = int(np.max(data[:, 5])) + 1
    omegas = np.zeros(n_omega, dtype=float)
    out: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    for row in data:
        key = tuple(int(row[i]) for i in range(5))
        iw = int(row[5])
        omegas[iw] = row[6]
        if key not in out:
            out[key] = np.zeros(n_omega, dtype=np.complex128)
        out[key][iw] = row[7] + 1j * row[8]
    return omegas, out


def read_ac_output(path: Path) -> dict[tuple[int, int, int, int, int, int], tuple[float, complex, complex]]:
    data = np.loadtxt(path, comments="#")
    out: dict[tuple[int, int, int, int, int, int], tuple[float, complex, complex]] = {}
    for row in data:
        key = tuple(int(row[i]) for i in range(6))
        out[key] = (float(row[6]), row[7] + 1j * row[8], row[10] + 1j * row[11])
    return out


def thiele_fit_indices(xs: np.ndarray, ys: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = xs[indices].copy()
    py0 = ys[indices].copy()
    n_params = len(indices)
    g = np.zeros((n_params, n_params), dtype=np.complex128)
    g[:, 0] = py0
    for ipar in range(1, n_params):
        for i in range(ipar, n_params):
            g[i, ipar] = (g[ipar - 1, ipar - 1] - g[i, ipar - 1]) / (
                (px[i] - px[ipar - 1]) * g[i, ipar - 1]
            )
    return px, np.diag(g).copy()


def thiele_fit(xs: np.ndarray, ys: np.ndarray, n_params: int) -> tuple[np.ndarray, np.ndarray]:
    n_data = len(ys)
    n_params = min(max(n_params, 1), n_data)
    if n_data <= n_params:
        indices = np.arange(n_data, dtype=int)
    elif n_params == 1:
        indices = np.arange(1, dtype=int)
    else:
        step = n_data // (n_params - 1)
        indices = np.asarray([i * step for i in range(n_params - 1)] + [n_data - 1], dtype=int)
    return thiele_fit_indices(xs, ys, indices)


def thiele_eval(fit: tuple[np.ndarray, np.ndarray], x: complex) -> tuple[complex, dict[str, float]]:
    px, py = fit
    tmp = 1.0 + 0.0j
    for ipar in range(len(py) - 1, 0, -1):
        tmp = 1.0 + py[ipar] * (x - px[ipar - 1]) / tmp
    return py[0] / tmp, {"coeff_norm": float(np.linalg.norm(py)), "den_abs": abs(tmp)}


def averaged_thiele_fits(xs: np.ndarray, ys: np.ndarray, method: Method) -> list[tuple[np.ndarray, np.ndarray]]:
    n_data = len(ys)
    n_min = min(max(method.avg_n_min, 2), n_data)
    n_max = min(max(method.avg_n_max, n_min), n_data)
    seen: set[tuple[int, ...]] = set()
    fits: list[tuple[np.ndarray, np.ndarray]] = []

    def add_indices(indices: np.ndarray) -> None:
        if len(indices) < 2:
            return
        key = tuple(int(i) for i in indices)
        if key in seen:
            return
        seen.add(key)
        try:
            fits.append(thiele_fit_indices(xs, ys, np.asarray(indices, dtype=int)))
        except FloatingPointError:
            return

    for n_params in range(n_min, n_max + 1):
        # LibRPA-like full-range subset.
        add_indices(np.rint(np.linspace(0, n_data - 1, n_params)).astype(int))

        # Low-frequency contiguous windows. These are deliberately deterministic
        # so that pairwise replay is reproducible.
        max_offset = min(4, n_data - n_params)
        for offset in range(max_offset + 1):
            add_indices(np.arange(offset, offset + n_params, dtype=int))

        # A middle-frequency full-range variant catches fits that are sensitive
        # to the very first Matsubara point without ignoring the high tail.
        if n_data - n_params > 4:
            add_indices(np.rint(np.linspace(1, n_data - 1, n_params)).astype(int))
            add_indices(np.rint(np.linspace(0, n_data - 2, n_params)).astype(int))

    return fits


def complex_component_median(values: np.ndarray) -> complex:
    return complex(float(np.median(values.real)), float(np.median(values.imag)))


def averaged_thiele_eval(fits: list[tuple[np.ndarray, np.ndarray]], x: complex, method: Method) -> tuple[complex, dict[str, float]]:
    values: list[complex] = []
    coeff_norms: list[float] = []
    den_values: list[float] = []
    finite_count = 0
    for fit in fits:
        value, diag = thiele_eval(fit, x)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            continue
        finite_count += 1
        den_abs = diag.get("den_abs", 0.0)
        if den_abs < method.avg_den_cut:
            continue
        if abs(value) > method.avg_max_abs:
            continue
        values.append(value)
        coeff_norms.append(diag.get("coeff_norm", math.nan))
        den_values.append(den_abs)

    if not values:
        return math.nan + 1j * math.nan, {
            "coeff_norm": math.nan,
            "den_abs": math.nan,
            "avg_kept": 0.0,
            "avg_total": float(len(fits)),
            "avg_finite": float(finite_count),
        }

    arr = np.asarray(values, dtype=np.complex128)
    if len(arr) >= 4 and method.avg_trim > 0.0:
        center = complex_component_median(arr)
        dist = np.abs(arr - center)
        keep_quantile = min(max(1.0 - method.avg_trim, 0.5), 1.0)
        cutoff = float(np.quantile(dist, keep_quantile))
        arr = arr[dist <= cutoff]

    value = complex_component_median(arr)
    return value, {
        "coeff_norm": quantile([x for x in coeff_norms if math.isfinite(x)], 0.5),
        "den_abs": min(den_values) if den_values else math.nan,
        "avg_kept": float(len(values)),
        "avg_total": float(len(fits)),
        "avg_finite": float(finite_count),
    }


def ridge_fit(xs: np.ndarray, ys: np.ndarray, method: Method) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_data = len(ys)
    n_unknown = min(max(method.n_params, 1), n_data)
    n_num = (n_unknown + 1) // 2
    n_den = n_unknown - n_num
    x_scale = max(float(np.max(np.abs(xs))), 1.0e-30)
    y_scale = max(float(np.max(np.abs(ys))), 1.0e-30)
    u = xs / x_scale
    y = ys / y_scale
    a = np.zeros((n_data, n_unknown), dtype=np.complex128)
    for i in range(n_data):
        powers = np.ones(max(n_num, n_den + 1), dtype=np.complex128)
        for p in range(1, len(powers)):
            powers[p] = powers[p - 1] * u[i]
        a[i, :n_num] = powers[:n_num]
        for j in range(n_den):
            a[i, n_num + j] = -y[i] * powers[j + 1]
    normal = a.conj().T @ a
    rhs = a.conj().T @ y
    lam = max(method.ridge_lambda, 0.0)
    for j in range(1, n_num):
        normal[j, j] += lam * j * j
    for j in range(n_den):
        power = j + 1
        normal[n_num + j, n_num + j] += lam * max(method.den_weight, 0.0) * power * power
    try:
        sol = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.solve(normal + np.eye(n_unknown) * 1.0e-12, rhs)
    num = sol[:n_num]
    den = np.concatenate((np.array([1.0 + 0.0j]), sol[n_num:]))
    return num, den, x_scale, y_scale


def poly_eval(coeff: np.ndarray, x: complex) -> complex:
    value = 0.0 + 0.0j
    for c in coeff[::-1]:
        value = value * x + c
    return value


def finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def ridge_eval(fit: tuple[np.ndarray, np.ndarray, float, float], x: complex, method: Method) -> tuple[complex, dict[str, float]]:
    num, den, x_scale, y_scale = fit
    u = x / x_scale
    p = poly_eval(num, u)
    q = poly_eval(den, u)
    q_abs = abs(q)
    floor_hit = 0.0
    if method.den_floor > 0.0 and q_abs < method.den_floor:
        floor_hit = 1.0
        q = method.den_floor + 0.0j if q_abs == 0.0 else q * (method.den_floor / q_abs)
    coeff_norm = float(np.linalg.norm(num) + np.linalg.norm(den[1:]))
    return y_scale * p / q, {"coeff_norm": coeff_norm, "den_abs": q_abs, "floor_hit": floor_hit}


def fit_one(xs: np.ndarray, ys: np.ndarray, method: Method):
    if method.kind == "thiele":
        return thiele_fit(xs, ys, method.n_params)
    if method.kind == "avg":
        return averaged_thiele_fits(xs, ys, method)
    if method.kind == "guard":
        return thiele_fit(xs, ys, method.n_params), ridge_fit(xs, ys, method)
    return ridge_fit(xs, ys, method)


def eval_one(fit, x: complex, method: Method) -> tuple[complex, dict[str, float]]:
    if method.kind == "thiele":
        return thiele_eval(fit, x)
    if method.kind == "avg":
        return averaged_thiele_eval(fit, x, method)
    if method.kind == "guard":
        thiele_value, thiele_diag = thiele_eval(fit[0], x)
        den_abs = thiele_diag.get("den_abs", math.inf)
        if math.isfinite(thiele_value.real) and math.isfinite(thiele_value.imag) and den_abs >= method.guard_den_cut:
            thiele_diag["guard_used_ridge"] = 0.0
            return thiele_value, thiele_diag
        ridge_value, ridge_diag = ridge_eval(fit[1], x, method)
        ridge_diag["guard_used_ridge"] = 1.0
        return ridge_value, ridge_diag
    return ridge_eval(fit, x, method)


def quantile(values: list[float], q: float) -> float:
    finite_values = [x for x in values if math.isfinite(x)]
    if not finite_values:
        return math.nan
    return float(np.quantile(np.asarray(finite_values, dtype=float), q))


def summarize_method(
    run_root: Path,
    left: str,
    right: str,
    iteration: int,
    method: Method,
) -> dict[str, str]:
    pattern = f"iter{iteration:03d}_spin00_k0000.dat"
    omegas_l, input_l = read_ac_input(run_root / left / f"sigc_ac_input_{pattern}")
    omegas_r, input_r = read_ac_input(run_root / right / f"sigc_ac_input_{pattern}")
    xs_l = 1j * omegas_l
    xs_r = 1j * omegas_r
    out_l = read_ac_output(run_root / left / f"sigc_ac_output_{pattern}")
    out_r = read_ac_output(run_root / right / f"sigc_ac_output_{pattern}")

    qp_diffs: list[float] = []
    zero_diffs: list[float] = []
    shifts_qp: list[float] = []
    shifts_zero: list[float] = []
    coeff_norms: list[float] = []
    den_abs: list[float] = []
    avg_kept: list[float] = []
    avg_finite: list[float] = []
    avg_total: list[float] = []
    floor_hits = 0
    guard_ridge_uses = 0
    eval_failures = 0
    max_qp = (-1.0, None, 0.0)
    max_zero = (-1.0, None, 0.0)

    common = sorted(set(out_l) & set(out_r))
    fits_l = {}
    fits_r = {}
    for key6 in common:
        key5 = key6[:5]
        if key5 not in fits_l:
            fits_l[key5] = fit_one(xs_l, input_l[key5], method)
            fits_r[key5] = fit_one(xs_r, input_r[key5], method)
        energy = out_l[key6][0]
        yl_qp, diag_l_qp = eval_one(fits_l[key5], energy + 0.0j, method)
        yr_qp, diag_r_qp = eval_one(fits_r[key5], energy + 0.0j, method)
        yl_0, diag_l_0 = eval_one(fits_l[key5], 0.0 + 0.0j, method)
        yr_0, diag_r_0 = eval_one(fits_r[key5], 0.0 + 0.0j, method)
        if finite_complex(yl_qp) and finite_complex(yr_qp):
            dq = abs(yl_qp - yr_qp)
            shifts_qp.extend([float(abs(yl_qp - out_l[key6][1])), float(abs(yr_qp - out_r[key6][1]))])
        else:
            dq = math.nan
            eval_failures += 1
        if finite_complex(yl_0) and finite_complex(yr_0):
            dz = abs(yl_0 - yr_0)
            shifts_zero.extend([float(abs(yl_0 - out_l[key6][2])), float(abs(yr_0 - out_r[key6][2]))])
        else:
            dz = math.nan
            eval_failures += 1
        qp_diffs.append(float(dq))
        zero_diffs.append(float(dz))
        for diag in [diag_l_qp, diag_r_qp, diag_l_0, diag_r_0]:
            coeff_norms.append(diag.get("coeff_norm", math.nan))
            den_abs.append(diag.get("den_abs", math.nan))
            floor_hits += int(diag.get("floor_hit", 0.0))
            guard_ridge_uses += int(diag.get("guard_used_ridge", 0.0))
            if "avg_kept" in diag:
                avg_kept.append(diag["avg_kept"])
                avg_finite.append(diag.get("avg_finite", math.nan))
                avg_total.append(diag.get("avg_total", math.nan))
        pre = float(np.max(np.abs(input_l[key5] - input_r[key5])))
        if math.isfinite(dq) and dq > max_qp[0]:
            max_qp = (float(dq), key6, pre)
        if math.isfinite(dz) and dz > max_zero[0]:
            max_zero = (float(dz), key6, pre)

    def fmt(value: float) -> str:
        if value is None or not math.isfinite(value):
            return "nan"
        return f"{value:.10e}"

    return {
        "run_root": str(run_root),
        "pair": f"{left}_vs_{right}",
        "iteration": str(iteration),
        "method": method.name,
        "count": str(len(common)),
        "qp_max": fmt(max_qp[0]),
        "qp_p99": fmt(quantile(qp_diffs, 0.99)),
        "qp_p95": fmt(quantile(qp_diffs, 0.95)),
        "qp_key": ",".join(str(x) for x in max_qp[1]) if max_qp[1] else "",
        "qp_same_elem_pre_max": fmt(max_qp[2]),
        "qp_amp": fmt(max_qp[0] / max_qp[2] if max_qp[2] else math.nan),
        "zero_max": fmt(max_zero[0]),
        "zero_p99": fmt(quantile(zero_diffs, 0.99)),
        "zero_key": ",".join(str(x) for x in max_zero[1]) if max_zero[1] else "",
        "zero_same_elem_pre_max": fmt(max_zero[2]),
        "zero_amp": fmt(max_zero[0] / max_zero[2] if max_zero[2] else math.nan),
        "max_shift_from_dump_qp": fmt(max(shifts_qp) if shifts_qp else math.nan),
        "p99_shift_from_dump_qp": fmt(quantile(shifts_qp, 0.99)),
        "p95_shift_from_dump_qp": fmt(quantile(shifts_qp, 0.95)),
        "max_shift_from_dump_zero": fmt(max(shifts_zero) if shifts_zero else math.nan),
        "p99_shift_from_dump_zero": fmt(quantile(shifts_zero, 0.99)),
        "coeff_norm_p99": fmt(quantile([x for x in coeff_norms if math.isfinite(x)], 0.99)),
        "min_den_abs": fmt(min([x for x in den_abs if math.isfinite(x)], default=math.nan)),
        "floor_hits": str(floor_hits),
        "guard_ridge_uses": str(guard_ridge_uses),
        "eval_failures": str(eval_failures),
        "avg_kept_min": fmt(min([x for x in avg_kept if math.isfinite(x)], default=math.nan)),
        "avg_finite_min": fmt(min([x for x in avg_finite if math.isfinite(x)], default=math.nan)),
        "avg_total_max": fmt(max([x for x in avg_total if math.isfinite(x)], default=math.nan)),
    }


def write_outputs(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    tsv = out_dir / "ac_regularization_compare.tsv"
    with tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    md = out_dir / "ac_regularization_compare.md"
    lines = [
        "# AC regularization comparison",
        "",
        "| pair | iter | method | QP max | QP p99 | QP amp | zero max | zero p99 | dump shift QP max/p99 | coeff p99 | min |Q| | failures | avg kept | floor | guard uses |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['pair']}` | {row['iteration']} | `{row['method']}` | "
            f"`{row['qp_max']}` | `{row['qp_p99']}` | `{row['qp_amp']}` | "
            f"`{row['zero_max']}` | `{row['zero_p99']}` | `{row['max_shift_from_dump_qp']}` / `{row['p99_shift_from_dump_qp']}` | "
            f"`{row['coeff_norm_p99']}` | `{row['min_den_abs']}` | `{row['eval_failures']}` | "
            f"`{row['avg_kept_min']}` / `{row['avg_total_max']}` | `{row['floor_hits']}` | `{row['guard_ridge_uses']}` |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--pairs", required=True, help="comma list like t32_a:t32_b,t1_a:t32_a")
    ap.add_argument("--iterations", default="1")
    ap.add_argument("--methods", default="thiele:32,ridge:14:1e-8:10,ridge:16:1e-8:10,ridge:16:1e-6:10")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    methods = parse_methods(args.methods)
    iterations = [int(x) for x in args.iterations.split(",") if x]
    pairs = [tuple(x.split(":", 1)) for x in args.pairs.split(",") if x]
    rows: list[dict[str, str]] = []
    for left, right in pairs:
        for iteration in iterations:
            for method in methods:
                rows.append(summarize_method(args.run_root, left, right, iteration, method))
    write_outputs(rows, args.out_dir)
    print(args.out_dir / "ac_regularization_compare.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
