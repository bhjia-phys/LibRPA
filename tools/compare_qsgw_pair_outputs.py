#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def keyed_values(path: Path, kind: str) -> dict[tuple[int, ...], complex]:
    data = read_table(path)
    out: dict[tuple[int, ...], complex] = {}
    if kind == "eigenvalues":
        for row in np.atleast_2d(data):
            key = tuple(int(row[i]) for i in range(4))
            out[key] = complex(float(row[4]), 0.0)
    elif kind in {"h0_gw", "vc_gw"}:
        for row in np.atleast_2d(data):
            key = tuple(int(row[i]) for i in range(5))
            out[key] = complex(float(row[5]), float(row[6]))
    else:
        raise ValueError(f"unknown kind: {kind}")
    return out


def summarize_pair(left_path: Path, right_path: Path, kind: str) -> dict[str, str]:
    left = keyed_values(left_path, kind)
    right = keyed_values(right_path, kind)
    common = sorted(set(left) & set(right))
    diffs = np.asarray([abs(left[k] - right[k]) for k in common], dtype=float)
    values = np.asarray([max(abs(left[k]), abs(right[k])) for k in common], dtype=float)
    if len(diffs) == 0:
        return {
            "kind": kind,
            "count": "0",
            "max_abs": "nan",
            "rms_abs": "nan",
            "p99_abs": "nan",
            "max_rel": "nan",
            "max_key": "",
        }
    max_idx = int(np.argmax(diffs))
    rel = diffs / np.maximum(values, 1.0e-30)
    return {
        "kind": kind,
        "count": str(len(common)),
        "max_abs": f"{float(np.max(diffs)):.10e}",
        "rms_abs": f"{float(math.sqrt(np.mean(diffs * diffs))):.10e}",
        "p99_abs": f"{float(np.quantile(diffs, 0.99)):.10e}",
        "max_rel": f"{float(np.max(rel)):.10e}",
        "max_key": ",".join(str(x) for x in common[max_idx]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--iterations", default="1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, str]] = []
    for iteration in [int(x) for x in args.iterations.split(",") if x]:
        pattern = f"iter{iteration:03d}_spin00_k0000.dat"
        for kind in ["eigenvalues", "h0_gw", "vc_gw"]:
            left_path = args.run_root / args.left / f"{kind}_{pattern}"
            right_path = args.run_root / args.right / f"{kind}_{pattern}"
            row = {
                "run_root": str(args.run_root),
                "pair": f"{args.left}_vs_{args.right}",
                "iteration": str(iteration),
            }
            row.update(summarize_pair(left_path, right_path, kind))
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
