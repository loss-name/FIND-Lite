"""Stage-one dimensional and budget boundary scan for frozen FIND-Lite."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import benchmarks
from find_lite import find_lite

DEFAULT_FUNCTIONS = ("sphere", "rotated_ellipsoid", "shifted_ackley", "rosenbrock")


def stat(values):
    x = np.asarray(values, float)
    return {"median": float(np.median(x)), "q1": float(np.quantile(x, .25)),
            "q3": float(np.quantile(x, .75)), "mean": float(x.mean()),
            "std": float(x.std()), "values": x.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", nargs="+", type=int, required=True)
    p.add_argument("--budget-factors", nargs="+", type=int, default=[50, 100, 500, 1000])
    p.add_argument("--functions", nargs="+", choices=benchmarks.NAMES, default=DEFAULT_FUNCTIONS)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=6000)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if any(d < 2 for d in a.dims) or any(k <= 0 for k in a.budget_factors):
        raise ValueError("dimensions must be >= 2 and budget factors positive")
    out = {"protocol": {"stage": 1, "algorithm": "FIND_Lite frozen default", "dims": a.dims,
                        "budget_factors": a.budget_factors, "functions": a.functions,
                        "runs": a.runs, "seeds": [a.seed_start, a.seed_start+a.runs-1]}, "results": {}}
    for dim in a.dims:
        for name in a.functions:
            f, bounds = benchmarks.make(name, dim)
            for factor in a.budget_factors:
                scores, calls = [], []
                for seed in range(a.seed_start, a.seed_start+a.runs):
                    row = find_lite(f, bounds, factor*dim, seed)
                    if row["evaluations"] > factor*dim: raise RuntimeError("budget violation")
                    scores.append(row["best_f"]); calls.append(row["evaluations"])
                out["results"].setdefault(str(dim), {}).setdefault(name, {})[str(factor)] = {"score": stat(scores), "evaluations": stat(calls)}
                print(f"d={dim} {name} budget={factor}d median={np.median(scores):.6g}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
