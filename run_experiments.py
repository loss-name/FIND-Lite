"""Reproducible FIND-Lite benchmark runner."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import benchmarks
from baselines import de_rand_1_bin, pso_global
from find_lite import find_lite, find_lite_annealed, find_lite_roles

SUITES = {
    "smoke": ("sphere", "rastrigin", "ackley"),
    "core": benchmarks.NAMES,
}
METHODS = {
    "FIND_Lite": find_lite,
    "FIND_ThreeAnchor": lambda f,b,n,s: find_lite_roles(f,b,n,s,exploration_floor=1e-15,enable_continuation=False),
    "FIND_Annealed": find_lite_annealed,
    "DE_rand_1_bin": de_rand_1_bin,
    "PSO_global": pso_global,
}


def summary(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "q1": float(np.quantile(values,.25)), "q3": float(np.quantile(values,.75)), "mean": float(values.mean()), "values": values.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=SUITES, default="smoke")
    p.add_argument("--dims", nargs="+", type=int, default=None)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--budget-factor", type=int, default=500)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); dims = a.dims or ([2, 10] if a.suite == "smoke" else [10, 30])
    result = {"protocol": {"suite": a.suite, "dims": dims, "runs": a.runs, "seeds": [a.seed_start, a.seed_start+a.runs-1], "budget": f"{a.budget_factor}d", "methods": a.methods}, "results": {}}
    for dim in dims:
        for name in SUITES[a.suite]:
            f, bounds = benchmarks.make(name, dim); case = {}
            for method in a.methods:
                scores = [METHODS[method](f, bounds, a.budget_factor*dim, seed)["best_f"] for seed in range(a.seed_start, a.seed_start+a.runs)]
                case[method] = summary(scores)
            result["results"].setdefault(str(dim), {})[name] = case
            print(f"d={dim} {name}: " + " | ".join(f"{m}={case[m]['median']:.6g}" for m in a.methods), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
