"""Stage-two confirmation against DE and PSO at pre-selected boundary cases."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import benchmarks
from find_lite import find_lite
from baselines import de_rand_1_bin, pso_global

METHODS = {"FIND_Lite": find_lite, "DE_rand_1_bin": de_rand_1_bin, "PSO_global": pso_global}


def summary(values):
    x = np.asarray(values, float)
    return {"median": float(np.median(x)), "q1": float(np.quantile(x,.25)), "q3": float(np.quantile(x,.75)), "mean": float(x.mean()), "std": float(x.std()), "values": x.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--function", choices=benchmarks.NAMES, required=True)
    p.add_argument("--budget-factor", type=int, required=True)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=7000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    a = p.parse_args(); f, bounds = benchmarks.make(a.function, a.dim); result = {"protocol": {"stage": 2, "dim": a.dim, "function": a.function, "budget": f"{a.budget_factor}d", "runs": a.runs, "seeds": [a.seed_start,a.seed_start+a.runs-1], "methods": a.methods}, "results": {}}
    for name in a.methods:
        method = METHODS[name]
        scores=[]
        for seed in range(a.seed_start,a.seed_start+a.runs):
            row=method(f,bounds,a.budget_factor*a.dim,seed)
            if row["evaluations"]>a.budget_factor*a.dim: raise RuntimeError("budget violation")
            scores.append(row["best_f"])
        result["results"][name]=summary(scores)
        print(f"d={a.dim} {a.function} {name} median={np.median(scores):.6g}",flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
