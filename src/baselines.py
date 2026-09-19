"""Fixed, transparent reference implementations for the release benchmark."""
from __future__ import annotations
import numpy as np


def _evaluate(f, points): return np.asarray([float(f(x)) for x in points])


def de_rand_1_bin(f, bounds, budget, seed=0):
    rng = np.random.default_rng(seed); lo, hi = np.asarray(bounds, float).T; d, n = len(bounds), 2*len(bounds)
    pop = rng.uniform(lo, hi, (n, d)); values = _evaluate(f, pop); calls = n
    while calls + n <= budget:
        for i in range(n):
            others = np.delete(np.arange(n), i); a, b, c = rng.choice(others, 3, replace=False)
            mutant = np.clip(pop[a] + .5*(pop[b]-pop[c]), lo, hi); mask = rng.random(d) < .9; mask[rng.integers(d)] = True
            trial = np.where(mask, mutant, pop[i]); value = float(f(trial)); calls += 1
            if value <= values[i]: pop[i], values[i] = trial, value
    return {"best_f": float(values.min()), "evaluations": calls}


def pso_global(f, bounds, budget, seed=0):
    rng = np.random.default_rng(seed); lo, hi = np.asarray(bounds, float).T; d, n = len(bounds), 2*len(bounds)
    pos = rng.uniform(lo, hi, (n, d)); velocity = rng.uniform(-(hi-lo), hi-lo, (n, d))*.1; values = _evaluate(f, pos); calls = n
    personal, pvalues = pos.copy(), values.copy(); global_x = personal[int(pvalues.argmin())].copy(); global_f = float(pvalues.min())
    while calls + n <= budget:
        velocity = .7298*velocity + 1.49618*rng.random((n,d))*(personal-pos) + 1.49618*rng.random((n,d))*(global_x-pos)
        pos = np.clip(pos + velocity, lo, hi); values = _evaluate(f, pos); calls += n
        better = values < pvalues; personal[better], pvalues[better] = pos[better], values[better]
        j = int(pvalues.argmin())
        if pvalues[j] < global_f: global_x, global_f = personal[j].copy(), float(pvalues[j])
    return {"best_f": global_f, "evaluations": calls}
