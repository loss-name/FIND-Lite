"""FIND-Lite: a compact two-point random line-search optimizer.

The algorithm minimizes an objective within box bounds.  Its design rests on
four invariants:

1. ``2 * dimension`` evaluated Latin-hypercube points become permanent
   references.  They are never replaced by better points, so the search does
   not discard its initial high-dimensional directional coverage.
2. Each iteration chooses one real reference ``r`` and the current evaluated
   incumbent ``x``.  The two trial points are ``x + s(r-x)`` and
   ``x - s(r-x)``.  Thus every search direction comes from two real evaluated
   points; no gradient, surrogate direction, or synthetic vector is used.
3. A single feasible scalar truncates each trial ray at the box boundary.  It
   keeps the evaluated point on the intended line, unlike coordinate-wise
   clipping that can bend a ray.
4. Every reference owns its own scalar scale.  A successful line grows only
   its own scale; an unsuccessful line shrinks only its own scale.

This is an improvement-only local line search.  It has no global-convergence
guarantee and can remain in a local basin on strongly multimodal problems.
"""
from __future__ import annotations
import operator
import numpy as np


def find_lite(objective, bounds, max_evals, seed=0):
    """Optimize ``objective`` and return the best evaluated point.

    ``bounds`` has shape ``(d, 2)`` and ``max_evals`` includes initialization.
    The frozen update rule is: initial scale ``0.25``; multiply by ``1.5`` if
    either trial improves the incumbent, otherwise multiply by ``0.5``; clamp
    the scale to ``[1e-15, 2]``.  The positive-side point is inside the segment
    when its scale is below one and may pass the reference after repeated
    success.  Both sides always use the same pre-evaluation incumbent.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0:
        raise ValueError("bounds must have shape (positive dimension, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:,1] <= bounds[:,0]):
        raise ValueError("bounds must be finite with lower < upper")
    try:
        budget = operator.index(max_evals)
    except TypeError as exc:
        raise ValueError("max_evals must be an integer") from exc
    dim, count = len(bounds), 2*len(bounds)
    if budget < count:
        raise ValueError(f"budget must cover {count} initialization evaluations")
    low, high = bounds[:,0], bounds[:,1]
    width = high-low
    if not np.all(np.isfinite(width)):
        raise ValueError("bound widths must be finite")
    rng = np.random.default_rng(seed)

    # Latin hypercube initialization supplies two evaluated reference points
    # per coordinate on average.  Keeping them fixed is deliberate: replacing
    # them by current elites was the rank-collapse mechanism in FIND-VI.
    unit = np.empty((count, dim))
    for axis in range(dim):
        unit[:,axis] = (rng.permutation(count)+rng.random(count))/count
    references = low+unit*width
    calls, score, best = 0, float("inf"), None
    history = []

    def evaluate(point):
        """Use exactly one budget unit and update the best-ever incumbent."""
        nonlocal calls, score, best
        if calls >= budget:
            raise RuntimeError("internal budget violation")
        value = float(objective(point.copy()))
        calls += 1
        if not np.isfinite(value):
            raise ValueError("objective returned non-finite value")
        if value < score:
            score, best = value, point.copy()
        history.append((calls, score))

    for point in references:
        evaluate(point)
    # One scale per reference is the only adaptive state.  It assigns feedback
    # to the line that produced it instead of mixing unrelated lines globally.
    scales = np.full(count, .25)
    successes, skipped, reason = 0, 0, "budget"
    while calls < budget:
        index = int(rng.integers(count))
        # ``anchor`` remains fixed while evaluating both sides.  Therefore C
        # and D are a genuine paired line experiment rather than two serial
        # moves that happen to use a similar direction.
        anchor, anchor_score = best.copy(), score
        direction = references[index]-anchor
        if np.linalg.norm(direction/width) < 1e-14:
            skipped += 1
            if skipped >= 10*count:
                reason = "indistinguishable_reference_points"
                break
            continue
        skipped = 0
        # Both proposals use the same anchor even when C succeeds.
        for sign in (1., -1.):
            if calls >= budget:
                break
            delta = sign*scales[index]*direction
            nonzero = np.abs(delta) > 1e-300
            if not np.any(nonzero):
                reason = "step_underflow"
                break
            # Intersect the ray with the box using one common scalar.  The
            # resulting point stays collinear with ``anchor`` and ``reference``.
            limits = np.where(delta[nonzero] > 0,
                              high[nonzero]-anchor[nonzero],
                              low[nonzero]-anchor[nonzero])/delta[nonzero]
            fraction = min(1., max(0., float(np.min(limits))))
            # Single scalar preserves the line; clip removes roundoff only.
            evaluate(np.clip(anchor+fraction*delta, low, high))
        if reason == "step_underflow":
            break
        # The line receives binary feedback: at least one of its two points
        # improved the incumbent, or neither did.  Objective magnitudes never
        # enter this update, preserving invariance to positive rescaling.
        if score < anchor_score:
            successes += 1
            scales[index] = min(scales[index]*1.5, 2.)
        else:
            scales[index] = max(scales[index]*.5, 1e-15)
    return dict(best_f=float(score), best_x=best.tolist(), evaluations=calls,
                history=history, termination=reason, successful_lines=successes,
                reference_count=count)


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = find_lite(lambda x: float(x@x),
                       np.tile([-100.,100.], (args.dim,1)), args.budget,args.seed)
    print(json.dumps({k:v for k,v in result.items() if k != "history"},indent=2))


def find_lite_roles(objective, bounds, max_evals, seed=0, *,
                    exploration_floor=0.05, enable_continuation=True):
    """Experimental three-anchor FIND-Lite with explicitly separate roles.

    This is deliberately separate from :func:`find_lite`: it is an ablation
    vehicle, not the default algorithm.  The three anchors share permanent
    Latin-hypercube references and take turns under one common evaluation
    budget:

    * group 0 (exploration) never contracts a reference-line scale below
      ``exploration_floor``;
    * group 1 (balance) follows the unmodified Lite line rule;
    * group 2 (development) stores the two real endpoints of a successful
      move.  On its next turn it evaluates exactly one point beyond the new
      endpoint on that same real segment.  A failed continuation is discarded.

    No group migrates its incumbent to another group.  Setting
    ``exploration_floor=1e-15`` and ``enable_continuation=False`` yields the
    plain independent-three-anchor control (apart from round-robin scheduling).
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0:
        raise ValueError("bounds must have shape (positive dimension, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be finite with lower < upper")
    if not np.isfinite(exploration_floor) or not 0 < exploration_floor <= 2:
        raise ValueError("exploration_floor must lie in (0, 2]")
    try:
        budget = operator.index(max_evals)
    except TypeError as exc:
        raise ValueError("max_evals must be an integer") from exc
    dim, count = len(bounds), 2 * len(bounds)
    if budget < count:
        raise ValueError(f"budget must cover {count} initialization evaluations")
    low, high = bounds[:, 0], bounds[:, 1]
    width = high - low
    rng = np.random.default_rng(seed)

    unit = np.empty((count, dim))
    for axis in range(dim):
        unit[:, axis] = (rng.permutation(count) + rng.random(count)) / count
    references = low + unit * width
    values = np.asarray([float(objective(point.copy())) for point in references])
    if not np.all(np.isfinite(values)):
        raise ValueError("objective returned non-finite value")
    calls = count
    best_index = int(np.argmin(values))
    best_value, best_point = float(values[best_index]), references[best_index].copy()

    # Initialize one good and two geometrically distant anchors.  Distance is
    # normalized by box width, so this selection is invariant to unit changes.
    normalized = (references - low) / width
    selected = [best_index]
    while len(selected) < 3:
        distances = np.min(np.linalg.norm(
            normalized[:, None, :] - normalized[selected][None, :, :], axis=2), axis=1)
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    anchors = references[selected].copy()
    anchor_values = values[selected].astype(float).copy()
    scales = np.full((3, count), 0.25)
    continuation = None  # (old_endpoint, new_endpoint, scale), only group 2
    role_successes = [0, 0, 0]
    continuation_attempts = continuation_successes = 0
    turn = 0

    def feasible_point(anchor, direction, scale):
        """One scalar boundary intersection preserves the intended line."""
        delta = scale * direction
        active = np.abs(delta) > 1e-300
        if not np.any(active):
            return anchor.copy()
        limits = np.where(delta[active] > 0,
                          (high[active] - anchor[active]) / delta[active],
                          (low[active] - anchor[active]) / delta[active])
        fraction = min(1.0, max(0.0, float(np.min(limits))))
        return np.clip(anchor + fraction * delta, low, high)

    while calls < budget:
        group = turn % 3
        turn += 1
        # Development continuation has priority for precisely one turn.  It
        # evaluates only beyond x_new, never an invented rotated direction.
        if group == 2 and enable_continuation and continuation is not None:
            old, new, scale = continuation
            candidate = feasible_point(new, new - old, scale)
            value = float(objective(candidate.copy()))
            calls += 1
            if not np.isfinite(value):
                raise ValueError("objective returned non-finite value")
            continuation_attempts += 1
            if value < anchor_values[2]:
                anchors[2], anchor_values[2] = candidate, value
                best_value, best_point = (value, candidate.copy()) if value < best_value else (best_value, best_point)
                continuation = (new, candidate, min(2.0, scale * 1.5))
                continuation_successes += 1
            else:
                continuation = None
            continue
        if calls + 2 > budget:
            break
        index = int(rng.integers(count))
        old_anchor, old_value = anchors[group].copy(), float(anchor_values[group])
        direction = references[index] - old_anchor
        plus = feasible_point(old_anchor, direction, scales[group, index])
        minus = feasible_point(old_anchor, -direction, scales[group, index])
        plus_value, minus_value = float(objective(plus.copy())), float(objective(minus.copy()))
        calls += 2
        if not np.isfinite(plus_value) or not np.isfinite(minus_value):
            raise ValueError("objective returned non-finite value")
        candidate, value = (plus, plus_value) if plus_value <= minus_value else (minus, minus_value)
        if value < old_value:
            anchors[group], anchor_values[group] = candidate, value
            role_successes[group] += 1
            scales[group, index] = min(2.0, scales[group, index] * 1.5)
            if value < best_value:
                best_value, best_point = value, candidate.copy()
            if group == 2 and enable_continuation:
                continuation = (old_anchor, candidate.copy(), scales[group, index])
        else:
            floor = exploration_floor if group == 0 else 1e-15
            scales[group, index] = max(floor, scales[group, index] * 0.5)
            if group == 2:
                continuation = None
    return dict(best_f=float(best_value), best_x=best_point.tolist(), evaluations=calls,
                reference_count=count, anchor_count=3, role_successes=role_successes,
                continuation_attempts=continuation_attempts,
                continuation_successes=continuation_successes)


def find_lite_annealed(objective, bounds, max_evals, seed=0):
    """Experimental FIND-Lite with Metropolis acceptance of line candidates.

    Geometry, permanent references, and per-line scale feedback are identical
    to :func:`find_lite`.  The only difference is that a best-of-two candidate
    may replace the current *search state* despite being worse.  The
    best-ever evaluated point is still returned and never discarded.

    The initial temperature is the interquartile range of initial objective
    values, making it equivariant to positive objective rescaling.  It cools
    exponentially to one percent of that value over the allocated budget.
    This function is intentionally experimental: formal paired tests show it
    helps Ackley but is destructive on Sphere and Rosenbrock, so it is not the
    default optimizer.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0:
        raise ValueError("bounds must have shape (positive dimension, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be finite with lower < upper")
    try:
        budget = operator.index(max_evals)
    except TypeError as exc:
        raise ValueError("max_evals must be an integer") from exc
    dim, count = len(bounds), 2 * len(bounds)
    if budget < count:
        raise ValueError(f"budget must cover {count} initialization evaluations")
    low, high = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(seed)
    unit = np.empty((count, dim))
    for axis in range(dim):
        unit[:, axis] = (rng.permutation(count) + rng.random(count)) / count
    references = low + unit * (high - low)
    values = np.asarray([float(objective(point.copy())) for point in references])
    if not np.all(np.isfinite(values)):
        raise ValueError("objective returned non-finite value")
    calls = count
    index = int(np.argmin(values))
    best_value, best_point = float(values[index]), references[index].copy()
    state, state_value = best_point.copy(), best_value
    q1, q3 = np.quantile(values, [.25, .75])
    initial_temperature = max(float(q3 - q1), np.finfo(float).eps * max(1.0, abs(float(np.median(values)))))
    scales = np.full(count, .25)
    accepted_worse = 0

    def feasible_point(anchor, reference, scale, sign):
        delta = sign * scale * (reference - anchor)
        active = np.abs(delta) > 1e-300
        if not np.any(active):
            return anchor.copy()
        limits = np.where(delta[active] > 0,
                          (high[active] - anchor[active]) / delta[active],
                          (low[active] - anchor[active]) / delta[active])
        return np.clip(anchor + min(1.0, max(0.0, float(np.min(limits)))) * delta, low, high)

    while calls + 2 <= budget:
        line = int(rng.integers(count))
        plus = feasible_point(state, references[line], scales[line], 1.0)
        minus = feasible_point(state, references[line], scales[line], -1.0)
        plus_value, minus_value = float(objective(plus.copy())), float(objective(minus.copy()))
        calls += 2
        if not np.isfinite(plus_value) or not np.isfinite(minus_value):
            raise ValueError("objective returned non-finite value")
        candidate, value = (plus, plus_value) if plus_value <= minus_value else (minus, minus_value)
        strict_improvement = value < state_value
        progress = (calls - count) / max(1, budget - count)
        temperature = initial_temperature * (.01 ** progress)
        accept = strict_improvement or rng.random() < np.exp(-min(
            700.0, max(0.0, value - state_value) / max(temperature, np.finfo(float).tiny)))
        if accept:
            accepted_worse += int(value > state_value)
            state, state_value = candidate, value
        scales[line] = min(2.0, scales[line] * 1.5) if strict_improvement else max(1e-15, scales[line] * .5)
        if value < best_value:
            best_value, best_point = value, candidate.copy()
    return dict(best_f=float(best_value), best_x=best_point.tolist(), evaluations=calls,
                reference_count=count, initial_temperature=float(initial_temperature),
                accepted_worse=accepted_worse)

