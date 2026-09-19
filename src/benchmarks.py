"""Deterministic continuous benchmark functions used by the release protocol."""
from __future__ import annotations
import numpy as np


def _shift(dim: int, key: int, amplitude: float) -> np.ndarray:
    return np.random.default_rng(4400 + 17 * dim + key).uniform(-amplitude, amplitude, dim)


def _rotation(dim: int) -> np.ndarray:
    return np.linalg.qr(np.random.default_rng(8191 + dim).normal(size=(dim, dim)))[0]


def sphere(x): return float(x @ x)
def rastrigin(x): return float(10 * x.size + np.sum(x*x - 10*np.cos(2*np.pi*x)))
def rosenbrock(x): return float(np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))
def ackley(x):
    n = x.size
    return float(-20*np.exp(-.2*np.sqrt(np.sum(x*x)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e)


NAMES = ("sphere", "shifted_sphere", "rosenbrock", "shifted_rosenbrock", "rastrigin", "shifted_rastrigin", "ackley", "shifted_ackley", "rotated_ellipsoid")


def make(name: str, dim: int):
    if name == "sphere": return sphere, [(-100., 100.)] * dim
    if name == "shifted_sphere":
        s = _shift(dim, 1, 40.); return lambda x: sphere(x-s), [(-100., 100.)] * dim
    if name == "rosenbrock": return rosenbrock, [(-5., 10.)] * dim
    if name == "shifted_rosenbrock":
        s = _shift(dim, 2, 2.); return lambda x: rosenbrock(x-s), [(-5., 10.)] * dim
    if name == "rastrigin": return rastrigin, [(-5.12, 5.12)] * dim
    if name == "shifted_rastrigin":
        s = _shift(dim, 3, 2.); return lambda x: rastrigin(x-s), [(-5.12, 5.12)] * dim
    if name == "ackley": return ackley, [(-32.768, 32.768)] * dim
    if name == "shifted_ackley":
        s = _shift(dim, 4, 10.); return lambda x: ackley(x-s), [(-32.768, 32.768)] * dim
    if name == "rotated_ellipsoid":
        q, weights = _rotation(dim), 10**np.linspace(0, 6, dim)
        return lambda x: float(np.sum(weights * (q @ x)**2)), [(-5., 5.)] * dim
    raise ValueError(f"unknown benchmark: {name}")
