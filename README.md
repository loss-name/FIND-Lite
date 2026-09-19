# FIND-Lite

FIND-Lite is a lightweight, derivative-free line-search optimizer for bounded continuous optimization. It selects two evaluated population points to define a search line and performs local exploration along that geometry.

> Research release: this project is intended for reproducible experiments, not a claim of universal global-optimum performance.

## Scope

FIND-Lite is most suitable for smooth, bounded, single-objective problems where local exploitation is valuable. It can be competitive on Sphere-like and rotated ill-conditioned quadratic landscapes. Performance may degrade on multimodal, shifted multimodal, strongly coupled, and high-dimensional curved-valley problems.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from find_lite import find_lite

result = find_lite(objective, bounds, max_evals=10000, seed=42)
print(result["best_f"], result["best_x"])
```

## Reproduce experiments

```bash
python run_experiments.py --suite smoke --runs 3 --out results/smoke.json
```

See [reproducibility](docs/REPRODUCIBILITY.md) and [performance positioning](docs/PERFORMANCE_POSITIONING.md).

## Documentation

- [中文说明](README.zh-CN.md)
- [Algorithm](docs/ALGORITHM.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Changelog](CHANGELOG.md)

## Status

Version 0.1.0 is a research-oriented public release. Results in `results/` are evidence for the stated scope, not a universal benchmark ranking.

## License

Apache License 2.0. See [LICENSE](LICENSE).
