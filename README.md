# FIND-Lite 开源验证项目

本目录是与历史研究目录隔离的发布准备空间。它冻结当前可复现的 FIND-Lite 实现，并以固定协议验证其性能边界；这里的结果才可作为未来开源说明的依据。

## 当前定位

FIND-Lite 是一个无梯度、盒约束、两真实点确定一条搜索线的局部线搜索算法。其强项是 Sphere 与旋转病态二次的高精度开发；它不是通用全局优化器。多峰、平移多峰、变量耦合和高维弯曲谷地均可能落后于 DE、PSO。

不要宣称全局收敛、全面优于 DE/PSO，或在未经协议验证的函数上作性能承诺。

## 目录

- `src/find_lite.py`：冻结候选实现；默认入口为 `find_lite`。
- `src/benchmarks.py`：确定性的原始、平移与旋转测试函数。
- `src/baselines.py`：固定参数的 DE/rand/1/bin 与 gbest PSO 对照。
- `run_experiments.py`：可复现的批量测试入口。
- `docs/`：协议、适用范围、消融和发布清单。
- `results/`：仅存放可复现实验的汇总 JSON/报告，不保存临时数据。


## 最小调用

```python
from find_lite import find_lite

result = find_lite(objective, bounds, max_evals=10000, seed=42)
print(result["best_f"], result["best_x"])
```

返回值包含 `best_f`、`best_x`、`evaluations` 和单调的 `history`。安装元数据见 `pyproject.toml`；基础测试可用 `pytest tests` 运行。

## 快速验证

```powershell
D:\electron\Python\python.exe run_experiments.py --suite smoke --runs 3 --out results/smoke.json
```

完整协议、固定种子、维度、预算与结果解释见 [测试协议](docs/TEST_PROTOCOL.md)。发布前必须完成 [发布清单](docs/RELEASE_CHECKLIST.md)。许可证尚未选择，见 `LICENSE_PENDING.md`。

## 当前证据

阶段一边界扫描、阶段二30种子确认及阶段三未见函数/参数敏感性检查已完成。正式适用范围、反例与对照结果见 [性能定位](docs/PERFORMANCE_POSITIONING.md) 和 [阶段三报告](docs/STAGE3_SUMMARY.md)。这些结果支持有条件定位，不构成通用全局优化声明。 汇总说明见 [算法数据报告](docs/FIND_Lite_算法数据报告.md)。
