# FIND-Lite（中文说明）

FIND-Lite 是一个轻量级、无梯度、盒约束连续优化算法。算法从种群中选取两个真实点确定搜索直线，并沿该几何方向进行局部探索。

这是一个可复现实验版本，不宣称在所有问题上优于 DE、PSO 或其他通用优化器。

## 适用范围

适合平滑、连续、有界、单目标问题，尤其是 Sphere 类和旋转病态二次函数。多峰、平移多峰、强变量耦合、弯曲峡谷和高维问题上可能明显落后于通用全局优化器。

## 安装与运行

```bash
pip install -e .
python run_experiments.py --suite smoke --runs 3 --out results/smoke.json
```

完整算法、实验协议和限制说明见 `docs/` 英文文档目录。

## 许可证

Apache License 2.0，允许商业和非商业使用、修改与再发布，但需保留版权和许可证声明。
