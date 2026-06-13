# 贝叶斯深度学习代码模板

本文档是代码模板的索引。每个模板的完整可运行代码位于 [assets/templates/](../assets/templates/) 目录。

## 模板索引

| 模板 | 文件 | 说明 | 适用场景 |
|------|------|------|----------|
| 1. SVI 回归 (生产级) | [svi_regression.py](../assets/templates/svi_regression.py) | Checkpoint / Early Stopping / LR调度 / 日志 / NaN检测 全集成 | 回归任务, 快速迭代 |
| 2. MCMC 回归 | [mcmc_regression.py](../assets/templates/mcmc_regression.py) | NUTS采样 / 后验持久化 / ArviZ导出 | 需要精确后验的小数据 |
| 3. MC Dropout | [mc_dropout.py](../assets/templates/mc_dropout.py) | 轻量级不确定性, 训练时保留Dropout | 大模型, 大数据 |
| 4. Deep Ensembles | [deep_ensembles.py](../assets/templates/deep_ensembles.py) | 多模型集成, 独立随机种子 | 需要高质量不确定性 |
| 5. 实验管理器 | [experiment_runner.py](../assets/templates/experiment_runner.py) | 网格搜索 / 多种子 / 结果汇总 | 超参数调优 |

## 使用方法

1. 选择合适的模板文件
2. 根据任务修改 \CONFIG\ 字典中的超参数
3. 填充数据加载部分 (\X_train\, \y_train\ 等)
4. 直接运行: \python svi_regression.py
## 模板选择决策

\需要不确定性量化?
├── 数据量 < 1000, 需精确后验 → 模板2 (MCMC)
├── 数据量 1000-100000 → 模板1 (SVI)
├── 大模型/大数据 → 模板3 (MC Dropout) 或 模板4 (Deep Ensembles)
└── 超参数搜索 → 模板5 (ExperimentRunner)
\