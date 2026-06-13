# BDLtrain

VS Code Copilot Agent Skill —— 指导 agent 编写生产级贝叶斯深度学习（Bayesian Deep Learning）训练代码。

加载此 Skill 后，agent 在编写 BNN 训练脚本时会自动关注数据预处理、先验选择、推理方法、收敛诊断、不确定性校准，以及 checkpoint、early stopping、学习率调度等训练基础设施，输出可直接用于实验的完整代码。

## 安装

将本文件夹复制到 VS Code 的 skills 目录：

```
.github/skills/BDLtrain/          # 项目级别
~/.copilot/skills/BDLtrain/       # 用户级别（跨项目）
```

加载方式：
- 在 Copilot Chat 中使用 `/bayes-deeplearning` 手动调用
- 对话中提及 BNN、SVI、MCMC、Bayesian regression 等关键词时自动匹配

## 覆盖范围

| 步骤 | 内容 |
|------|------|
| 方法选择 | MCMC (NUTS) / SVI (变分推断) / MC Dropout / Deep Ensembles / Laplace 近似 |
| 数据预处理 | 标准化策略、训练/校准/测试三分割、mini-batch 配置 |
| 模型定义 | 全连接 BNN、层次先验、异方差模型、分类 BNN |
| 先验选择 | 权重/偏置/噪声的先验推荐、敏感性分析、非中心参数化 |
| 推理训练 | Pyro SVI 与 MCMC 的完整流程、autoguide 选型 |
| 训练基础设施 | Checkpoint 断点续训、Early Stopping、LR 调度（warmup/衰减/余弦退火）、实验日志、NaN/Inf 检测与自动恢复 |
| 收敛诊断 | R-hat、ESS、发散率（MCMC）；ELBO 平稳性、过拟合检测（SVI）；ArviZ 集成 |
| 评估校准 | 预测区间覆盖率、ECE、后验预测检查（PPC）、温度缩放 |
| 可视化 | ELBO 曲线、Trace 图、预测 vs 真实、残差分布、校准曲线 |

## 目录结构

```
├── SKILL.md                       # Skill 入口（9 步工作流，每步有完成检查点）
├── references/
│   ├── core-training.md           # 数据预处理、先验选择、模型架构、训练稳定性、超参数调优
│   ├── training-infra.md          # Checkpoint、Early Stopping、LR 调度、日志、异常恢复、效率优化
│   ├── bdl-methods.md             # 各推理方法的原理、超参数、适用场景与代码模板
│   ├── bdl-frameworks.md          # Pyro / NumPyro / PyMC / TFP / Stan 对比与选型
│   ├── diagnostics.md             # 收敛诊断、后验预测检查、校准评估、问题排查表
│   ├── model-architecture.md      # 常用 BNN 架构（全连接、层次先验、异方差、分类）
│   └── code-templates.md          # 模板索引，指向 assets/templates/
└── assets/templates/
    ├── svi_regression.py           # SVI 回归（含 checkpoint、early stopping、LR 调度、NaN 恢复）
    ├── mcmc_regression.py          # MCMC 回归（NUTS 采样、后验持久化、ArviZ 导出）
    ├── mc_dropout.py               # MC Dropout 轻量级不确定性估计
    ├── deep_ensembles.py           # Deep Ensembles 集成不确定性
    └── experiment_runner.py        # 超参数网格搜索与实验管理
```

## 环境依赖

代码模板基于 Pyro + PyTorch：

- Python ≥ 3.10
- Pyro ≥ 1.9
- PyTorch ≥ 2.0
- NumPyro / JAX（可选，用于 GPU 加速 MCMC）
- ArviZ（可选，用于详细诊断与 NetCDF 导出）

