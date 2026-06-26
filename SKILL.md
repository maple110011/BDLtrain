---
name: bayes-deeplearning
description: 'Write production-grade Bayesian Deep Learning training code with Pyro/PyTorch. Use when: training BNN, MCMC (NUTS/HMC), Variational Inference (SVI), uncertainty quantification, posterior predictive checks, BDL diagnostics (R-hat/ESS/ELBO/calibration), Bayesian regression/classification, prior selection, heteroskedastic models, MC Dropout, Deep Ensembles, Laplace approximation, hierarchical priors, autoguide selection, NumPyro, ArviZ diagnostics, temperature scaling, checkpoint/resume training, early stopping, learning rate scheduling. Covers: data preprocessing, prior sensitivity, numerical stability, GPU, memory, reproducibility. Do NOT use for: standard deep learning without Bayesian components, pure MCMC without neural networks, pure Stan/PyMC.'
argument-hint: '[任务: 回归BNN / 分类BNN / MCMC vs SVI / MC Dropout / 异方差模型 / 层次先验]'
---

# 贝叶斯深度学习模型训练

本 Skill 指导 agent 编写**完整且注意实际训练中各种注意事项**的贝叶斯深度学习训练代码。

## 何时使用 / 何时不用

| ✅ 使用 | ❌ 不用 |
|---------|---------|
| 用户要求 BNN / 贝叶斯神经网络 | 标准 (非贝叶斯) DNN 训练 |
| 需要不确定性量化的深度学习 | 纯 MCMC 统计建模 (无神经网络) |
| Pyro / NumPyro / TFP 训练代码 | 纯 Stan / PyMC (无深度学习组件) |
| BDL 模型诊断与校准 | 通用 Python 脚本调试 |

## 核心原则

1. **完整流程**: 数据预处理 → 模型 → 训练+Checkpoint → 诊断 → 预测 → 校准
2. **记录运行环境**: 训练开始时必须捕获并打印 CPU/GPU/RAM/PyTorch 版本等硬件信息, 不同设备上的耗时才有可比性
3. **必须有 Checkpoint**: 定期保存 (参数+optimizer+RNG), 支持断点续训
4. **不跳过诊断**: MCMC 查 R-hat/ESS/发散; SVI 查 ELBO 收敛
5. **先验需有理由**: 不可随意 `Normal(0,1)`, 需说明依据
6. **量化不确定性**: 不仅报告点估计, 必须报告预测区间和校准质量

## 工作流

### 第 1 步: 理解需求并选择方法

询问或推断: 任务类型 / 数据规模 / 精度要求 / 部署约束

| 条件 | 推荐方法 |
|------|----------|
| N < 10³, 需精确后验 | MCMC (NUTS) |
| N 10³-10⁵, 快速迭代 | SVI (Pyro) |
| N > 10⁵, 大模型 | MC Dropout / Deep Ensembles |
| 需异方差不确定性 | Heteroskedastic BNN |
| 需自动正则化 | 层次先验 BNN |

详见 [bdl-methods.md](./references/bdl-methods.md) 和 [bdl-frameworks.md](./references/bdl-frameworks.md).

> ✅ **完成标志**: 已选定推理方法 + 框架, 并向用户确认任务类型

### 第 2 步: 数据预处理 ⚠️

参见 [core-training.md](./references/core-training.md) §1.

1. 标准化输入 — 用**训练集** mean/std, `clamp(min=1e-6)` 防除零
2. 标准化目标 — y→N(0,1), 预测后逆变换; 噪声用 `HalfNormal(0.5)`
3. 三分割 — 68% train / 12% cal / 20% test
4. 设备管理 — GPU 可用则移至 GPU

> ✅ **完成标志**: 数据已标准化; 验证 `x_std > 0` 且 `y_std > 0`

### 第 3 步: 模型定义

参见 [model-architecture.md](./references/model-architecture.md) 和 [core-training.md](./references/core-training.md) §2-§3.

选择架构模板: 基础回归 / 层次先验 / 异方差 / 分类 / Mini-batch兼容

**先验指南**: 权重 `Normal(0,1)`; 噪声 `HalfNormal(0.5)`; MCMC 场景用非中心参数化.

> ✅ **完成标志**: 模型函数可通过 `pyro.poutine.trace` 验证; 先验选择有注释说明

### 第 4 步: 推理训练

参见 [bdl-methods.md](./references/bdl-methods.md).

**SVI** (默认): `AutoDiagonalNormal` + `ClippedAdam(clip_norm=10.0)` + `Trace_ELBO()`
**MCMC**: `NUTS(adapt_step_size=True, jit_compile=True)` + `num_chains ≥ 2`

> ✅ **完成标志**: 训练循环运行无报错; ELBO/Loss 已记录

### 第 5 步: 训练基础设施 ⚠️

参见 [training-infra.md](./references/training-infra.md). 在训练循环中集成:

| 机制 | 要点 |
|------|------|
| **运行环境** | ⚠️ 训练开始前捕获 CPU/GPU/RAM/PyTorch版本等硬件信息, 保存为 environment.json (§13.3) |
| **Checkpoint** | 每500步保存 (参数+optimizer+RNG); 保留最佳; 启动时自动恢复 (§10) |
| **Early Stopping** | 基于验证集指标, 设最大训练时间 (§11) |
| **LR 调度** | Warmup + 阶梯衰减/余弦退火 (§12) |
| **实验日志** | logging + CSV + 元数据 (git/pkg/硬件) (§13) |
| **NaN 检测** | ELBO异常→回退checkpoint + 降lr (§15) |

> ✅ **完成标志**: Checkpoint 文件可正常 save/load; 中断重启训练可恢复; 环境信息已打印并保存

### 第 6 步: 收敛诊断 ⚠️

详见 [diagnostics.md](./references/diagnostics.md).

**MCMC**: R-hat<1.05, ESS>100, 发散率<1%, Trace 图无趋势
**SVI**: ELBO 移动平均平稳 (CV<0.001), 验证集无过拟合

> ✅ **完成标志**: 所有诊断指标通过阈值; 不通过时有解决方案

### 第 7 步: 预测与不确定性

`Predictive(model, guide, num_samples=500)`. 回归需逆标准化.

> ✅ **完成标志**: `pred_mean` 和 `pred_std` 形状正确; NaN-free

### 第 8 步: 评估与校准

RMSE/Accuracy + 95%PI覆盖率 + ECE<0.05 + PPC. 详见 [diagnostics.md](./references/diagnostics.md) §3-§4.

> ✅ **完成标志**: 覆盖率≈名义值 (误差<5%); ECE<0.1

### 第 9 步: 可视化

ELBO/Trace + 预测vs真实 + 残差 + 校准曲线.

> ✅ **完成标志**: 至少4张诊断图保存为PNG

## 常见错误排查

| 症状 | 解决 |
|------|------|
| ELBO→NaN | ClippedAdam + 降lr + 检查标准化 |
| MCMC 全发散 | 非中心参数化, target_accept_prob→0.95 |
| 预测过于自信 | 换 Full-rank guide 或 MCMC |
| 断电丢失 | 集成 CheckpointManager, 每500步保存 |
| 训练不收敛 | LR 阶梯衰减/余弦退火 |
| 验证集性能下降 | Early Stopping (基于验证指标) |
| GPU OOM | 降 batch_size, 梯度累积 |
| 局部最优 | 降 lr / 从 checkpoint 恢复后调整 |

## 代码输出要求

Agent 生成的代码必须在训练完成后产生以下**全部**输出物（详见 [training-infra.md](./references/training-infra.md) §17）:

1. **完整可运行** — import + 数据 + 训练 + 评估
2. **运行环境记录** — `environment.json` + `seed.json` (§13.3)
3. **Checkpoint 系统** — 中间 checkpoint + 最佳 checkpoint (`ckpt_best.tar`) + RNG state (§10)
4. **模型权重导出** — `best_model.pt` (最佳) + `final_model.pt` (最终) — 便于部署和分享
5. **训练日志** — `training.log` (完整) + `errors.log` (仅错误, 单独文件) (§13.1)
6. **逐 epoch 指标** — `metrics.csv` (epoch, train_elbo, val_rmse, lr) (§13.2)
7. **Early Stopping + LR调度** (§11-§12)
8. **NaN/Inf 检测与恢复** (§15)
9. **评估报告** — `evaluation_report.json` (结构化 JSON, 含所有指标+训练信息, 可脚本化批量对比) (§17.3)
10. **诊断** — SVI: ELBO 收敛判断; MCMC: `diagnostics.json` + `posterior.pt` + `posterior.nc` (ArviZ)
11. **图表** — ELBO曲线 / 预测vs真实 / 残差 / 校准曲线 → 保存到 `figures/`
12. **可复现** — 固定种子, 保存 RNG state, 记录完整 config

## 参考资源

| 资源 | 内容 |
|------|------|
| [bdl-frameworks.md](./references/bdl-frameworks.md) | 框架对比与选型 |
| [bdl-methods.md](./references/bdl-methods.md) | 推理方法详解 |
| [core-training.md](./references/core-training.md) | 数据预处理(§1) / 先验(§2) / 架构(§3) / 稳定性(§4) / 诊断(§5) / 校准(§6) / 效率(§7) / 超参(§8) / 代码组织(§9) |
| [training-infra.md](./references/training-infra.md) | Checkpoint(§10) / EarlyStopping(§11) / LR调度(§12) / 运行环境+种子(§13.3) / 日志+错误日志(§13.1-13.2) / DataLoader(§14) / NaN检测(§15) / 效率(§16) / **输出物清单(§17)** |
| [diagnostics.md](./references/diagnostics.md) | 诊断/PPC/校准/排查表 |
| [model-architecture.md](./references/model-architecture.md) | BNN 架构模式 |
| [code-templates.md](./references/code-templates.md) | 模板索引 → [assets/templates/](./assets/templates/) |

## 示例用法

```
# 回归 BNN with SVI + Checkpoint
/bayes-deeplearning 写一个回归 BNN, 用 SVI 训练, 需要 checkpoint 和 early stopping

# 分类 BNN with MCMC
/bayes-deeplearning 写一个分类 BNN, 用 MCMC (NUTS) 推理, 带完整诊断

# 对比实验
/bayes-deeplearning 写一个脚本对比 MCMC 和 SVI 在 BNN 回归上的速度和效果

# MC Dropout 快速原型
/bayes-deeplearning 用 MC Dropout 实现一个带有不确定性估计的回归模型

# 异方差模型
/bayes-deeplearning 写一个能同时预测均值和方差的异方差 BNN
```
