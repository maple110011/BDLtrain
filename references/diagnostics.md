# 贝叶斯深度学习诊断与调试

## 1. MCMC 诊断清单

### 1.1 必须检查的指标

```python
# ⚠️ 每次 MCMC 运行后必须检查以下内容

# ---- 使用 Pyro 内置诊断 ----
mcmc.summary()  # 打印 r_hat, n_eff 等

# ---- 使用 ArviZ 详细诊断 (推荐) ----
import arviz as az

# 转换为 ArviZ InferenceData
idata = az.from_pyro(mcmc)

# 1. R-hat 统计量 (收敛诊断)
rhat = az.rhat(idata)
print(f"Max R-hat: {rhat.max().values:.3f}")  # 应 < 1.01 (严格) 或 < 1.05 (宽松)
bad_params = rhat[rhat > 1.05].count()
if bad_params > 0:
    print(f"⚠️ {bad_params} 个参数未收敛 (R-hat > 1.05)")

# 2. 有效样本量 (ESS)
ess = az.ess(idata)
print(f"Min ESS: {ess.min().values:.0f}")  # 应 > 100
# Bulk ESS (评估均值) 和 Tail ESS (评估尾部)
ess_bulk = az.ess(idata, method="bulk")
ess_tail = az.ess(idata, method="tail")

# 3. Monte Carlo 标准误 (MCSE)
mcse = az.mcse(idata)
print(f"Max MCSE/Posterior SD: {(mcse / idata.posterior.std()).max().values:.3f}")
# 应 < 0.05 说明估计足够精确

# 4. 发散样本检测
divergences = idata.sample_stats.get("diverging", None)
if divergences is not None:
    n_div = divergences.sum().values
    total = divergences.size
    print(f"发散率: {n_div}/{total} = {n_div/total*100:.2f}%")
    if n_div / total > 0.01:
        print("⚠️ 发散过多! 解决方法见下文")
```

### 1.2 可视化诊断

```python
# 必须绘制的诊断图
# 1. Trace 图 — 检查混合与平稳性
az.plot_trace(idata, var_names=["w1", "b1"])
# 好的 trace: 看起来像"毛虫", 围绕均值上下波动
# 坏的 trace: 有趋势、跳跃、或粘滞在一个区域

# 2. 自相关图 — 检查样本独立性
az.plot_autocorr(idata, var_names=["w1"])
# 自相关应在 lag≈10 内衰减到 0
# 高自相关 → 增大 warmup_steps 或 thin

# 3. 后验密度图
az.plot_posterior(idata, var_names=["sigma"])
# 检查是否合理 (如 sigma > 0)

# 4. 平行坐标图 — 检测发散
az.plot_parallel_coordinate(idata)
# 发散的样本会显示为与其他样本不同的模式

# 5. 对图 — 检测参数间相关性
az.plot_pair(idata, var_names=["w1", "b1"])
# 强相关性 → 考虑非中心参数化
```

### 1.3 发散问题解决流程

```
检测到发散样本?
├── 发散率 < 1%
│   └── 可接受的噪声, 无需处理
├── 发散率 1-10%
│   ├── 增大 target_accept_prob: 0.9 → 0.95
│   ├── 增大 warmup_steps: ×2
│   └── 检查先验是否过弱
└── 发散率 > 10%
    ├── 使用非中心参数化 (Non-centered)
    ├── 简化模型 (减少层数/宽度)
    ├── 加强先验 (减小方差)
    └── 考虑改用 VI 代替 MCMC
```

---

## 2. SVI 诊断清单

### 2.1 ELBO 监控

```python
# ⚠️ ELBO 收敛判断 (非数值诊断)
import numpy as np

def check_elbo_convergence(elbo_history, window=200, tol=1e-3):
    """
    检查 ELBO 是否收敛
    返回: (是否收敛, 诊断信息)
    """
    if len(elbo_history) < 2 * window:
        return False, "训练步数不足, 继续训练"
    
    # 移动平均
    ma = np.convolve(elbo_history, np.ones(window)/window, mode='valid')
    
    # 近期波动的相对大小
    recent = ma[-window:]
    std = np.std(recent)
    mean_abs = np.abs(np.mean(recent))
    cv = std / max(mean_abs, 1e-8)
    
    if cv < tol:
        return True, f"ELBO 已收敛 (CV={cv:.6f})"
    
    # 检查是否还在改善
    first_half = np.mean(ma[:len(ma)//2])
    second_half = np.mean(ma[len(ma)//2:])
    improvement = (second_half - first_half) / max(abs(first_half), 1e-8)
    
    if improvement < tol:
        return True, f"ELBO 改善可忽略 (Δ={improvement:.6f})"
    
    return False, f"ELBO 仍在改善 (Δ={improvement:.6f}), CV={cv:.6f}"
```

### 2.2 过拟合检测

```python
# ⚠️ BNN 训练中常被忽略的关键检查
# 在训练集和验证集上计算 ELBO
def compute_validation_elbo(model, guide, X_val, y_val, num_particles=10):
    """在验证集上评估 ELBO (固定模型参数)"""
    predictive = Predictive(model, guide=guide, num_samples=num_particles)
    samples = predictive(X_val)
    # 计算 log predictive density
    log_prob = dist.Normal(samples["obs"].mean(0), samples["obs"].std(0)).log_prob(y_val)
    return log_prob.mean().item()

# 每 N 个 epoch 评估一次
# 如果训练 ELBO 持续下降但验证 ELBO 上升 → 过拟合
```

### 2.3 Guide 选择诊断

```python
# 比较不同 guide 的 ELBO
guides = {
    "Diagonal": AutoDiagonalNormal(model),
    "Multivariate": AutoMultivariateNormal(model),
    "LowRank(r=20)": AutoLowRankMultivariateNormal(model, rank=20),
}

for name, guide_cls in guides.items():
    guide = guide_cls
    # 训练并记录最终 ELBO
    # 选择 ELBO 最低 (最好) 的 guide
    # 但 Multivariate 参数量大, 可能过拟合
```

---

## 3. 后验预测检查 (PPC)

### 3.1 基本 PPC

```python
# ⚠️ 每次 BDL 训练后必须执行 PPC
def posterior_predictive_check(model, posterior_samples, X_test, y_test):
    """
    后验预测检查:
    1. 从后验预测分布采样
    2. 比较预测分布与观测数据的统计量
    """
    predictive = Predictive(model, posterior_samples)
    ppc_samples = predictive(X_test)
    
    # 预测均值 vs 观测均值
    pred_mean = ppc_samples["obs"].mean()
    obs_mean = y_test.mean()
    print(f"预测均值: {pred_mean:.3f}, 观测均值: {obs_mean:.3f}")
    
    # 预测标准差 vs 观测标准差
    pred_std = ppc_samples["obs"].std()
    obs_std = y_test.std()
    print(f"预测标准差: {pred_std:.3f}, 观测标准差: {obs_std:.3f}")
    
    # ⚠️ 如果观测统计量在预测分布的极端尾部 → 模型设定错误
    # 可能是: 先验不当、似然函数错误、模型过于简单
```

### 3.2 残差分析

```python
# 贝叶斯残差: (y - pred_mean) / pred_std
residuals = (y_test - pred_mean) / pred_std
# 应该近似 N(0,1)
# 检查: QQ plot, histogram vs N(0,1)
# 非正态残差 → 模型设定错误 (如应该用异方差模型)
```

---

## 4. 校准诊断

### 4.1 回归校准
```python
# 名义覆盖率 vs 实际覆盖率
def calibration_coverage(y_true, pred_mean, pred_std, alphas=[0.1, 0.2, ..., 0.95]):
    coverages = {}
    for alpha in alphas:
        z = norm.ppf(1 - alpha/2)  # 如 0.95 → 1.96
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        coverage = ((y_true >= lower) & (y_true <= upper)).float().mean()
        coverages[alpha] = coverage.item()
    return coverages
# 绘制: x=名义覆盖率, y=实际覆盖率
# 理想: y=x 对角线
# 曲线上凸 (y < x) → 过度自信; 下凹 (y > x) → 过于保守
```

### 4.2 分类校准 (ECE)
```python
# 期望校准误差 (Expected Calibration Error)
# pip install netcal
from netcal.metrics import ECE

ece_metric = ECE(bins=15)
ece = ece_metric.measure(probabilities, labels)
print(f"ECE: {ece:.4f}")  # < 0.05 良好, > 0.1 需校准
```

---

## 5. 常见问题排查表

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| MCMC 发散率 > 10% | 后验几何复杂 | 非中心参数化, target_accept_prob→0.95 |
| R-hat > 1.1 | 链未混合 | 增大 warmup_steps, 检查多模态 |
| ESS < 50 | 高自相关 | 增大 num_samples, thinning, 非中心参数化 |
| SVI ELBO → NaN | 梯度爆炸 | 降低 lr, 梯度裁剪, 检查数据标准化 |
| SVI ELBO 波动大 | batch太小 | 增大 batch_size 或 num_particles |
| 预测不确定性过大 | 先验过弱 | 加强先验, 检查似然噪声设定 |
| 预测不确定性过小 (过度自信) | VI 低估方差 | 用 Full-rank guide, 或改用 MCMC |
| 训练慢 | 参数量大 | 降 hidden_dim, 用 Diagonal guide, mini-batch |
| 后验预测覆盖差 | 模型设定错误 | 检查似然函数, 增加异方差, 检查先验 |
