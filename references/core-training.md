# 贝叶斯深度学习实际训练注意事项

本文档涵盖编写和生产级贝叶斯深度学习训练代码时**必须注意**的全部实际考量。

---

## 1. 数据预处理

### 1.1 标准化 (Standardization)
```python
# ⚠️ 贝叶斯模型对输入尺度敏感
# 使用训练集的统计量标准化, 不要用全局统计量!
x_mean = X_train.mean(dim=0)
x_std = X_train.std(dim=0).clamp(min=1e-6)  # 防止除零
X_train_norm = (X_train - x_mean) / x_std
X_test_norm = (X_test - x_mean) / x_std
```

### 1.2 目标变量处理
- 回归: 标准化 y 到 N(0,1), 预测后逆变换
- 分类: 无需标准化, 但注意类别不平衡
- ⚠️ 使用 `HalfNormal(1.0)` 作为噪声先验 (对应标准化后的 y)

### 1.3 训练/验证/测试分割
```python
# ⚠️ 贝叶斯模型需要校准集 (calibration set)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.15)
# 比例: 68% train / 12% cal / 20% test
```

### 1.4 Mini-batch 设置
```python
# SVI 中使用 pyro.plate 进行 mini-batch 子采样
batch_size = 128  # ⚠️ 太小→ELBO 噪声大; 太大→慢
with pyro.plate("data", len(X_train), subsample_size=batch_size):
    # 模型定义
```

---

## 2. 先验选择

### 2.1 先验对结果影响巨大
```python
# ❌ 坏: 无信息先验 —— 可能导致后验弥散
w = pyro.sample("w", dist.Normal(0, 100))

# ✅ 好: 弱信息先验 —— 提供正则化但不过分约束
w = pyro.sample("w", dist.Normal(0, 1))

# ✅ 好: 层次先验 —— 让数据决定正则化强度
sigma_w = pyro.sample("sigma_w", dist.HalfNormal(0.5))
w = pyro.sample("w", dist.Normal(0, sigma_w).expand([out, in]).to_event(2))
```

### 2.2 网络权重先验指南

| 网络组件 | 推荐先验 | 说明 |
|----------|----------|------|
| 权重 (浅层) | `Normal(0, 1)` | 标准弱信息 |
| 权重 (深层) | `Normal(0, 1/sqrt(fan_in))` | Xavier/He 尺度 |
| 偏置 | `Normal(0, 1)` | 常用 |
| 噪声标准差 σ (回归) | `HalfNormal(0.5)` 或 `HalfCauchy(1)` | 后者有更重尾部 |
| 噪声标准差 σ (分类) | 固定为 1 (softmax 固有噪声) | — |

### 2.3 先验敏感性分析
```python
# ⚠️ 必须检查先验选择对结果的影响
for prior_scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
    # 重新训练并比较后验
    # 如果后验对 prior_scale 敏感, 说明数据信息不足
```

### 2.4 共轭与半共轭
- 使用 `HalfNormal` / `HalfCauchy` 作为尺度参数先验
- 避免使用 `Uniform(0, large_number)` — 边界处有不良行为

---

## 3. 模型架构设计

### 3.1 BNN 网络设计原则
```python
# ⚠️ BNN 通常比同任务 DNN 更窄/更浅
# 原因: 每个权重现在是随机变量, 参数空间膨胀
# 规则: 隐藏层维度 = DNN 版本 × 0.5~0.7
hidden_dim = int(dnn_hidden_dim * 0.6)
```

### 3.2 激活函数
- `tanh`: 配合标准正态先验, 对称性好 (推荐)
- `relu`: 与正态先验配合可能导致后验不对称
- `swish/silu`: 平滑, 适合深度 BNN

### 3.3 权重参数化技巧
```python
# 非中心参数化 (Non-centered Parameterization)
# ⚠️ 有助于 MCMC 采样效率, 减少漏斗状几何
w_raw = pyro.sample("w_raw", dist.Normal(0, 1).expand(shape).to_event(2))
sigma_w = pyro.sample("sigma_w", dist.HalfNormal(0.5))
w = w_raw * sigma_w  # 确定性变换
```

---

## 4. 训练稳定性

### 4.1 数值稳定性
```python
# ⚠️ 始终使用 log-space 计算
# log_softmax 而非 softmax 后取 log
log_p = dist.Normal(mu, sigma.clamp(min=1e-6)).log_prob(y)

# ⚠️ 梯度裁剪 (对 SVI 尤其重要)
from pyro.optim import ClippedAdam
optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})

# 或在优化器中设置
optimizer = torch.optim.Adam(params, lr=0.01)
torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
```

### 4.2 初始化
```python
# ⚠️ Pyro autoguide 的参数初始化影响收敛
# 可以使用 init_loc_fn 自定义
from pyro.infer.autoguide import init_to_mean, init_to_median

guide = AutoDiagonalNormal(
    model,
    init_loc_fn=init_to_median  # 比 init_to_mean 更鲁棒
)
```

### 4.3 SVI 训练不稳定
- **症状**: ELBO 突变为 NaN 或极大值
- **原因**: 梯度爆炸, 步长过大, 数值下溢
- **解决**:
  1. 降低 learning rate
  2. 开启梯度裁剪
  3. 检查数据标准化
  4. 使用 `pyro.poutine.trace` 调试

### 4.4 MCMC 发散
```python
# ⚠️ 检测发散样本
mcmc.run(X, y)
diagnostics = mcmc.diagnostics()
divergences = diagnostics.get("divergences", {}).get("chain 1", [])
div_rate = sum(divergences) / len(divergences)
if div_rate > 0.01:
    print(f"⚠️ 发散率 {div_rate:.3f}, 建议:")
    print("  1. 增大 target_accept_prob → 0.9 或 0.95")
    print("  2. 增大 warmup_steps")
    print("  3. 使用非中心参数化")
    print("  4. 简化模型/更强先验")
```

---

## 5. 收敛诊断

### 5.1 MCMC 诊断
```python
# ⚠️ 必须检查以下指标
# 1. R-hat (Gelman-Rubin 统计量)
r_hat = az.rhat(posterior_samples)  # 应 < 1.01 (严格) 或 < 1.05 (宽松)

# 2. 有效样本量 (ESS)
ess = az.ess(posterior_samples)  # 应 > 100 per chain

# 3. Trace 图 (肉眼检查)
# 应看起来像"毛虫", 无明显趋势或跳跃

# 4. 发散检查
# Pyro: mcmc.diagnostics()
# ArviZ: az.plot_parallel_coordinate, az.plot_pair
```

### 5.2 SVI 诊断
```python
# ⚠️ ELBO 收敛判断
# 1. 移动平均平缓
elbo_ma = np.convolve(elbo_hist, np.ones(200)/200, mode='valid')
recent_std = np.std(elbo_ma[-200:])
recent_mean = np.abs(np.mean(elbo_ma[-200:]))
if recent_std / max(recent_mean, 1e-8) < 0.001:
    print("ELBO 已收敛")

# 2. Predictive 检查
# 在验证集上评估; 如果 ELBO 继续下降但验证性能变差 → 过拟合
```

---

## 6. 不确定性量化与校准

### 6.1 校准评估
```python
# ⚠️ 贝叶斯模型可能校准不佳 (尤其是 VI)
# 回归: 检查预测区间覆盖率
coverage = ((y_test > lower) & (y_test < upper)).float().mean()
# 应接近名义覆盖率 (如 0.95)

# 分类: 可靠性图 (Reliability Diagram)
from sklearn.calibration import calibration_curve
# 或使用 netcal 库

# 校准误差 (ECE — Expected Calibration Error)
# ECE < 0.05 较好; > 0.1 需重新校准
```

### 6.2 温度缩放 (Temperature Scaling)
```python
# ⚠️ 对分类 BNN (尤其 VI) 的后验校准
# 在 calibration set 上优化温度参数 T
class TemperatureScaled(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
# 在校准集上用 NLL 优化 temperature, 冻结 model
```

### 6.3 后验预测检查 (PPC)
```python
# ⚠️ 必须执行 PPC 验证模型充分性
# 从后验预测分布采样
ppc_samples = predictive(X_test)
# 比较 ppc_samples 的汇总统计量与观测数据
# 如果观测数据在 PPC 分布尾部 → 模型设定错误
```

---

## 7. 计算效率

### 7.1 GPU 利用
```python
# ⚠️ 确保数据和模型在同一设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Pyro MCMC 主要 CPU; 如需 GPU MCMC → NumPyro + JAX
# Pyro SVI 天然支持 GPU
X, y = X.to(device), y.to(device)
```

### 7.2 SVI 子采样
```python
# ⚠️ subsample_size 对 ELBO 收敛速度影响大
# 经验法则: batch_size ≥ 100 (噪声可控)
# batch_size = N (全批量): ELBO 最稳定, 但 O(N) 每步
# batch_size = 32~256: 随机梯度, 更快但噪声更大
```

### 7.3 MCMC 加速技巧
- 减少 `num_samples` + 增大 `warmup_steps`
- 使用 `jit_compile=True` (Pyro ≥ 1.9)
- 薄化 (thinning): 如果内存受限, `mcmc.get_samples(group_by_chain=True)`
- 使用 NumPyro 替代 Pyro (GPU MCMC)

### 7.4 内存管理
```python
# ⚠️ MCMC 样本可能耗尽内存
# 估计: num_samples × num_chains × num_params × 8 bytes (float64)
# 例如: 1000 × 2 × 10000 × 8 = 160 MB
# 节省: 使用 float32, thinning, 或仅保存部分参数
```

---

## 8. 超参数调优

### 8.1 调优范围参考

| 超参数 | 搜索范围 | 说明 |
|--------|----------|------|
| 隐藏层维度 | [8, 16, 32, 64, 128] | BNN 用较小网络 |
| 隐藏层数 | [1, 2, 3] | 深度 > 3 时 VI 难训练 |
| 先验尺度 | [0.1, 0.5, 1.0, 2.0] | Log-uniform 搜索 |
| SVI lr | [0.001, 0.005, 0.01, 0.05] | 默认 0.01 |
| SVI epochs | [3000, 5000, 10000] | 看 ELBO 收敛 |
| MCMC samples | [500, 1000, 2000] | 权衡精度和速度 |
| MCMC warmup | [200, 500, 1000] | ≥ samples/2 |
| Batch size | [64, 128, 256, 512] | GPU 内存允许取大 |
| Dropout rate (MC Dropout) | [0.05, 0.1, 0.2, 0.3] | 网格搜索 |

### 8.2 ⚠️ 贝叶斯模型不宜用过多超参数调优
- 贝叶斯理念是通过先验分布表达不确定性
- 过度调优违背贝叶斯原则, 导致过拟合
- 建议: 先固定合理默认值, 仅在证据不足时微调

---

## 9. 代码组织最佳实践

### 9.1 模型函数结构
```python
def bnn_model(x, y=None, hidden_dim=20, num_layers=2):
    """
    贝叶斯神经网络模型

    Args:
        x: 输入 (N, input_dim)
        y: 观测目标 (可选, 用于训练)
        hidden_dim: 隐藏层宽度
        num_layers: 隐藏层数

    Returns: (仅当 y=None 时返回) 预测 mu
    """
    # ⚠️ 使用 pyro.sample 注册所有随机变量
    # ⚠️ 使用 pyro.plate 处理数据
    # ⚠️ 使用 pyro.deterministic 记录确定性变换
```

### 9.2 可复现性
```python
# ⚠️ 固定所有随机种子
import random
import numpy as np
import torch
import pyro

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ⚠️ 对于生产代码, 记录所有超参数
import json
config = {"hidden_dim": 20, "lr": 0.01, "seed": 42, ...}
with open("config.json", "w") as f:
    json.dump(config, f)
```

### 9.3 模型保存与加载
```python
# MCMC: 保存采样结果
torch.save(mcmc.get_samples(), "mcmc_posterior.pt")

# SVI: 保存 guide 参数
torch.save({"guide_state": pyro.get_param_store().get_state()}, "svi_guide.pt")
# 加载:
pyro.get_param_store().load_state(torch.load("svi_guide.pt")["guide_state"])
```

---

