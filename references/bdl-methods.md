# 贝叶斯深度学习方法详解

## 方法分类

```
贝叶斯深度学习推理方法
├── 精确推断 (仅限简单模型, 不适用于 DNN)
├── 马尔可夫链蒙特卡洛 (MCMC)
│   ├── HMC (哈密顿蒙特卡洛)
│   ├── NUTS (No-U-Turn Sampler) ← 推荐默认
│   └── SGHMC (随机梯度 HMC)
├── 变分推断 (VI)
│   ├── 均值场 (Mean-Field) / AutoDiagonalNormal
│   ├── 全协方差 (Full-Rank) / AutoMultivariateNormal
│   ├── 低秩协方差 / AutoLowRankMultivariateNormal
│   ├── 归一化流 (Normalizing Flows)
│   └── 随机变分推断 (SVI, 支持 mini-batch)
├── 近似推断
│   ├── MC Dropout (Monte Carlo Dropout)
│   ├── Deep Ensembles (深度集成)
│   ├── Laplace 近似
│   └── SWAG (Stochastic Weight Averaging Gaussian)
└── 确定性方法
    └── SGD 轨迹的协方差
```

## 1. MCMC (NUTS/HMC)

### 原理
通过构造哈密顿动力学模拟在参数空间中采样，收敛到真实后验分布。

### Pyro 代码模板
```python
from pyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model, adapt_step_size=True, max_tree_depth=7)
mcmc = MCMC(
    nuts_kernel,
    num_samples=1000,      # 保留样本数
    warmup_steps=500,      # 预热/调适步数
    num_chains=1,          # 链数 (多链用于诊断)
)
mcmc.run(data)
posterior_samples = mcmc.get_samples()
```

### 关键超参数
| 参数 | 建议值 | 说明 |
|------|--------|------|
| `num_samples` | 500-2000 | 后验样本数, 越多越精确但越慢 |
| `warmup_steps` | ≥ num_samples/2 | 预热期, 用于调步长; 默认丢弃 |
| `num_chains` | 2-4 | 多链用于 R-hat 诊断 |
| `max_tree_depth` | 5-10 | NUTS 树深度; 过大→慢, 过小→有偏 |
| `adapt_step_size` | True | 自动调步长, 推荐开启 |
| `target_accept_prob` | 0.8 | 目标接受率; 复杂后验用 0.9-0.95 |
| `jit_compile` | True | JIT 编译加速 (Pyro ≥ 1.9) |

### ⚠️ 实际训练注意事项
1. **维度诅咒**: 参数量 > 10³ 时 NUTS 极慢, 考虑 VI 或子采样
2. **发散 (Divergence)**: 检查 `mcmc.diagnostics()`; 发散数应 < 1%
3. **步长调适**: `adapt_step_size=True` 在 warmup 期间自动调整
4. **内存**: 样本全部存内存; 1000样本×10⁴参数 ≈ 80MB float64
5. **GPU 利用**: Pyro MCMC 主要 CPU; NumPyro 可 GPU 加速

### 诊断清单
```python
# 收敛诊断
mcmc.summary()  # 检查 r_hat < 1.01, n_eff > 100
diagnostics = mcmc.diagnostics()
if diagnostics.get("divergences", 0) > 0:
    print("⚠️ 存在发散样本, 增大 target_accept_prob 或 warmup_steps")
```

## 2. 变分推断 (SVI)

### 原理
用参数化分布 q_φ(θ) 近似真实后验 p(θ|D), 最小化 KL(q||p) 等价于最大化 ELBO。

### Pyro Autoguide 选择

| Guide | 参数量 | 协方差结构 | 适用场景 |
|-------|--------|-----------|----------|
| `AutoDiagonalNormal` | 2|θ| | 对角 | 快速; 大参数空间; 默认首选 |
| `AutoMultivariateNormal` | 2|θ| + |θ|² | 全协方差 | 高精度; 参数 < 10³ |
| `AutoLowRankMultivariateNormal` | 2|θ| + r·|θ| | 低秩 | 折中方案; rank=20-50 |
| `AutoNormalizingFlow` | 取决于流 | 任意 | 复杂后验; 多模态 |

### Pyro 代码模板
```python
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.contrib.autoguide import AutoDiagonalNormal

guide = AutoDiagonalNormal(model)
optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# 训练循环
elbo_history = []
for epoch in range(num_epochs):
    loss = svi.step(data)
    elbo_history.append(loss)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, ELBO: {loss:.1f}")
```

### ⚠️ 实际训练注意事项

#### 学习率调优
- 初始: `lr=0.01` (Adam)
- ELBO 不收敛 → 降低 lr (×0.1)
- ELBO 收敛过慢 → 提高 lr (×2) + 增大 batch_size
- 推荐使用 `ClippedAdam` 防梯度爆炸

#### 收敛监控
```python
# ⚠️ ELBO 本身含噪声, 移动平均查看趋势
import numpy as np
elbo_ma = np.convolve(elbo_history, np.ones(100)/100, mode='valid')
if np.std(elbo_ma[-100:]) / np.abs(np.mean(elbo_ma[-100:])) < 0.001:
    print("ELBO 已稳定收敛")
```

#### 局部最优
- 多次随机初始化, 取最佳 ELBO 的 run
- 学习率退火 (Cosine Annealing / ReduceLROnPlateau)
- `AutoMultivariateNormal` 比 `AutoDiagonalNormal` 更容易陷入局部最优

#### ELBO 估计
- `Trace_ELBO` 默认使用 1 个 Monte Carlo 样本估计梯度
- `num_particles=10` 可降低梯度方差但 ×10 计算量
- Mini-batch ELBO 是无偏的但方差大

### 预测
```python
from pyro.infer import Predictive

# ⚠️ num_samples 是预测时的采样数, 与训练分离
predictive = Predictive(model, guide=guide, num_samples=500)
predictions = predictive(X_test)
pred_mean = predictions["obs"].mean(dim=0)
pred_std = predictions["obs"].std(dim=0)
```

## 3. MC Dropout

### 原理
训练时使用 Dropout, 推理时保持 Dropout 开启, 多次前向传播得到预测分布。

### 实现
```python
class MCDropoutModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 训练和推理都保持开启
        return self.fc2(x)

# 预测时多次前向传播
def mc_predict(model, x, num_samples=100):
    model.train()  # ⚠️ 保持 train 模式以启用 dropout
    preds = torch.stack([model(x) for _ in range(num_samples)])
    return preds.mean(0), preds.std(0)
```

### ⚠️ 实际注意事项
- Dropout rate 需调优 (网格搜索 0.05-0.5)
- 可能需要比普通 DNN 更宽的网络以补偿 Dropout
- 不给出完整后验, 仅近似预测不确定性

## 4. Deep Ensembles

### 原理
训练 M 个独立初始化（和不同数据顺序）的 DNN, 集成预测。

### 实现
```python
def train_ensemble(model_class, data, num_models=5):
    models = []
    for i in range(num_models):
        model = model_class()
        # ⚠️ 不同随机种子是关键
        torch.manual_seed(i * 42)
        train_single(model, data)
        models.append(model)
    return models

def ensemble_predict(models, x):
    preds = torch.stack([m(x) for m in models])
    return preds.mean(0), preds.std(0)
```

### ⚠️ 实际注意事项
- 计算量 ×M, 但天然可并行
- 超参数: M=5-10, 模型需要足够多样化
- 可与 MC Dropout 结合 (Deep Ensembles + MC Dropout)
- 集成成员的 calibration 很重要

## 5. Laplace 近似

### 原理
在 MAP 估计处用 Hessian 构造高斯近似后验。

### 库推荐
- `laplace-torch` (PyTorch): `pip install laplace-torch`
- `backpack-for-pytorch`: 高效 Hessian 对角近似

### 适用场景
- 大模型微调后的不确定性估计
- 计算资源有限时
- 后验近似质量一般, 尤其对于深度非线性模型

## 方法选择决策指南

```
数据量 < 10³ 且参数 < 10³
└── → MCMC (NUTS), 最精确

数据量 10³-10⁵ 且需要不确定性估计
├── → SVI (AutoDiagonalNormal), 快速迭代
└── → NumPyro NUTS (GPU), 如需精确后验

数据量 > 10⁵ 或 DNN 参数 > 10⁴
├── → MC Dropout, 最简单
├── → Deep Ensembles, 质量最佳
└── → Laplace 近似, 最轻量

需要不确定性校准?
├── MCMC → 精确后验, 天然校准
├── SVI → 需检查校准 (可能过度自信)
├── MC Dropout → 需温度缩放 (temperature scaling)
└── Deep Ensembles → 校准通常较好
```
