# 贝叶斯深度学习框架对比与选型

## 框架总览

| 框架 | 语言 | 推理方法 | GPU 支持 | 适用场景 |
|------|------|----------|----------|----------|
| **Pyro** | Python | SVI, MCMC (NUTS/HMC), 重要性采样 | ✅ (PyTorch 后端) | 通用 BDL, 灵活模型, 研究原型 |
| **NumPyro** | Python/JAX | SVI, MCMC (NUTS/HMC) | ✅ (JAX 后端, 极快) | 大规模 MCMC, 需要高性能 |
| **PyMC** | Python/Aesara | MCMC (NUTS/HMC), SVI | ❌ (有限) | 经典贝叶斯统计, 非深度学习 |
| **TensorFlow Probability** | Python/TF | VI, MCMC (HMC), Bijectors | ✅ (TF 后端) | TF 生态用户, 生产部署 |
| **Stan** | C++/多语言 | MCMC (NUTS/HMC), VI, ADVI | ❌ | 严格贝叶斯工作流, 非深度 |
| **Branches** | Python | 各种方法包装 | 取决于后端 | 快速原型 |

## 选型决策流程

```
需要灵活定义复杂概率模型?
├── 是 → Pyro (PyTorch) 或 NumPyro (JAX)
│   ├── 需要极快 MCMC（大规模数据）? → NumPyro
│   └── 需要 PyTorch 生态集成? → Pyro
├── 否 → 传统贝叶斯统计?
│   ├── 需要深度神经网络? → TFP 或 Pyro
│   └── 标准统计模型? → PyMC 或 Stan
└── 生产部署?
    ├── PyTorch 栈 → Pyro + TorchScript/ONNX
    └── TensorFlow 栈 → TFP + TF Serving
```

## Pyro (推荐用于本 Skill)

### 核心优势
- **PyTorch 原生**: 无缝集成 PyTorch 的自动微分、GPU 加速、神经网络模块
- **灵活的推理**: SVI (变分推断) + MCMC (NUTS/HMC) 双引擎
- **plate 记号**: 高效 mini-batch 和数据子采样
- **丰富的分布库**: 所有常见分布 + 变换 (Transform)

### 关键 API
```python
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.optim import Adam, ClippedAdam
```

### 版本注意事项
- Pyro ≥ 1.8: 推荐使用 `AutoNormal`, `AutoMultivariateNormal` 等新版 autoguide
- `pyro.plate` 用于 mini-batch; 在 MCMC 中自动处理
- `Predictive` 类 (Pyro ≥ 1.5) 替代旧的 ` Predictive` 导入

## NumPyro (JAX 加速版)

### 何时选择 NumPyro
- 数据量 > 10⁴ 且需要 MCMC
- 需要 JAX 的 JIT 编译加速
- 需要并行链 (chains) 在 GPU 上批量运行

### 与 Pyro 的 API 差异
```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# NumPyro 使用 JAX 随机数
rng_key = jax.random.PRNGKey(0)
```

### 注意事项
- 模型函数需要是纯函数 (JAX 要求)
- 不支持 PyTorch 的 `nn.Module`
- 需要用 `jax.numpy` 而非 `torch.*`

## 环境中可用框架

根据当前 Python 环境 (`E:\software\Python\python.exe`):

| 框架 | 已安装 | 版本 |
|------|--------|------|
| Pyro | ✅ | 1.9.1 |
| PyTorch | ✅ | 2.12.0 |
| NumPyro | ❌ (需 `pip install numpyro`) | — |
| PyMC | ❌ (需 `pip install pymc`) | — |
| TFP | ❌ (需 `pip install tensorflow-probability`) | — |

**默认使用 Pyro** 因为已安装且与 PyTorch 生态无缝集成。
如需安装其他框架，使用:
```bash
pip install numpyro jax jaxlib  # NumPyro
pip install pymc                 # PyMC
pip install tensorflow-probability  # TFP
```

## 混合策略

实际项目中可使用多种方法互补:
1. **先用 SVI 快速迭代原型** (Pyro, 秒级)
2. **再用 MCMC 验证后验质量** (NumPyro, 分钟级)
3. **部署用 MC Dropout 或 Laplace 近似** (轻量)
