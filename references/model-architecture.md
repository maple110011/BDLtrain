# 贝叶斯神经网络架构模式

## 1. 基础 BNN (全连接)

```python
def bnn_fully_connected(x, y=None, hidden_dims=[32, 16], activation="tanh"):
    """
    灵活的多层全连接 BNN
    hidden_dims: 各隐藏层维度列表, 如 [32, 16] 表示 2 隐藏层
    """
    input_dim = x.shape[-1]
    output_dim = 1 if y is None else y.shape[-1] if y.dim() > 1 else 1
    
    # 噪声先验
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    
    h = x
    dims = [input_dim] + hidden_dims
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        w = pyro.sample(
            f"w{i+1}",
            dist.Normal(0, 1).expand([d_out, d_in]).to_event(2)
        )
        b = pyro.sample(
            f"b{i+1}",
            dist.Normal(0, 1).expand([d_out]).to_event(1)
        )
        h = h @ w.t() + b
        if activation == "tanh":
            h = torch.tanh(h)
        elif activation == "relu":
            h = torch.relu(h)
    
    # 输出层
    w_out = pyro.sample(
        f"w_out",
        dist.Normal(0, 1).expand([output_dim, hidden_dims[-1]]).to_event(2)
    )
    b_out = pyro.sample(
        f"b_out",
        dist.Normal(0, 1).expand([output_dim]).to_event(1)
    )
    mu = h @ w_out.t() + b_out
    
    # 观测模型
    with pyro.plate("data", x.shape[0]):
        if y is not None:
            pyro.sample("obs", dist.Normal(mu.squeeze(-1), sigma), obs=y)
    return mu.squeeze(-1)
```

## 2. 层次先验 BNN (Hierarchical BNN)

```python
def bnn_hierarchical(x, y=None, hidden_dim=20):
    """
    使用层次先验的 BNN:
    每个权重的尺度由共享的超先验决定
    好处: 自动正则化, 减少手动调先验的需要
    """
    input_dim = x.shape[-1]
    output_dim = 1
    
    # 层次先验 —— 超先验
    # 权重尺度先验
    tau_w = pyro.sample("tau_w", dist.HalfNormal(0.5))
    # 偏置尺度先验
    tau_b = pyro.sample("tau_b", dist.HalfNormal(0.5))
    
    # 噪声
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    
    # 第一层
    w1 = pyro.sample("w1",
        dist.Normal(0, tau_w).expand([hidden_dim, input_dim]).to_event(2))
    b1 = pyro.sample("b1",
        dist.Normal(0, tau_b).expand([hidden_dim]).to_event(1))
    
    # 输出层
    w2 = pyro.sample("w2",
        dist.Normal(0, tau_w).expand([output_dim, hidden_dim]).to_event(2))
    b2 = pyro.sample("b2",
        dist.Normal(0, tau_b).expand([output_dim]).to_event(1))
    
    h = torch.tanh(x @ w1.t() + b1)
    mu = (h @ w2.t() + b2).squeeze(-1)
    
    with pyro.plate("data", x.shape[0]):
        if y is not None:
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    return mu
```

## 3. 异方差噪声 BNN (Heteroskedastic)

```python
def bnn_heteroskedastic(x, y=None, hidden_dim=20):
    """
    同时预测均值和方差 (异方差不确定性)
    输出: mu 和 log_sigma (每个数据点有不同噪声)
    """
    input_dim = x.shape[-1]
    
    # 共享的隐藏表示
    w1 = pyro.sample("w1", dist.Normal(0, 1).expand([hidden_dim, input_dim]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))
    h = torch.tanh(x @ w1.t() + b1)
    
    # 均值头
    w_mu = pyro.sample("w_mu", dist.Normal(0, 1).expand([1, hidden_dim]).to_event(2))
    b_mu = pyro.sample("b_mu", dist.Normal(0, 1).expand([1]).to_event(1))
    mu = (h @ w_mu.t() + b_mu).squeeze(-1)
    
    # 方差头 (log-space)
    w_logvar = pyro.sample("w_logvar", dist.Normal(0, 1).expand([1, hidden_dim]).to_event(2))
    b_logvar = pyro.sample("b_logvar", dist.Normal(0, 1).expand([1]).to_event(1))
    log_sigma = (h @ w_logvar.t() + b_logvar).squeeze(-1)
    sigma = torch.exp(log_sigma).clamp(min=1e-4)
    
    with pyro.plate("data", x.shape[0]):
        if y is not None:
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    return mu, sigma
```

## 4. 分类 BNN

```python
def bnn_classifier(x, y=None, hidden_dim=20, num_classes=3):
    """
    多分类 BNN
    注意: 分类不需要 sigma 噪声参数
    """
    input_dim = x.shape[-1]
    
    w1 = pyro.sample("w1", dist.Normal(0, 1).expand([hidden_dim, input_dim]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))
    
    w2 = pyro.sample("w2", dist.Normal(0, 1).expand([num_classes, hidden_dim]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0, 1).expand([num_classes]).to_event(1))
    
    h = torch.tanh(x @ w1.t() + b1)
    logits = h @ w2.t() + b2
    
    with pyro.plate("data", x.shape[0]):
        if y is not None:
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
    return logits  # 返回 logits 而非概率
```

## 5. Mini-batch 兼容版本

```python
def bnn_model_minibatch(x, y=None, hidden_dim=20, subsample_size=None):
    """
    支持 mini-batch 训练的 BNN
    使用 pyro.plate 的 subsample 参数
    """
    input_dim = x.shape[-1]
    
    w1 = pyro.sample("w1", dist.Normal(0, 1).expand([hidden_dim, input_dim]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, 1).expand([hidden_dim]).to_event(1))
    w2 = pyro.sample("w2", dist.Normal(0, 1).expand([1, hidden_dim]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0, 1).expand([1]).to_event(1))
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    
    h = torch.tanh(x @ w1.t() + b1)
    mu = (h @ w2.t() + b2).squeeze(-1)
    
    # ⚠️ subsample_size 用于 mini-batch SVI
    # MCMC 中 subsample_size 自动忽略 (使用全批量)
    with pyro.plate("data", x.shape[0], subsample_size=subsample_size):
        if y is not None:
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    return mu
```

## 架构选择指南

| 场景 | 推荐架构 | 隐藏层维度 |
|------|----------|-----------|
| 一维回归 (如 sin(x)) | 单隐藏层 10-20 | 10-20 |
| 多维回归 | 双隐藏层 | [32, 16] 或 [64, 32] |
| 图像分类 | 贝叶斯 CNN (见下方) | — |
| 时间序列 | BNN + RNN 结构 | — |
| 大规模数据 | 用 MC Dropout / Deep Ensembles | 接近 DNN |
| 异方差数据 | Heteroskedastic BNN | [32, 16] |

## 贝叶斯 CNN 参考架构 (Pyro)

```python
# ⚠️ BNN-CNN 参数极多, MCMC 不可行, 只用 SVI
# 建议: 在 CNN 特征提取器后用 BNN 分类头 (最后一层贝叶斯化)
```
