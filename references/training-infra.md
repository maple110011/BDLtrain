# 贝叶斯深度学习训练基础设施

本文档涵盖生产级 BDL 训练中必须集成的 Checkpoint、Early Stopping、LR 调度、日志等基础设施。

---

## 10. Checkpoint 与断点续训 (⚠️ 生产级训练必备)

### 10.1 为什么 BDL 训练需要 Checkpoint
- SVI 训练可能需要数千甚至数万 epoch, 耗时长 (分钟~小时)
- MCMC 采样代价极高, 中途中断则前功尽弃
- 超参数搜索需要多次独立训练, 容易遗漏中间结果
- 断电/系统崩溃/显存溢出 (OOM) 等意外中断

### 10.2 SVI Checkpoint 完整实现

```python
import os
import json
from datetime import datetime

class SVICheckpointManager:
    """
    SVI 训练检查点管理器
    功能: 定期保存 / 断点续训 / 保留最佳 / 自动清理
    """
    def __init__(self, save_dir="checkpoints", keep_best_n=3,
                 save_interval_epochs=500):
        self.save_dir = save_dir
        self.keep_best_n = keep_best_n
        self.save_interval = save_interval_epochs
        self.best_elbo = float("inf")
        self.best_epoch = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, epoch, elbo_history, config, guide, optimizer,
             is_best=False):
        """保存完整训练状态"""
        checkpoint = {
            "epoch": epoch,
            "elbo_history": elbo_history,
            "config": config,
            "guide_state": pyro.get_param_store().get_state(),
            "optimizer_state": optimizer.get_state(),
            "best_elbo": self.best_elbo,
            "best_epoch": self.best_epoch,
            "timestamp": datetime.now().isoformat(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "pyro": pyro.get_rng_state(),
            }
        }
        
        # ⚠️ 使用 .tar 而非 .pt —— 方便检查文件完整性
        path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.tar")
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoint_best.tar")
            torch.save(checkpoint, best_path)
        
        # 清理旧 checkpoint (保留最近 N 个)
        self._cleanup()
        return path
    
    def load(self, path=None):
        """加载检查点, 恢复训练状态"""
        if path is None:
            # 自动查找最新 checkpoint
            path = self._find_latest()
            if path is None:
                return None  # 无 checkpoint, 从头训练
        
        # ⚠️ 校验文件完整性
        try:
            checkpoint = torch.load(path, map_location="cpu",
                                    weights_only=False)
        except Exception as e:
            print(f"⚠️ Checkpoint 损坏 ({path}): {e}, 从头训练")
            return None
        
        # 恢复 Pyro 参数存储
        pyro.get_param_store().load_state(checkpoint["guide_state"])
        
        # 恢复 RNG 状态 (保证可复现)
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"]["torch"])
            pyro.set_rng_state(checkpoint["rng_state"]["pyro"])
        
        self.best_elbo = checkpoint.get("best_elbo", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        
        return checkpoint
    
    def should_save(self, epoch):
        return epoch % self.save_interval == 0
    
    def update_best(self, epoch, val_metric):
        """记录最佳模型 (val_metric 越小越好)"""
        if val_metric < self.best_elbo:
            self.best_elbo = val_metric
            self.best_epoch = epoch
            return True
        return False
    
    def _find_latest(self):
        """查找最新的 checkpoint 文件"""
        files = [f for f in os.listdir(self.save_dir)
                 if f.startswith("checkpoint_epoch") and f.endswith(".tar")]
        if not files:
            return None
        files.sort(key=lambda f: int(f.split("epoch")[1].split(".")[0]))
        return os.path.join(self.save_dir, files[-1])
    
    def _cleanup(self):
        """保留最近 N 个 epoch checkpoint + best checkpoint"""
        files = [f for f in os.listdir(self.save_dir)
                 if f.startswith("checkpoint_epoch") and f.endswith(".tar")]
        files.sort(key=lambda f: int(f.split("epoch")[1].split(".")[0]),
                   reverse=True)
        for f in files[self.keep_best_n:]:
            os.remove(os.path.join(self.save_dir, f))


# ─── 使用示例: SVI 训练循环 ───
ckpt_mgr = SVICheckpointManager(save_dir="checkpoints/bnn_svi",
                                 save_interval_epochs=500)

# 尝试断点续训
resume = ckpt_mgr.load()
start_epoch = resume["epoch"] + 1 if resume else 0
if resume:
    print(f"✅ 从 epoch {resume['epoch']} 恢复训练 (best_elbo={resume['best_elbo']:.1f})")
    elbo_history = resume["elbo_history"]
    # 重新创建 optimizer (Pyro SVI 的 optimizer state 恢复复杂, 建议重新创建)
    optimizer = ClippedAdam({"lr": config["lr"], "clip_norm": 10.0})
    svi = SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())
else:
    elbo_history = []

for epoch in range(start_epoch, config["num_epochs"]):
    loss = svi.step(X_train, y_train)
    elbo_history.append(loss)
    
    # 定期保存 checkpoint
    if ckpt_mgr.should_save(epoch):
        is_best = False
        ckpt_mgr.save(epoch, elbo_history, config, guide, optimizer,
                      is_best=is_best)
    
    # 验证并更新最佳
    if (epoch + 1) % 1000 == 0:
        val_metric = compute_validation_metric()  # 自定义
        if ckpt_mgr.update_best(epoch, val_metric):
            ckpt_mgr.save(epoch, elbo_history, config, guide, optimizer,
                          is_best=True)
            print(f"  🏆 新最佳模型 (epoch {epoch}, metric={val_metric:.4f})")

# 训练结束前最后一次保存
ckpt_mgr.save(config["num_epochs"] - 1, elbo_history, config, guide,
              optimizer, is_best=True)
```

### 10.3 MCMC Checkpoint

```python
# ⚠️ Pyro MCMC 不支持原生 checkpoint
# 替代方案: 训练完成后立即持久化后验样本

import hashlib

def save_mcmc_posterior(mcmc, config, save_path):
    """
    持久化 MCMC 后验样本及元数据
    包含模型配置的哈希, 便于追溯
    """
    # 计算配置哈希 (确保结果可追溯)
    config_hash = hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]
    
    posterior = {
        "samples": mcmc.get_samples(),
        "config": config,
        "config_hash": config_hash,
        "diagnostics": mcmc.diagnostics() if hasattr(mcmc, "diagnostics") else None,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(posterior, save_path)
    print(f"后验样本已保存至 {save_path} (hash={config_hash})")
    
    # ⚠️ 同时保存为 ArviZ NetCDF (更好的互操作性)
    try:
        import arviz as az
        idata = az.from_pyro(mcmc)
        nc_path = save_path.replace(".pt", ".nc")
        az.to_netcdf(idata, nc_path)
        print(f"ArviZ InferenceData 已保存至 {nc_path}")
    except ImportError:
        pass  # arviz 未安装则跳过

# 使用:
save_mcmc_posterior(mcmc, MCMC_CONFIG, "checkpoints/mcmc_posterior.pt")
```

### 10.4 Checkpoint 最佳实践

| 实践 | 说明 |
|------|------|
| **定期保存** | SVI 每 500-1000 epoch; MCMC 完成后立即保存 |
| **保存完整状态** | 不仅 parameters, 还需 optimizer state + RNG state + config |
| **保留最佳** | 基于验证集指标 (不是训练 ELBO) 保存最佳模型 |
| **文件命名规范** | `{model_name}_epoch{num}_{timestamp}.tar` |
| **完整性校验** | `torch.load` 包裹在 try-except, 失败时从头训练 |
| **自动清理** | 仅保留最近 N 个 checkpoint + best, 防止磁盘爆满 |
| **config 哈希** | 在文件名或元数据中记录 config 的 hash, 便于追溯 |
| **分离存储** | 不同实验的 checkpoint 存不同目录 |
| **云端备份** | 重要结果同步到云存储 (rclone / cloud SDK) |

---

## 11. Early Stopping (早停) 与训练时长控制

### 11.1 BDL 中的 Early Stopping 特殊性

⚠️ 与标准 DNN 不同, BDL 的 early stopping 需要更谨慎:
- ELBO 不是直接的泛化指标; ELBO 下降不一定意味着更好的预测
- 验证集上的 log predictive density 或 RMSE 是更好的早停指标
- VI 可能 ELBO 仍在改善但模型已过拟合

```python
class EarlyStopper:
    """
    贝叶斯模型专用 Early Stopping
    基于验证集指标 (ELBO/预测性能), 而非训练 ELBO
    """
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
        """
        patience: 容忍多少个评估周期无改善
        min_delta: 最小改善阈值 (绝对值)
        mode: "min" (越小越好, 如 RMSE) 或 "max" (越大越好)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, metric):
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop

# 使用:
stopper = EarlyStopper(patience=10, min_delta=1e-4)
for epoch in range(max_epochs):
    loss = svi.step(X_train, y_train)
    
    if epoch % eval_interval == 0:
        val_metric = compute_validation_metric()
        if stopper(val_metric):
            print(f"Early stopping at epoch {epoch}")
            break
```

### 11.2 最大训练时间限制

```python
import time

# ⚠️ 设置硬性时间上限, 防止训练无限循环
MAX_TRAINING_TIME = 3600 * 4  # 4 小时
training_start = time.time()

for epoch in range(max_epochs):
    # ... 训练代码 ...
    
    if time.time() - training_start > MAX_TRAINING_TIME:
        print(f"⚠️ 达到最大训练时间 {MAX_TRAINING_TIME/3600:.1f}h, 停止训练")
        break
```

---

## 12. 学习率调度 (Learning Rate Scheduling)

### 12.1 BDL 中学习率调度的特殊性

⚠️ Pyro SVI 使用的 `PyroOptim` (如 `ClippedAdam`) 不完全兼容 PyTorch 原生 scheduler。
推荐使用 **手动学习率衰减** 或 **Pyro 的 `pyro.optim.LambdaLR`**。

```python
# 方法 1: 手动阶段性衰减 (最可靠)
def get_lr(epoch, base_lr=0.01, milestones={3000: 0.5, 8000: 0.1}):
    """阶梯式学习率衰减"""
    factor = 1.0
    for milestone, decay in sorted(milestones.items()):
        if epoch >= milestone:
            factor = decay
    return base_lr * factor

# 在训练循环中:
for epoch in range(num_epochs):
    lr = get_lr(epoch)
    # 重新创建 optimizer (Pyro 限制 —— 但代价低)
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    # ... 继续训练 ...


# 方法 2: 余弦退火 (Cosine Annealing)
def cosine_lr(epoch, base_lr=0.01, min_lr=1e-4, total_epochs=10000):
    """余弦退火: 平滑衰减到最小值"""
    import math
    progress = epoch / total_epochs
    return min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


# 方法 3: ELBO 平台检测自动降 lr
def auto_reduce_lr(elbo_history, window=500, factor=0.5,
                   threshold=0.001):
    """
    如果 ELBO 在最近 window 步内改善不足 threshold,
    则将 lr 乘以 factor
    """
    if len(elbo_history) < 2 * window:
        return False  # 数据不足
    prev = np.mean(elbo_history[-2*window:-window])
    recent = np.mean(elbo_history[-window:])
    improvement = (prev - recent) / max(abs(prev), 1e-8)
    return improvement < threshold
```

### 12.2 学习率 Warmup

```python
# ⚠️ SVI 初期梯度估计噪声大, 建议 warmup
def warmup_lr(epoch, base_lr=0.01, warmup_epochs=500):
    """线性 warmup: 从 lr/100 线性增长到 base_lr"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr
```

---

## 13. 实验追踪与日志 (Experiment Tracking)

### 13.1 日志级别和内容

```python
import logging

# ⚠️ 配置分级日志: 训练日志(INFO+) + 错误日志(ERROR+ 单独文件)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler 1: 控制台 — INFO 及以上
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console)

# Handler 2: 训练日志文件 — DEBUG 及以上 (完整记录)
train_fh = logging.FileHandler("training.log", encoding="utf-8")
train_fh.setLevel(logging.DEBUG)
train_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(train_fh)

# Handler 3: 错误日志 — ERROR 及以上, 单独文件
error_fh = logging.FileHandler("errors.log", encoding="utf-8")
error_fh.setLevel(logging.ERROR)
error_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(error_fh)

# 训练中使用:
logger.info(f"开始训练 | config={config}")
logger.info(f"Epoch {epoch}/{total} | ELBO={loss:.1f} | lr={lr:.5f}")
logger.warning(f"检测到 ELBO 波动过大 (std={elbo_std:.1f})")
logger.error(f"NaN detected at epoch {epoch}")  # 自动写入 errors.log
```

### 13.2 CSV 日志 (轻量级, 无需外部依赖)

```python
import csv

class CSVLogger:
    """简单 CSV 日志记录器"""
    def __init__(self, filepath, fields):
        self.filepath = filepath
        self.fields = fields
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
    
    def log(self, **kwargs):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(kwargs)

# 使用:
logger = CSVLogger("metrics.csv",
                   ["epoch", "train_elbo", "val_rmse", "lr", "timestamp"])
logger.log(epoch=100, train_elbo=-45.2, val_rmse=0.12, lr=0.01,
           timestamp=time.time())
```

### 13.3 运行环境记录 (⚠️ 不同设备间实验可比性的基础)

> **为什么必须记录**: 笔记本 CPU、Colab GPU、专用计算卡之间性能差距可达 10-100 倍，不记录硬件信息则所有耗时数据毫无参照价值。每次训练都必须输出完整运行环境。

```python
import platform, psutil, subprocess

def capture_environment():
    """
    采集当前运行环境的完整信息。
    在训练脚本开头调用, 将结果同时打印到日志和保存为 JSON。
    这样在不同设备上运行的结果才能互相参照。
    """
    env_info = {
        # ── 操作系统 ──
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": sys.version,
        # ── CPU ──
        "cpu": {
            "model": platform.processor() or "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
        },
        # ── GPU ──
        "gpu": {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        # ── 软件版本 ──
        "packages": {
            "torch": torch.__version__,
            "pyro": pyro.__version__,
            "numpy": np.__version__,
        },
        # ── PyTorch 编译信息 (决定CPU指令集优化) ──
        "torch_config": {
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "mkl_available": torch.backends.mkl.is_available() if hasattr(torch.backends, "mkl") else False,
            "mkldnn_available": torch.backends.mkldnn.is_available(),
        },
    }

    # 逐卡记录 GPU 详细信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env_info["gpu"][f"device_{i}"] = {
                "name": props.name,
                "vram_total_gb": round(props.total_mem / (1024**3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }

    # 可选: 尝试检测是否在 Colab / Kaggle 环境
    try:
        import IPython
        env_info["runtime"] = "Colab" if "google.colab" in sys.modules else "Local"
    except ImportError:
        env_info["runtime"] = "Local"

    return env_info


# ─── 在训练脚本中使用 ───
env = capture_environment()
logger.info("=" * 60)
logger.info("运行环境:")
logger.info(f"  CPU: {env['cpu']['model']} "
            f"({env['cpu']['physical_cores']}C/{env['cpu']['logical_cores']}T, "
            f"{env['cpu']['ram_total_gb']}GB RAM)")
if env["gpu"]["available"]:
    for i in range(env["gpu"]["count"]):
        g = env["gpu"][f"device_{i}"]
        logger.info(f"  GPU[{i}]: {g['name']} ({g['vram_total_gb']}GB, CC {g['compute_capability']})")
else:
    logger.info("  GPU: 无 (CPU only)")
logger.info(f"  PyTorch {env['packages']['torch']}, "
            f"Pyro {env['packages']['pyro']}, "
            f"CUDA {env['torch_config']['cuda_version']}")
logger.info(f"  运行时: {env['runtime']}")
logger.info("=" * 60)

# 保存为 JSON, 方便后续汇总对比
with open("environment.json", "w", encoding="utf-8") as f:
    json.dump(env, f, indent=2, ensure_ascii=False, default=str)

# ⚠️ 同时保存种子信息 (单独文件, 便于审计)
with open("seed.json", "w") as f:
    json.dump({"seed": CONFIG["seed"], "rng_state_note":
               "RNG state 已嵌入 checkpoint .tar 文件中"}, f, indent=2)
```

> ⚠️ `capture_environment()` 依赖 `psutil`，需 `pip install psutil`。如果不想引入额外依赖，至少记录 `torch.cuda.get_device_name()` 和 `platform.processor()`。还应增加 `psutil.disk_usage(".").free` 以记录可用磁盘空间。

### 13.4 实验元数据记录 (完整版)

在以上环境信息基础上，生成训练脚本时还应一并记录：

```python
# ⚠️ 每个实验必须记录完整的元数据, 便于后续复现和比较
EXPERIMENT_METADATA = {
    "experiment_name": "bnn_regression_v1",
    "description": "BNN 回归: 1隐层20单元, SVI Diagonal guide",
    "date": datetime.now().isoformat(),
    "git_commit": subprocess.getoutput("git rev-parse HEAD"),
    "environment": capture_environment(),  # ← 嵌入环境信息
    "dataset": {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "input_dim": X_train.shape[1],
    },
    "config": config,  # 模型超参数
}

with open("experiment_metadata.json", "w") as f:
    json.dump(EXPERIMENT_METADATA, f, indent=2, default=str)
```

> ✅ **关键原则**: 实验报告或论文中描述训练耗时时，必须附带环境信息，否则数据无意义。建议在训练日志开头和 `experiment_metadata.json` 中均记录环境，方便事后查证。

### 13.5 与 wandb / TensorBoard 集成 (可选)

```python
# ⚠️ wandb 集成 — 推荐用于正式实验
# pip install wandb
try:
    import wandb
    wandb.init(project="bayes-dl", config=config, name="bnn_svi_v1")
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 训练循环中:
if WANDB_AVAILABLE:
    wandb.log({
        "train/elbo": loss,
        "train/lr": current_lr,
        "val/rmse": val_rmse,
        "val/coverage": coverage,
        "epoch": epoch,
    })
```

---

## 14. 数据加载最佳实践

### 14.1 PyTorch DataLoader 与 Pyro plate 的正确配合

```python
from torch.utils.data import TensorDataset, DataLoader

# ⚠️ Pyro plate 的 subsample 与 DataLoader 不要混用!
# 推荐: 使用 DataLoader 手动迭代, 不在 plate 中设置 subsample_size

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,            # ⚠️ 必须 shuffle
    num_workers=2,           # ⚠️ 多进程加载, 0=主进程
    pin_memory=True,         # ⚠️ GPU 训练时开启
    drop_last=True,          # ⚠️ 避免最后一批大小不一致
)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # Pyro model 内部使用 plate 但不再用 subsample_size
        loss = svi.step(batch_x, batch_y)
```

### 14.2 大数据的流式加载

```python
# ⚠️ 当数据无法全部加载到内存时
# 使用 HDF5 / memory-mapped numpy / PyTorch IterableDataset

# 方案 1: Memory-mapped numpy
X_mmap = np.load("data/X.npy", mmap_mode="r")  # 仅索引, 不加载

# 方案 2: HDF5 (推荐用 h5py)
import h5py
with h5py.File("data.h5", "r") as f:
    X = f["X"]  # 惰性加载
    # 按需索引: X[start:end]
```

---

## 15. 数值异常检测与自动恢复

### 15.1 NaN/Inf 实时监控

```python
# ⚠️ BDL 训练中 NaN 可能悄无声息地出现
def check_nan_inf(tensor, name="tensor"):
    """检查张量中是否有 NaN 或 Inf"""
    if torch.isnan(tensor).any():
        logger.warning(f"⚠️ {name} 包含 NaN! shape={tensor.shape}")
        return True
    if torch.isinf(tensor).any():
        logger.warning(f"⚠️ {name} 包含 Inf! shape={tensor.shape}")
        return True
    return False

# 在每个 epoch 后检查 ELBO
if np.isnan(loss) or np.isinf(loss):
    logger.error(f"Epoch {epoch}: ELBO = {loss}, 尝试恢复...")
    # 回退到上一个 checkpoint
    resume = ckpt_mgr.load()
    if resume:
        logger.info(f"已从 checkpoint epoch {resume['epoch']} 恢复")
        continue
    else:
        logger.error("无可用 checkpoint, 终止训练")
        break
```

### 15.2 梯度异常处理

```python
# ⚠️ 监控梯度范数
def get_grad_norm(guide):
    """计算 guide 参数的梯度范数"""
    total_norm = 0.0
    for name, param in pyro.get_param_store().named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# 训练循环中:
grad_norm = get_grad_norm(guide)
if grad_norm > 1000:  # 梯度爆炸
    logger.warning(f"梯度爆炸 (norm={grad_norm:.0f}), 跳过此步")
    continue  # 跳过, 依赖 ClippedAdam 的 clip_norm 防护
```

### 15.3 自动回退机制总结

```
训练步骤出错?
├── NaN in ELBO → 从上个 checkpoint 恢复, lr × 0.5
├── 梯度爆炸   → 跳过当前 step, 降低 lr
├── OOM (显存)  → 降低 batch_size, 从 checkpoint 恢复
├── 断电/崩溃   → 重启后自动加载最新 checkpoint
└── 连续异常   → 终止训练, 发送告警
```

---

## 16. 训练效率进阶技巧

### 16.1 混合精度训练 (Mixed Precision)

```python
# ⚠️ Pyro SVI 对混合精度支持有限
# 仅在模型前向传播部分使用 autocast
from torch.cuda.amp import autocast

def bnn_model_amp(x, y=None):
    """AMP 兼容的 BNN 模型 (仅前向部分用 autocast)"""
    # ... 参数采样 (保持 float32) ...
    with autocast():
        h = torch.tanh(x @ w1.t() + b1)
        mu = (h @ w2.t() + b2).squeeze(-1)
    # ... 观测模型 ...
```

### 16.2 多 GPU 策略

```python
# ⚠️ Pyro SVI 不支持原生多 GPU 数据并行
# 替代方案:
# 1. MCMC: 用 num_chains 在不同 GPU 上运行独立链
# 2. Deep Ensembles: 各成员在不同 GPU 上并行训练
# 3. 超参数搜索: 不同试验在不同 GPU 上

import multiprocessing as mp

def run_experiment_on_gpu(gpu_id, config):
    """在指定 GPU 上运行实验"""
    torch.cuda.set_device(gpu_id)
    # ... 完整训练流程 ...
    return results

# 并行运行:
with mp.Pool(num_gpus) as pool:
    results = pool.starmap(run_experiment_on_gpu,
                           [(i, config) for i in range(num_gpus)])
```

### 16.3 梯度累积 (Gradient Accumulation)

```python
# ⚠️ 当 GPU 显存不足以支持大 batch_size 时
# 用梯度累积模拟更大的有效 batch_size

accumulation_steps = 4  # 有效 batch_size = batch_size × accumulation_steps
optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

for epoch in range(num_epochs):
    for i, (batch_x, batch_y) in enumerate(dataloader):
        loss = svi.step(batch_x, batch_y)
        # ⚠️ Pyro 的优化步骤在 svi.step 中自动执行
        # 梯度累积需要通过自定义 loss 缩放实现
        # 或用 PyTorch 原生的 optimizer.zero_grad() / step() 模式
```

### 16.4 性能 Profiling

```python
# ⚠️ 定位训练瓶颈
import cProfile
import pstats

# 仅 profile 少量 epoch 以免 .prof 文件过大
profiler = cProfile.Profile()
profiler.enable()
for epoch in range(10):
    svi.step(X_train, y_train)
profiler.disable()
profiler.dump_stats("svi_profile.prof")

# 或用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for epoch in range(10):
        svi.step(X_train, y_train)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 16.5 内存泄漏检测

```python
# ⚠️ 长时间训练可能出现内存泄漏
import gc

def log_memory(step=""):
    """记录当前内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{step}] GPU: {allocated:.2f}GB allocated, "
              f"{reserved:.2f}GB reserved")

# 每 1000 epoch 检查一次
if epoch % 1000 == 0:
    log_memory(f"epoch {epoch}")
    gc.collect()  # 显式触发垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

---

## 17. 训练输出物清单 (⚠️ 必须全部生成)

每次训练完成后，输出目录应包含以下文件。缺少任何一项意味着实验不可复现或无法跨设备比较。

### 17.1 SVI 训练输出物

```
outputs/
├── environment.json          # 运行环境 (CPU/GPU/RAM/磁盘/PyTorch版本)
├── seed.json                 # 随机种子 (显式记录, 便于审计)
├── experiment_metadata.json  # 完整元数据 (嵌入 environment + dataset + config)
├── training.log              # 完整训练日志 (DEBUG 级别)
├── errors.log                # 错误日志 (仅 ERROR 级别, 空文件=无错误)
├── metrics.csv               # 逐 epoch 指标 (epoch, train_elbo, val_rmse, lr)
├── checkpoints/
│   ├── ckpt_epoch0500.tar    # 中间 checkpoint (含 RNG state + guide_state)
│   ├── ckpt_epoch1000.tar
│   ├── ...                   # 保留最近 N 个
│   └── ckpt_best.tar         # 最佳模型 checkpoint
├── best_model.pt             # 最佳模型权重 (仅 guide_state, 轻量)
├── final_model.pt            # 最终模型权重 (训练结束时)
├── evaluation_report.json    # 评估报告 (结构化)
└── figures/
    ├── elbo_curve.png        # ELBO 收敛曲线
    ├── prediction_vs_true.png # 预测 vs 真实
    ├── residuals.png         # 残差分布
    └── calibration.png       # 校准曲线
```

### 17.2 MCMC 训练输出物

```
outputs/
├── environment.json          # 同上
├── seed.json                 # 同上
├── experiment_metadata.json  # 同上
├── posterior.pt              # 后验样本 (PyTorch 格式, 含 config hash)
├── posterior.nc              # ArviZ InferenceData (NetCDF)
├── diagnostics.json          # R-hat / ESS / 发散率 (结构化)
├── evaluation_report.json    # 评估报告
└── figures/
    ├── trace_*.png           # Trace 图
    ├── posterior_*.png       # 后验密度图
    └── prediction_vs_true.png
```

### 17.3 评估报告格式

```python
# ⚠️ 评估完成后生成结构化报告, 方便脚本化对比多个实验
def generate_evaluation_report(metrics, env, config, output_path):
    """生成 JSON 格式的评估报告"""
    report = {
        "experiment": config.get("experiment", "unnamed"),
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "config": config,
        "metrics": {
            "test_rmse": float(metrics["test_rmse"]),
            "test_mae": float(metrics.get("test_mae", float("nan"))),
            "coverage_95pct": float(metrics["coverage_95pct"]),
            "coverage_90pct": float(metrics.get("coverage_90pct", float("nan"))),
            "ece": float(metrics.get("ece", float("nan"))),
        },
        "training": {
            "total_epochs": metrics.get("total_epochs", 0),
            "final_elbo": float(metrics.get("final_elbo", float("nan"))),
            "elbo_converged": metrics.get("elbo_converged", False),
            "early_stopped": metrics.get("early_stopped", False),
            "training_time_seconds": metrics.get("training_time_seconds", 0),
            "best_epoch": metrics.get("best_epoch", 0),
        },
        "diagnostics": {
            "mcmc_r_hat_max": metrics.get("r_hat_max"),
            "mcmc_ess_min": metrics.get("ess_min"),
            "mcmc_divergence_rate": metrics.get("divergence_rate"),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"评估报告已保存至 {output_path}")


# ─── SVI 训练结束后调用 ───
generate_evaluation_report(
    metrics={
        "test_rmse": rmse.item(),
        "coverage_95pct": coverage.item(),
        "total_epochs": final_epoch + 1,
        "final_elbo": elbo_history[-1],
        "elbo_converged": cv < 0.001,
        "early_stopped": stopper.should_stop,
        "training_time_seconds": time.time() - training_start,
        "best_epoch": ckpt_mgr.best_epoch,
    },
    env=env,
    config=CONFIG,
    output_path="evaluation_report.json"
)

# ─── MCMC 训练结束后调用 ───
generate_evaluation_report(
    metrics={
        "test_rmse": rmse.item(),
        "coverage_95pct": coverage.item(),
        "training_time_seconds": elapsed,
        "r_hat_max": float(rhat.max().values) if 'rhat' in dir() else None,
        "ess_min": float(ess.min().values) if 'ess' in dir() else None,
        "divergence_rate": n_div / diverging.size if 'n_div' in dir() else None,
    },
    env=env,
    config=MCMC_CONFIG,
    output_path="evaluation_report.json"
)
```

### 17.4 输出物检查清单

| 类别 | 文件 | SVI | MCMC |
|------|------|:---:|:---:|
| 环境 | `environment.json` | ✅ | ✅ |
| 种子 | `seed.json` | ✅ | ✅ |
| 元数据 | `experiment_metadata.json` | ✅ | ✅ |
| 训练日志 | `training.log` | ✅ | ✅ |
| 错误日志 | `errors.log` | ✅ | ✅ |
| 指标 | `metrics.csv` | ✅ | — |
| 中间检查点 | `checkpoints/ckpt_epoch*.tar` | ✅ | — |
| 最佳模型 | `checkpoints/ckpt_best.tar` | ✅ | — |
| 最佳权重 | `best_model.pt` | ✅ | — |
| 最终权重 | `final_model.pt` | ✅ | — |
| 后验样本 | `posterior.pt` | — | ✅ |
| 后验(ArviZ) | `posterior.nc` | — | ✅ |
| 诊断 | `diagnostics.json` | — | ✅ |
| 评估报告 | `evaluation_report.json` | ✅ | ✅ |
| 图表 | `figures/*.png` | ✅ | ✅ |
