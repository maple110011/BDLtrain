# Template 1: Pyro + SVI 完整训练流程 (回归) — 生产级
# Extracted from code-templates.md

"""
Pyro BNN 回归 —— 生产级模板 (v2)
新增: Checkpoint断点续训 / Early Stopping / LR调度 / CSV日志 / NaN检测 / 运行环境记录
"""
import os, sys, json, time, logging, warnings, platform, subprocess
from datetime import datetime
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 0. 日志系统
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log", encoding="utf-8"),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 1. 配置与种子
# ═══════════════════════════════════════════════════════════
CONFIG = {
    "experiment": "bnn_regression_v2",
    "seed": 42,
    "hidden_dim": 20,
    "base_lr": 0.01,
    "num_epochs": 10000,
    "batch_size": None,           # None = 全批量
    "num_predictive_samples": 500,
    # ── Checkpoint ──
    "checkpoint_dir": "checkpoints",
    "save_interval": 500,         # 每 N epoch 保存
    "keep_best_n": 3,
    # ── Early Stopping ──
    "early_stop_patience": 15,    # 容忍 N 个评估周期
    "eval_interval": 500,         # 每 N epoch 评估一次
    # ── LR 调度 ──
    "warmup_epochs": 300,
    "lr_milestones": {5000: 0.5, 8000: 0.2},
    # ── 最大训练时间 ──
    "max_training_hours": 4,
}
pyro.set_rng_seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"设备: {device} | 实验: {CONFIG['experiment']}")

# ═══════════════════════════════════════════════════════════
# 1.5 运行环境记录 (⚠️ 不同设备间实验可比性的基础)
# ═══════════════════════════════════════════════════════════
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning("psutil 未安装, CPU 详细信息将不可用。安装: pip install psutil")

def capture_environment():
    """采集当前运行环境的完整信息。不同设备上的耗时才有可比性。"""
    env = {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "cpu": {"model": platform.processor() or "Unknown"},
        "gpu": {"available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0},
        "packages": {"torch": torch.__version__, "pyro": pyro.__version__,
                      "numpy": np.__version__},
        "torch_config": {
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "mkldnn_available": torch.backends.mkldnn.is_available(),
        },
    }
    if _HAS_PSUTIL:
        env["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
        env["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
        env["cpu"]["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env["gpu"][f"device_{i}"] = {
                "name": props.name,
                "vram_total_gb": round(props.total_mem / (1024**3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
            }
    return env

env = capture_environment()
logger.info("=" * 60)
logger.info(f"CPU: {env['cpu']['model']} "
            f"({env['cpu'].get('physical_cores','?')}C/{env['cpu'].get('logical_cores','?')}T"
            + (f", {env['cpu'].get('ram_total_gb','?')}GB RAM)" if _HAS_PSUTIL else ")"))
if env["gpu"]["available"]:
    for i in range(env["gpu"]["count"]):
        g = env["gpu"][f"device_{i}"]
        logger.info(f"GPU[{i}]: {g['name']} ({g['vram_total_gb']}GB, CC {g['compute_capability']})")
else:
    logger.info("GPU: 无 (CPU only)")
logger.info(f"PyTorch {env['packages']['torch']}, Pyro {env['packages']['pyro']}, "
            f"CUDA {env['torch_config']['cuda_version']}")
logger.info("=" * 60)
json.dump(env, open("environment.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False, default=str)

# ═══════════════════════════════════════════════════════════
# 2. 数据预处理 (假设 X_train, y_train, X_test, y_test 已有)
# ═══════════════════════════════════════════════════════════
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0).clamp(min=1e-6)
y_mean, y_std = y_train.mean(), max(y_train.std(), 1e-6)
X_train_n = (X_train - X_mean) / X_std
y_train_n = (y_train - y_mean) / y_std
X_test_n = (X_test - X_mean) / X_std
y_test_n = (y_test - y_mean) / y_std

# ═══════════════════════════════════════════════════════════
# 3. 模型定义
# ═══════════════════════════════════════════════════════════
def bnn_model(x, y=None):
    input_dim = x.shape[-1]
    h_dim = CONFIG["hidden_dim"]
    w1 = pyro.sample("w1", dist.Normal(0, 1).expand([h_dim, input_dim]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, 1).expand([h_dim]).to_event(1))
    w2 = pyro.sample("w2", dist.Normal(0, 1).expand([1, h_dim]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0, 1).expand([1]).to_event(1))
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    h = torch.tanh(x @ w1.t() + b1)
    mu = (h @ w2.t() + b2).squeeze(-1)
    with pyro.plate("data", x.shape[0], subsample_size=CONFIG["batch_size"]):
        if y is not None:
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    return mu

# ═══════════════════════════════════════════════════════════
# 4. Checkpoint 管理器
# ═══════════════════════════════════════════════════════════
class CheckpointManager:
    def __init__(self, save_dir, keep_best_n=3):
        self.save_dir = save_dir
        self.keep_best_n = keep_best_n
        self.best_val_metric = float("inf")
        self.best_epoch = 0
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, elbo_hist, guide, optimizer, is_best=False):
        ckpt = {
            "epoch": epoch, "elbo_history": elbo_hist,
            "config": CONFIG, "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "guide_state": pyro.get_param_store().get_state(),
            "optimizer_state": optimizer.get_state(),
            "rng_state": {"torch": torch.get_rng_state(),
                          "pyro": pyro.get_rng_state()},
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.save_dir, f"ckpt_epoch{epoch}.tar")
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, os.path.join(self.save_dir, "ckpt_best.tar"))
        self._cleanup()
        return path

    def load(self, path=None):
        if path is None:
            files = sorted(
                [f for f in os.listdir(self.save_dir) if f.startswith("ckpt_epoch")],
                key=lambda f: int(f.split("epoch")[1].split(".")[0])
            )
            path = os.path.join(self.save_dir, files[-1]) if files else None
        if path is None: return None
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            pyro.get_param_store().load_state(ckpt["guide_state"])
            torch.set_rng_state(ckpt["rng_state"]["torch"])
            pyro.set_rng_state(ckpt["rng_state"]["pyro"])
            self.best_val_metric = ckpt.get("best_val_metric", float("inf"))
            self.best_epoch = ckpt.get("best_epoch", 0)
            return ckpt
        except Exception as e:
            logger.warning(f"Checkpoint 损坏: {e}, 从头开始")
            return None

    def update_best(self, epoch, val_metric):
        if val_metric < self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_epoch = epoch
            return True
        return False

    def _cleanup(self):
        files = sorted(
            [f for f in os.listdir(self.save_dir) if f.startswith("ckpt_epoch")],
            key=lambda f: int(f.split("epoch")[1].split(".")[0]), reverse=True
        )
        for f in files[self.keep_best_n:]:
            os.remove(os.path.join(self.save_dir, f))

# ═══════════════════════════════════════════════════════════
# 5. LR 调度 & Early Stopping
# ═══════════════════════════════════════════════════════════
def get_lr(epoch):
    """Warmup + 阶梯衰减"""
    base = CONFIG["base_lr"]
    if epoch < CONFIG["warmup_epochs"]:
        return base * (epoch + 1) / CONFIG["warmup_epochs"]
    factor = 1.0
    for milestone, decay in sorted(CONFIG["lr_milestones"].items()):
        if epoch >= milestone:
            factor = decay
    return base * factor

class EarlyStopper:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = float("inf"); self.counter = 0; self.should_stop = False
    def __call__(self, metric):
        if metric < self.best - self.min_delta:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ═══════════════════════════════════════════════════════════
# 5. 训练准备
# ═══════════════════════════════════════════════════════════
guide = AutoDiagonalNormal(bnn_model)
ckpt_mgr = CheckpointManager(CONFIG["checkpoint_dir"], CONFIG["keep_best_n"])
stopper = EarlyStopper(CONFIG["early_stop_patience"])
training_start = time.time()
max_seconds = CONFIG["max_training_hours"] * 3600

# 尝试断点续训
resume = ckpt_mgr.load()
if resume:
    start_epoch = resume["epoch"] + 1
    elbo_history = resume["elbo_history"]
    logger.info(f"✅ 从 epoch {resume['epoch']} 恢复 "
                f"(best_val={resume['best_val_metric']:.4f})")
else:
    start_epoch = 0
    elbo_history = []

# ═══════════════════════════════════════════════════════════
# 6. 训练循环
# ═══════════════════════════════════════════════════════════
logger.info(f"{'Epoch':<8} {'ELBO':<12} {'LR':<10} {'状态'}")
logger.info("-" * 50)

for epoch in range(start_epoch, CONFIG["num_epochs"]):
    # ── LR 更新 ──
    lr = get_lr(epoch)
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(bnn_model, guide, optimizer, loss=Trace_ELBO())

    # ── 训练一步 ──
    loss = svi.step(X_train_n, y_train_n)

    # ⚠️ NaN 检测
    if np.isnan(loss) or np.isinf(loss):
        logger.error(f"Epoch {epoch}: ELBO={loss}, 尝试从 checkpoint 恢复")
        resume = ckpt_mgr.load()
        if resume:
            start_epoch = resume["epoch"] + 1
            elbo_history = resume["elbo_history"]
            CONFIG["base_lr"] *= 0.5  # 降 lr 重试
        else:
            logger.error("无可恢复 checkpoint, 终止")
            break
        continue

    elbo_history.append(loss)

    # ── 定期日志 ──
    if (epoch + 1) % 200 == 0:
        logger.info(f"{epoch+1:<8} {loss:<12.1f} {lr:<10.5f} 训练中")

    # ── 评估 + Early Stopping ──
    if (epoch + 1) % CONFIG["eval_interval"] == 0:
        with torch.no_grad():
            pred = Predictive(bnn_model, guide=guide, num_samples=50)(
                X_test_n)
            val_rmse = ((pred["obs"].mean(0) - y_test_n)**2).mean().sqrt()
        logger.info(f"  >> Epoch {epoch+1}: val_rmse={val_rmse:.4f}")

        if stopper(val_rmse.item()):
            logger.info(f"🛑 Early stopping at epoch {epoch+1}")
            break

        # 更新最佳
        if ckpt_mgr.update_best(epoch, val_rmse.item()):
            ckpt_mgr.save(epoch, elbo_history, guide, optimizer, is_best=True)
            logger.info(f"  🏆 新最佳 (val_rmse={val_rmse:.4f})")

    # ── 定期 Checkpoint ──
    if (epoch + 1) % CONFIG["save_interval"] == 0:
        ckpt_mgr.save(epoch, elbo_history, guide, optimizer)

    # ── 最大时间限制 ──
    if time.time() - training_start > max_seconds:
        logger.warning(f"⚠️ 达到最大训练时间 {CONFIG['max_training_hours']}h")
        break

# 最终保存
final_epoch = len(elbo_history) - 1
ckpt_mgr.save(final_epoch, elbo_history, guide, optimizer, is_best=True)
logger.info(f"训练完成, 共 {final_epoch+1} epochs, 耗时 "
            f"{(time.time()-training_start)/60:.1f} min")

# ═══════════════════════════════════════════════════════════
# 7. 收敛检查
# ═══════════════════════════════════════════════════════════
elbo_arr = np.array(elbo_history)
ma = np.convolve(elbo_arr, np.ones(200)/200, mode='valid')
cv = np.std(ma[-200:]) / max(abs(np.mean(ma[-200:])), 1e-8)
if cv < 0.001:
    logger.info(f"✅ ELBO 已收敛 (CV={cv:.6f})")
else:
    logger.warning(f"⚠️ ELBO 可能未完全收敛 (CV={cv:.6f})")

# ═══════════════════════════════════════════════════════════
# 8. 预测与评估 (加载最佳 checkpoint)
# ═══════════════════════════════════════════════════════════
best_ckpt = torch.load(
    os.path.join(CONFIG["checkpoint_dir"], "ckpt_best.tar"),
    map_location="cpu", weights_only=False
)
pyro.get_param_store().load_state(best_ckpt["guide_state"])
logger.info(f"已加载最佳模型 (epoch {best_ckpt['best_epoch']}, "
            f"val_metric={best_ckpt['best_val_metric']:.4f})")

predictive = Predictive(bnn_model, guide=guide,
                        num_samples=CONFIG["num_predictive_samples"])
preds = predictive(X_test_n)
pred_mean_n = preds["obs"].mean(dim=0)
pred_std_n = preds["obs"].std(dim=0)
pred_mean = pred_mean_n * y_std + y_mean
pred_std = pred_std_n * y_std

rmse = ((pred_mean - y_test)**2).mean().sqrt()
logger.info(f"测试 RMSE: {rmse:.4f}")

z = 1.96
lower = pred_mean - z * pred_std
upper = pred_mean + z * pred_std
coverage = ((y_test >= lower) & (y_test <= upper)).float().mean()
logger.info(f"95% PI 覆盖率: {coverage:.4f} (理想 0.95)")

# ═══════════════════════════════════════════════════════════
# 9. 可视化
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(elbo_history, alpha=0.5)
axes[0].set_title("ELBO 收敛曲线")
axes[1].scatter(y_test_n, pred_mean_n, alpha=0.5)
axes[1].plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'r--')
axes[1].set_title("预测 vs 真实")
axes[2].hist((y_test_n - pred_mean_n).cpu(), bins=30, alpha=0.7)
axes[2].set_title("残差分布")
plt.tight_layout(); plt.savefig("bnn_svi_results.png", dpi=150)
logger.info("结果已保存至 bnn_svi_results.png")