# Template 2: Pyro + MCMC 完整训练流程 (含后验持久化)
# Extracted from code-templates.md

"""
Pyro BNN MCMC (NUTS) —— 生产级模板 (v2)
新增: 后验持久化 / 配置哈希追溯 / ArviZ NetCDF / 运行环境 / 评估报告 / seed记录
"""
import os, sys, json, time, hashlib, logging, platform
from datetime import datetime
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import numpy as np

# 日志 (训练 + 错误分离)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_ch = logging.StreamHandler(); _ch.setLevel(logging.INFO); _ch.setFormatter(_fmt)
logger.addHandler(_ch)
_fh1 = logging.FileHandler("training.log", encoding="utf-8")
_fh1.setLevel(logging.DEBUG); _fh1.setFormatter(_fmt); logger.addHandler(_fh1)
_fh2 = logging.FileHandler("errors.log", encoding="utf-8")
_fh2.setLevel(logging.ERROR); _fh2.setFormatter(_fmt); logger.addHandler(_fh2)

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
MCMC_CONFIG = {
    "experiment": "bnn_mcmc_v1",
    "seed": 42,
    "hidden_dim": 20,
    "num_samples": 1000,
    "warmup_steps": 500,
    "num_chains": 2,
    "max_tree_depth": 7,
    "target_accept_prob": 0.8,
    "jit_compile": True,
    "output_dir": "outputs/mcmc",
}
pyro.set_rng_seed(MCMC_CONFIG["seed"])
torch.manual_seed(MCMC_CONFIG["seed"])
os.makedirs(MCMC_CONFIG["output_dir"], exist_ok=True)

# 保存配置 (包含哈希用于追溯)
config_hash = hashlib.md5(
    json.dumps(MCMC_CONFIG, sort_keys=True).encode()
).hexdigest()[:8]
logger.info(f"实验: {MCMC_CONFIG['experiment']} (hash={config_hash})")

# ═══════════════════════════════════════════════════════════
# 运行环境记录
# ═══════════════════════════════════════════════════════════
env = {
    "platform": platform.platform(),
    "hostname": platform.node(),
    "cpu_model": platform.processor() or "Unknown",
    "gpu": {"available": torch.cuda.is_available(),
            "count": torch.cuda.device_count() if torch.cuda.is_available() else 0},
    "packages": {"torch": torch.__version__, "pyro": pyro.__version__,
                  "numpy": np.__version__},
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        env["gpu"][f"device_{i}"] = {
            "name": props.name,
            "vram_gb": round(props.total_mem / (1024**3), 1),
        }
logger.info(f"环境: {env['cpu_model']} | "
            + (f"GPU: {env['gpu']['device_0']['name']}" if env["gpu"]["available"] else "CPU only"))
json.dump(env, open(os.path.join(MCMC_CONFIG["output_dir"], "environment.json"), "w"),
          indent=2, default=str)

# 保存配置
with open(os.path.join(MCMC_CONFIG["output_dir"], "config.json"), "w") as f:
    json.dump({**MCMC_CONFIG, "hash": config_hash}, f, indent=2)
# 显式记录种子
json.dump({"seed": MCMC_CONFIG["seed"]}, 
          open(os.path.join(MCMC_CONFIG["output_dir"], "seed.json"), "w"), indent=2)
# 输出目录
os.makedirs(os.path.join(MCMC_CONFIG["output_dir"], "figures"), exist_ok=True)

# ═══════════════════════════════════════════════════════════
# MCMC 运行 (数据准备见模板1)
# ═══════════════════════════════════════════════════════════
nuts_kernel = NUTS(
    bnn_model,
    adapt_step_size=True,
    max_tree_depth=MCMC_CONFIG["max_tree_depth"],
    target_accept_prob=MCMC_CONFIG["target_accept_prob"],
    jit_compile=MCMC_CONFIG["jit_compile"],
)
mcmc = MCMC(
    nuts_kernel,
    num_samples=MCMC_CONFIG["num_samples"],
    warmup_steps=MCMC_CONFIG["warmup_steps"],
    num_chains=MCMC_CONFIG["num_chains"],
)

t0 = time.time()
logger.info("开始 MCMC 采样...")
mcmc.run(X_train_n, y_train_n)
elapsed = time.time() - t0
logger.info(f"MCMC 完成, 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

# ═══════════════════════════════════════════════════════════
# 诊断
# ═══════════════════════════════════════════════════════════
mcmc.summary()

# ⚠️ 后验持久化 (关键步骤!)
posterior_path = os.path.join(
    MCMC_CONFIG["output_dir"],
    f"posterior_{config_hash}.pt"
)
posterior_package = {
    "samples": mcmc.get_samples(),
    "config": MCMC_CONFIG,
    "config_hash": config_hash,
    "diagnostics": mcmc.diagnostics() if hasattr(mcmc, "diagnostics") else {},
    "elapsed_seconds": elapsed,
    "timestamp": datetime.now().isoformat(),
}
torch.save(posterior_package, posterior_path)
logger.info(f"后验样本已保存至 {posterior_path}")

# ⚠️ 同时导出 ArviZ NetCDF (更好的互操作性)
try:
    import arviz as az
    idata = az.from_pyro(mcmc)
    nc_path = posterior_path.replace(".pt", ".nc")
    az.to_netcdf(idata, nc_path)
    logger.info(f"ArviZ InferenceData 已导出至 {nc_path}")

    # ArviZ 诊断
    rhat = az.rhat(idata)
    logger.info(f"R-hat 范围: [{rhat.min().values:.3f}, {rhat.max().values:.3f}]")

    ess = az.ess(idata, method="bulk")
    logger.info(f"ESS 范围: [{ess.min().values:.0f}, {ess.max().values:.0f}]")
    diverging = idata.sample_stats.get("diverging")
    r_hat_max, ess_min, div_rate = None, None, None
    if diverging is not None:
        n_div = diverging.sum().values
        div_rate = n_div / diverging.size
        logger.info(f"发散: {n_div}/{diverging.size} ({100*div_rate:.1f}%)")
    if 'rhat' in dir():
        r_hat_max = float(az.rhat(idata).max().values)
    if 'ess' in dir():
        ess_min = float(az.ess(idata, method="bulk").min().values)
    # 保存结构化诊断
    json.dump({"r_hat_max": r_hat_max, "ess_min": ess_min,
               "divergence_rate": div_rate},
              open(os.path.join(MCMC_CONFIG["output_dir"], "diagnostics.json"), "w"), indent=2)
except ImportError:
    logger.info("ArviZ 未安装, 跳过 NetCDF 导出和详细诊断")

# ═══════════════════════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════════════════════
predictive = Predictive(bnn_model, mcmc.get_samples())
preds = predictive(X_test_n)
pred_mean = preds["obs"].mean(dim=0)
pred_std = preds["obs"].std(dim=0)
rmse = ((pred_mean - y_test_n)**2).mean().sqrt()
coverage = ((y_test_n >= pred_mean - 1.96 * pred_std) &
            (y_test_n <= pred_mean + 1.96 * pred_std)).float().mean()
logger.info(f"MCMC 预测完成 | RMSE: {rmse:.4f} | 95% PI 覆盖率: {coverage:.4f}")

# ═══════════════════════════════════════════════════════════
# 评估报告
# ═══════════════════════════════════════════════════════════
eval_report = {
    "experiment": MCMC_CONFIG["experiment"],
    "timestamp": datetime.now().isoformat(),
    "environment": env, "config": MCMC_CONFIG,
    "metrics": {"test_rmse": float(rmse), "coverage_95pct": float(coverage)},
    "mcmc": {"num_samples": MCMC_CONFIG["num_samples"],
             "warmup_steps": MCMC_CONFIG["warmup_steps"],
             "num_chains": MCMC_CONFIG["num_chains"],
             "elapsed_seconds": elapsed,
             "r_hat_max": r_hat_max, "ess_min": ess_min,
             "divergence_rate": div_rate},
}
json.dump(eval_report,
          open(os.path.join(MCMC_CONFIG["output_dir"], "evaluation_report.json"), "w"),
          indent=2, default=str)
logger.info(f"评估报告已保存")