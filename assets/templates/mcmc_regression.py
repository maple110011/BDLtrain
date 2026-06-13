# Template 2: Pyro + MCMC 完整训练流程 (含后验持久化)
# Extracted from code-templates.md

"""
Pyro BNN MCMC (NUTS) —— 生产级模板 (v2)
新增: 后验持久化 / 配置哈希追溯 / ArviZ NetCDF 导出
"""
import os, json, time, hashlib, logging
from datetime import datetime
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

with open(os.path.join(MCMC_CONFIG["output_dir"], "config.json"), "w") as f:
    json.dump({**MCMC_CONFIG, "hash": config_hash}, f, indent=2)

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
    if diverging is not None:
        n_div = diverging.sum().values
        logger.info(f"发散: {n_div}/{diverging.size} ({100*n_div/diverging.size:.1f}%)")

except ImportError:
    logger.info("ArviZ 未安装, 跳过 NetCDF 导出和详细诊断")

# ═══════════════════════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════════════════════
predictive = Predictive(bnn_model, mcmc.get_samples())
preds = predictive(X_test_n)
pred_mean = preds["obs"].mean(dim=0)
pred_std = preds["obs"].std(dim=0)
logger.info(f"MCMC 预测完成")