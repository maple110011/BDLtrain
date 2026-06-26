# Template 5: 实验运行管理器 (一键训练+记录)
# Extracted from code-templates.md

"""
实验管理器 —— 管理多次 BDL 训练的完整生命周期
功能: 超参数网格搜索 / 多次随机种子 / 结果汇总 / 模型选优 / 运行环境记录
"""
import os, json, time, copy, itertools, platform
from datetime import datetime
import numpy as np
import torch
import pyro
from collections import defaultdict

class ExperimentRunner:
    """
    BDL 实验管理器
    - 自动管理不同实验的目录、checkpoint、日志
    - 支持网格搜索和多次随机种子
    - 自动汇总结果
    """
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        self.results = []
        os.makedirs(base_dir, exist_ok=True)
        # 记录全局运行环境 (所有实验共享)
        self._env = {
            "platform": platform.platform(),
            "cpu_model": platform.processor() or "Unknown",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "torch": torch.__version__, "pyro": pyro.__version__,
            "timestamp": datetime.now().isoformat(),
        }
        json.dump(self._env, open(os.path.join(base_dir, "environment.json"), "w"),
                  indent=2, default=str)
    
    def run_grid_search(self, train_fn, param_grid, seeds=[42]):
        """
        网格搜索
        train_fn(config, seed, exp_dir) -> dict of metrics
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        total = len(list(itertools.product(*values))) * len(seeds)
        count = 0
        
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            for seed in seeds:
                config["seed"] = seed
                exp_name = "_".join(f"{k}={v}" for k, v in config.items()
                                    if k != "seed")
                exp_dir = os.path.join(self.base_dir, exp_name,
                                       f"seed{seed}")
                os.makedirs(exp_dir, exist_ok=True)
                
                # 保存配置
                with open(os.path.join(exp_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                count += 1
                print(f"\n{'='*60}")
                print(f"实验 {count}/{total}: {exp_name} (seed={seed})")
                print(f"{'='*60}")
                
                t0 = time.time()
                try:
                    metrics = train_fn(config, seed, exp_dir)
                    metrics["status"] = "success"
                except Exception as e:
                    metrics = {"status": "failed", "error": str(e)}
                    print(f"❌ 实验失败: {e}")
                
                metrics["elapsed_min"] = (time.time() - t0) / 60
                metrics["experiment"] = exp_name
                metrics["seed"] = seed
                metrics["exp_dir"] = exp_dir
                metrics["timestamp"] = datetime.now().isoformat()
                self.results.append(metrics)
                
                # 增量保存结果 (防止中途崩溃丢失)
                with open(os.path.join(self.base_dir, "results.json"), "w") as f:
                    json.dump(self.results, f, indent=2, default=str)
        
        return self.summarize()
    
    def summarize(self):
        """汇总所有实验结果"""
        successful = [r for r in self.results if r.get("status") == "success"]
        failed = [r for r in self.results if r.get("status") != "success"]
        
        print(f"\n{'='*60}")
        print(f"实验汇总: {len(successful)} 成功, {len(failed)} 失败")
        print(f"{'='*60}")
        
        if successful:
            # 按验证指标排序
            metric_keys = [k for k in successful[0].keys()
                          if k.startswith("val_") and
                          isinstance(successful[0][k], (int, float))]
            for mk in metric_keys:
                sorted_results = sorted(successful,
                                       key=lambda r: r.get(mk, float("inf")))
                best = sorted_results[0]
                print(f"\n最佳 {mk}: {best.get(mk):.4f}")
                print(f"  实验: {best['experiment']}, seed={best['seed']}")
                print(f"  耗时: {best['elapsed_min']:.1f} min")
        
        # 保存完整汇总 (含环境信息, 便于跨设备对比)
        summary = {
            "environment": self._env,
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "results": self.results,
        }
        with open(os.path.join(self.base_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


# ─── 使用示例 ───
def train_single_bnn(config, seed, exp_dir):
    """单个 BNN 训练函数 (适配 ExperimentRunner 接口)"""
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    
    # ... 数据准备 ...
    # ... 模型定义 ...
    # ... SVI 训练 (使用 config["hidden_dim"], config["lr"] 等) ...
    # ... 评估 ...
    
    metrics = {
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "coverage_95": coverage,
        "final_elbo": elbo_history[-1],
    }
    return metrics


# 运行网格搜索
runner = ExperimentRunner(base_dir="experiments/bnn_tuning")
param_grid = {
    "hidden_dim": [16, 32, 64],
    "lr": [0.005, 0.01, 0.02],
}
summary = runner.run_grid_search(train_single_bnn, param_grid, seeds=[42, 123])