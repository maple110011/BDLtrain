# BDLtrain

给 Copilot agent 用的贝叶斯深度学习训练 Skill。让 agent 写出来的 BNN 训练代码不再是玩具，而是带着 checkpoint、早停、诊断、校准这些实际训练该有的东西。

## 能干什么

写贝叶斯神经网络训练代码时，agent 会自动注意以下问题：
- 数据该怎么标准化、先验怎么选
- 训练中途断了能不能接着跑（checkpoint）
- 模型到底收敛了没（R-hat / ELBO 诊断）
- 训出来的不确定性靠不靠谱（校准、PPC）
- 学习率该不该降、什么时候该停（LR 调度、early stopping）

## 怎么用

把这个文件夹丢到 VS Code 能识别的 skills 路径下：

```
.github/skills/BDLtrain/        # 项目级别
~/.copilot/skills/BDLtrain/     # 全局
```

然后在 Copilot Chat 里 `/bayes-deeplearning` 就能调用了。或者在对话里提到 BNN、SVI、MCMC 这些词，agent 也会自动加载。

## 里面有什么

```
├── SKILL.md                  # 主文件，agent 优先读这个
├── references/               # 参考文档，用到才加载
│   ├── core-training.md      # 数据、先验、架构、稳定性这些基础的东西
│   ├── training-infra.md     # checkpoint、早停、日志、异常恢复
│   ├── bdl-methods.md        # MCMC / SVI / MC Dropout 等方法怎么选
│   ├── bdl-frameworks.md     # Pyro / NumPyro / PyMC 框架对比
│   ├── diagnostics.md        # 收敛诊断、校准、排查表
│   └── model-architecture.md # 几种 BNN 结构的写法
└── assets/templates/         # 能直接跑的代码模板
    ├── svi_regression.py     # SVI 回归，checkpoint/早停全都有
    ├── mcmc_regression.py    # MCMC 回归，带后验持久化
    ├── mc_dropout.py         # MC Dropout 快速原型
    ├── deep_ensembles.py     # Deep Ensembles
    └── experiment_runner.py  # 网格搜索和实验管理
```

## 依赖

代码模板基于 Pyro + PyTorch，本地环境已有 Pyro 1.9.1 + PyTorch 2.12.0。如果用 NumPyro 或 TFP 需要另外装。
