# Template 4: Deep Ensembles 实现
# Extracted from code-templates.md

"""
Deep Ensembles —— 通过多模型集成获得不确定性
注意: 训练完成后建议记录运行环境以便跨设备对比
"""
import torch
import torch.nn as nn
import platform, json

def train_ensemble(model_class, model_kwargs, X_train, y_train,
                   num_models=5, epochs=100, lr=0.01):
    """
    训练深度集成
    ⚠️ 每个模型需要不同的随机种子以确保多样性
    """
    models = []
    for i in range(num_models):
        torch.manual_seed(i * 42 + 7)
        model = model_class(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_train).squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        models.append(model)
        print(f"Ensemble 成员 {i+1}/{num_models} 训练完成")
    
    return models

def ensemble_predict(models, X):
    """集成预测: 均值 = 均值, 不确定性 = 标准差"""
    with torch.no_grad():
        preds = torch.stack([m(X).squeeze() for m in models])
    return preds.mean(dim=0), preds.std(dim=0)

def save_environment(save_path="environment.json"):
    """训练结束后保存硬件信息"""
    env = {
        "platform": platform.platform(),
        "cpu_model": platform.processor() or "Unknown",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "torch_version": torch.__version__,
    }
    json.dump(env, open(save_path, "w"), indent=2, default=str)