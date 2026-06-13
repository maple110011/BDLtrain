# Template 3: MC Dropout 实现
# Extracted from code-templates.md

"""
MC Dropout——轻量级不确定性估计
适用: 大模型、大数据、快速原型
"""
import torch.nn as nn

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.dropout_p = dropout_p
    
    def forward(self, x):
        return self.net(x)

def train_mc_dropout(model, X_train, y_train, epochs=100, lr=0.01):
    """标准 DNN 训练 (MSE loss)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred.squeeze(), y_train)
        loss.backward()
        optimizer.step()

def mc_predict(model, X, num_samples=100):
    """
    MC Dropout 预测
    ⚠️ model.train() 保持 dropout 开启
    """
    model.train()  # 关键!
    preds = torch.stack([model(X).squeeze().detach() for _ in range(num_samples)])
    return preds.mean(dim=0), preds.std(dim=0)