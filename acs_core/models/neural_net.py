# -*- coding: utf-8 -*-
# acs_core/models/neural_net.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from acs_core.registries import register_model
from acs_core.types import FeatureName, ModelName  # ✅ 正确拼写

# 1) 简单三层 MLP（CPU/GPU 自适应）
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 2) 训练 + 预测函数（注册为模型）
@register_model(ModelName("neural_net"))
def train_and_predict_nn(df: pd.DataFrame, features: list[FeatureName]) -> pd.DataFrame:
    """
    使用简单全连接网络做三分类：标签取值为 {-1, 0, 1}。
    约定输入 df 至少包含：
      - 特征列：features 列表指向的列
      - 标签列：'label'（取值 -1/0/1）
    返回：
      - 概率表 DataFrame，列名为 [-1, 0, 1]，与项目其它头保持一致。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据准备 ---
    X = df[features].to_numpy(dtype=float)
    # 标签期望是 -1/0/1；CrossEntropyLoss 需要 [0..C-1]，所以做平移映射
    y_raw = df["label"].astype(int).to_numpy()
    y_mapped = np.clip(y_raw + 1, 0, 2)  # -1->0, 0->1, 1->2

    # --- 标准化（神经网络对尺度敏感）---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 转张量并放置到设备
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_mapped, dtype=torch.long, device=device)

    # --- 模型定义 ---
    input_size = X.shape[1]
    hidden_size = 64     # 可调
    num_classes = 3      # ✅ 三分类：{-1,0,1}
    learning_rate = 1e-3
    num_epochs = 50      # 可调；CPU 环境可以先小一点，比如 20

    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 训练 ---
    model.train()
    for _ in range(num_epochs):
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- 预测（概率）---
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.softmax(logits, dim=1).cpu().numpy()

    # 列名映射回 {-1,0,1} 的自然语义
    prob_df = pd.DataFrame(prob, index=df.index, columns=[-1, 0, 1])
    return prob_df
