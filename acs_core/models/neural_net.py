
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from acs_core.registries import register_model
from acs_core.types import FeatrueName, ModelName

# 1. 定义神经网络结构
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
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

# 2. 编写模型训练和预测的函数，并注册
@register_model(ModelName("neural_net"))
def train_and_predict_nn(df: pd.DataFrame, features: list[FeatrueName]):
    """
    使用一个简单的神经网络模型进行训练和预测。
    """
    # --- 数据准备 ---
    X = df[features].values
    y = df["label"].values

    # **非常重要**: 神经网络对特征的尺度非常敏感，必须进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 将数据转换为PyTorch Tensors
    X_tensor = torch.FloatTensor(X_scaled)
    # PyTorch的CrossEntropyLoss期望的标签是0, 1, 2...，而我们的是-1, 0, 1，所以需要+1处理
    y_tensor = torch.LongTensor(y) + 1

    # --- 模型定义 ---
    input_size = X.shape[1]
    hidden_size = 64  # 隐藏层大小，可以调整
    num_classes = 3   # 类别数 (-1, 0, 1)
    learning_rate = 0.001
    num_epochs = 50   # 训练轮数，可以调整

    model = SimpleMLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 训练循环 ---
    model.train()
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- 预测 ---
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        # 使用softmax将输出转换为概率
        probabilities = nn.functional.softmax(logits, dim=1)

    # 将结果转为pandas DataFrame，并保持和项目其他模型一致的格式
    prob_df = pd.DataFrame(probabilities.numpy(), index=df.index, columns=[-1, 0, 1])
    
    return prob_df
