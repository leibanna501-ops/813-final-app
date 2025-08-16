# -*- coding: utf-8 -*-
"""
更可靠的二分类神经网络（残差MLP版）：
- 时间序列交叉验证(TimeSeriesSplit)
- 残差块 + BatchNorm + Dropout
- AdamW + 余弦退火学习率
- 早停(EarlyStopping)
- 温度校准(Temperature Scaling)
返回：按索引对齐的概率 DataFrame，列为 [0, 1]
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ===== 模型结构：残差 MLP（更深但稳定）=====
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)      # BatchNorm 稳定训练
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = F.gelu(self.fc1(out))
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.fc2(out)
        return F.gelu(out + identity)       # 残差连接：更易训练

class TabResNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_blocks: int = 4, num_classes: int = 2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout=dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)  # 输出 logits（未过softmax）

# ===== 实用工具：早停、温度校准、训练一步 =====
class EarlyStopper:
    """早停：若验证集 loss 连续 patience 轮未提升则停"""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0

    def step(self, val_loss) -> bool:
        if val_loss + self.min_delta < self.best:
            self.best = val_loss
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience

@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, in loader:
        xb = xb.to(device)
        outs.append(model(xb).cpu().numpy())
    return np.vstack(outs)

def learn_temperature(val_logits: torch.Tensor, val_y: torch.Tensor, max_iter=200) -> float:
    """
    温度校准：优化标量 T 以最小化验证集 NLL
    - logits_new = logits / T
    """
    T = torch.nn.Parameter(torch.tensor(1.0))  # 初始温度为1
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        # 为保证T>0，用softplus约束；也可以直接对logT优化
        T_pos = F.softplus(T)
        loss = F.cross_entropy(val_logits / T_pos, val_y)
        loss.backward()
        return loss

    opt.step(_closure)
    T_final = F.softplus(T).item()
    return float(T_final)

def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    in_dim: int, hidden=128, num_blocks=4,
    epochs=120, batch_size=512, lr=3e-3, weight_decay=1e-3,
    dropout=0.2, label_smoothing=0.05, device=None
) -> Tuple[np.ndarray, float, float]:
    """
    训练单个时间折：返回（校准后验证集概率，最佳val_loss，对应温度）
    """
    device = device or torch.device("cpu")
    model = TabResNet(in_dim, hidden, num_blocks, num_classes=2, dropout=dropout).to(device)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32))
    train_ys = torch.tensor(y_tr.astype(int), dtype=torch.long)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), train_ys),
                              batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 余弦退火学习率（单周期）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    stopper = EarlyStopper(patience=15, min_delta=1e-4)

    best_state = None
    best_val_loss = np.inf

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 梯度裁剪
            opt.step()
        sched.step()

        # 简单用训练末期模型在验证集评估
        with torch.no_grad():
            model.eval()
            val_logits = []
            for (xv,) in val_loader:
                xv = xv.to(device)
                val_logits.append(model(xv))
            val_logits = torch.cat(val_logits, dim=0)
            val_loss = F.cross_entropy(val_logits, torch.tensor(y_val.astype(int), dtype=torch.long, device=device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_loss):
            break

    # 恢复最佳权重
    model.load_state_dict({k: v for k, v in best_state.items()})
    model.eval()

    # 温度校准（在验证集上学习 T）
    with torch.no_grad():
        val_logits = []
        for (xv,) in val_loader:
            xv = xv.to(device)
            val_logits.append(model(xv))
        val_logits = torch.cat(val_logits, dim=0)
    T = learn_temperature(val_logits, torch.tensor(y_val.astype(int), dtype=torch.long))
    probs_val = F.softmax(val_logits / T, dim=1).cpu().numpy()
    return probs_val, best_val_loss, T

def train_and_predict_nn_v2(
    df: pd.DataFrame,
    features: List[str],
    n_splits: int = 5,
    epochs: int = 120,
    hidden: int = 128,
    num_blocks: int = 4,
    batch_size: int = 512,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.2,
    label_smoothing: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    入口函数（与旧版保持风格一致）：
    参数：
      - df：包含特征列 + 'label'(0/1) 的 DataFrame
      - features：特征列名列表
    返回：
      - 概率 DataFrame（两列：0、1），索引与 df 对齐，CV 预测填充，其余为 NaN（可前向填充）
    """
    assert "label" in df.columns, "df 必须包含 label 列(0/1)"

    # 防御：如果样本数不足以进行 n_splits 折交叉验证，则直接返回空结果
    if df.shape[0] < n_splits:
        return pd.DataFrame(index=df.index, columns=[0, 1], dtype=float)

    X_all = df[features].values.astype(np.float32)
    y_all = df["label"].values.astype(int)
    idx_all = df.index

    # 结果容器（按索引对齐）
    prob = pd.DataFrame(index=idx_all, columns=[0, 1], dtype=float)

    # 时间序列交叉验证（不打乱，严格过去→未来）
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 逐折训练→验证，生成 out-of-fold 概率
    for fold, (tr, te) in enumerate(tscv.split(X_all)):
        # === 仅用训练折拟合 scaler（防泄漏）===
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr])
        X_te = scaler.transform(X_all[te])

        # === 训练单折模型（含校准）===
        probs_val, val_loss, T = train_one_fold(
            X_tr, y_all[tr], X_te, y_all[te],
            in_dim=X_tr.shape[1], hidden=hidden, num_blocks=num_blocks,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            dropout=dropout, label_smoothing=label_smoothing,
            device=torch.device("cpu")  # 无独显环境，强制CPU
        )

        # === 写回该折的验证集预测 ===
        prob.loc[idx_all[te], 0] = probs_val[:, 0]
        prob.loc[idx_all[te], 1] = probs_val[:, 1]

    return prob
