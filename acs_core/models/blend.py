# -*- coding: utf-8 -*-
# acs_core/models/blend.py
import numpy as np, pandas as pd

def blend_prob_with_vol(prob_trend: pd.Series, prob_react: pd.Series, atr_series: pd.Series, k=6.0):
    """ATR 越高越偏向反应头：w=sigmoid(k*atr_z)；prob=(1-w)*trend+w*react"""
    atr_z = (atr_series - atr_series.rolling(60, min_periods=20).mean())/(
        atr_series.rolling(60, min_periods=20).std() + 1e-9
    )
    w = 1.0/(1.0+np.exp(-k*atr_z))
    w = w.clip(0.0, 1.0).fillna(0.0)
    return (1.0-w)*prob_trend + w*prob_react

def force_exit_rule(df: pd.DataFrame, prob: pd.Series, sell_thr=0.45, atr_k=1.0):
    """硬止损：价格 < YTD-AVWAP - k*ATR 或 概率<阈值"""
    tr = df.assign(h_l=df["high"]-df["low"],
                   h_pc=(df["high"]-df["close"].shift(1)).abs(),
                   l_pc=(df["low"]-df["close"].shift(1)).abs())
    atr14 = tr[["h_l","h_pc","l_pc"]].max(axis=1).rolling(14, min_periods=5).mean()
    vol = df["volume"].astype(float)
    y = df.index.to_series().dt.year
    avwap = ((df["close"]*vol).groupby(y).cumsum()/(vol.groupby(y).cumsum().replace(0,np.nan))).ffill()
    hard = df["close"] < (avwap - atr_k*atr14)
    rule = (prob < sell_thr)
    return (hard | rule).astype(int)
