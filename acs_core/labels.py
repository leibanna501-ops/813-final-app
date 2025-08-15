# -*- coding: utf-8 -*-
# acs_core/labels.py
import numpy as np, pandas as pd
from .config import LabelConfig

EPS = 1e-12

def build_weak_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """环境自适应弱标签（与你现实现保持一致的思想）"""
    close = df["close"]
    high_fwd = df["high"].shift(-1).rolling(cfg.horizon).max()
    low_fwd  = df["low"].shift(-1).rolling(cfg.horizon).min()
    ret_up = high_fwd/close - 1.0
    dd_min = low_fwd/close - 1.0

    vol20 = close.pct_change().rolling(20).std()
    vol_ref = vol20.rolling(250, min_periods=40).median()
    vol_scale = (vol20/(vol_ref+EPS)).clip(0.7, 1.5).fillna(1.0)

    # 一年位置分段
    def _rolling_rank(x, win=250):
        def _rank_last(a):
            if len(a)<=1 or np.all(np.isnan(a)): return np.nan
            last=a[-1]; arr=a[:-1]; arr=arr[~np.isnan(arr)]
            if len(arr)==0: return np.nan
            return (arr<last).sum()/max(len(arr),1)
        return x.rolling(win, min_periods=40).apply(_rank_last, raw=True)

    pos_1y = _rolling_rank(close, 250)
    bull = (pos_1y >= 0.60).astype(float)
    bear = (pos_1y <= 0.40).astype(float)

    up_adj = 1.0 - 0.05*bull + 0.05*bear
    dd_adj = 1.0 - 0.10*bull + 0.10*bear

    min_up_s = (cfg.min_up*vol_scale*up_adj).astype(float)
    max_dd_s = (cfg.max_dd*vol_scale*dd_adj).astype(float)

    label = ((ret_up >= min_up_s) & (dd_min >= -max_dd_s)).astype(float)
    label[(high_fwd.isna()) | (low_fwd.isna())] = np.nan
    return label
