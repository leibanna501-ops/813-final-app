# -*- coding: utf-8 -*-
# acs_core/features/fast.py
import numpy as np, pandas as pd
from ..registries import register_feature


@register_feature("ret_1d", fast=True)
def f_ret_1d(df):
    return df["close"].pct_change(1).shift(1)


@register_feature("ret_3d", fast=True)
def f_ret_3d(df):
    return df["close"].pct_change(3).shift(1)


@register_feature("atr14", fast=True)
def f_atr14(df):
    tr = (df["high"] - df["low"]).fillna(0)
    tr1 = (df["high"] - df["close"].shift(1)).abs()
    tr2 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=5).mean().shift(1)


@register_feature("vol_z5", fast=True)
def f_vol_z5(df):
    vol = df["volume"].astype(float)
    return ((vol - vol.rolling(5, min_periods=3).mean()) / (
        vol.rolling(5, min_periods=3).std() + 1e-9
    )).shift(1)


@register_feature("ud_vol_ratio_5", fast=True)
def f_ud_vol_ratio_5(df):
    vol = df["volume"].astype(float)
    up_vol = vol.where(df["close"] >= df["close"].shift(1), 0.0)
    dn_vol = vol.where(df["close"] < df["close"].shift(1), 0.0)
    ratio = (up_vol.rolling(5, min_periods=3).sum() + 1.0) / (
        dn_vol.rolling(5, min_periods=3).sum() + 1.0
    )
    return np.log(ratio).shift(1)


@register_feature("dist_ytd_avwap", fast=True)
def f_dist_ytd_avwap(df):
    year = df.index.to_series().dt.year
    vol = df["volume"].astype(float)
    cum_vol = vol.groupby(year).cumsum()
    avwap = (df["close"] * vol).groupby(year).cumsum() / (cum_vol.replace(0, np.nan))
    return (df["close"] / avwap - 1.0).shift(1)
