# -*- coding: utf-8 -*-
# acs_core/features/builtins.py
import numpy as np, pandas as pd
from ..registries import register_feature

EPS = 1e-12


def _zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=max(10, win // 5)).mean()
    s = x.rolling(win, min_periods=max(10, win // 5)).std()
    return (x - m) / (s.replace(0, np.nan))


def _rolling_rank(x: pd.Series, win: int) -> pd.Series:
    def _rank_last(a: np.ndarray):
        if len(a) <= 1 or np.all(np.isnan(a)):
            return np.nan
        last = a[-1]
        arr = a[:-1]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        return (arr < last).sum() / max(len(arr), 1)

    return x.rolling(win, min_periods=max(20, win // 4)).apply(_rank_last, raw=True)


def _safe_div(a, b):
    return a / (b.replace(0, np.nan))


def _avwap_ytd(close, volume):
    years = pd.DatetimeIndex(close.index).year
    return _safe_div(
        (close * volume).groupby(years).cumsum(), volume.groupby(years).cumsum()
    )


@register_feature("pos_1y")
def f_pos_1y(df: pd.DataFrame) -> pd.Series:
    return _rolling_rank(df["close"], 250)


@register_feature("vol_drop")
def f_vol_drop(df):
    vol20 = df["close"].pct_change().rolling(20).std()
    return -_zscore(vol20, 60)


@register_feature("ma_converge")
def f_ma_converge(df):
    close = df["close"]
    fast = close.rolling(10).mean()
    mid = close.rolling(20).mean()
    slow = close.rolling(60).mean()
    return -(
        _zscore((fast - mid).abs() / close, 60)
        + _zscore((mid - slow).abs() / close, 60)
    )


@register_feature("slope_mid")
def f_slope_mid(df):
    mid = df["close"].rolling(20).mean()
    return (mid - mid.shift(10)) / (mid.shift(10) + EPS)


@register_feature("ud_amt_ratio_z")
def f_ud_amt_ratio_z(df):
    up = df["close"] >= df["close"].shift(1)
    up_m = df["amount"].where(up, np.nan).rolling(40).median()
    dn_m = df["amount"].where(~up, np.nan).rolling(40).median()
    rob = _safe_div(up_m, dn_m + EPS)
    return _zscore(np.log1p(rob), 60)


@register_feature("signed_vol_z")
def f_signed_vol_z(df):
    open_, high, low, close, vol = (
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
    )
    tr_raw = (high - low).abs()
    tr_alt = pd.concat(
        [
            (close - close.shift(1)).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr_raw.where(tr_raw > tr_alt, tr_alt).fillna(tr_raw)
    dirc = ((close - open_) / (tr + EPS)).clip(-3, 3)
    dirc = np.tanh(dirc)
    return _zscore(dirc * vol, 60)


@register_feature("amihud_z")
def f_amihud_z(df):
    ret = df["close"].pct_change().fillna(0.0)
    amihud = (ret.abs() / (df["amount"].replace(0, np.nan))).rolling(20).mean()
    return -_zscore(np.log(amihud.replace(0, np.nan)), 60)


@register_feature("kyle_z")
def f_kyle_z(df):
    ret = df["close"].pct_change().fillna(0.0)
    open_, high, low, close, vol = (
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
    )
    tr_raw = (high - low).abs()
    tr_alt = pd.concat(
        [
            (close - close.shift(1)).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr_raw.where(tr_raw > tr_alt, tr_alt).fillna(tr_raw)
    dirc = ((close - open_) / (tr + EPS)).clip(-3, 3)
    dirc = np.tanh(dirc)
    s_vol = dirc * vol
    cov = ret.rolling(60).cov(s_vol)
    var = s_vol.rolling(60).var()
    lam = cov / (var + EPS)
    return -_zscore(lam, 60)


@register_feature("vwap_dist")
def f_vwap_dist(df):
    vwap = _avwap_ytd(df["close"], df["volume"])
    return (df["close"] - vwap) / (vwap + EPS)


@register_feature("box_tight_z")
def f_box_tight_z(df):
    width = (df["high"].rolling(40).max() - df["low"].rolling(40).min()) / (
        df["close"] + EPS
    )
    return -_zscore(width, 60)


@register_feature("below_avwap_evt")
def f_below_avwap_evt(df):
    # 简化：从你现实现逻辑抽象，事件后相对 AVWAP 的位置（空=0）
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    evt = (close.pct_change() >= 0.09) & (((close - low) / (high - low + EPS)) > 0.70)
    seg = evt.astype(int).cumsum()
    pv_cum = (close * vol).groupby(seg).cumsum()
    v_cum = vol.groupby(seg).cumsum()
    avwap_evt = pv_cum / (v_cum + EPS)
    avwap_evt[seg == 0] = np.nan
    out = ((close - avwap_evt) / (avwap_evt + EPS)).clip(-1, 1)
    return out.fillna(0.0)


@register_feature("cooldown_penalty")
def f_cooldown_penalty(df):
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    evt = (close.pct_change() >= 0.09) & (((close - low) / (high - low + EPS)) > 0.70)
    seg = evt.astype(int).cumsum()
    days = seg.groupby(seg).cumcount()
    days[seg == 0] = np.nan
    pv_cum = (close * vol).groupby(seg).cumsum()
    v_cum = vol.groupby(seg).cumsum()
    avwap_evt = pv_cum / (v_cum + EPS)
    avwap_evt[seg == 0] = np.nan
    below = ((close - avwap_evt) / (avwap_evt + EPS)) < -0.01
    penalty = ((days <= 20) & below).astype(float)
    return -penalty.fillna(0.0)


@register_feature("rs_z")
def f_rs_z(df):
    rs = df["close"] / (df["close"].rolling(60).mean() + EPS)
    return _zscore(rs, 60)


@register_feature("mkt_ma200_slope")
def f_mkt_ma200_slope(df):
    ma200 = df["close"].rolling(200).mean()
    return (ma200 - ma200.shift(20)) / (ma200.shift(20) + EPS)


@register_feature("mkt_vol_z")
def f_mkt_vol_z(df):
    vol20 = df["close"].pct_change().rolling(20).std()
    return _zscore(vol20, 60)


@register_feature("regime_prob")
def f_regime_prob(df):
    # 组合：上行斜率↑、一年位置↑、高波动↓
    ma200 = df["close"].rolling(200).mean()
    slope = (ma200 - ma200.shift(20)) / (ma200.shift(20) + EPS)
    pos1y = _rolling_rank(df["close"], 250).fillna(0.5) - 0.5
    volz = _zscore(df["close"].pct_change().rolling(20).std(), 60).fillna(0.0)
    score = 1.2 * slope.fillna(0) + 0.6 * pos1y - 0.3 * volz
    return 1.0 / (1.0 + np.exp(-score))
