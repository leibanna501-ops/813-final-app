# -*- coding: utf-8 -*-
# acs_core/models/trend.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# Scikit-learn imports will be inside the function to match original file structure
from sklearn.base import BaseEstimator, TransformerMixin

EPS = 1e-12

@dataclass
class TrendTrainResult:
    proba: pd.Series
    auc_raw: float | None
    auc_cal: float | None
    mode: str
    calibration: str
    tail: pd.Series

class FoldSafeSelector(BaseEstimator, TransformerMixin):
    """
    每折内的安全特征筛选：
    - 丢掉“当折训练集里整列都是 NaN”的列（避免 SimpleImputer 警告）
    - 丢掉“当折训练集里方差为 0”的列（避免无效特征）
    仅用训练折数据拟合，不泄漏。
    """
    def __init__(self, drop_allnan=True, drop_constant=True):
        self.drop_allnan = drop_allnan
        self.drop_constant = drop_constant
        self.keep_idx_ = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        keep = np.ones(Xdf.shape[1], dtype=bool)
        if self.drop_allnan:
            keep &= Xdf.notna().any(axis=0).values
        if self.drop_constant:
            var = Xdf.var(axis=0, skipna=True).values
            keep &= np.nan_to_num(var) > 0.0
        self.keep_idx_ = np.where(keep)[0]
        if self.keep_idx_.size == 0:
            self.keep_idx_ = np.arange(Xdf.shape[1])
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X)
        return Xdf.iloc[:, self.keep_idx_].values

def fit_trend_head(
    feat_df: pd.DataFrame,
    feat_cols: list,
    label: pd.Series,
    *,
    n_splits=5,
    C=1.5,
    horizon=60,
    block=120,
    min_rows_oof=200,
    min_rows_insample=40,
    feature_lag=1
) -> TrendTrainResult:
    """
    完全基于 1.py 中的 fit_probability 函数重写，确保算法一致性。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from math import floor

    metrics = {"auc_raw": None, "auc_cal": None, "mode": "logreg_oof+calib+tail", "calibration": "none"}
    
    X_all = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    if feature_lag > 0:
        X_all = X_all.shift(feature_lag)

    valid_row = X_all.notna().any(axis=1)
    mask_lbl = label.notna() & valid_row

    if mask_lbl.sum() < min_rows_insample:
        Z = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan)
        Z = (Z - Z.mean()) / (Z.std() + EPS)
        
        def _col_or_zero(df, name):
            return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

        lin = (
            0.9 * _col_or_zero(Z, "signed_vol_z").fillna(0)
            + 0.8 * _col_or_zero(Z, "amihud_z").fillna(0)
            + 0.8 * _col_or_zero(Z, "kyle_z").fillna(0)
            + 0.6 * _col_or_zero(Z, "ud_amt_ratio_z").fillna(0)
            + 0.5 * _col_or_zero(Z, "box_tight_z").fillna(0)
            + 0.4 * _col_or_zero(Z, "ma_converge").fillna(0)
            + 0.3 * _col_or_zero(Z, "vol_drop").fillna(0)
            + 0.2 * _col_or_zero(Z, "vwap_dist").fillna(0)
            + 0.2 * _col_or_zero(Z, "slope_mid").fillna(0)
            + 0.1 * _col_or_zero(Z, "pos_1y").fillna(0)
            + 0.2 * _col_or_zero(Z, "regime_prob").fillna(0)
            + 0.2 * _col_or_zero(Z, "mkt_ma200_slope").fillna(0)
        )
        proba = 1.0 / (1.0 + np.exp(-lin))
        return TrendTrainResult(
            proba=proba, auc_raw=None, auc_cal=None,
            mode="sigmoid_combo(fallback)", calibration="none", tail=proba
        )

    X_l = X_all.loc[mask_lbl].copy()
    y_l = label.loc[mask_lbl].astype(float).copy()
    n_train = len(X_l)

    pipe = Pipeline(steps=[
        ("foldsafe", FoldSafeSelector()),
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, C=C, class_weight="balanced", random_state=42))
    ])

    block_eff = int(min(block, max(30, n_train // 6)))
    n_splits_eff = int(min(n_splits, max(3, n_train // max(80, block_eff))))
    
    MIN_TEST_FOLD = max(40, min_rows_insample)
    gap_eff = int(horizon)

    tscv, use_manual_purge = None, False
    ns_try, gap_try, made = n_splits_eff, gap_eff, False

    def _try_make_tscv(ns, gap_val, ts, n, manual_ok=True):
        try:
            tscv = TimeSeriesSplit(n_splits=ns, gap=gap_val, test_size=ts)
            _ = list(tscv.split(np.arange(n)))
            return tscv, False
        except TypeError:
            try:
                tscv = TimeSeriesSplit(n_splits=ns, gap=gap_val)
                _ = list(tscv.split(np.arange(n)))
                return tscv, False
            except TypeError:
                if not manual_ok: raise
                tscv = TimeSeriesSplit(n_splits=ns)
                _ = list(tscv.split(np.arange(n)))
                return tscv, True

    while ns_try >= 2:
        ts_try = max(MIN_TEST_FOLD, n_train // (ns_try + 2))
        try:
            tscv, use_manual_purge = _try_make_tscv(ns_try, gap_try, ts_try, n_train)
            made = True
            break
        except ValueError:
            ns_try -= 1
            continue
    
    if not made:
        ns_try = max(2, ns_try)
        while gap_try > max(10, horizon // 2):
            gap_try = max(10, gap_try // 2)
            try:
                ts_try = max(MIN_TEST_FOLD, n_train // (ns_try + 2))
                tscv, use_manual_purge = _try_make_tscv(ns_try, gap_try, ts_try, n_train)
                made = True
                break
            except Exception:
                continue

    if not made:
        # Fallback to full-train if TSCV fails
        pipe.fit(X_l.values, y_l.values)
        p_tail = pipe.predict_proba(X_all.values)[:, 1]
        proba_final = pd.Series(p_tail, index=feat_df.index)
        metrics["note"] = f"oof_fallback(n_splits=auto_failed, block={block_eff})"
        return TrendTrainResult(proba=proba_final, auc_raw=None, auc_cal=None, mode=metrics.get("mode"), calibration="none", tail=proba_final)

    groups = np.arange(n_train) // block_eff
    proba_oof_raw = pd.Series(np.nan, index=X_l.index)
    used_folds = 0

    for tr, te in tscv.split(np.arange(n_train)):
        if use_manual_purge:
            te_start = te.min()
            tr = tr[tr < te_start - gap_try]
        
        g_te = set(groups[te])
        tr = tr[~np.isin(groups[tr], list(g_te))]

        if (len(tr) < min_rows_insample) or (len(te) < 20) or (np.unique(y_l.iloc[tr].values).size < 2):
            continue

        pipe.fit(X_l.iloc[tr].values, y_l.iloc[tr].values)
        p = pipe.predict_proba(X_l.iloc[te].values)[:, 1]
        proba_oof_raw.iloc[te] = p
        used_folds += 1

    if used_folds == 0:
        pipe.fit(X_l.values, y_l.values)
        p_tail = pipe.predict_proba(X_all.values)[:, 1]
        proba_final = pd.Series(p_tail, index=feat_df.index)
        metrics["note"] = f"oof_fallback(n_splits={ns_try}, block={block_eff})"
        return TrendTrainResult(proba=proba_final, auc_raw=None, auc_cal=None, mode=metrics.get("mode"), calibration="none", tail=proba_final)

    m_valid = proba_oof_raw.notna().values
    p_valid = proba_oof_raw.values[m_valid]
    y_valid = y_l.values[m_valid]

    if p_valid.size >= 10 and np.unique(y_valid).size == 2:
        metrics["auc_raw"] = float(roc_auc_score(y_valid, p_valid))

    proba_oof_cal = proba_oof_raw.copy()
    cal_method = "none"
    iso, platt_model = None, None
    
    if (p_valid.size >= max(50, min_rows_oof)) and (np.unique(y_valid).size == 2):
        try:
            from sklearn.linear_model import LogisticRegression as _LR
            platt_model = _LR(max_iter=200)
            platt_model.fit(p_valid.reshape(-1, 1), y_valid)
            proba_oof_cal.values[m_valid] = platt_model.predict_proba(p_valid.reshape(-1, 1))[:, 1]
            cal_method = "platt"
        except Exception:
            platt_model = None
            try:
                from sklearn.isotonic import IsotonicRegression
                x_ext = np.r_[0.0, p_valid, 1.0]
                y_ext = np.r_[0.0, y_valid, 1.0]
                w_ext = np.r_[1e-3, np.ones_like(p_valid), 1e-3]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x_ext, y_ext, sample_weight=w_ext)
                proba_oof_cal.values[m_valid] = iso.transform(p_valid)
                cal_method = "isotonic_anchor"
            except Exception:
                iso = None
                cal_method = "none"

    metrics["calibration"] = cal_method
    if cal_method != "none":
        try:
            metrics["auc_cal"] = float(roc_auc_score(y_valid, proba_oof_cal.values[m_valid]))
        except Exception:
            metrics["auc_cal"] = None

    pipe.fit(X_l.values, y_l.values)
    p_tail = pipe.predict_proba(X_all.values)[:, 1]
    
    if cal_method == "platt" and platt_model is not None:
        p_tail = platt_model.predict_proba(p_tail.reshape(-1, 1))[:, 1]
    elif cal_method.startswith("isotonic") and iso is not None:
        p_tail = iso.transform(p_tail)

    proba_final = pd.Series(np.nan, index=feat_df.index)
    proba_final.loc[X_l.index] = proba_oof_cal
    need_tail = proba_final.isna().values
    proba_final.values[need_tail] = p_tail[need_tail]

    return TrendTrainResult(
        proba=proba_final,
        auc_raw=metrics["auc_raw"],
        auc_cal=metrics["auc_cal"],
        mode=metrics.get("mode", "logreg_oof+calib+tail"),
        calibration=cal_method,
        tail=pd.Series(p_tail, index=feat_df.index)
    )