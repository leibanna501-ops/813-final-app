# -*- coding: utf-8 -*-
# acs_core/pipeline.py
# 统一管线：取数→特征→标签→趋势头→（可选）反应头→融合→止损→导出标准产物
import os, json
import pandas as pd
from typing import Dict, Any
from .config import PipelineConfig
from .io_utils import write_csv, write_json, ensure_dir
from .data import fetch_daily
from .labels import build_weak_labels
from .registries import FEATURE_REGISTRY, FEATURE_FAST_REGISTRY
from .models.trend import fit_trend_head
from .models.reactive import fit_reactive_head
from .models.blend import blend_prob_with_vol, force_exit_rule

def _series_spec_dict(series: Dict[str, pd.Series]) -> Dict[str, Any]:
    """把可视化序列打包成可供 UI 消费的规范：名称→元数据"""
    spec = {}
    for name, s in series.items():
        spec[name] = {
            "min": float(pd.to_numeric(s, errors="coerce").min()) if s.notna().any() else None,
            "max": float(pd.to_numeric(s, errors="coerce").max()) if s.notna().any() else None,
            "kind": "line",   # 现阶段统一按线条，可扩展 "bar"/"area"
            "scale": "auto",  # 或 "0-1" 等
        }
    return spec

def compute(symbol: str, start, end, cfg: PipelineConfig) -> Dict[str, Any]:
    """主入口：返回 {df, paths, metrics, series}；同时根据 cfg.output 决定是否落盘。"""
    # 1) 取数
    mkt = fetch_daily(symbol, start, end, cfg.data)

    # 2) 特征（主 + 快）
    feat_main = {}
    for name in cfg.features.enabled:
        fn = FEATURE_REGISTRY.get(name)
        if fn is None: continue
        val = fn(mkt)
        feat_main[name] = val
    feat_df = pd.DataFrame(feat_main, index=mkt.index)

    feat_fast = {}
    for name in cfg.features.enabled_fast:
        fn = FEATURE_FAST_REGISTRY.get(name)
        if fn is None: continue
        val = fn(mkt)
        feat_fast[name] = val
    fast_df = pd.DataFrame(feat_fast, index=mkt.index)

    # 3) 标签
    label = build_weak_labels(mkt, cfg.labels)

    # 4) 趋势头（OOF+校准+尾端）
    trend = fit_trend_head(
        feat_df=feat_df, feat_cols=list(feat_df.columns), label=label,
        n_splits=cfg.trend.n_splits, C=cfg.trend.C,
        horizon=cfg.trend.horizon_gap, block=cfg.trend.block,
        min_rows_oof=cfg.trend.min_rows_oof, min_rows_insample=cfg.trend.min_rows_insample,
        feature_lag=cfg.data.feature_lag
    )
    prob_trend = trend.proba

    # 5) 反应头（可选）
    if cfg.reactive.use and fast_df.shape[1]>0:
        prob_react = fit_reactive_head(fast_df, label, time_index=feat_df.index,
                                       C=cfg.reactive.C, lambda_decay=cfg.reactive.lambda_decay)
    else:
        prob_react = prob_trend.copy()

    # 6) 融合 + 止损
    atr_series = fast_df["atr14"] if "atr14" in fast_df.columns else mkt["close"].rolling(14).std()
    prob = blend_prob_with_vol(prob_trend, prob_react, atr_series, k=cfg.blend.k_by_vol)
    sell_signal = force_exit_rule(mkt[["close","high","low","volume"]].loc[feat_df.index],
                                  prob, sell_thr=cfg.blend.sell_thr, atr_k=cfg.blend.atr_k)

    # 7) 预计算所有可视化序列
    price_ma_windows = [5, 10, 20, 60, 120, 200]
    prob_ma_windows = [5, 10, 20, 30, 60, 120]
    
    series_dict = {"acs_prob": prob}
    series_dict.update({f"price_ma_{w}": mkt["close"].rolling(w).mean() for w in price_ma_windows})
    series_dict.update({f"prob_ma_{w}": prob.rolling(w).mean() for w in prob_ma_windows})
    
    series_df = pd.DataFrame(series_dict, index=prob.index)

    # 8) 汇总结果（主 CSV）
    base_mkt = mkt.reindex(prob.index)[["open","high","low","close","volume","turnover"]]
    labels_df = pd.DataFrame({"label_weak": label.reindex(prob.index), "sell_signal": sell_signal}, index=prob.index)
    out = pd.concat([base_mkt, labels_df, series_df], axis=1).reset_index().rename(columns={"index":"date"})

    # 9) 指标与元数据
    metrics = {
        "auc_raw": trend.auc_raw, "auc_cal": trend.auc_cal,
        "mode": trend.mode, "calibration": trend.calibration
    }
    series_meta = {
        "prob": prob,
        **{k: v for k, v in series_df.items() if 'prob_ma' in k},
        **{k: v for k, v in series_df.items() if 'price_ma' in k},
        "dist_ytd_avwap": fast_df.get("dist_ytd_avwap", pd.Series(index=prob.index)),
    }

    # 10) 落盘
    paths = {}
    run_dir = os.path.join(cfg.output.root, cfg.output.run_subdir or f"{symbol}")
    ensure_dir(run_dir)

    if cfg.output.save_csv:
        p = os.path.join(run_dir, f"{symbol}_acs_prob.csv")
        write_csv(out, p); paths["csv"] = p
    if cfg.output.save_metrics_json:
        p = os.path.join(run_dir, f"{symbol}_metrics.json")
        write_json(metrics, p); paths["metrics"] = p
    if cfg.output.save_series_json:
        spec = _series_spec_dict(series_meta)
        p = os.path.join(run_dir, f"{symbol}_series.json")
        write_json({"series": list(series_meta.keys()), "spec": spec}, p); paths["series"] = p

    return {"df": out, "paths": paths, "metrics": metrics, "series": series_meta}
