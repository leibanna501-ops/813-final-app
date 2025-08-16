# -*- coding: utf-8 -*-
# acs_core/config.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    # 数据获取配置（可切换 akshare 或自有数据源）
    provider: str = "akshare"  # 目前仅支持 akshare
    adjust: str = "qfq"  # 复权模式
    feature_lag: int = 1  # 评估/部署统一滞后（EOD→T+1）


@dataclass
class LabelConfig:
    horizon: int = 20
    min_up: float = 0.06
    max_dd: float = 0.12


@dataclass
class TrendHeadConfig:
    n_splits: int = 5
    C: float = 1.5
    block: int = 120
    horizon_gap: int = 60
    min_rows_oof: int = 200
    min_rows_insample: int = 40


@dataclass
class ReactiveHeadConfig:
    C: float = 3.0
    lambda_decay: float = 0.03  # 最近样本权重大
    use: bool = True


@dataclass
class BlendConfig:
    k_by_vol: float = 6.0  # ATR 门控强度（越大越偏向反应头）
    sell_thr: float = 0.45  # 概率止损阈值
    atr_k: float = 1.0  # YTD-AVWAP - k*ATR 的硬止损线


@dataclass
class FeatureConfig:
    # 启用的特征名列表（来自注册表）
    enabled: List[str] = field(
        default_factory=lambda: [
            "pos_1y",
            "vol_drop",
            "ma_converge",
            "slope_mid",
            "box_tight_z",
            "vwap_dist",
            "signed_vol_z",
            "amihud_z",
            "kyle_z",
            "ud_amt_ratio_z",
            "below_avwap_evt",
            "cooldown_penalty",
            "rs_z",
            "mkt_ma200_slope",
            "mkt_vol_z",
            "regime_prob",
        ]
    )
    # 为神经网络单独配置的特征集，移除了长周期特征以保证数据量
    enabled_for_nn: List[str] = field(
        default_factory=lambda: [
            # 使用最小安全特征集进行调试
            "slope_mid",        # ~30天回看
            "below_avwap_evt",  # 内部 fillna(0)
            "cooldown_penalty", # 内部 fillna(0)
        ]
    )
    enabled_fast: List[str] = field(
        default_factory=lambda: [
            "ret_1d",
            "ret_3d",
            "atr14",
            "vol_z5",
            "ud_vol_ratio_5",
            "dist_ytd_avwap",
        ]
    )


@dataclass
class OutputConfig:
    root: str = "./artifacts"  # 输出根目录
    save_csv: bool = True
    save_parquet: bool = False
    save_series_json: bool = True
    save_metrics_json: bool = True
    run_subdir: Optional[str] = None  # 不设则按 symbol 自动创建


@dataclass
class PipelineConfig:
    # 注意：以下所有默认值都用 default_factory，避免“可变默认值”报错
    data: "DataConfig" = field(default_factory=DataConfig)
    labels: "LabelConfig" = field(default_factory=LabelConfig)
    features: "FeatureConfig" = field(default_factory=FeatureConfig)
    trend: "TrendHeadConfig" = field(default_factory=TrendHeadConfig)
    reactive: "ReactiveHeadConfig" = field(default_factory=ReactiveHeadConfig)
    blend: "BlendConfig" = field(default_factory=BlendConfig)
    output: "OutputConfig" = field(default_factory=OutputConfig)
    # 字典本身也必须用 default_factory
    extras: Dict[str, Any] = field(default_factory=dict)
