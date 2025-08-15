# -*- coding: utf-8 -*-
# acs_core/registries.py
# 统一的“可插拔注册表”：新增特征/模型，只需 @register_feature / @register_model
from typing import Callable, Dict

FEATURE_REGISTRY: Dict[str, Callable] = {}
FEATURE_FAST_REGISTRY: Dict[str, Callable] = {}
MODEL_REGISTRY: Dict[str, Callable] = {}  # name -> trainer/predictor 工厂

def register_feature(name: str, fast: bool = False):
    """装饰器：注册特征构造函数；函数签名必须为 f(df) -> pd.Series/pd.DataFrame"""
    def deco(fn):
        (FEATURE_FAST_REGISTRY if fast else FEATURE_REGISTRY)[name] = fn
        return fn
    return deco

def register_model(name: str):
    """装饰器：注册模型头构造器；签名 f(config_dict) -> object(须实现 fit/predict_proba)"""
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco
