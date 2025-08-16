# -*- coding: utf-8 -*-
# acs_core/types.py
# 说明：这些只是“类型别名”，运行时等同于 str，用于增加代码可读性。
# 对现有注册器（register_model(name: str)）完全兼容。

from typing import NewType

FeatureName = NewType("FeatureName", str)  # 特征名（实质是 str）
ModelName   = NewType("ModelName", str)    # 模型名（实质是 str）

# 简化方案（也可用）：如果不想依赖 NewType，直接：
# FeatureName = str
# ModelName   = str
