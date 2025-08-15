# -*- coding: utf-8 -*-
# acs_core/io_utils.py
import os, json
import pandas as pd
from typing import Dict, Any

def ensure_dir(p: str):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str, encoding="utf-8-sig"):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding=encoding)

def write_json(data: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
