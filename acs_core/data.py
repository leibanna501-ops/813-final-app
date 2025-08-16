# -*- coding: utf-8 -*-
# acs_core/data.py
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, time
import pytz
from typing import Optional, List, Set
from .config import DataConfig

try:
    import akshare as ak
except Exception:
    ak = None

EPS = 1e-12

# --- 交易时间判断模块 ---
_trade_dates_cache: Set[datetime.date] = set()
_trade_dates_file = os.path.join("data", "trade_dates.json")

def _load_trade_dates():
    global _trade_dates_cache
    if _trade_dates_cache:
        return
    try:
        if os.path.exists(_trade_dates_file):
            with open(_trade_dates_file, "r") as f:
                dates_str = json.load(f)
                _trade_dates_cache = {datetime.strptime(d, "%Y-%m-%d").date() for d in dates_str}
                # 如果缓存的日历不是今年的，就认为它过期了
                if datetime.now().year not in {d.year for d in _trade_dates_cache}:
                    print("[Cache] Trading calendar is outdated, refetching.")
                    _trade_dates_cache = set()
    except Exception as e:
        print(f"[Cache] Failed to load trade dates cache: {e}")
        _trade_dates_cache = set()

    if not _trade_dates_cache:
        if ak is None: raise RuntimeError("akshare is required to fetch trade dates.")
        try:
            df = ak.tool_trade_date_hist_sina()
            dates = pd.to_datetime(df["trade_date"]).dt.date
            _trade_dates_cache = set(dates)
            os.makedirs(os.path.dirname(_trade_dates_file), exist_ok=True)
            with open(_trade_dates_file, "w") as f:
                json.dump([d.strftime("%Y-%m-%d") for d in sorted(list(_trade_dates_cache))], f)
            print(f"[Cache] Fetched and cached {len(_trade_dates_cache)} trade dates.")
        except Exception as e:
            print(f"[Error] Failed to fetch trade dates from akshare: {e}")

def _is_trade_time_now(tz_str="Asia/Shanghai") -> bool:
    """判断当前是否为A股交易时间"""
    _load_trade_dates()
    if not _trade_dates_cache:
        # 如果无法获取交易日历，为安全起见，默认在工作日的9-16点都实时获取
        now = datetime.now(pytz.timezone(tz_str))
        return now.weekday() < 5 and now.hour >= 9 and now.hour < 16

    now = datetime.now(pytz.timezone(tz_str))
    today = now.date()

    if today not in _trade_dates_cache:
        return False

    current_time = now.time()
    # 用户指定的交易时间段
    morning_start, morning_end = time(9, 0), time(12, 0)
    afternoon_start, afternoon_end = time(13, 0), time(15, 30)

    is_morning = morning_start <= current_time <= morning_end
    is_afternoon = afternoon_start <= current_time <= afternoon_end

    return is_morning or is_afternoon

# --- 数据获取函数 ---

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def _to_datestr_yyyymmdd(x) -> str:
    ts = pd.to_datetime(x) if not pd.isna(x) else pd.Timestamp.today()
    return ts.strftime("%Y%m%d")

def fetch_daily(symbol: str, start, end, cfg: DataConfig) -> pd.DataFrame:
    """拉 A 股日线，带交易时间感知的智能缓存。"""
    cache_dir = os.path.join("data", "raw_cache")
    os.makedirs(cache_dir, exist_ok=True)

    start_str = _to_datestr_yyyymmdd(start)
    end_str = _to_datestr_yyyymmdd(end)
    cache_file = f"{symbol}_{start_str}_{end_str}_{cfg.adjust}.parquet"
    cache_path = os.path.join(cache_dir, cache_file)

    end_date = pd.to_datetime(end).date()
    today = datetime.now(pytz.timezone("Asia/Shanghai")).date()

    # 缓存策略
    use_cache = True
    if end_date == today and _is_trade_time_now():
        use_cache = False # 结束日期是今天，并且在交易时段内，不使用缓存
    
    if use_cache and os.path.exists(cache_path):
        print(f"[Cache] Hit: {cache_file}")
        return pd.read_parquet(cache_path)

    # --- 网络获取 ---
    if use_cache is False:
        print(f"[Network] Fetching live data during trade hours for {symbol}")
    else:
        print(f"[Network] Fetching data for {symbol} from {start_str} to {end_str}")

    if cfg.provider != "akshare": raise NotImplementedError("仅支持 akshare")
    if ak is None: raise RuntimeError("未安装 akshare")

    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust=cfg.adjust)
    if df is None or df.empty: raise RuntimeError("akshare 返回空数据")

    rename = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "均价": "avg", "换手率": "turnover"
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # —— 将数值列转为浮点；turnover 可能是 '3.21%' 这样的字符串 —— 
    for c in ["open", "high", "low", "close", "volume", "amount", "avg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # 专门处理换手率：去掉 '%' 并转成小数；失败就留空（不作为必需列）
    if "turnover" in df.columns:
        s = df["turnover"].astype(str).str.replace("%", "", regex=False)
        df["turnover"] = pd.to_numeric(s, errors="coerce") / 100.0

    df = df.replace([np.inf, -np.inf], np.nan)

    # —— 只把 K 线与量作为“必需列”，避免换手率导致整表被 drop —— 
    need = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df.dropna(subset=need)

    # 若仍然为空，就不要写缓存，直接报错，便于追查
    if df.empty:
        raise RuntimeError("fetch_daily 得到空表，请检查代码与数据源参数。")

    # 写入缓存
    df.to_parquet(cache_path)
    print(f"[Cache] Saved to {cache_path}")
    return df