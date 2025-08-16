# -*- coding: utf-8 -*-
"""
精简与优化版 Dash 应用（保留原有功能与布局）
- 删除无效/重复代码段，统一命名，保证参数直观
- 阈值高亮改为柔和底纹，更易观察
- 默认展示：价格MA(5/20/60)、概率主线、概率MA(10/60)，其余通过图例切换
"""

import os
import glob
import calendar
import traceback
import time
import hashlib
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update

# --- 兼容 dash < 2.9 与 ≥ 2.9 的触发上下文 ---
try:
    from dash import ctx as dash_ctx
except ImportError:
    dash_ctx = None
from dash import callback_context as cb_ctx

# --- 与 acs_core 对接（保持原逻辑） ---
from acs_core.pipeline import compute as acs_compute
from acs_core.config import PipelineConfig
from acs_core.data import _is_trade_time_now

# =========================
#       全局常量
# =========================

DEBUG_VERSION = "v2.5.0 (Relative Path Cache)"

# 核心代码文件，用于计算哈希以判断缓存是否失效
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_FILES = [
    "acs_core/pipeline.py",
    "acs_core/config.py",
    "acs_core/models/neural_net_v2.py",
    "acs_core/data.py",
    "acs_core/types.py",
    "acs_core/__init__.py",
    "acs_core/features/__init__.py",
    "acs_core/features/builtins.py",
    "acs_core/features/fast.py",
    "acs_core/io_utils.py",
    "acs_core/labels.py",
    "acs_core/models/blend.py",
    "acs_core/models/reactive.py",
    "acs_core/models/trend.py",
    "acs_core/registries.py"
]

# 颜色：遵循A股常见配色（红涨绿/青跌）
INC_COLOR = "#ff4d4f"   # 上涨红
DEC_COLOR = "#17becf"   # 下跌青
HIGHLIGHT_FILL = "rgba(13,110,253,0.0)"  # 概率阈值区间柔和底纹

# 调色板
PRICE_MA_PALETTE = ["#FFBF00", "#E83F6F", "#2274A5", "#32936F", "#DA4167", "#08BDBD"]
PROB_MA_PALETTE  = ["#6c757d", "#fca311", "#9d4edd", "#c9184a", "#023e8a"]

# 概率均线可选窗口（列名需与数据对应）
PROB_MA_WINDOWS: List[int] = [5, 10, 20, 30, 60, 120]
# 默认展示哪几条概率均线（可在 on_data_change 里改）
DEFAULT_PROB_MA_VISIBLE: Set[int] = {10, 60}
# 默认展示价格均线
DEFAULT_PRICE_MA_VISIBLE: Set[int] = {5, 20, 60}

# 搜索CSV时的优先关键字
PREFER_KEYWORDS = ("acs_prob", "prob", "acs")


# =========================
#    工具函数（IO/选择）
# =========================
def get_code_hash(files: List[str], root_dir: str) -> str:
    """计算核心代码文件的哈希值"""
    hasher = hashlib.sha256()
    for fn in sorted(files):
        abs_path = os.path.join(root_dir, fn)
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'rb') as f:
                    hasher.update(f.read())
            except Exception as e:
                print(f"Error reading file {abs_path} for hashing: {e}")
    return hasher.hexdigest()


def list_csvs(root: str, code: Optional[str] = None) -> List[str]:
    """
    在 root 下递归搜索 CSV。
    - 若提供 code，则匹配形如 '{code}*.csv'；
    - 否则回退到 '*acs*.csv'。
    """
    pattern = os.path.join(root or ".", "**", f"{str(code).strip()}*.csv" if code else "*acs*.csv")
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith("~")]


def _score_csv_path(path: str) -> Tuple[int, int, float]:
    """
    根据关键字命中、路径深度、修改时间进行打分（越大越优）。
    """
    base = os.path.basename(path).lower()
    kw_hits = sum(1 for kw in PREFER_KEYWORDS if kw in base)  # 关键字命中数
    depth = -len(os.path.normpath(path).split(os.sep))        # 层级越深，分越小（取负）
    try:
        mtime = os.path.getmtime(path)                        # 越新越好
    except Exception:
        mtime = 0.0
    return (kw_hits, depth, mtime)


def pick_best_csv(files: List[str]) -> Optional[str]:
    """在候选文件中挑选最优的一份。"""
    return sorted(files, key=_score_csv_path, reverse=True)[0] if files else None


def read_csv_smart(path: str) -> pd.DataFrame:
    """
    以多编码尝试读取 CSV，最大化兼容性。
    """
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030", "cp936", "mbcs", "cp1252", "latin1")
    tried, last_err = [], None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            tried.append(enc)
            last_err = e
    raise RuntimeError(f"读取CSV失败，尝试编码={tried}；最后错误：{last_err}")



# =========================
#      图形绘制工具
# =========================

def _subplot_params(separate_prob_panel: bool, show_volume: bool, show_prob_diff: bool):
    """
    生成子图网格配置。
    返回: rows, row_heights, specs, prob_row, vol_row, diff_row
    - prob_row: 概率曲线所在行号
    - vol_row : 成交量所在行号
    - diff_row: 概率差值所在行号
    """
    rows = 1
    specs = [[{"secondary_y": True}]]  # 第一行：价格K线，带副Y轴
    prob_row, vol_row, diff_row = 1, None, None

    if separate_prob_panel:
        rows += 1
        prob_row = rows
        specs.append([{"secondary_y": False}])

    if show_volume:
        rows += 1
        vol_row = rows
        specs.append([{"secondary_y": True}])

    if show_prob_diff:
        rows += 1
        diff_row = rows
        specs.append([{"secondary_y": False}])

    # 行高比例：保持与原版接近的视觉
    if rows == 1:
        row_heights = [1.0]
    elif rows == 2:
        row_heights = [0.46, 0.54] if separate_prob_panel else [0.8, 0.2]
    elif rows == 3:
        if separate_prob_panel and (show_volume or show_prob_diff):
            row_heights = [0.48, 0.36, 0.16]
        else:
            row_heights = [0.7, 0.15, 0.15]
    else:  # rows == 4
        row_heights = [0.4, 0.36, 0.12, 0.12]

    return rows, row_heights, specs, prob_row, vol_row, diff_row


def _highlight_threshold(fig: go.Figure, series: pd.Series, thr: Optional[float], row=1, col=1):
    """
    在 series >= thr 的连续区间添加竖向高亮矩形（柔和底纹）。
    注：原版为透明填充相当于无效果，这里改为轻微底纹，便于观察且不喧宾夺主。
    """
    if thr is None:
        return
    mask = (series >= float(thr)).astype(int)
    if mask.sum() == 0:
        return

    # 用 diff 边界找区间
    edges = mask.diff().fillna(mask.iloc[0]).astype(int)
    starts = mask.index[edges == 1]
    ends = mask.index[edges == -1]

    if mask.iloc[0] == 1:
        starts = starts.insert(0, mask.index[0])
    if mask.iloc[-1] == 1 and len(ends) < len(starts):
        ends = ends.append(pd.Index([mask.index[-1]]))

    for s, e in zip(starts, ends):
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=HIGHLIGHT_FILL,  # 柔和填充
            line_width=0,
            layer="below",
            row=row, col=col,
        )


def _hide_non_trading(fig: go.Figure, idx: pd.Index):
    """
    对日期轴应用非交易日隐藏（避免周末/节假日留白）。
    """
    trading = pd.DatetimeIndex(pd.to_datetime(idx).normalize().unique())
    if len(trading) == 0:
        return
    all_days = pd.date_range(trading.min(), trading.max(), freq="D")
    non_trading_days = all_days.difference(trading)
    fig.update_xaxes(rangebreaks=[dict(values=non_trading_days.tolist())])

def _tail_texts(x_len: int, label: str):
    """返回一个长度与数据相同的文本数组，只有最后一个点显示 label，其余为空字符串。"""
    arr = [""] * int(x_len)
    if x_len:
        arr[-1] = label
    return arr

def make_figure(
    df: pd.DataFrame,
    title: str,
    visible_price_ma_set: Set[int],
    visible_prob_ma_set: Set[int],
    separate_prob_panel: bool,
    show_volume: bool,
    smooth_win: int,
    threshold: Optional[float],
    date_start: Optional[str],
    date_end: Optional[str],
    show_nn_prob: bool,
    show_prob_diff: bool,
) -> go.Figure:
    """
    统一的图表生成函数：
    - 默认第一行价格K线（副轴可放概率主线/均线）
    - 可选分离概率面板/成交量面板/概率差值面板
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")

    # 过滤日期段
    if date_start:
        data = data.loc[data.index >= pd.to_datetime(date_start)]
    if date_end:
        data = data.loc[data.index <= pd.to_datetime(date_end)]

    if data.empty:
        return go.Figure().update_layout(
            title="所选时间段无数据",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
        )

    # 预处理显示列（存在才绘制，避免零值图形）
    if "turnover" in data.columns:
        data["turnover_display"] = data["turnover"] * 100  # 百分比→%
    if "volume" in data.columns:
        data["volume_display"] = data["volume"] / 10000.0  # 换算为“万”

    # --- 新增：计算涨跌幅 ---
    data["pct_change"] = data["close"].pct_change() * 100

    # --- 新增：生成悬浮文本 ---
    hover_texts = []
    for i in range(len(data)):
        row = data.iloc[i]
        hover_texts.append(
            '<span style="font-weight: bold; font-size: 1.1em;">收盘价: {close:.2f}</span><br>'
            '<span style="font-weight: bold; font-size: 1.1em;">涨跌幅: {pct:.2f}%</span><br>'
            '换手率: {trn:.2f}%<br>'
            '开盘价: {opn:.2f}<br>'
            '最高价: {hgh:.2f}<br>'
            '最低价: {lw:.2f}'.format(
                close=row['close'],
                pct=row['pct_change'],
                trn=row.get('turnover_display', float('nan')),
                opn=row['open'],
                hgh=row['high'],
                lw=row['low']
            )
        )

    # 概率平滑
    prob_smoothed = data["acs_prob"].rolling(max(1, int(smooth_win)), min_periods=1).mean()

    # --- 子图布局 ---
    rows, row_heights, specs, prob_row, vol_row, diff_row = _subplot_params(
        separate_prob_panel, show_volume, show_prob_diff
    )
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.02, specs=specs
    )

    # ==============
    #   价格主图
    # ==============
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"], high=data["high"],
            low=data["low"], close=data["close"],
            increasing=dict(line=dict(color=INC_COLOR, width=1), fillcolor=INC_COLOR),
            decreasing=dict(line=dict(color=DEC_COLOR, width=1), fillcolor=DEC_COLOR),
            name="", showlegend=False,
            hovertext=hover_texts,
            hoverinfo='text'
        ),
        row=1, col=1, secondary_y=False,
    )

    # 价格均线
    for i, w in enumerate(sorted(visible_price_ma_set)):
        col_name = f"price_ma_{w}"
        if col_name in data.columns:
            label = f"MA{w}"
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[col_name],
                    name=label,
                    mode="lines",  # 仅画线，不在末端叠加文本
                    line=dict(color=PRICE_MA_PALETTE[i % len(PRICE_MA_PALETTE)], width=1.2, dash="dot"),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",  # 注意：Plotly 花括号需双写以逃逸
                    legendgroup=f"price_ma_{w}",
                ),
                row=1, col=1, secondary_y=False,
            )


    # 概率主线（若不分离，则放副轴）
    fig.add_trace(
        go.Scatter(
            x=data.index, y=prob_smoothed,
            name="ACS概率",
            mode="lines",
            line=dict(color="#0D6EFD", width=1.5 if not separate_prob_panel else 1.0),
            hovertemplate="%{y:.2f}<extra></extra>",
            legendgroup="prob",
            showlegend=not separate_prob_panel,  # 分离面板时避免重复图例
        ),
        row=prob_row, col=1, secondary_y=(not separate_prob_panel),
    )

    # 纯NN概率（存在才显示）
    if show_nn_prob and "acs_prob_nn_only" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["acs_prob_nn_only"],
                name="NN概率",
                mode="lines",
                line=dict(color="#d62728", width=2.0),
                hovertemplate="%{y:.2f}<extra></extra>",
                legendgroup="prob_nn", showlegend=True,
            ),
            row=prob_row, col=1, secondary_y=(not separate_prob_panel),
        )

    # 概率均线（以窗口整数集合控制显示）
    for i, w in enumerate(PROB_MA_WINDOWS):
        col_name = f"prob_ma_{w}"
        if (w in visible_prob_ma_set) and (col_name in data.columns):
            label = f"MA{w}"  # 你想显示“ma5/ma10”这种短名，就用 MA{w}
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data[col_name],
                    name=f"ProbMA{w}",
                    mode="lines",  # 仅画线
                    # 去掉 text / textposition / textfont，避免末端标注
                    line=dict(color=PROB_MA_PALETTE[i % len(PROB_MA_PALETTE)], width=1.2),
                    hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>",
                    legendgroup=f"prob_ma_{w}",
                ),
                row=prob_row, col=1, secondary_y=(not separate_prob_panel),
            )


    # 概率阈值区间：默认标在第一行（价格图）更直观
    _highlight_threshold(fig, prob_smoothed, threshold, row=1, col=1)

    # 成交量/换手率（存在相应列才绘制，避免“零柱”）
    if show_volume and vol_row is not None:
        if "volume_display" in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data["volume_display"],
                    name="成交量(万)",
                    marker=dict(color=np.where(data["close"] >= data["open"], INC_COLOR, DEC_COLOR)),
                    opacity=0.7,
                    hovertemplate="%{y:.2f}万<extra></extra>",
                ),
                row=vol_row, col=1, secondary_y=False,
            )
        if "turnover_display" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["turnover_display"],
                    name="换手率(%)",
                    mode="lines",
                    line=dict(color="#FFBF00", width=1.2),
                    hovertemplate="%{y:.2f}%<extra></extra>",
                ),
                row=vol_row, col=1, secondary_y=True,
            )

    
    # 概率差值（存在列才显示）
    if show_prob_diff and diff_row is not None and "prob_diff" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["prob_diff"],
                name="概率差值",
                mode="lines",
                line=dict(color="#9467bd", width=1.5, dash="dot"),
                hovertemplate="%{y:.2f}<extra></extra>",
                legendgroup="prob_diff",
                showlegend=True,
            ),
            row=diff_row, col=1,
        )

    # 布局/坐标轴
    stock_code = title.split("_")[0] if title else ""
    fig.update_layout(
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
            font=dict(size=11), borderwidth=0,
        ),
        margin=dict(l=60, r=120, t=40, b=20),
        font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        hovermode="x unified",
        annotations=[
            dict(
                text=stock_code, align="center", showarrow=False,
                xref="paper", yref="paper", x=0.5, y=1.07,
                font=dict(size=14, color="#6C757D"),
            )
        ],
    )
    for r in range(1, rows + 1):
        fig.update_yaxes(showgrid=True, gridcolor="#E9ECEF", zerolinecolor="#DEE2E6", row=r, col=1)

    fig.update_yaxes(title_text="价格", row=1, col=1, secondary_y=False, color="#212529")
    fig.update_yaxes(title_text="概率", row=1, col=1, secondary_y=True, range=[0, 1], color="#0D6EFD")
    if separate_prob_panel:
        fig.update_yaxes(title_text="概率(均线)", row=prob_row, col=1, range=[0, 1], color="#212529")
    if show_volume and vol_row is not None:
        fig.update_yaxes(title_text="成交量(万)", row=vol_row, col=1, secondary_y=False,
                         rangemode="tozero", color="#212529", showgrid=False)
        fig.update_yaxes(title_text="换手率(%)", row=vol_row, col=1, secondary_y=True,
                         rangemode="tozero", color="#FFBF00", showgrid=False)
    if show_prob_diff and diff_row is not None:
        fig.update_yaxes(title_text="概率差值", row=diff_row, col=1, color="#9467bd", showgrid=False)

    _hide_non_trading(fig, data.index)
    fig.update_xaxes(hoverformat="%Y.%-m.%-d")
    return fig


# =========================
#        Dash App
# =========================

def build_app(title_default: str, root_dir: str):
    """
    构建 Dash 应用（保持原布局/交互，样式与回调小幅清理）
    """
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = title_default

    # --- 样式常量 ---
    COLORS = {
        "background": "#F8F9FA",
        "surface": "#FFFFFF",
        "primary": "#0D6EFD",
        "secondary": "#6C757D",
        "text": "#212529",
        "light_text": "#6C757D",
        "border": "#DEE2E6",
        "hover_bg": "#E9ECEF",
    }
    FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
    SIDEBAR_W = 420
    GAP = 12
    SIDEBAR_PAD = 18  # 侧栏内边距，用于动作条贴边计算
    
    # 容器样式
    SIDEBAR_STYLE_OPEN = {
        "boxSizing": "border-box", "position": "fixed", "top": f"{GAP}px", "bottom": f"{GAP}px",
        "left": f"{GAP}px", "width": f"{SIDEBAR_W}px", "overflowY": "auto",
        "maxHeight": f"calc(100vh - {GAP*2}px)", "padding": f"{SIDEBAR_PAD}px",
        "background": COLORS["surface"], "border": "none",
        "borderRadius": "14px", "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
        "zIndex": 9998, "transition": "left .3s ease",
        # 防止 sticky 子元素在圆角边缘出现细微外溢
        "overflowX": "hidden",
    }


    SIDEBAR_STYLE_CLOSED = {**SIDEBAR_STYLE_OPEN, "left": f"-{SIDEBAR_W+GAP}px"}

    MAIN_STYLE_OPEN = {
        "boxSizing": "border-box", "position": "fixed", "top": f"{GAP}px", "bottom": f"{GAP}px",
        "right": f"{GAP}px", "left": f"{SIDEBAR_W + GAP*2}px",
        "background": COLORS["surface"], "borderRadius": "14px", "overflow": "hidden",
        "border": "none", "boxShadow": "0 1px 3px rgba(0,0,0,0.04)",
        "transition": "left .3s ease, right .3s ease",
    }
    MAIN_STYLE_CLOSED = {**MAIN_STYLE_OPEN, "left": f"{GAP}px"}

    # 组件通用样式
    LABEL_STYLE = {"fontWeight": "500", "marginBottom": "6px", "color": COLORS["text"], "fontSize": "14px"}
    INPUT_STYLE = {
        "boxSizing": "border-box", "width": "100%", "padding": "8px 12px",
        "borderRadius": "8px", "border": f"1px solid {COLORS['border']}",
        "background": COLORS["surface"], "color": COLORS["text"], "fontSize": "14px",
    }
    BUTTON_STYLE = {
        "background": COLORS["primary"], "color": "white", "border": "none",
        "borderRadius": "10px",
        "height": "36px", "minWidth": "88px",  # 统一高度与最小宽度
        "padding": "0 14px", "fontSize": "14px", "fontWeight": "500",
        "cursor": "pointer", "transition": "background-color .2s ease",
    }
    BUTTON_SECONDARY_STYLE = {**BUTTON_STYLE, "background": COLORS["secondary"]}



    DROP_STYLE = {"flex": "1 1 0", "minWidth": "92px"}
    ROW_FLEX   = {"display": "flex", "gap": "8px", "alignItems": "center"}

    # ========== 日期辅助 ==========
    def _days_in_month(year: int, month: int) -> int:
        """给定年月返回天数（用于下拉选项）。"""
        return calendar.monthrange(int(year), int(month))[1] if year and month else 31

    def _ymd_to_iso(y, m, d) -> str:
        """安全拼装 YYYY-MM-DD（若日超出当月天数则自动压缩）。"""
        if not (y and m and d):
            return pd.Timestamp.today().strftime("%Y-%m-%d")
        y, m, d = int(y), int(m), int(d)
        return f"{y:04d}-{m:02d}-{min(d, _days_in_month(y, m)):02d}"

    _now = pd.Timestamp.today()
    year_opts  = [{"label": f"{y}", "value": y} for y in range(2005, _now.year + 1)]
    month_opts = [{"label": f"{m:02d}", "value": m} for m in range(1, 13)]
    start_y, start_m, start_d = 2019, 1, 1
    end_y,   end_m,   end_d   = _now.year, _now.month, _now.day

    # 初始“日”选项（未做联动，但 _ymd_to_iso 会自动纠正超出天数）
    start_day_opts = [{"label": f"{d:02d}", "value": d} for d in range(1, _days_in_month(start_y, start_m) + 1)]
    end_day_opts   = [{"label": f"{d:02d}", "value": d} for d in range(1, _days_in_month(end_y, end_m) + 1)]

    # ========== 布局 ==========
    app.layout = html.Div(
        [
            # 内存态
            dcc.Store(id="df-store"),
            dcc.Store(id="csv-path-store"),
            dcc.Store(id="title-store"),
            dcc.Store(id="uirevision-store"),

            # 侧栏
            html.Div(
                [
                    # 顶部动作条：按钮居中 + 真贴边 + 不透底
                    html.Div(
                        [
                            # 按钮组：单独一行，水平居中
                            html.Div(
                                [
                                    html.Button("运算", id="btn-compute", n_clicks=0, style=BUTTON_STYLE),
                                    html.Button("搜索CSV", id="btn-search", n_clicks=0, style=BUTTON_SECONDARY_STYLE),
                                    html.Button("载入所选文件", id="btn-load", n_clicks=0, style=BUTTON_SECONDARY_STYLE),
                                ],
                                style={
                                    "display": "flex", "gap": "8px",
                                    "justifyContent": "center", "alignItems": "center",
                                    "width": "100%",
                                },
                            ),
                            # 提示文本：在下一行居中显示
                            html.Span(
                                id="compute-msg",
                                style={
                                    "color": COLORS["light_text"], "fontSize": 12,
                                    "textAlign": "center", "width": "100%", "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                        id="action-bar",
                        style={
                            # 使用 sticky 吸顶，但考虑到侧栏有内边距，这里向上“抵消”内边距实现真贴边
                            "position": "sticky", "top": f"-{SIDEBAR_PAD}px", "zIndex": 10000,

                            # 纯色背景防止底层透出；去掉模糊滤镜（backdropFilter）
                            "background": COLORS["surface"],

                            # 左右扩展到侧栏内边距之外，上下留出与正文的分隔
                            "margin": f"0 -{SIDEBAR_PAD}px 8px",
                            "padding": f"{SIDEBAR_PAD}px {SIDEBAR_PAD}px 10px",

                            "borderBottom": "1px solid rgba(15,23,42,0.06)",

                            # 纵向堆叠：第一行是按钮组，第二行是提示文本
                            "display": "flex", "flexDirection": "column",
                            "alignItems": "center", "gap": "8px",
                        },
                    ),

                    # === 侧栏其余内容（保持原有顺序） ===
                    html.Label("数据根目录", style=LABEL_STYLE),
                    dcc.Input(id="root-dir", type="text", value=root_dir, style={**INPUT_STYLE, "marginBottom": "12px"}),

                    html.Label("标的代码（如 603533）", style=LABEL_STYLE),
                    dcc.Input(
                        id="code-input", type="text", placeholder="输入后点击下方按钮",
                        debounce=True, style={**INPUT_STYLE, "marginBottom": "12px"}
                    ),

                    html.Label("日期范围", style=LABEL_STYLE),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("开始", style={"fontSize": "12px", "color": "#6b7280", "marginBottom": "4px"}),
                                    html.Div(
                                        [
                                            dcc.Dropdown(id="start-year",  options=year_opts,  value=start_y,  clearable=False, style=DROP_STYLE),
                                            dcc.Dropdown(id="start-month", options=month_opts, value=start_m,  clearable=False, style=DROP_STYLE),
                                            dcc.Dropdown(id="start-day",   options=start_day_opts, value=start_d, clearable=False, style=DROP_STYLE),
                                        ],
                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                                    ),
                                ],
                                className="section",
                            ),
                            html.Div(
                                [
                                    html.Div("结束", style={"fontSize": "12px", "color": "#6b7280", "marginBottom": "4px"}),
                                    html.Div(
                                        [
                                            dcc.Dropdown(id="end-year",  options=year_opts,  value=end_y, clearable=False, style=DROP_STYLE),
                                            dcc.Dropdown(id="end-month", options=month_opts, value=end_m, clearable=False, style=DROP_STYLE),
                                            dcc.Dropdown(id="end-day",   options=end_day_opts, value=end_d, clearable=False, style=DROP_STYLE),
                                        ],
                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "8px"},
                                    ),
                                ],
                                className="section",
                            ),
                        ],
                        style={"background": "transparent", "border": "none", "borderRadius": "8px", "padding": "6px 0", "marginBottom": "8px"},
                    ),

                    html.Div(style={"height": "8px"}),  # 留白

                    html.Label("调试信息", style=LABEL_STYLE),
                    dcc.Textarea(
                        id="debug-box", value="",
                        style={**INPUT_STYLE, "height": "120px", "whiteSpace": "pre", "fontFamily": "Menlo, Consolas, monospace"},
                    ),

                    html.Div(style={"height": "8px"}),  # 留白

                    html.H4("切换/载入已存在的CSV", style={"color": COLORS["text"]}),
                    html.Div(style={"height": "4px"}),
                    html.Div(id="search-msg", style={"color": COLORS["light_text"], "fontSize": 12}),
                    dcc.Dropdown(
                        id="csv-dropdown", options=[], value=None, placeholder="先搜索再选择",
                        style={"boxSizing": "border-box", "width": "100%", "marginTop": "8px", "marginBottom": "12px"},
                    ),

                    html.Div(style={"height": "8px"}),  # 留白

                    html.Div(
                        [
                            html.H4("可视化选项", style={"color": COLORS["text"], "marginTop": "0px", "marginBottom": "15px"}),
                            html.Label("图表设置", style=LABEL_STYLE),
                            dcc.Checklist(
                                id="options",
                                options=[
                                    {"label": "单独概率面板", "value": "separate"},
                                    {"label": "成交量面板", "value": "showvol"},
                                    {"label": "显示纯NN概率", "value": "show_nn_prob"},
                                    {"label": "显示概率差值", "value": "show_prob_diff"},
                                ],
                                value=["separate", "showvol", "show_nn_prob"],
                                style={"marginBottom": "15px"},
                                labelStyle={"display": "inline-block", "marginRight": "15px", "fontSize": "13px"},
                            ),
                            html.Label("神经网络权重", style=LABEL_STYLE),
                            dcc.Slider(
                                id="nn-weight-slider", min=0.0, max=1.0, step=0.05, value=0.0,
                                marks={i/10: str(i/10) for i in range(0, 11, 2)}, className="custom-slider",
                            ),
                            html.Div(style={"height": "12px"}),
                            html.Label("概率阈值高亮 (≥)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="thr", min=0.0, max=1.0, step=0.01, value=0.70,
                                marks={0.0: "0.0", 0.5: "0.5", 1.0: "1.0"}, className="custom-slider",
                            ),
                            html.Div(style={"height": "12px"}),
                            html.Label("概率平滑窗口（天）", style=LABEL_STYLE),
                            dcc.Slider(
                                id="smooth", min=1, max=12, step=1, value=3,
                                marks={i: str(i) for i in [1, 3, 5, 7, 9, 12]}, className="custom-slider",
                            ),
                        ],
                        style={"background": "transparent", "border": "none", "borderRadius": "8px", "padding": "6px 0"},
                    ),

                    html.Div(id="load-msg", style={"marginTop": "8px", "color": COLORS["light_text"], "fontSize": 12}),
                ],
                id="sidebar", className="sidebar", style=SIDEBAR_STYLE_OPEN,
            ),


            # 主区：图表
            html.Div(
                [
                    html.Button(
                        "‹", id="btn-toggle-sidebar", n_clicks=0, title="收起/展开侧边栏",
                        style={"position": "absolute", "top": "12px", "left": "12px", "zIndex": 10000,
                               "fontSize": "18px", "padding": "4px 10px", "background": COLORS["hover_bg"],
                               "border": f"1px solid {COLORS['border']}", "borderRadius": "8px", "cursor": "pointer"},
                    ),
                    dcc.Graph(id="chart", config={"scrollZoom": True}, style={"height": "100%", "width": "100%"}),
                ],
                id="main", style=MAIN_STYLE_OPEN,
            ),
        ],
        style={"fontFamily": FONT_FAMILY, "background": COLORS["background"]},
    )

    # ========== 回调：侧边栏展开/收起 ==========
    @app.callback(
        [Output("sidebar", "style"), Output("main", "style"), Output("btn-toggle-sidebar", "children")],
        [Input("btn-toggle-sidebar", "n_clicks")],
        [State("sidebar", "style")],
        prevent_initial_call=True,
    )
    def toggle_sidebar(n_clicks, current_style):
        """展开/收起侧栏。"""
        if n_clicks:
            if current_style and current_style.get("left") == f"{GAP}px":
                return SIDEBAR_STYLE_CLOSED, MAIN_STYLE_CLOSED, "›"
            return SIDEBAR_STYLE_OPEN, MAIN_STYLE_OPEN, "‹"
        return no_update, no_update, no_update

    # ========== 回调：搜索CSV ==========
    @app.callback(
        [Output("csv-dropdown", "options"), Output("csv-dropdown", "value"), Output("search-msg", "children")],
        [Input("btn-search", "n_clicks")],
        [State("root-dir", "value"), State("code-input", "value")],
        prevent_initial_call=True,
    )
    def on_search_csv(n_clicks, root_dir, code):
        """搜索符合条件的CSV并自动挑选最优。"""
        if not n_clicks:
            return no_update, no_update, no_update

        root = root_dir or "."
        try:
            files = list_csvs(root=root, code=code)
            if not files:
                return [], None, "在指定目录中未找到匹配的CSV文件。"

            options = [{"label": os.path.relpath(f, root), "value": f} for f in sorted(files)]
            best_pick = pick_best_csv(files)
            msg = f"✅ 找到 {len(files)} 个文件，已为您选中最优结果。"
            return options, best_pick, msg

        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return [], None, f"搜索文件时出错: {e}"

    # ========== 回调：载入或运算 ==========
    @app.callback(
        [
            Output("df-store", "data"),
            Output("csv-path-store", "data"),
            Output("title-store", "data"),
            Output("uirevision-store", "data"),
            Output("load-msg", "children"),
            Output("compute-msg", "children"),
            Output("debug-box", "value"),
        ],
        [Input("btn-load", "n_clicks"), Input("btn-compute", "n_clicks")],
        [
            State("csv-dropdown", "value"),
            State("code-input", "value"),
            State("root-dir", "value"),
            State("start-year", "value"),
            State("start-month", "value"),
            State("start-day", "value"),
            State("end-year", "value"),
            State("end-month", "value"),
            State("end-day", "value"),
            State("nn-weight-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def on_load_or_compute(
        n_load, n_comp, csv_path, code, root_dir, sy, sm, sd, ey, em, ed, nn_weight
    ):
        """
        两个按钮共用一个回调，依赖触发来源做分支。
        - btn-load: 读取已有CSV
        - btn-compute: 调用 pipeline 计算（交易时段优先实时；非交易时段命中缓存则用缓存）
        """
        trig = (
            dash_ctx.triggered_id
            if dash_ctx
            else (cb_ctx.triggered[0]["prop_id"].split(".")[0] if cb_ctx.triggered else None)
        )
        start_iso, end_iso = _ymd_to_iso(sy, sm, sd), _ymd_to_iso(ey, em, ed)
        debug_lines = [f"[trigger] {trig}", f"[dates] {start_iso}..{end_iso}"]
        root = root_dir or "."

        try:
            if trig == "btn-compute":
                if not code or not str(code).strip():
                    msg = "请输入代码再点击‘运算’。"
                    debug_lines.append(f"[error] {msg}")
                    return (no_update, no_update, no_update, no_update, msg, msg, "\n".join(debug_lines))

                # 智能缓存：非交易时段命中缓存则直接载入
                potential_caches = list_csvs(root=root, code=code)
                best_cache = pick_best_csv(potential_caches)
                
                cache_is_valid = False
                if best_cache and not _is_trade_time_now():
                    debug_lines.append(f"[cache] 非交易时间，命中缓存: {best_cache}")
                    hash_path = best_cache + ".hash"
                    if os.path.exists(hash_path):
                        with open(hash_path, "r") as f:
                            stored_hash = f.read()
                        current_hash = get_code_hash(CORE_FILES, PROJECT_ROOT)
                        if stored_hash == current_hash:
                            cache_is_valid = True
                            debug_lines.append(f"[cache] 哈希校验通过")
                        else:
                            debug_lines.append(f"[cache] 哈希校验失败，重新计算")
                    else:
                        debug_lines.append(f"[cache] 未找到哈希文件，重新计算")

                if cache_is_valid:
                    df = read_csv_smart(best_cache)
                    data_json = df.to_json(orient="split", date_format="iso")
                    ttl = os.path.splitext(os.path.basename(best_cache))[0]
                    note = f"✅ 已从缓存载入: {best_cache}"
                    debug_lines.append(f"[done] {note}")
                    return (data_json, best_cache, ttl, str(best_cache), "", "", "\n".join(debug_lines))

                # 交易时段或无缓存 → 实时计算
                debug_lines.append("[info] 交易时间或无缓存，执行实时计算。")
                cfg = PipelineConfig()
                cfg.output.root = root
                cfg.extras["nn_weight"] = nn_weight

                debug_lines.append(f"[compute] acs_compute(symbol={code}, nn_weight={nn_weight})")
                result = acs_compute(str(code).strip(), start_iso, end_iso, cfg)

                df_res = result["df"]
                saved_path = result["paths"].get("csv", "N/A")
                metrics = result.get("metrics", {})

                if saved_path != "N/A":
                    current_hash = get_code_hash(CORE_FILES, PROJECT_ROOT)
                    hash_path = saved_path + ".hash"
                    with open(hash_path, "w") as f:
                        f.write(current_hash)
                    debug_lines.append(f"[cache] 哈希已保存到: {hash_path}")

                if metrics.get("nn_used", False):
                    debug_lines.append("[info] 神经网络模型已启用")
                else:
                    debug_lines.append("[info] 数据量不足，跳过神经网络模型")

                data_json = df_res.to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(saved_path))[0]
                auc_val = metrics.get("auc_cal", metrics.get("auc_raw"))
                auc_part = f" | OOF AUC≈{auc_val:.3f}" if isinstance(auc_val, float) else ""
                note = f"✅ 运算完成: {saved_path}｜模式: {metrics.get('mode','?')}{auc_part}"
                debug_lines.append(f"[done] {note}")
                return (data_json, saved_path, ttl, str(time.time()), "", note, "\n".join(debug_lines))

            if trig == "btn-load":
                if not csv_path:
                    msg = "请选择一个CSV文件。"
                    debug_lines.append(f"[error] {msg}")
                    return (no_update, no_update, no_update, no_update, msg, "", "\n".join(debug_lines))

                df = read_csv_smart(csv_path)
                data_json = df.to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(csv_path))[0]
                note = f"✅ 已载入: {csv_path}"
                debug_lines.append(f"[done] {note}")
                return (data_json, csv_path, ttl, str(csv_path), note, "", "\n".join(debug_lines))

        except Exception as e:
            tb_str = traceback.format_exc()
            debug_lines.append(f"[CRITICAL] 操作失败: {e}\n{tb_str}")
            return (no_update, no_update, no_update, no_update, str(e), str(e), "\n".join(debug_lines))

        return no_update, no_update, no_update, no_update, "", "", ""

    # ========== 回调：数据变化 -> 重绘图 ==========
    @app.callback(
        Output("chart", "figure"),
        Input("df-store", "data"),
        Input("title-store", "data"),
        Input("options", "value"),
        Input("smooth", "value"),
        Input("thr", "value"),
        State("start-year", "value"),
        State("start-month", "value"),
        State("start-day", "value"),
        State("end-year", "value"),
        State("end-month", "value"),
        State("end-day", "value"),
        State("uirevision-store", "data"),
    )
    def on_data_change(
        data_json, title, opts, smooth, thr, sy, sm, sd, ey, em, ed, uirevision
    ):
        """
        数据或选项变更 → 生成新图。
        说明：
        - 价格均线默认显示 {5,20,60}
        - 概率均线默认显示 {10,60}
        - 其他曲线可在图例中自行切换（legendonly）
        """
        if not data_json:
            return go.Figure(layout={"title": "请先载入或运算数据", "uirevision": uirevision})

        start_iso = f"{int(sy):04d}-{int(sm):02d}-{int(sd):02d}"
        end_iso   = f"{int(ey):04d}-{int(em):02d}-{int(ed):02d}"

        df = pd.read_json(data_json, orient="split")

        fig = make_figure(
            df=df,
            title=title,
            visible_price_ma_set=DEFAULT_PRICE_MA_VISIBLE,
            visible_prob_ma_set=DEFAULT_PROB_MA_VISIBLE,
            separate_prob_panel=("separate" in (opts or [])),
            show_volume=("showvol" in (opts or [])),
            smooth_win=int(smooth or 1),
            threshold=thr,
            date_start=start_iso,
            date_end=end_iso,
            show_nn_prob=("show_nn_prob" in (opts or [])),
            show_prob_diff=("show_prob_diff" in (opts or [])),
        )
        fig.update_layout(uirevision=uirevision)  # 保持缩放状态
        return fig

    # ========== 回调：缩放联动修正 ==========
    @app.callback(
        Output("chart", "figure", allow_duplicate=True),
        Input("chart", "relayoutData"),
        [State("df-store", "data"), State("chart", "figure")],
        prevent_initial_call=True,
    )
    def adjust_yaxis_on_zoom(relayout_data, data_json, current_fig):
        """
        修正双Y轴缩放时的自适应范围（并修复 plotly 某些版本 rangeslider 注入 yaxis2 的小坑）。
        """
        if not relayout_data or not data_json or not current_fig:
            return no_update

        # 清理潜在的无效属性
        rangeslider_obj = current_fig.get("layout", {}).get("xaxis", {}).get("rangeslider")
        if isinstance(rangeslider_obj, dict):
            rangeslider_obj.pop("yaxis2", None)

        fig = go.Figure(current_fig)

        # 双击还原
        if relayout_data.get("xaxis.autorange"):
            fig.update_layout(yaxis={"autorange": True})
            if "yaxis2" in fig.layout:
                fig.update_layout(yaxis2={"autorange": True})
            fig.update_layout(hovermode="x unified")
            return fig

        # 只处理缩放/平移事件
        if ("xaxis.range[0]" not in relayout_data) or ("xaxis.range[1]" not in relayout_data):
            return no_update

        try:
            x_start = pd.to_datetime(relayout_data["xaxis.range[0]"])
            x_end   = pd.to_datetime(relayout_data["xaxis.range[1]"])
            df = (
                pd.read_json(data_json, orient="split")
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .set_index("date")
            )
            visible_df = df.loc[x_start:x_end]
            if visible_df.empty:
                return no_update

            # 主Y轴（价格）
            y_min, y_max = visible_df["low"].min(), visible_df["high"].max()
            pad = (y_max - y_min) * 0.05
            fig.update_layout(yaxis={"autorange": False, "range": [y_min - pad, y_max + pad]})

            # 副Y轴（概率）存在时自适应
            if "yaxis2" in fig.layout:
                prob_traces = [
                    t for t in fig.data
                    if t.yaxis == "y2" and (t.visible is None or t.visible != "legendonly")
                ]
                if prob_traces:
                    y2_min, y2_max = 1.0, 0.0
                    for t in prob_traces:
                        trace_y = pd.Series(t.y, index=pd.to_datetime(t.x))
                        visible_y = trace_y.loc[x_start:x_end]
                        if not visible_y.empty:
                            y2_min = min(y2_min, visible_y.min())
                            y2_max = max(y2_max, visible_y.max())
                    if y2_max > y2_min:
                        pad2 = (y2_max - y2_min) * 0.1
                        fig.update_layout(yaxis2={"autorange": False, "range": [max(0, y2_min - pad2), min(1, y2_max + pad2)]})

            fig.update_layout(hovermode="x unified")
            return fig
        except (KeyError, TypeError, ValueError):
            return no_update

    return app


# =========================
#         入口
# =========================

if __name__ == "__main__":
    # 默认输出目录为 'data'（可在界面上修改）
    app = build_app(title_default="ACS 概率面板", root_dir="data")
    app.run(debug=True, port=8050)
