# -*- coding: utf-8 -*-
import os
import glob
import calendar
import traceback
import time
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update

# === 颜色/窗口常量（模块级，避免函数内重复定义） ===
INC_COLOR = "#ff4d4f"  # 上涨红
DEC_COLOR = "#17becf"  # 下跌青

PRICE_MA_PALETTE = ["#FFBF00", "#E83F6F", "#2274A5", "#32936F", "#DA4167", "#08BDBD"]
PROB_MA_PALETTE = ["#6c757d", "#fca311", "#9d4edd", "#c9184a", "#023e8a"]
PROB_MA_WINDOWS = [5, 10, 20, 30, 60, 120]

# 兼容 dash < 2.9 与 ≥ 2.9 的触发上下文
try:
    from dash import ctx as dash_ctx
except ImportError:
    dash_ctx = None
from dash import callback_context as cb_ctx

# --- 从 acs_core 导入 ---
from acs_core.pipeline import compute as acs_compute
from acs_core.config import PipelineConfig

DEBUG_VERSION = "v2.2.0 (Optimized)"

# ------------------------------
# 工具：CSV 搜索与读取 (不变)
# ------------------------------
PREFER_KEYWORDS = ("acs_prob", "prob", "acs")


def _subplot_params(separate_prob_panel: bool, show_volume: bool):
    """
    返回: rows, row_heights, specs, prob_row, vol_row
    - prob_row: 概率曲线所在的行号（分面板时为第2行，否则第1行）
    - vol_row : 成交量所在行号（未显示则为 None）
    """
    # 基础：仅价格+概率（同一面板）
    rows, row_heights, specs = 1, [1.0], [[{"secondary_y": True}]]
    prob_row, vol_row = 1, None

    if separate_prob_panel:
        rows, row_heights, specs = (
            2,
            [0.25, 0.75],
            [
                [{"secondary_y": True}],  # 行1：价格(secondary_y 可给概率均线)
                [{"secondary_y": False}],  # 行2：概率面板
            ],
        )
        prob_row = 2

    if show_volume:
        vol_spec = {"secondary_y": True}
        if separate_prob_panel:
            rows, row_heights, specs = (
                3,
                [0.475, 0.375, 0.15],
                [
                    [{"secondary_y": True}],  # 行1：价格
                    [{"secondary_y": False}],  # 行2：概率(独立面板)
                    [vol_spec],  # 行3：量/换手
                ],
            )
            prob_row, vol_row = 2, 3
        else:
            rows, row_heights, specs = (
                2,
                [0.8, 0.2],
                [
                    [{"secondary_y": True}],  # 行1：价格+概率
                    [vol_spec],  # 行2：量/换手
                ],
            )
            prob_row, vol_row = 1, 2

    return rows, row_heights, specs, prob_row, vol_row


def _highlight_threshold(fig, x_index, series, thr, row=1, col=1):
    """
    在 series >= thr 的连续区间内添加竖向高亮矩形，保持现有视觉不变。
    """
    if thr is None:
        return
    m = (series >= float(thr)).astype(int)
    if m.sum() == 0:
        return

    # 利用 diff 找到区间起止边界
    edges = m.diff().fillna(m.iloc[0]).astype(int)
    starts = m.index[edges == 1]
    ends = m.index[edges == -1]

    # 处理起/末端落在高亮区间内的情况
    if m.iloc[0] == 1:
        starts = starts.insert(0, m.index[0])
    if m.iloc[-1] == 1 and len(ends) < len(starts):
        ends = ends.append(pd.Index([m.index[-1]]))

    for s, e in zip(starts, ends):
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor="rgba(255, 77, 79, 0.1)",
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )


def list_csvs(root: str, code: Optional[str] = None) -> List[str]:
    if code and str(code).strip():
        pattern = os.path.join(root, "**", f"{code}*.csv")
    else:
        pattern = os.path.join(root, "**", f"*acs*.csv")
    files = glob.glob(pattern, recursive=True)
    return [
        f
        for f in files
        if os.path.isfile(f) and not os.path.basename(f).startswith("~")
    ]


def _score_csv_path(path: str) -> Tuple[int, int, float]:
    base = os.path.basename(path).lower()
    kw_hits = sum(1 for kw in PREFER_KEYWORDS if kw in base)
    depth = -len(os.path.normpath(path).split(os.sep))
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    return (kw_hits, depth, mtime)


def pick_best_csv(files: List[str]) -> Optional[str]:
    if not files:
        return None
    return sorted(files, key=_score_csv_path, reverse=True)[0]


def read_csv_smart(path: str) -> pd.DataFrame:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "gbk",
        "gb18030",
        "cp936",
        "mbcs",
        "cp1252",
        "latin1",
    )
    tried, last_err = [], None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            tried.append(enc)
            last_err = e
    raise RuntimeError(f"读取CSV失败，尝试编码={tried}；最后错误：{last_err}")


# ------------------------------
# 可视化 (重写以使用预计算数据)
# ------------------------------
def _hide_non_trading(fig, idx):
    trading = pd.DatetimeIndex(pd.to_datetime(idx).normalize().unique())
    if len(trading) == 0:
        return
    all_days = pd.date_range(trading.min(), trading.max(), freq="D")
    non_trading_days = all_days.difference(trading)
    fig.update_xaxes(rangebreaks=[dict(values=non_trading_days.tolist())])


def make_figure(
    df: pd.DataFrame,
    title: str,
    visible_prob_set: Set[str],
    visible_price_ma_set: Set[int],
    separate_prob_panel: bool,
    show_volume: bool,
    smooth_win: int,
    threshold: Optional[float],
    date_start: Optional[str],
    date_end: Optional[str],
):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")

    if date_start:
        data = data.loc[data.index >= pd.to_datetime(date_start)]
    if date_end:
        data = data.loc[data.index <= pd.to_datetime(date_end)]
    if data.empty:
        return go.Figure().update_layout(
            title="所选时间段无数据", paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA"
        )

    # 涨跌幅（%）
    data["daily_change"] = data["close"].pct_change().mul(100).fillna(0)

    # 平滑后的概率
    prob_smoothed = (
        data["acs_prob"].rolling(max(1, int(smooth_win)), min_periods=1).mean()
    )

    # === 统一生成子图布局 ===
    rows, row_heights, specs, prob_row, vol_row = _subplot_params(
        separate_prob_panel, show_volume
    )
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        specs=specs,
    )

    # === K线（不变） ===
    hover_texts = []
    for idx, row in data.iterrows():
        color = INC_COLOR if row["daily_change"] >= 0 else DEC_COLOR
        hover_texts.append(
            f"<b>{idx:%Y-%m-%d}</b><br>"
            f"开盘: {row['open']:.2f}<br>"
            f"<b>收盘: {row['close']:.2f}</b><br>"
            f"最高: {row['high']:.2f}<br>"
            f"最低: {row['low']:.2f}<br>"
            f"<b><span style='color:{color}'>涨跌幅: {row['daily_change']:.2f}%</span></b><br>"
            f"换手率: {row.get('turnover', 0):.2f}%"
        )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            increasing=dict(line=dict(color=INC_COLOR, width=1), fillcolor=INC_COLOR),
            decreasing=dict(line=dict(color=DEC_COLOR, width=1), fillcolor=DEC_COLOR),
            name="K线",
            showlegend=False,
            hovertext=hover_texts,
            hoverinfo="text",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # === 价格均线（不变，只是用模块级调色板） ===
    for i, w in enumerate(sorted(list(visible_price_ma_set))):
        col_name = f"price_ma_{w}"
        if col_name in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col_name],
                    name=f"价格MA{w}",
                    mode="lines",
                    line=dict(
                        color=PRICE_MA_PALETTE[i % len(PRICE_MA_PALETTE)],
                        width=1.2,
                        dash="dot",
                    ),
                    legendgroup=f"price_ma_{w}",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    # === 概率主线：只添加一次，通过行号与 secondary_y 控制 ===
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=prob_smoothed,
            name="概率",
            mode="lines",
            line=dict(color="#0D6EFD", width=1.5 if not separate_prob_panel else 1),
            fill=("tozeroy" if separate_prob_panel else None),
            fillcolor=("rgba(77, 143, 253, 0.1)" if separate_prob_panel else None),
            hovertemplate="日期=%{x|%Y-%m-%d}<br>概率=%{y:.2f}<extra></extra>",
            legendgroup="prob",
            # 与原始逻辑保持一致：仅当非独立面板时显示在图例（避免重复图例）
            showlegend=not separate_prob_panel,
            visible=("概率" in visible_prob_set),
        ),
        row=prob_row,
        col=1,
        secondary_y=(not separate_prob_panel),
    )

    # === 概率均线（统一行号/secondary_y） ===
    for i, w in enumerate(PROB_MA_WINDOWS):
        name, col_name = f"MA{w}", f"prob_ma_{w}"
        if name in visible_prob_set and col_name in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col_name],
                    name=name,
                    mode="lines",
                    line=dict(
                        color=PROB_MA_PALETTE[i % len(PROB_MA_PALETTE)], width=1.2
                    ),
                    opacity=0.95,
                    legendgroup=f"prob_ma_{w}",
                ),
                row=prob_row,
                col=1,
                secondary_y=(not separate_prob_panel),
            )

    # === 阈值高亮（向量化） ===
    _highlight_threshold(fig, data.index, prob_smoothed, threshold, row=1, col=1)

    # === 成交量/换手率（与原逻辑等价） ===
    if show_volume and vol_row is not None:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data.get("volume", pd.Series(0, index=data.index)),
                name="成交量",
                marker=dict(
                    color=np.where(data["close"] >= data["open"], INC_COLOR, DEC_COLOR)
                ),
                opacity=0.7,
                hovertemplate="日期=%{x|%Y-%m-%d}<br>量=%{y:.0f}<extra></extra>",
            ),
            row=vol_row,
            col=1,
            secondary_y=False,
        )
        if "turnover" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["turnover"],
                    name="换手率",
                    mode="lines",
                    line=dict(color="#FFBF00", width=1.2),
                    hovertemplate="日期=%{x|%Y-%m-%d}<br>换手率=%{y:.2f}%<extra></extra>",
                ),
                row=vol_row,
                col=1,
                secondary_y=True,
            )

    # === 布局/坐标轴（与原参数一致） ===
    stock_code = title.split("_")[0] if title else ""
    fig.update_layout(
        title=None,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
        margin=dict(l=60, r=120, t=40, b=20),
        font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        hovermode="x unified",
        annotations=[
            dict(
                text=stock_code,
                align="right",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.00,
                y=1.07,
                font=dict(size=14, color="#6C757D"),
            )
        ],
    )
    for r in range(1, rows + 1):
        fig.update_yaxes(
            showgrid=True, gridcolor="#E9ECEF", zerolinecolor="#DEE2E6", row=r, col=1
        )

    fig.update_yaxes(
        title_text="价格", row=1, col=1, secondary_y=False, color="#212529"
    )
    fig.update_yaxes(
        title_text="概率", row=1, col=1, secondary_y=True, range=[0, 1], color="#0D6EFD"
    )
    if separate_prob_panel:
        fig.update_yaxes(
            title_text="概率(均线)", row=prob_row, col=1, range=[0, 1], color="#212529"
        )
    if show_volume and vol_row is not None:
        fig.update_yaxes(
            title_text="成交量",
            row=vol_row,
            col=1,
            secondary_y=False,
            rangemode="tozero",
            color="#212529",
            showgrid=False,
        )
        fig.update_yaxes(
            title_text="换手率(%)",
            row=vol_row,
            col=1,
            secondary_y=True,
            rangemode="tozero",
            color="#FFBF00",
            showgrid=False,
        )

    _hide_non_trading(fig, data.index)
    return fig


# ------------------------------
# Dash App
# ------------------------------
def build_app(title_default: str, root_dir: str):
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = title_default

    # --- 样式定义 ---
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

    SIDEBAR_STYLE_OPEN = {
        "boxSizing": "border-box",
        "position": "fixed",
        "top": f"{GAP}px",
        "bottom": f"{GAP}px",
        "left": f"{GAP}px",
        "width": f"{SIDEBAR_W}px",
        "overflowY": "auto",
        "maxHeight": f"calc(100vh - {GAP*2}px)",
        "padding": "20px",
        "background": COLORS["surface"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "16px",
        "boxShadow": "0 8px 24px rgba(0,0,0,0.05)",
        "zIndex": 9998,
        "transition": "left .3s ease",
    }
    SIDEBAR_STYLE_CLOSED = {**SIDEBAR_STYLE_OPEN, "left": f"-{SIDEBAR_W+GAP}px"}

    MAIN_STYLE_OPEN = {
        "boxSizing": "border-box",
        "position": "fixed",
        "top": f"{GAP}px",
        "bottom": f"{GAP}px",
        "right": f"{GAP}px",
        "left": f"{SIDEBAR_W + GAP*2}px",
        "background": COLORS["surface"],
        "borderRadius": "16px",
        "overflow": "hidden",
        "border": f"1px solid {COLORS['border']}",
        "transition": "left .3s ease, right .3s ease",
    }
    MAIN_STYLE_CLOSED = {**MAIN_STYLE_OPEN, "left": f"{GAP}px"}

    # --- 组件通用样式 ---
    LABEL_STYLE = {
        "fontWeight": "500",
        "marginBottom": "6px",
        "color": COLORS["text"],
        "fontSize": "14px",
    }
    INPUT_STYLE = {
        "boxSizing": "border-box",
        "width": "100%",
        "padding": "8px 12px",
        "borderRadius": "8px",
        "border": f"1px solid {COLORS['border']}",
        "background": COLORS["surface"],
        "color": COLORS["text"],
        "fontSize": "14px",
    }
    BUTTON_STYLE = {
        "background": COLORS["primary"],
        "color": "white",
        "border": "none",
        "borderRadius": "8px",
        "padding": "10px 16px",
        "fontWeight": "500",
        "cursor": "pointer",
        "transition": "background-color .2s ease",
    }
    BUTTON_SECONDARY_STYLE = {**BUTTON_STYLE, "background": COLORS["secondary"]}

    DROP_STYLE, ROW_FLEX = {"flex": "1 1 0", "minWidth": "92px"}, {
        "display": "flex",
        "gap": "8px",
        "alignItems": "center",
    }

    def _days_in_month(year: int, month: int) -> int:
        return calendar.monthrange(int(year), int(month))[1] if year and month else 31

    def _ymd_to_iso(y, m, d) -> str:
        if not (y and m and d):
            return pd.Timestamp.today().strftime("%Y-%m-%d")
        y, m, d = int(y), int(m), int(d)
        return f"{y:04d}-{m:02d}-{min(d, _days_in_month(y, m)):02d}"

    _now = pd.Timestamp.today()
    year_opts = [{"label": f"{y} 年", "value": y} for y in range(2005, _now.year + 1)]
    month_opts = [{"label": f"{m:02d} 月", "value": m} for m in range(1, 13)]
    start_y, start_m, start_d = 2019, 1, 1
    end_y, end_m, end_d = _now.year, _now.month, _now.day
    start_day_opts = [
        {"label": f"{d:02d} 日", "value": d}
        for d in range(1, _days_in_month(start_y, start_m) + 1)
    ]
    end_day_opts = [
        {"label": f"{d:02d} 日", "value": d}
        for d in range(1, _days_in_month(end_y, end_m) + 1)
    ]

    app.layout = html.Div(
        [
            dcc.Store(id="df-store"),
            dcc.Store(id="csv-path-store"),
            dcc.Store(id="title-store"),
            dcc.Store(id="uirevision-store"),
            html.Div(
                [
                    html.H3(
                        "一键 ACS 面板",
                        style={"color": COLORS["text"], "marginBottom": "4px"},
                    ),
                    html.Div(
                        f"Debug {DEBUG_VERSION}",
                        style={
                            "fontSize": "11px",
                            "color": COLORS["light_text"],
                            "marginBottom": "16px",
                        },
                    ),
                    html.Label("数据根目录", style=LABEL_STYLE),
                    dcc.Input(
                        id="root-dir",
                        type="text",
                        value=root_dir,
                        style={**INPUT_STYLE, "marginBottom": "12px"},
                    ),
                    html.Label("标的代码（如 603533）", style=LABEL_STYLE),
                    dcc.Input(
                        id="code-input",
                        type="text",
                        placeholder="输入后点击下方按钮",
                        debounce=True,
                        style={**INPUT_STYLE, "marginBottom": "12px"},
                    ),
                    html.Label("日期范围", style=LABEL_STYLE),
                    html.Div(
                        [
                            html.Div(
                                [
                                    "开始",
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="start-year",
                                                options=year_opts,
                                                value=start_y,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                            dcc.Dropdown(
                                                id="start-month",
                                                options=month_opts,
                                                value=start_m,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                            dcc.Dropdown(
                                                id="start-day",
                                                options=start_day_opts,
                                                value=start_d,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                        ],
                                        style={**ROW_FLEX, "marginBottom": "6px"},
                                    ),
                                ],
                                style={"fontSize": "12px", "color": "#666"},
                            ),
                            html.Div(
                                [
                                    "结束",
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="end-year",
                                                options=year_opts,
                                                value=end_y,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                            dcc.Dropdown(
                                                id="end-month",
                                                options=month_opts,
                                                value=end_m,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                            dcc.Dropdown(
                                                id="end-day",
                                                options=end_day_opts,
                                                value=end_d,
                                                clearable=False,
                                                style=DROP_STYLE,
                                            ),
                                        ],
                                        style=ROW_FLEX,
                                    ),
                                ],
                                style={"fontSize": "12px", "color": "#666"},
                            ),
                        ],
                        style={
                            "background": "#fafcff",
                            "border": "1px solid #eef2f7",
                            "borderRadius": "10px",
                            "padding": "10px",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.Button(
                                "运算", id="btn-compute", n_clicks=0, style=BUTTON_STYLE
                            ),
                            html.Span(
                                id="compute-msg",
                                style={
                                    "color": COLORS["light_text"],
                                    "fontSize": 12,
                                    "marginLeft": 12,
                                    "verticalAlign": "middle",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Label("调试信息", style=LABEL_STYLE),
                    dcc.Textarea(
                        id="debug-box",
                        value="",
                        style={
                            **INPUT_STYLE,
                            "height": "120px",
                            "whiteSpace": "pre",
                            "fontFamily": "Menlo, Consolas, monospace",
                        },
                    ),
                    html.Hr(
                        style={"margin": "20px 0", "borderColor": COLORS["border"]}
                    ),
                    html.H4("切换/载入已存在的CSV", style={"color": COLORS["text"]}),
                    html.Div(
                        [
                            html.Button(
                                "搜索CSV",
                                id="btn-search",
                                n_clicks=0,
                                style={**BUTTON_SECONDARY_STYLE, "marginRight": "8px"},
                            ),
                            html.Button(
                                "载入所选文件",
                                id="btn-load",
                                n_clicks=0,
                                style=BUTTON_SECONDARY_STYLE,
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        id="search-msg", style={"color": COLORS["light_text"], "fontSize": 12}
                    ),
                    dcc.Dropdown(
                        id="csv-dropdown",
                        options=[],
                        value=None,
                        placeholder="先搜索再选择",
                        style={
                            "boxSizing": "border-box",
                            "width": "100%",
                            "marginTop": "8px",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Hr(
                        style={"margin": "20px 0", "borderColor": COLORS["border"]}
                    ),
                    html.Div(
                        [
                            html.H4(
                                "可视化选项",
                                style={
                                    "color": COLORS["text"],
                                    "marginTop": "0px",
                                    "marginBottom": "15px",
                                },
                            ),
                            # --- 其他选项 ---
                            html.Label("图表设置", style=LABEL_STYLE),
                            dcc.Checklist(
                                id="options",
                                options=[
                                    {"label": "单独概率面板", "value": "separate"},
                                    {"label": "成交量面板", "value": "showvol"},
                                ],
                                value=["separate", "showvol"],
                                style={"marginBottom": "15px"},
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "15px",
                                    "fontSize": "13px",
                                },
                            ),
                            # --- 阈值滑块 ---
                            html.Label("概率阈值高亮 (≥)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="thr",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=0.70,
                                marks={0.0: "0.0", 0.5: "0.5", 1.0: "1.0"},
                                className="custom-slider",
                            ),
                            html.Div(style={"height": "12px"}),
                            # --- 平滑窗口 ---
                            html.Label("概率平滑窗口（天）", style=LABEL_STYLE),
                            dcc.Slider(
                                id="smooth",
                                min=1,
                                max=12,
                                step=1,
                                value=3,
                                marks={i: str(i) for i in [1, 3, 5, 7, 9, 12]},
                                className="custom-slider",
                            ),
                        ],
                        style={
                            "background": "#fafcff",
                            "border": "1px solid #eef2f7",
                            "borderRadius": "10px",
                            "padding": "15px",
                            "paddingTop": "10px",
                        },
                    ),
                    html.Div(
                        id="load-msg",
                        style={
                            "marginTop": "8px",
                            "color": COLORS["light_text"],
                            "fontSize": 12,
                        },
                    ),
                ],
                id="sidebar",
                style=SIDEBAR_STYLE_OPEN,
            ),
            html.Div(
                [
                    html.Button(
                        "‹",
                        id="btn-toggle-sidebar",
                        n_clicks=0,
                        title="收起/展开侧边栏",
                        style={
                            "position": "absolute",
                            "top": "12px",
                            "left": "12px",
                            "zIndex": 10000,
                            "fontSize": "18px",
                            "padding": "4px 10px",
                            "background": COLORS["hover_bg"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "8px",
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Graph(
                        id="chart",
                        config={"scrollZoom": True},
                        style={"height": "100%", "width": "100%"},
                    ),
                ],
                id="main",
                style=MAIN_STYLE_OPEN,
            ),
        ],
        style={"fontFamily": FONT_FAMILY, "background": COLORS["background"]},
    )

    # ... (Callbacks for date pickers and search are the same)

    @app.callback(
        [
            Output("sidebar", "style"),
            Output("main", "style"),
            Output("btn-toggle-sidebar", "children"),
        ],
        [Input("btn-toggle-sidebar", "n_clicks")],
        [State("sidebar", "style")],
        prevent_initial_call=True,
    )
    def toggle_sidebar(n_clicks, current_style):
        if n_clicks:
            if current_style and current_style.get("left") == f"{GAP}px":
                return SIDEBAR_STYLE_CLOSED, MAIN_STYLE_CLOSED, "›"
            return SIDEBAR_STYLE_OPEN, MAIN_STYLE_OPEN, "‹"
        return no_update, no_update, no_update

    @app.callback(
        [
            Output("csv-dropdown", "options"),
            Output("csv-dropdown", "value"),
            Output("search-msg", "children"),
        ],
        [Input("btn-search", "n_clicks")],
        [State("root-dir", "value"), State("code-input", "value")],
        prevent_initial_call=True,
    )
    def on_search_csv(n_clicks, root_dir, code):
        if not n_clicks:
            return no_update, no_update, no_update

        root = root_dir or "."
        try:
            files = list_csvs(root=root, code=code)
            if not files:
                return [], None, "在指定目录中未找到匹配的CSV文件。"

            options = [
                {"label": os.path.relpath(f, root), "value": f} for f in sorted(files)
            ]
            best_pick = pick_best_csv(files)
            msg = f"✅ 找到 {len(files)} 个文件，已为您选中最优结果。"
            return options, best_pick, msg

        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            return [], None, f"搜索文件时出错: {e}"

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
        ],
        prevent_initial_call=True,
    )
    def on_load_or_compute(
        n_load, n_comp, csv_path, code, root_dir, sy, sm, sd, ey, em, ed
    ):
        trig = (
            dash_ctx.triggered_id
            if dash_ctx
            else (
                cb_ctx.triggered[0]["prop_id"].split(".")[0]
                if cb_ctx.triggered
                else None
            )
        )
        start_iso, end_iso = _ymd_to_iso(sy, sm, sd), _ymd_to_iso(ey, em, ed)
        debug_lines = [f"[trigger] {trig}", f"[dates] {start_iso}..{end_iso}"]
        root = root_dir or "."

        try:
            if trig == "btn-compute":
                if not code or not str(code).strip():
                    msg = "请输入代码再点击‘运算’。"
                    debug_lines.append(f"[error] {msg}")
                    return (
                        no_update, no_update, no_update, no_update, msg, msg, "\n".join(debug_lines),
                    )

                cfg = PipelineConfig()
                cfg.output.root = root

                debug_lines.append(
                    f"[compute] 调用 acs_core.pipeline.compute, symbol={code}"
                )
                result = acs_compute(str(code).strip(), start_iso, end_iso, cfg)

                df_res = result["df"]
                saved_path = result["paths"].get("csv", "N/A")
                metrics = result["metrics"]

                prob_stats = df_res["acs_prob"].describe().to_string()
                debug_lines.append("[debug] `acs_prob` column stats:\n" + prob_stats)

                data_json = df_res.to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(saved_path))[0]
                auc_part = (
                    f" | OOF AUC≈{metrics.get('auc_cal', metrics.get('auc_raw', 'N/A')):.3f}"
                    if isinstance(metrics.get("auc_cal"), float)
                    or isinstance(metrics.get("auc_raw"), float)
                    else ""
                )
                note = f"✅ 运算完成: {saved_path}｜模式: {metrics.get('mode','?')}" + auc_part
                debug_lines.append(f"[done] {note}")
                return (
                    data_json, saved_path, ttl, str(time.time()), "", note, "\n".join(debug_lines),
                )

            if trig == "btn-load":
                if not csv_path:
                    msg = "请选择一个CSV文件。"
                    debug_lines.append(f"[error] {msg}")
                    return (
                        no_update, no_update, no_update, no_update, msg, "", "\n".join(debug_lines),
                    )

                df = read_csv_smart(csv_path)
                data_json = df.to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(csv_path))[0]
                note = f"✅ 已载入: {csv_path}"
                debug_lines.append(f"[done] {note}")
                return (
                    data_json, csv_path, ttl, str(csv_path), note, "", "\n".join(debug_lines),
                )

        except Exception as e:
            tb_str = traceback.format_exc()
            debug_lines.append(f"[CRITICAL] 操作失败: {e}\n{tb_str}")
            return (
                no_update, no_update, no_update, no_update, str(e), str(e), "\n".join(debug_lines),
            )

        return no_update, no_update, no_update, no_update, "", "", ""

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
        data_json,
        title,
        opts,
        smooth,
        thr,
        sy,
        sm,
        sd,
        ey,
        em,
        ed,
        uirevision,
    ):
        if not data_json:
            return go.Figure(
                layout={"title": "请先载入或运算数据", "uirevision": uirevision}
            )

        start_iso, end_iso = _ymd_to_iso(sy, sm, sd), _ymd_to_iso(ey, em, ed)
        df = pd.read_json(data_json, orient="split")

        # All lines are generated, visibility is controlled by legend click
        # Default visible lines
        visible_prob = {"概率", "MA5", "MA10", "MA20", "MA30", "MA60", "MA120"}
        visible_price_ma = {5, 20, 60}

        fig = make_figure(
            df,
            title,
            visible_prob,
            visible_price_ma,
            "separate" in (opts or []),
            "showvol" in (opts or []),
            smooth,
            thr,
            start_iso,
            end_iso,
        )
        fig.update_layout(uirevision=uirevision)  # Apply uirevision
        return fig

    @app.callback(
        Output("chart", "figure", allow_duplicate=True),
        Input("chart", "relayoutData"),
        [State("df-store", "data"), State("chart", "figure")],
        prevent_initial_call=True,
    )
    def adjust_yaxis_on_zoom(relayout_data, data_json, current_fig):
        if not relayout_data or not data_json or not current_fig:
            return no_update

        fig = go.Figure(current_fig)

        # 双击还原
        if relayout_data.get("xaxis.autorange"):
            fig.update_layout(yaxis={"autorange": True})
            return fig

        # 非缩放事件
        if (
            "xaxis.range[0]" not in relayout_data
            or "xaxis.range[1]" not in relayout_data
        ):
            return no_update

        try:
            x_start = pd.to_datetime(relayout_data["xaxis.range[0]"])
            x_end = pd.to_datetime(relayout_data["xaxis.range[1]"])
            df = (
                pd.read_json(data_json, orient="split")
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .set_index("date")
            )
            visible_df = df.loc[x_start:x_end]
            if visible_df.empty:
                return no_update

            y_min, y_max = visible_df["low"].min(), visible_df["high"].max()
            pad = (y_max - y_min) * 0.05
            fig.update_layout(
                yaxis={"autorange": False, "range": [y_min - pad, y_max + pad]}
            )
            return fig
        except (KeyError, TypeError, ValueError):
            return no_update

    return app


if __name__ == "__main__":
    # 设置默认输出目录为 'data'
    app = build_app(title_default="ACS 概率面板", root_dir="data")
    app.run(debug=True, port=8050)