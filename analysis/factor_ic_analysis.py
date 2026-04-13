"""
单因子 IC 分析

对所有自定义因子（及可选的 Alpha158 子集）计算截面 IC 时间序列，
输出：
  - 每个因子的 IC 均值、ICIR、IC>0 比例、IC>0.02 比例
  - 因子间相关性矩阵（识别冗余）
  - 各因子 IC 时间序列折线图（HTML）
  - 汇总 CSV

用法：
    python analysis/factor_ic_analysis.py
    python analysis/factor_ic_analysis.py --start 2021-01-01 --end 2024-12-31
    python analysis/factor_ic_analysis.py --include-alpha158-sample
"""

import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
from loguru import logger

from utils.helpers import setup_logger, get_model_config, get_data_config, ensure_dir, PROJECT_ROOT


# Alpha158 中有代表性的子集，用于对比自定义因子是否有增量信息
ALPHA158_SAMPLE = [
    ("$close/Ref($close,5)-1",   "A158_ROC5"),
    ("$close/Ref($close,20)-1",  "A158_ROC20"),
    ("$close/Ref($close,60)-1",  "A158_ROC60"),
    ("Std($close/Ref($close,1)-1,5)",  "A158_STD5"),
    ("Std($close/Ref($close,1)-1,20)", "A158_STD20"),
    ("Mean($turnover,5)",         "A158_TURN5"),
    ("Mean($turnover,20)",        "A158_TURN20"),
]


def _compute_label(instruments: list, start: str, end: str) -> pd.DataFrame:
    """计算原始 label（未经预处理）"""
    from qlib.data import D

    m_config = get_model_config()
    label_expr = m_config["label"]["expression"]

    raw = D.features(instruments, [label_expr], start_time=start, end_time=end)
    if raw is None or raw.empty:
        raise RuntimeError("无法计算 label，请检查 Qlib 数据是否完整")

    raw.columns = ["label"]
    if list(raw.index.names) == ["instrument", "datetime"]:
        raw = raw.swaplevel().sort_index()
    return raw


def _compute_factor(instruments: list, start: str, end: str, expr: str) -> pd.Series:
    """计算单个因子的截面值"""
    from qlib.data import D

    df = D.features(instruments, [expr], start_time=start, end_time=end)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df.columns = ["value"]
    if list(df.index.names) == ["instrument", "datetime"]:
        df = df.swaplevel().sort_index()
    return df["value"]


def _cross_sectional_ic(factor: pd.Series, label: pd.Series, method: str = "spearman") -> pd.Series:
    """逐日计算截面 IC"""
    combined = pd.DataFrame({"factor": factor, "label": label}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    ic = combined.groupby(level=0).apply(
        lambda x: x["factor"].corr(x["label"], method=method)
        if len(x) >= 5 else float("nan")
    )
    return ic.dropna()


def _summarize_ic(ic_series: pd.Series) -> dict:
    """从 IC 时间序列计算汇总指标"""
    if ic_series.empty:
        return {k: float("nan") for k in ["IC均值", "IC_std", "ICIR", "IC>0比例", "IC>0.02比例", "有效天数"]}

    return {
        "IC均值":      ic_series.mean(),
        "IC_std":      ic_series.std(),
        "ICIR":        ic_series.mean() / ic_series.std() if ic_series.std() > 0 else float("nan"),
        "IC>0比例":    (ic_series > 0).mean(),
        "IC>0.02比例": (ic_series > 0.02).mean(),
        "有效天数":    len(ic_series),
    }


def run_factor_ic_analysis(
    start: str = None,
    end: str = None,
    instruments: str = "csi300",
    include_alpha158_sample: bool = False,
    method: str = "spearman",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        (summary_df, ic_series_df)
        summary_df: 每个因子的汇总指标，index=因子名
        ic_series_df: 每个因子的 IC 时间序列，columns=因子名
    """
    from data.data_loader import get_data_loader
    get_data_loader().init_qlib()

    d_config = get_data_config()
    if start is None:
        start = d_config["split"]["test"]["start"]
    if end is None:
        end = d_config["split"]["test"]["end"]

    logger.info(f"分析区间: {start} ~ {end}")

    # 获取标的列表
    from qlib.data import D
    inst_list = D.list_instruments(
        D.instruments(instruments), start_time=start, end_time=end, as_list=True
    )
    logger.info(f"标的数量: {len(inst_list)}")

    # 计算 label
    logger.info("计算 label ...")
    label_df = _compute_label(inst_list, start, end)

    # 构建待分析因子列表
    from factors.custom_factors import get_custom_factor_expressions
    custom_factors = get_custom_factor_expressions()

    factor_list = list(custom_factors)
    if include_alpha158_sample:
        factor_list = ALPHA158_SAMPLE + factor_list

    logger.info(f"共 {len(factor_list)} 个因子待分析")

    summary_rows = []
    ic_series_dict = {}

    for expr, name in factor_list:
        logger.info(f"  计算因子 {name} ...")
        try:
            factor_vals = _compute_factor(inst_list, start, end, expr)
            if factor_vals.empty:
                logger.warning(f"  {name}: 因子值为空，跳过")
                continue

            ic = _cross_sectional_ic(factor_vals, label_df["label"], method=method)
            stats = _summarize_ic(ic)
            stats["因子名"] = name
            stats["表达式"] = expr
            summary_rows.append(stats)
            ic_series_dict[name] = ic

            logger.info(
                f"  {name}: IC均值={stats['IC均值']:.4f}, "
                f"ICIR={stats['ICIR']:.4f}, "
                f"IC>0={stats['IC>0比例']:.1%}"
            )
        except Exception as e:
            logger.warning(f"  {name}: 计算失败 ({e})")

    if not summary_rows:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("因子名")
        .sort_values("ICIR", ascending=False)
    )

    ic_series_df = pd.DataFrame(ic_series_dict)

    return summary_df, ic_series_df


def _compute_correlation_matrix(ic_series_df: pd.DataFrame) -> pd.DataFrame:
    """计算因子间 IC 时间序列相关性矩阵（识别冗余因子）"""
    return ic_series_df.corr(method="pearson")


def _save_results(
    summary_df: pd.DataFrame,
    ic_series_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    ensure_dir(output_dir)
    paths = {}

    # 汇总 CSV
    csv_path = output_dir / "factor_ic_summary.csv"
    summary_df.to_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"因子 IC 汇总已保存: {csv_path}")
    paths["summary_csv"] = str(csv_path)

    # IC 时间序列 CSV
    ts_csv = output_dir / "factor_ic_series.csv"
    ic_series_df.to_csv(ts_csv, encoding="utf-8-sig")
    paths["series_csv"] = str(ts_csv)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_factors = len(ic_series_df.columns)

        # ── 图1: 每个因子的 IC 时间序列 ──────────────────────────
        rows = max(1, (n_factors + 1) // 2)
        fig1 = make_subplots(
            rows=rows, cols=2,
            subplot_titles=list(ic_series_df.columns),
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )

        for i, col in enumerate(ic_series_df.columns):
            row = i // 2 + 1
            c = i % 2 + 1
            ic = ic_series_df[col].dropna()
            ic_mean = ic.mean()
            color = "#2196F3" if ic_mean >= 0 else "#F44336"

            # 柱状图（负值红色，正值蓝色）
            fig1.add_trace(
                go.Bar(
                    x=ic.index.astype(str),
                    y=ic.values,
                    name=col,
                    marker_color=[("#F44336" if v < 0 else "#2196F3") for v in ic.values],
                    showlegend=False,
                ),
                row=row, col=c,
            )
            # 均值水平线
            fig1.add_hline(
                y=ic_mean, line_dash="dash", line_color=color,
                annotation_text=f"均值={ic_mean:.4f}",
                annotation_position="top right",
                row=row, col=c,
            )
            # 零线
            fig1.add_hline(y=0, line_color="black", line_width=0.5, row=row, col=c)

        fig1.update_layout(
            title_text="各因子 IC 时间序列（负=红，正=蓝，虚线=均值）",
            height=max(400, rows * 250),
            font=dict(family="sans-serif", size=11),
        )

        ic_chart_path = output_dir / "factor_ic_series.html"
        fig1.write_html(str(ic_chart_path))
        logger.info(f"IC 时间序列图已保存: {ic_chart_path}")
        paths["ic_chart"] = str(ic_chart_path)

        # ── 图2: 汇总条形图（ICIR 排序）────────────────────────────
        if "ICIR" in summary_df.columns:
            sorted_df = summary_df["ICIR"].dropna().sort_values()
            colors = ["#F44336" if v < 0 else "#2196F3" for v in sorted_df.values]

            fig2 = go.Figure(go.Bar(
                x=sorted_df.values,
                y=sorted_df.index,
                orientation="h",
                marker_color=colors,
            ))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.add_vline(x=0.3, line_dash="dash", line_color="green",
                           annotation_text="ICIR=0.3(可用)", annotation_position="top")
            fig2.update_layout(
                title_text="因子 ICIR 排序（绝对值越大越好）",
                xaxis_title="ICIR",
                height=max(300, len(sorted_df) * 28 + 100),
                font=dict(family="sans-serif"),
            )

            bar_path = output_dir / "factor_icir_bar.html"
            fig2.write_html(str(bar_path))
            logger.info(f"ICIR 条形图已保存: {bar_path}")
            paths["icir_bar"] = str(bar_path)

        # ── 图3: 因子相关性热力图 ─────────────────────────────────
        if len(ic_series_df.columns) > 1:
            corr = _compute_correlation_matrix(ic_series_df)
            corr_csv = output_dir / "factor_correlation.csv"
            corr.to_csv(corr_csv, encoding="utf-8-sig")
            paths["corr_csv"] = str(corr_csv)

            import plotly.figure_factory as ff
            fig3 = ff.create_annotated_heatmap(
                z=corr.values.round(2),
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu_r",
                zmid=0,
                showscale=True,
            )
            fig3.update_layout(
                title_text="因子 IC 相关性矩阵（>0.7 视为高度冗余）",
                height=max(400, len(corr) * 40 + 150),
                font=dict(family="sans-serif", size=10),
            )

            heatmap_path = output_dir / "factor_correlation.html"
            fig3.write_html(str(heatmap_path))
            logger.info(f"相关性热力图已保存: {heatmap_path}")
            paths["heatmap"] = str(heatmap_path)

    except ImportError:
        logger.warning("plotly 未安装，跳过图表生成")

    return paths


def main():
    parser = argparse.ArgumentParser(description="单因子 IC 分析")
    parser.add_argument("--start", default=None, help="分析开始日期，默认使用 data_config 测试集开始")
    parser.add_argument("--end",   default=None, help="分析结束日期，默认使用 data_config 测试集结束")
    parser.add_argument("--instruments", default="csi300", help="标的池")
    parser.add_argument("--include-alpha158-sample", action="store_true",
                        help="同时分析 Alpha158 代表性因子，用于对比自定义因子的增量信息")
    parser.add_argument("--method", default="spearman",
                        choices=["spearman", "pearson"],
                        help="IC 计算方法，默认 spearman（Rank IC）")
    parser.add_argument("--output-dir", default="reports", help="输出目录")
    args = parser.parse_args()

    setup_logger()
    logger.info("=" * 60)
    logger.info("单因子 IC 分析")
    logger.info("=" * 60)

    summary_df, ic_series_df = run_factor_ic_analysis(
        start=args.start,
        end=args.end,
        instruments=args.instruments,
        include_alpha158_sample=args.include_alpha158_sample,
        method=args.method,
    )

    if summary_df.empty:
        logger.error("分析失败，无结果")
        return

    logger.info("\n" + "=" * 60)
    logger.info("因子 IC 分析汇总（按 ICIR 排序）")
    logger.info("=" * 60)

    display_cols = ["IC均值", "ICIR", "IC>0比例", "IC>0.02比例", "有效天数"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    logger.info("\n" + summary_df[display_cols].to_string(float_format="{:.4f}".format))

    # 识别高度冗余因子（IC 相关性 > 0.7）
    if len(ic_series_df.columns) > 1:
        corr = _compute_correlation_matrix(ic_series_df)
        high_corr_pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if abs(c) > 0.7:
                    high_corr_pairs.append((cols[i], cols[j], c))
        if high_corr_pairs:
            logger.info("\n高度冗余因子对（|相关系数| > 0.7）：")
            for a, b, c in high_corr_pairs:
                logger.info(f"  {a} ↔ {b}: {c:.3f}")

    # 识别弱因子（|ICIR| < 0.1）
    if "ICIR" in summary_df.columns:
        weak = summary_df[summary_df["ICIR"].abs() < 0.1]
        if not weak.empty:
            logger.info(f"\n弱因子（|ICIR| < 0.1，建议考虑移除）：")
            for name in weak.index:
                icir = summary_df.loc[name, "ICIR"]
                logger.info(f"  {name}: ICIR={icir:.4f}")

    output_dir = PROJECT_ROOT / args.output_dir
    _save_results(summary_df, ic_series_df, output_dir)


if __name__ == "__main__":
    main()
