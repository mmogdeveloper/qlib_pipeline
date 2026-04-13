"""
可视化模块 - 集成 qlib.contrib.report

提供两类图表:
  1. IC 时间序列图 (score_ic_graph): 每日 IC / Rank IC 折线图
  2. 组合回测分析图 (report_graph): 累计收益、超额收益、回撤、换手率

所有图表保存为 HTML 文件，可在浏览器直接查看交互式 Plotly 图。
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from utils.helpers import ensure_dir, PROJECT_ROOT


def generate_ic_chart(
    pred_label: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """生成 IC 时间序列图

    使用 qlib.contrib.report.analysis_position.score_ic_graph，
    同时展示 Pearson IC 和 Spearman Rank IC 两条曲线。

    Args:
        pred_label: MultiIndex=(datetime, instrument) 或 (instrument, datetime),
                    columns 必须包含 'score' 和 'label'
        output_dir: 输出目录，默认为 reports/

    Returns:
        保存的 HTML 文件路径，失败返回 None
    """
    try:
        from qlib.contrib.report.analysis_position import score_ic_graph
    except ImportError:
        logger.warning("qlib.contrib.report 不可用，跳过 IC 图表生成")
        return None

    if pred_label is None or pred_label.empty:
        logger.warning("pred_label 为空，跳过 IC 图表生成")
        return None

    if not {"score", "label"}.issubset(pred_label.columns):
        logger.warning(
            f"pred_label 缺少必要列 'score'/'label'，当前列: {list(pred_label.columns)}"
        )
        return None

    out_dir = ensure_dir(
        Path(output_dir) if output_dir else PROJECT_ROOT / "reports"
    )

    try:
        result = score_ic_graph(pred_label, show_notebook=False)
        if result is None:
            return None
        figs = list(result) if not isinstance(result, (list, tuple)) else result
        html_path = out_dir / "ic_chart.html"
        _save_figures_to_html(figs, html_path, title="IC 时间序列分析")
        logger.info(f"IC 图表已保存: {html_path}")
        return str(html_path)
    except Exception as e:
        logger.warning(f"IC 图表生成失败: {e}")
        return None


def generate_portfolio_chart(
    report_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """生成组合回测分析图

    使用 qlib.contrib.report.analysis_position.report_graph，
    包含: 累计收益 vs 基准、超额收益、最大回撤区间、日换手率。

    Args:
        report_df: backtest_daily 返回的 portfolio_metric DataFrame
                   必须包含列: ['return', 'bench', 'cost', 'turnover']
        output_dir: 输出目录，默认为 reports/

    Returns:
        保存的 HTML 文件路径，失败返回 None
    """
    try:
        from qlib.contrib.report.analysis_position import report_graph
    except ImportError:
        logger.warning("qlib.contrib.report 不可用，跳过组合图表生成")
        return None

    if report_df is None or report_df.empty:
        logger.warning("report_df 为空，跳过组合图表生成")
        return None

    required = {"return", "bench", "cost", "turnover"}
    missing = required - set(report_df.columns)
    if missing:
        logger.warning(f"report_df 缺少列 {missing}，跳过组合图表生成")
        return None

    out_dir = ensure_dir(
        Path(output_dir) if output_dir else PROJECT_ROOT / "reports"
    )

    try:
        result = report_graph(report_df, show_notebook=False)
        if result is None:
            return None
        figs = list(result) if not isinstance(result, (list, tuple)) else result
        html_path = out_dir / "portfolio_chart.html"
        _save_figures_to_html(figs, html_path, title="组合回测分析")
        logger.info(f"组合图表已保存: {html_path}")
        return str(html_path)
    except Exception as e:
        logger.warning(f"组合图表生成失败: {e}")
        return None


def generate_all_charts(
    report_df: Optional[pd.DataFrame] = None,
    pred_label: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """一次性生成所有图表

    Args:
        report_df:  backtest_daily 返回的 portfolio_metric
        pred_label: 含 score/label 列的预测标签 DataFrame
        output_dir: 输出目录

    Returns:
        {"ic_chart": path_or_None, "portfolio_chart": path_or_None}
    """
    return {
        "ic_chart": generate_ic_chart(pred_label, output_dir),
        "portfolio_chart": generate_portfolio_chart(report_df, output_dir),
    }


def _save_figures_to_html(figs, path: Path, title: str = "Qlib Analysis") -> None:
    """将 Plotly figure 列表序列化为单个 HTML 文件"""
    try:
        import plotly.io as pio
    except ImportError:
        logger.warning("plotly 未安装，无法保存 HTML 图表")
        return

    parts = [
        f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>",
        f"<h2 style='font-family:sans-serif;padding:16px'>{title}</h2>",
    ]
    for fig in figs:
        if fig is not None:
            parts.append(
                pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            )
    parts.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
