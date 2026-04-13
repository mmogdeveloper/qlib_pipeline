"""
n_drop 敏感性分析

对同一份 pred.pkl 用不同的 n_drop 值（1/3/5/10/topk）分别回测，
量化换手约束对 alpha 的衰减效应。

用法：
    python analysis/ndrop_sensitivity.py
    python analysis/ndrop_sensitivity.py --model lgbm --n-drops 1 3 5 10 20
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
import pandas as pd
from loguru import logger

from utils.helpers import (
    setup_logger,
    get_backtest_config,
    get_strategy_config,
    set_config_override,
    clear_config_overrides,
    ensure_dir,
    PROJECT_ROOT,
)


DEFAULT_N_DROPS = [1, 3, 5, 10, 20]
# 当前策略默认 n_drop=3；若修改 strategy_config.yaml 请同步改此处
_CURRENT_N_DROP = 3


def _load_recorder(model_name: str, rolling: bool = False):
    """从 MLflow 加载最近一次实验的 recorder"""
    from qlib.workflow import R

    suffix = "_rolling" if rolling else ""
    exp_name = f"qlib_pipeline_{model_name}{suffix}"
    try:
        experiment = R.get_exp(experiment_name=exp_name)
        recorders = experiment.list_recorders(rtype=experiment.RT_L)
    except Exception as e:
        raise RuntimeError(f"找不到实验 '{exp_name}'，请先运行模型训练: {e}")

    if not recorders:
        raise RuntimeError(f"实验 '{exp_name}' 中无记录，请先运行模型训练")

    if isinstance(recorders, dict):
        candidates = list(recorders.values())
    else:
        candidates = recorders

    finished = [r for r in candidates if r.info.get("status") == "FINISHED"]
    pool = finished or candidates
    return max(pool, key=lambda r: r.info.get("start_time") or r.info.get("id", ""))


def _extract_key_metrics(report_df: pd.DataFrame) -> dict:
    """从 backtest_daily 的 report_df 提取关键指标"""
    from qlib.contrib.evaluate import risk_analysis

    ret_with_cost = report_df["return"] - report_df["cost"]
    excess = ret_with_cost - report_df["bench"]

    analysis = risk_analysis(excess)
    ra = analysis["risk"]

    turnover_mean = report_df["turnover"].mean() if "turnover" in report_df.columns else float("nan")

    return {
        "年化超额收益": ra.get("annualized_return", float("nan")),
        "超额Sharpe":   ra.get("information_ratio", float("nan")),
        "最大回撤":     ra.get("max_drawdown", float("nan")),
        "日均换手率":   turnover_mean,
        "胜率":         ra.get("win_rate", float("nan")),
    }


def run_sensitivity(
    model_name: str = "lgbm",
    n_drops: list[int] = None,
    rolling: bool = False,
) -> pd.DataFrame:
    """对同一份 pred.pkl 用不同 n_drop 值回测，返回对比表格"""
    from data.data_loader import get_data_loader
    from signal_gen.portfolio import run_backtest_from_recorder

    get_data_loader().init_qlib()
    recorder = _load_recorder(model_name, rolling)

    n_drops = n_drops or DEFAULT_N_DROPS
    topk = get_strategy_config()["topk"]

    rows = []
    for nd in n_drops:
        label = f"n_drop={nd}" if nd < topk else f"n_drop={nd}(无限制)"
        logger.info(f"回测 {label} ...")

        clear_config_overrides("strategy")
        set_config_override("strategy", {"n_drop": nd})

        try:
            result = run_backtest_from_recorder(recorder)
            portfolio_metric = result["portfolio_metric"]

            if isinstance(portfolio_metric, tuple):
                report_df = portfolio_metric[0]
            else:
                report_df = portfolio_metric

            if report_df is None or report_df.empty:
                logger.warning(f"  {label}: 回测结果为空，跳过")
                continue

            metrics = _extract_key_metrics(report_df)
            metrics["n_drop"] = nd
            rows.append(metrics)
            logger.info(
                f"  {label}: 超额年化={metrics['年化超额收益']:.4f}, "
                f"Sharpe={metrics['超额Sharpe']:.4f}, "
                f"最大回撤={metrics['最大回撤']:.4f}, "
                f"换手率={metrics['日均换手率']:.4f}"
            )
        except Exception as e:
            logger.error(f"  {label} 回测失败: {e}")

    clear_config_overrides("strategy")

    if not rows:
        logger.error("所有 n_drop 回测均失败，无结果")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("n_drop")
    return df


def _save_results(df: pd.DataFrame, output_dir: Path) -> dict:
    """保存对比表格（CSV）和可视化（HTML）"""
    ensure_dir(output_dir)
    paths = {}

    # CSV
    csv_path = output_dir / "ndrop_sensitivity.csv"
    df.to_csv(csv_path, encoding="utf-8-sig")
    logger.info(f"对比表格已保存: {csv_path}")
    paths["csv"] = str(csv_path)

    # HTML 图表（4 指标折线/柱状图）
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        metrics_to_plot = [
            ("年化超额收益", "超额年化收益"),
            ("超额Sharpe",   "超额 Sharpe 比率"),
            ("最大回撤",     "最大回撤（绝对值）"),
            ("日均换手率",   "日均换手率"),
        ]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m[1] for m in metrics_to_plot],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        x = df.index.tolist()
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for (col, title), (row, col_pos) in zip(metrics_to_plot, positions):
            if col not in df.columns:
                continue
            y = df[col].tolist()
            # 最大回撤取绝对值展示（负数更直观）
            if col == "最大回撤":
                y = [abs(v) for v in y]
            fig.add_trace(
                go.Bar(
                    x=[str(v) for v in x],
                    y=y,
                    name=title,
                    showlegend=False,
                    marker_color="#4C8BF5",
                ),
                row=row,
                col=col_pos,
            )
            # 标记当前配置（红点）
            if _CURRENT_N_DROP in x:
                cur_idx = x.index(_CURRENT_N_DROP)
                fig.add_trace(
                    go.Scatter(
                        x=[str(_CURRENT_N_DROP)],
                        y=[y[cur_idx]],
                        mode="markers",
                        marker=dict(color="red", size=10, symbol="circle"),
                        name=f"当前配置(n_drop={_CURRENT_N_DROP})" if col == "年化超额收益" else None,
                        showlegend=(col == "年化超额收益"),
                    ),
                    row=row,
                    col=col_pos,
                )

        fig.update_layout(
            title_text="n_drop 敏感性分析（横轴=n_drop 值，红点=当前配置）",
            height=600,
            font=dict(family="sans-serif"),
            legend=dict(x=0.01, y=-0.05, orientation="h"),
        )
        fig.update_xaxes(title_text="n_drop")

        html_path = output_dir / "ndrop_sensitivity.html"
        fig.write_html(str(html_path))
        logger.info(f"可视化图表已保存: {html_path}")
        paths["html"] = str(html_path)

    except ImportError:
        logger.warning("plotly 未安装，跳过图表生成")

    return paths


def main():
    parser = argparse.ArgumentParser(description="n_drop 敏感性分析")
    parser.add_argument("--model", default="lgbm", help="模型名称")
    parser.add_argument(
        "--n-drops", nargs="+", type=int,
        default=DEFAULT_N_DROPS,
        help="要测试的 n_drop 值列表，默认 1 3 5 10 20",
    )
    parser.add_argument("--rolling", action="store_true", help="使用滚动验证的 recorder")
    parser.add_argument("--output-dir", default="reports", help="输出目录")
    args = parser.parse_args()

    setup_logger()
    logger.info("=" * 60)
    logger.info("n_drop 敏感性分析")
    logger.info(f"模型: {args.model}, n_drop 候选值: {args.n_drops}")
    logger.info("=" * 60)

    df = run_sensitivity(
        model_name=args.model,
        n_drops=args.n_drops,
        rolling=args.rolling,
    )

    if df.empty:
        logger.error("分析失败，无结果")
        return

    logger.info("\n" + "=" * 60)
    logger.info("n_drop 敏感性分析汇总")
    logger.info("=" * 60)
    logger.info("\n" + df.to_string(float_format="{:.4f}".format))

    output_dir = PROJECT_ROOT / args.output_dir
    _save_results(df, output_dir)

    # 输出结论
    best_ndrop = df["超额Sharpe"].idxmax()
    current_ndrop = _CURRENT_N_DROP
    current_sharpe = df.loc[current_ndrop, "超额Sharpe"] if current_ndrop in df.index else float("nan")
    best_sharpe = df.loc[best_ndrop, "超额Sharpe"]
    if best_ndrop != current_ndrop:
        logger.info(
            f"\n结论: 当前 n_drop={current_ndrop} 的超额 Sharpe={current_sharpe:.4f}，"
            f"最优 n_drop={best_ndrop} 的 Sharpe={best_sharpe:.4f}，"
            f"提升 {best_sharpe - current_sharpe:.4f}。"
            f"建议在 strategy_config.yaml 中将 n_drop 调整为 {best_ndrop}。"
        )
    else:
        logger.info(f"\n结论: 当前 n_drop={current_ndrop} 已是最优配置（Sharpe={best_sharpe:.4f}）。")


if __name__ == "__main__":
    main()
