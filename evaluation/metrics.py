"""
评估指标计算
核心指标通过 qlib.contrib.evaluate.risk_analysis() 获取，
仅补充 Qlib 未提供的少量自定义指标。
"""

from typing import Dict, Optional

import pandas as pd
from loguru import logger


def compute_metrics_from_report(
    report_df: pd.DataFrame,
) -> Dict[str, float]:
    """从 Qlib backtest_daily 返回的 report_df 计算全部指标

    直接调用 qlib.contrib.evaluate.risk_analysis()，
    它会计算：annualized_return, max_drawdown, sharpe, information_ratio 等。

    Args:
        report_df: backtest_daily 返回的 report DataFrame
                   columns: ['return', 'bench', 'cost', 'turnover', ...]

    Returns:
        指标字典
    """
    from qlib.contrib.evaluate import risk_analysis

    logger.info("使用 Qlib risk_analysis() 计算评估指标...")

    # risk_analysis 接受 report_df 整体，自动提取 return/bench 列
    # 返回 OrderedDict，包含子类别:
    #   'excess_return_without_cost', 'excess_return_with_cost',
    #   每个子类别包含: mean, std, annualized_return, max_drawdown, sharpe, information_ratio 等
    analysis_result = risk_analysis(report_df)

    # 提取关键指标
    metrics = {}

    # 从 excess_return_with_cost 子字典提取（含交易成本的指标更有意义）
    for category_name, category_df in analysis_result.items():
        if hasattr(category_df, 'to_dict'):
            for metric_name, value_series in category_df.items():
                if hasattr(value_series, 'iloc'):
                    val = value_series.iloc[0] if len(value_series) > 0 else value_series
                else:
                    val = value_series
                key = f"{category_name}/{metric_name}"
                metrics[key] = float(val) if pd.notna(val) else 0.0

    # 日志输出关键指标
    for name, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {name}: {val:.6f}")

    return metrics


def compute_metrics_from_returns(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """从收益率序列计算指标（兼容无 report_df 的情况）

    同样调用 risk_analysis，但先构造 report_df 格式。

    Args:
        returns: 策略日收益率序列
        benchmark_returns: 基准日收益率序列

    Returns:
        指标字典
    """
    from qlib.contrib.evaluate import risk_analysis

    if returns.empty or returns.isna().all():
        logger.error("收益率序列为空或全为 NaN")
        return {}

    logger.info("使用 Qlib risk_analysis() 计算评估指标...")

    # 构造 Qlib 期望的 report_df 格式
    report_df = pd.DataFrame({"return": returns})
    if benchmark_returns is not None:
        report_df["bench"] = benchmark_returns
    # cost 和 turnover 置零（如果没有的话）
    if "cost" not in report_df.columns:
        report_df["cost"] = 0.0
    if "turnover" not in report_df.columns:
        report_df["turnover"] = 0.0

    return compute_metrics_from_report(report_df)


def load_ic_from_recorder(recorder) -> Dict:
    """从 Qlib Recorder 加载 SigAnaRecord 已计算的 IC 指标

    SigAnaRecord.generate() 会将 IC/ICIR/Rank IC 等写入 recorder，
    无需手动重新计算。

    Args:
        recorder: Qlib Recorder 对象

    Returns:
        {"ic_mean": float, "icir": float, "rank_ic_mean": float, ...}
        如果获取失败返回 None
    """
    try:
        rec_metrics = recorder.list_metrics()
        if rec_metrics and ("IC" in rec_metrics or "ICIR" in rec_metrics):
            ic_summary = {
                "ic_mean": rec_metrics.get("IC", 0),
                "icir": rec_metrics.get("ICIR", 0),
                "rank_ic_mean": rec_metrics.get("Rank IC", 0),
                "rank_icir": rec_metrics.get("Rank ICIR", 0),
            }
            logger.info(f"IC 指标从 Recorder 获取: IC={ic_summary['ic_mean']:.4f}, "
                        f"ICIR={ic_summary['icir']:.4f}")
            return ic_summary
    except Exception as e:
        logger.warning(f"从 Recorder 获取 IC 失败: {e}")
    return None


def load_ic_series_from_recorder(recorder) -> Optional[pd.Series]:
    """从 Recorder 加载 IC 时间序列（用于可视化）

    SigAnaRecord 保存了 pred.pkl 和 label.pkl，
    可以用来计算每日 IC 序列。

    Args:
        recorder: Qlib Recorder 对象

    Returns:
        IC 时间序列 Series，失败返回 None
    """
    try:
        pred = recorder.load_object("pred.pkl")
        label = recorder.load_object("label.pkl")
        if pred is None or label is None:
            return None

        concat = pd.concat(
            [pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred,
             label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label],
            axis=1,
        )
        concat.columns = ["pred", "label"]
        concat = concat.dropna()

        ic = concat.groupby(level=0).apply(
            lambda x: x["pred"].corr(x["label"])
        )
        logger.info(f"IC 时间序列已计算: {len(ic)} 个交易日")
        return ic
    except Exception as e:
        logger.warning(f"计算 IC 时间序列失败: {e}")
        return None
