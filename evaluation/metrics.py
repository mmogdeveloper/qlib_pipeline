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

    调用 qlib.contrib.evaluate.risk_analysis() 分别计算：
    - 含成本超额收益指标 (excess_return_with_cost)
    - 不含成本超额收益指标 (excess_return_without_cost)

    注意: Qlib 0.9.7 的 risk_analysis(r) 接受 pd.Series，不是 DataFrame。
    必须从 report_df 中提取正确的收益率 Series 后再传入。

    Args:
        report_df: backtest_daily 返回的 report DataFrame
                   columns: ['return', 'bench', 'cost', 'turnover', ...]

    Returns:
        指标字典
    """
    from qlib.contrib.evaluate import risk_analysis

    logger.info("使用 Qlib risk_analysis() 计算评估指标...")

    # ── 校验 report_df 结构 ──────────────────────────────────
    required_cols = {"return", "bench", "cost"}
    missing = required_cols - set(report_df.columns)
    if missing:
        raise ValueError(
            f"report_df 缺少必要列 {missing}，实际列: {list(report_df.columns)}。"
            f"请检查 backtest_daily 返回结构。"
        )

    # 日收益率合理性检查：A股单日涨跌幅上限 ±20%（含 ST/创业板等）
    ret = report_df["return"]
    if ret.abs().max() > 0.3:
        logger.error(
            f"日收益率异常: max={ret.max():.6f}, min={ret.min():.6f}。"
            f"输入可能是资金量而非收益率。"
        )

    logger.info(
        f"report_df 概览: {len(report_df)} 天, "
        f"日均收益={ret.mean():.6f}, 日收益std={ret.std():.6f}"
    )

    # ── 计算各类指标 ─────────────────────────────────────────
    # Qlib 0.9.7: risk_analysis(r) 接受 Series，返回 DataFrame(column='risk')
    # 分别对三种收益率序列调用 risk_analysis
    metrics = {}

    # 1) 含成本超额收益 = return - bench（return 已扣成本? 否: return - cost - bench）
    #    Qlib report_df 中 return 是扣成本前的收益率，cost 是成本率
    excess_return_with_cost = report_df["return"] - report_df["cost"] - report_df["bench"]
    analysis = risk_analysis(excess_return_with_cost)
    for metric_name in analysis.index:
        metrics[f"excess_return_with_cost/{metric_name}"] = float(analysis.loc[metric_name, "risk"])

    # 2) 不含成本超额收益
    excess_return_without_cost = report_df["return"] - report_df["bench"]
    analysis = risk_analysis(excess_return_without_cost)
    for metric_name in analysis.index:
        metrics[f"excess_return_without_cost/{metric_name}"] = float(analysis.loc[metric_name, "risk"])

    # 3) 策略绝对收益（含成本）
    strategy_return = report_df["return"] - report_df["cost"]
    analysis = risk_analysis(strategy_return)
    for metric_name in analysis.index:
        metrics[f"return_with_cost/{metric_name}"] = float(analysis.loc[metric_name, "risk"])

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


def load_ic_series_from_recorder(recorder, use_raw_label: bool = False) -> Optional[pd.Series]:
    """从 Recorder 加载 IC 时间序列（用于可视化）

    Args:
        recorder: Qlib Recorder 对象
        use_raw_label: 若 True，用原始收益率而非 CSRankNorm 后的 label 计算 IC
                       这提供了一个不受 label 预处理影响的独立 IC 校验

    Returns:
        IC 时间序列 Series，失败返回 None
    """
    try:
        pred = recorder.load_object("pred.pkl")
        if pred is None:
            return None

        if use_raw_label:
            # 用原始 label（未经 CSRankNorm）重新计算 IC，作为独立校验
            label = _compute_raw_labels(pred)
            if label is None:
                logger.warning("无法计算原始 label，回退到 label.pkl")
                label = recorder.load_object("label.pkl")
        else:
            label = recorder.load_object("label.pkl")

        if label is None:
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
        label_type = "原始" if use_raw_label else "CSRankNorm"
        ic_mean = ic.mean()
        logger.info(f"IC 时间序列已计算({label_type} label): {len(ic)} 个交易日, "
                     f"IC均值={ic_mean:.4f}" if not pd.isna(ic_mean) else
                     f"IC 时间序列已计算({label_type} label): {len(ic)} 个交易日, IC均值=NaN")
        return ic
    except Exception as e:
        logger.warning(f"计算 IC 时间序列失败: {e}")
        return None


def _compute_raw_labels(pred: pd.DataFrame) -> Optional[pd.Series]:
    """从 Qlib 数据源重新计算原始 label（未经 CSRankNorm）

    用于独立校验 IC，避免 CSRankNorm 对 IC 的放大效应。
    """
    try:
        import qlib
        from qlib.data import D
        from utils.helpers import get_model_config

        m_config = get_model_config()
        label_expr = m_config["label"]["expression"]

        # 从 pred 的 index 提取 instruments 和时间范围
        dates = pred.index.get_level_values(0)
        instruments = pred.index.get_level_values(1).unique().tolist()
        start = dates.min()
        end = dates.max()

        # 直接从 Qlib 数据源计算原始 label
        raw_label = D.features(
            instruments=instruments,
            fields=[label_expr],
            start_time=start,
            end_time=end,
        )
        if raw_label is not None and not raw_label.empty:
            raw_label.columns = ["label"]
            return raw_label
    except Exception as e:
        logger.debug(f"计算原始 label 失败: {e}")
    return None
