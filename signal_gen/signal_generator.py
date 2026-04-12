"""
信号生成器（薄封装层）

实际的选股逻辑由 Qlib 的 TopkDropoutStrategy 内部处理，
本模块仅提供 pred_score 的预处理和格式校验。

TopkDropoutStrategy 在 qlib.contrib.strategy.signal_strategy 中实现，
其内部逻辑：
1. 每日按 signal(pred_score) 排序
2. 选出 topk 只股票
3. 对比上期持仓，最多换出 n_drop 只（降低换手率）
4. 生成买卖交易决策传给 Executor
"""

from typing import Optional

import pandas as pd
from loguru import logger

from utils.helpers import get_strategy_config


def validate_pred_score(pred_score: pd.DataFrame) -> pd.DataFrame:
    """校验预测分数格式是否符合 Qlib 策略要求

    Qlib TopkDropoutStrategy 期望 signal 格式:
    - DataFrame 或 Series
    - MultiIndex: (datetime, instrument)
    - 值为浮点数预测分数

    Args:
        pred_score: 模型预测输出

    Returns:
        校验通过的 pred_score

    Raises:
        ValueError: 格式不符
    """
    if pred_score is None or (hasattr(pred_score, 'empty') and pred_score.empty):
        raise ValueError("pred_score 为空")

    if not isinstance(pred_score.index, pd.MultiIndex):
        raise ValueError(
            f"pred_score 应为 MultiIndex (datetime, instrument)，"
            f"实际为 {type(pred_score.index)}"
        )

    n_dates = pred_score.index.get_level_values(0).nunique()
    n_instruments = pred_score.index.get_level_values(1).nunique()
    nan_pct = pred_score.isna().mean()
    if isinstance(nan_pct, pd.Series):
        nan_pct = nan_pct.iloc[0]

    logger.info(
        f"预测信号校验: {n_dates} 交易日, {n_instruments} 只股票, "
        f"NaN 占比 {nan_pct:.2%}"
    )

    config = get_strategy_config()
    if n_instruments < config["topk"]:
        logger.warning(
            f"股票数({n_instruments}) < TopK({config['topk']})，"
            f"策略可能无法选满 {config['topk']} 只"
        )

    return pred_score


def generate_trade_signals(pred_score: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """从模型预测分数生成买卖信号

    逻辑：
    1. 每个交易日按 score 降序排列
    2. 排名前 topk 的标的标记为「买入」信号
    3. 排名后 topk 的标的标记为「卖出」信号
    4. 其余标的标记为「持有/观望」
    5. 相邻交易日对比，新进入 topk 的为「新买入」，退出的为「新卖出」

    Args:
        pred_score: MultiIndex (datetime, instrument) 的预测分数
        config: 策略配置（topk, n_drop 等）

    Returns:
        DataFrame with columns:
            date, instrument, score, rank, signal, prev_signal
        signal 取值: 买入, 卖出, 观望
    """
    config = config or get_strategy_config()
    topk = config["topk"]

    if isinstance(pred_score, pd.Series):
        scores = pred_score.to_frame("score")
    elif isinstance(pred_score, pd.DataFrame):
        scores = pred_score.copy()
        if "score" not in scores.columns:
            scores.columns = ["score"]
    else:
        raise ValueError(f"pred_score 类型不支持: {type(pred_score)}")

    scores = scores.reset_index()
    scores.columns = ["date", "instrument", "score"]

    results = []
    prev_buy_set = set()

    for date, group in scores.groupby("date"):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        group["rank"] = range(1, len(group) + 1)

        n = len(group)
        buy_set = set(group.head(topk)["instrument"])
        sell_set = set(group.tail(topk)["instrument"]) if n > topk else set()

        for _, row in group.iterrows():
            inst = row["instrument"]
            if inst in buy_set:
                signal = "买入"
            elif inst in sell_set:
                signal = "卖出"
            else:
                signal = "观望"

            # 对比前一交易日信号变化
            if inst in buy_set and inst not in prev_buy_set:
                change = "新买入"
            elif inst not in buy_set and inst in prev_buy_set:
                change = "新卖出"
            else:
                change = ""

            results.append({
                "date": date,
                "instrument": inst,
                "score": row["score"],
                "rank": row["rank"],
                "signal": signal,
                "change": change,
            })

        prev_buy_set = buy_set

    df = pd.DataFrame(results)
    n_buy = (df["signal"] == "买入").sum()
    n_sell = (df["signal"] == "卖出").sum()
    n_new_buy = (df["change"] == "新买入").sum()
    n_new_sell = (df["change"] == "新卖出").sum()
    logger.info(f"信号生成完成: 买入 {n_buy}, 卖出 {n_sell}, 新买入 {n_new_buy}, 新卖出 {n_new_sell}")
    return df
