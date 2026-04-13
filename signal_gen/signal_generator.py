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


def _update_hold_with_ndrop(
    current_hold: set,
    ranked: list,
    topk: int,
    n_drop: int,
) -> tuple:
    """按 n_drop 约束更新持仓集合

    与 Qlib TopkDropoutStrategy 的逻辑保持一致：
    - 空仓时直接买入 topk 只（初始建仓，不受 n_drop 限制）
    - 已有持仓时，每期最多主动替换 n_drop 只
      * 候选卖出：当前持仓中跌出 top-k 的股票
      * 候选买入：top-k 中尚未持有的股票
      * 实际替换数 = min(候选卖出数, 候选买入数, n_drop)
      * 优先卖出排名最差（分数最低）的候选股

    注意：本函数不处理涨跌停/停牌等不可交易过滤，
    实盘实际换手以 Qlib 回测引擎执行结果为准。

    Args:
        current_hold: 当前持仓集合
        ranked:       今日按分数降序排列的股票列表
        topk:         持仓数量上限
        n_drop:       每期最多主动换出数量

    Returns:
        (new_hold, buys, sells)
        new_hold: 更新后的持仓集合
        buys:     本期新买入集合
        sells:    本期新卖出集合
    """
    topk_set = set(ranked[:topk])

    if not current_hold:
        new_hold = topk_set
        return new_hold, topk_set, set()

    outside_topk = [s for s in current_hold if s not in topk_set]
    entering_topk = [s for s in ranked[:topk] if s not in current_hold]
    n_replace = min(len(outside_topk), len(entering_topk), n_drop)

    if n_replace == 0:
        return set(current_hold), set(), set()

    rank_map = {s: i for i, s in enumerate(ranked)}
    outside_sorted = sorted(
        outside_topk,
        key=lambda s: rank_map.get(s, len(ranked)),
        reverse=True,
    )
    sells = set(outside_sorted[:n_replace])
    buys = set(entering_topk[:n_replace])
    new_hold = (current_hold - sells) | buys
    return new_hold, buys, sells


def generate_trade_signals(pred_score: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """从模型预测分数生成买卖信号（含 n_drop 换手约束）

    逻辑：
    1. 每个交易日按 score 降序排列
    2. 按 n_drop 约束维护实际持仓（与回测引擎行为一致）
    3. 持仓内标的标记为「买入」；分数最低的 topk 只标记为「卖出」（供参考）
    4. 相邻交易日对比实际持仓变化，新进入为「新买入」，退出为「新卖出」

    「卖出」列表仅为模型评分最低的 topk 只，供参考，
    并非实际卖出指令——真实卖出以「新卖出」变动为准。

    Args:
        pred_score: MultiIndex (datetime, instrument) 的预测分数
        config:     策略配置（topk, n_drop 等）

    Returns:
        DataFrame with columns:
            date, instrument, score, rank, signal, change
        signal 取值: 买入（持仓中）, 卖出（评分垫底）, 观望
        change 取值: 新买入, 新卖出, 空字符串
    """
    config = config or get_strategy_config()
    topk = config["topk"]
    n_drop = config.get("n_drop", topk)

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
    current_hold: set = set()

    for date, group in scores.groupby("date"):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        group["rank"] = range(1, len(group) + 1)
        ranked = group["instrument"].tolist()

        n = len(group)
        sell_display_set = set(group.tail(topk)["instrument"]) if n > topk else set()

        new_hold, buys, sells = _update_hold_with_ndrop(current_hold, ranked, topk, n_drop)

        for _, row in group.iterrows():
            inst = row["instrument"]
            if inst in new_hold:
                signal = "买入"
            elif inst in sell_display_set:
                signal = "卖出"
            else:
                signal = "观望"

            if inst in buys:
                change = "新买入"
            elif inst in sells:
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

        current_hold = new_hold

    df = pd.DataFrame(results)
    n_buy = (df["signal"] == "买入").sum()
    n_sell = (df["signal"] == "卖出").sum()
    n_new_buy = (df["change"] == "新买入").sum()
    n_new_sell = (df["change"] == "新卖出").sum()
    n_days = df["date"].nunique()
    logger.info(
        f"信号生成完成 (n_drop={n_drop}): 持仓 {n_buy}, 参考卖出 {n_sell}, "
        f"新买入 {n_new_buy} ({n_new_buy/n_days:.1f}/日), "
        f"新卖出 {n_new_sell} ({n_new_sell/n_days:.1f}/日)"
    )
    return df
