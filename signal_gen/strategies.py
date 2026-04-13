"""自定义策略

- TopkScoreWeightedStrategy: 选 Top-K, 按预测分数归一化加权
"""

from typing import Optional

import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


class TopkScoreWeightedStrategy(WeightStrategyBase):
    """Top-K 持仓 + 分数加权

    每个交易日:
      1. 对当日预测分数排序, 取最高的 topk 只
      2. 将这些股票的分数做 min-max 平移后归一化为权重 (sum=1)
         (平移避免负分数, 让排序与权重单调一致)
      3. 输出目标权重字典 {instrument: weight}
    """

    def __init__(self, *, topk: int, **kwargs):
        super().__init__(**kwargs)
        self.topk = int(topk)

    def generate_target_weight_position(
        self,
        score: pd.Series,
        current,
        trade_start_time,
        trade_end_time,
    ) -> dict:
        if score is None or len(score) == 0:
            return {}

        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]

        score = score.dropna()
        if score.empty:
            return {}

        top = score.nlargest(self.topk)
        # min-max 平移: 让最低分变 0, 最高分变 max-min, 保持单调
        shifted = top - top.min()
        total = shifted.sum()
        if total <= 0 or not np.isfinite(total):
            # 全部分数相同时退化为等权
            weights = pd.Series(1.0 / len(top), index=top.index)
        else:
            weights = shifted / total
            # 极端情况防御: 若某只权重为 0(最低分), 给一个最小占比避免组合空仓偏离
            min_w = 0.005
            weights = weights.clip(lower=min_w)
            weights = weights / weights.sum()

        return weights.to_dict()
