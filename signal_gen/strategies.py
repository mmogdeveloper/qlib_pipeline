"""自定义策略

- TopkScoreWeightedStrategy: 选 Top-K, 按预测分数归一化加权
- OptimizedWeightStrategy: 选 Top-K, 使用 PortfolioOptimizer 分配权重
  支持 gmv (最小方差) / mvo (均值方差) / rp (风险平价) / inv (逆波动率)
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


class OptimizedWeightStrategy(WeightStrategyBase):
    """Top-K 持仓 + 投资组合优化加权

    使用 Qlib 内置 PortfolioOptimizer 对 Top-K 股票分配最优权重。

    支持方法:
      - "inv": 逆波动率加权 (默认，无需全协方差矩阵，最稳健)
      - "rp":  风险平价 (等风险贡献)
      - "gmv": 全局最小方差
      - "mvo": 均值方差优化 (用预测分数作为期望收益输入)

    Args:
        topk:     持仓数量
        method:   优化方法，"inv"/"rp"/"gmv"/"mvo"
        lookback: 计算协方差使用的历史天数
        ret_df:   预计算的历史收益率矩阵 (datetime × instrument)，
                  由 portfolio._precompute_returns() 在回测前生成
    """

    def __init__(
        self,
        *,
        topk: int,
        method: str = "inv",
        lookback: int = 60,
        ret_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk = int(topk)
        self.method = method
        self.lookback = lookback
        self._ret_df = ret_df

        from qlib.contrib.strategy.optimizer import PortfolioOptimizer
        self._optimizer = PortfolioOptimizer(method=method)

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
        instruments = list(top.index)
        n = len(instruments)
        equal_w = {inst: 1.0 / n for inst in instruments}

        if self._ret_df is None or n < 2:
            return equal_w

        try:
            t = pd.Timestamp(trade_start_time)
            hist = self._ret_df.loc[self._ret_df.index <= t].tail(self.lookback)
            avail = [i for i in instruments if i in hist.columns]

            if len(avail) < 2:
                return equal_w

            hist_sub = hist[avail].dropna(how="all")
            if len(hist_sub) < 10:
                return equal_w

            cov_df = pd.DataFrame(
                hist_sub.cov().values,
                index=avail,
                columns=avail,
            )
            r_series = pd.Series(top[avail].values, index=avail) if self.method == "mvo" else None

            w = self._optimizer(S=cov_df, r=r_series)
            # w 是 pd.Series，index=avail
            result = {}
            for inst in instruments:
                if inst in avail and inst in w.index:
                    result[inst] = float(w[inst])
                else:
                    result[inst] = float(w.mean()) if len(w) > 0 else 1.0 / n

            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result

        except Exception:
            return equal_w
