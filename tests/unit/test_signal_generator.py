"""
信号生成器测试
覆盖 topk 选股、新买入检测、排序、格式校验
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from tests.fixtures.sample_data import make_trading_dates, make_instruments


class TestGenerateTradeSignals:
    """测试 generate_trade_signals"""

    def _make_pred(self, n_instruments=5, n_days=3, seed=42):
        """构造测试用 pred_score"""
        rng = np.random.RandomState(seed)
        dates = make_trading_dates(n_days)
        instruments = make_instruments(n_instruments)
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        scores = rng.randn(len(index))
        return pd.DataFrame({"score": scores}, index=index)

    def test_top_k_count(self):
        """topk=3, 5 只股票，每日恰好 3 个「买入」信号"""
        from signal_gen.signal_generator import generate_trade_signals

        pred = self._make_pred(n_instruments=5, n_days=3)
        config = {"topk": 3, "n_drop": 1}
        df = generate_trade_signals(pred, config=config)

        for date in df["date"].unique():
            day_buys = df[(df["date"] == date) & (df["signal"] == "买入")]
            assert len(day_buys) == 3, (
                f"日期 {date}: 买入信号数={len(day_buys)}, 预期=3"
            )

    def test_new_buy_detection(self):
        """某只股票从非 topk 升入 topk 应标记为「新买入」"""
        from signal_gen.signal_generator import generate_trade_signals

        dates = make_trading_dates(2)
        instruments = ["A", "B", "C", "D", "E"]
        # Day 1: A > B > C > D > E → topk=3 → {A, B, C}
        # Day 2: E > A > B > C > D → topk=3 → {E, A, B}  E 是新买入
        rows = []
        for inst, score in zip(instruments, [5, 4, 3, 2, 1]):
            rows.append({"datetime": dates[0], "instrument": inst, "score": score})
        for inst, score in zip(instruments, [4, 3, 2, 1, 5]):
            rows.append({"datetime": dates[1], "instrument": inst, "score": score})

        pred = pd.DataFrame(rows).set_index(["datetime", "instrument"])
        config = {"topk": 3, "n_drop": 1}
        df = generate_trade_signals(pred, config=config)

        day2 = df[df["date"] == dates[1]]
        e_row = day2[day2["instrument"] == "E"]
        assert len(e_row) == 1
        assert e_row.iloc[0]["change"] == "新买入"

        # C 退出 topk → 新卖出
        c_row = day2[day2["instrument"] == "C"]
        assert c_row.iloc[0]["change"] == "新卖出"

    def test_sorted_by_rank(self):
        """rank=1 对应 score 最高"""
        from signal_gen.signal_generator import generate_trade_signals

        pred = self._make_pred(n_instruments=5, n_days=2)
        config = {"topk": 3, "n_drop": 1}
        df = generate_trade_signals(pred, config=config)

        for date in df["date"].unique():
            day = df[df["date"] == date].sort_values("rank")
            scores = day["score"].values
            # rank=1 的 score 应 >= rank=2 的 score …
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"排名未按 score 降序: rank={i+1} score={scores[i]}, "
                    f"rank={i+2} score={scores[i+1]}"
                )

    def test_series_input(self):
        """pred_score 为 Series 时也能正常工作"""
        from signal_gen.signal_generator import generate_trade_signals

        pred = self._make_pred(n_instruments=3, n_days=2)
        series = pred["score"]
        config = {"topk": 2, "n_drop": 1}
        df = generate_trade_signals(series, config=config)
        assert len(df) > 0
        assert "signal" in df.columns


class TestValidatePredScore:
    """测试 validate_pred_score"""

    @patch("signal_gen.signal_generator.get_strategy_config")
    def test_rejects_non_multiindex(self, mock_config):
        """普通 Index 应抛 ValueError"""
        from signal_gen.signal_generator import validate_pred_score

        mock_config.return_value = {"topk": 3}
        df = pd.DataFrame({"score": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(ValueError, match="MultiIndex"):
            validate_pred_score(df)

    @patch("signal_gen.signal_generator.get_strategy_config")
    def test_rejects_empty(self, mock_config):
        """空 pred_score 应抛 ValueError"""
        from signal_gen.signal_generator import validate_pred_score

        mock_config.return_value = {"topk": 3}
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="为空"):
            validate_pred_score(df)

    @patch("signal_gen.signal_generator.get_strategy_config")
    def test_accepts_valid_multiindex(self, mock_config):
        """合法的 MultiIndex pred_score 应通过校验"""
        from signal_gen.signal_generator import validate_pred_score

        mock_config.return_value = {"topk": 3}
        dates = make_trading_dates(3)
        instruments = make_instruments(5)
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred = pd.DataFrame({"score": np.random.randn(len(index))}, index=index)
        result = validate_pred_score(pred)
        assert result is not None
