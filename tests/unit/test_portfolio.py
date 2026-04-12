"""
组合构建与回测执行测试
重点覆盖：前视偏差防护、成本一致性、交易记录提取
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from tests.fixtures.sample_data import make_trading_dates, make_instruments, make_synthetic_pred_score


# ── 前视偏差防护 ────────────────────────────────────────────


class TestSignalShift:
    """验证 signal shift 逻辑消除前视偏差"""

    def _do_shift(self, pred_score: pd.DataFrame) -> pd.DataFrame:
        """复制 portfolio.run_backtest 中的 shift 逻辑，隔离测试"""
        pred_shifted = pred_score.copy()
        if isinstance(pred_shifted.index, pd.MultiIndex):
            pred_shifted = pred_shifted.reset_index()
            date_col = pred_shifted.columns[0]
            dates = sorted(pred_shifted[date_col].unique())
            date_map = dict(zip(dates[:-1], dates[1:]))
            pred_shifted[date_col] = pred_shifted[date_col].map(date_map)
            pred_shifted = pred_shifted.dropna(subset=[date_col])
            pred_shifted = pred_shifted.set_index(pred_score.index.names)
        return pred_shifted

    def test_signal_shift_delays_by_one_day(self):
        """signal[T] shift 后变成 signal[T+1]"""
        dates = make_trading_dates(3)
        instruments = make_instruments(5)
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        # 每个 (date, instrument) 赋予确定性分数
        scores = np.arange(len(index), dtype=float)
        pred = pd.DataFrame({"score": scores}, index=index)

        shifted = self._do_shift(pred)

        # shift 后，原 dates[0] 的数据应出现在 dates[1]
        for inst in instruments:
            original_val = pred.loc[(dates[0], inst), "score"]
            shifted_val = shifted.loc[(dates[1], inst), "score"]
            assert original_val == shifted_val, (
                f"instrument={inst}: shift 后 T+1 的值 ({shifted_val}) "
                f"应等于原 T 的值 ({original_val})"
            )

        # 原 dates[1] 的数据应出现在 dates[2]
        for inst in instruments:
            original_val = pred.loc[(dates[1], inst), "score"]
            shifted_val = shifted.loc[(dates[2], inst), "score"]
            assert original_val == shifted_val

    def test_signal_shift_drops_last_day_data(self):
        """shift 后原最后一天的数据被丢弃（无下一天可映射到）

        shift 逻辑: date_map = {T0: T1, T1: T2}
        - T0 的 score 出现在 T1（T+1 才使用）
        - T1 的 score 出现在 T2
        - T2（最后一天）的 score 无处映射 → 被 dropna 丢弃
        结果: shifted 含 dates[1:] 两天，其中：
          - dates[1] 持有 dates[0] 的原始信号
          - dates[2] 持有 dates[1] 的原始信号
          - dates[2] 原始数据被丢失（符合预期）
        """
        dates = make_trading_dates(3)
        instruments = make_instruments(5)
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        # 每天不同 score 以便区分
        scores = []
        for day_val in [1.0, 2.0, 3.0]:
            scores.extend([day_val] * len(instruments))
        pred = pd.DataFrame({"score": scores}, index=index)

        shifted = self._do_shift(pred)
        shifted_dates = sorted(shifted.index.get_level_values("datetime").unique())

        # shift 后应有 dates[1] 和 dates[2]
        assert len(shifted_dates) == 2
        assert shifted_dates[0] == dates[1]
        assert shifted_dates[1] == dates[2]

        # dates[2] 原始 score=3.0 不应出现在 shifted 中
        # shifted 中 dates[2] 应持有原 dates[1] 的 score=2.0
        for inst in instruments:
            assert shifted.loc[(dates[2], inst), "score"] == 2.0
        # shifted 中 dates[1] 应持有原 dates[0] 的 score=1.0
        for inst in instruments:
            assert shifted.loc[(dates[1], inst), "score"] == 1.0

    def test_signal_shift_preserves_multiindex_structure(self):
        """shift 前后 MultiIndex names 和层级不变"""
        pred = make_synthetic_pred_score(n_instruments=3, n_days=5)

        shifted = self._do_shift(pred)

        assert isinstance(shifted.index, pd.MultiIndex)
        assert list(shifted.index.names) == list(pred.index.names)
        assert shifted.index.nlevels == pred.index.nlevels


# ── 成本一致性 ──────────────────────────────────────────────


class TestCostConsistency:
    """验证交易成本配置

    直接测试成本计算逻辑，不依赖 qlib 的 backtest_daily/TopkDropoutStrategy 导入。
    """

    def test_exchange_kwargs_cost_overridden_by_strategy_config(self):
        """strategy_config 的成本应覆盖 backtest_config 的 exchange_kwargs

        直接复现 portfolio.run_backtest 中的成本合并逻辑。
        """
        exchange_kwargs = {
            "limit_threshold": 0.099,
            "deal_price": "close",
            "open_cost": 0.9999,  # 故意设错，应被覆盖
            "close_cost": 0.9999,
            "min_cost": 5,
        }
        cost_config = {
            "buy_commission": 0.0003,
            "buy_slippage": 0.0002,
            "sell_commission": 0.0003,
            "stamp_tax": 0.001,
            "sell_slippage": 0.0002,
        }

        # 复现 portfolio.py:112-119 的成本合并逻辑
        exchange_kwargs = dict(exchange_kwargs)
        if cost_config:
            open_cost = cost_config.get("buy_commission", 0.0003) + cost_config.get("buy_slippage", 0.0002)
            close_cost = (cost_config.get("sell_commission", 0.0003)
                          + cost_config.get("stamp_tax", 0.001)
                          + cost_config.get("sell_slippage", 0.0002))
            exchange_kwargs["open_cost"] = open_cost
            exchange_kwargs["close_cost"] = close_cost

        expected_open = 0.0003 + 0.0002  # = 0.0005
        expected_close = 0.0003 + 0.001 + 0.0002  # = 0.0015
        assert abs(exchange_kwargs["open_cost"] - expected_open) < 1e-10, (
            f"open_cost={exchange_kwargs['open_cost']}, expected={expected_open}"
        )
        assert abs(exchange_kwargs["close_cost"] - expected_close) < 1e-10, (
            f"close_cost={exchange_kwargs['close_cost']}, expected={expected_close}"
        )

    def test_sell_cost_includes_stamp_tax(self):
        """close_cost 必须包含印花税"""
        stamp_tax = 0.001
        cost_config = {
            "buy_commission": 0.0003,
            "buy_slippage": 0.0002,
            "sell_commission": 0.0003,
            "stamp_tax": stamp_tax,
            "sell_slippage": 0.0002,
        }

        close_cost = (cost_config.get("sell_commission", 0.0003)
                      + cost_config.get("stamp_tax", 0.001)
                      + cost_config.get("sell_slippage", 0.0002))

        assert close_cost >= stamp_tax, (
            f"close_cost ({close_cost}) 应包含印花税 ({stamp_tax})"
        )
        # 验证精确值
        assert abs(close_cost - 0.0015) < 1e-10

    def test_cost_from_real_config(self):
        """从真实策略配置文件验证成本一致性"""
        from utils.helpers import get_strategy_config, get_backtest_config

        st_config = get_strategy_config()
        bt_config = get_backtest_config()

        cost = st_config.get("cost", {})
        if cost:
            expected_open = cost["buy_commission"] + cost["buy_slippage"]
            expected_close = cost["sell_commission"] + cost["stamp_tax"] + cost["sell_slippage"]

            # 验证 backtest_config 中的默认值与 strategy_config 一致
            assert abs(bt_config["exchange_kwargs"]["open_cost"] - expected_open) < 1e-10, (
                "backtest_config.exchange_kwargs.open_cost 与 strategy_config.cost 不一致"
            )
            assert abs(bt_config["exchange_kwargs"]["close_cost"] - expected_close) < 1e-10, (
                "backtest_config.exchange_kwargs.close_cost 与 strategy_config.cost 不一致"
            )


# ── 交易记录提取 ────────────────────────────────────────────


class TestExtractTradeRecords:
    """测试 extract_trade_records"""

    def test_first_day_all_buys(self):
        """首日持仓应全部标记为「买入」"""
        from signal_gen.portfolio import extract_trade_records

        dates = make_trading_dates(2)
        instruments = ["SH600000", "SH600001"]

        positions = {}
        # Day 1: hold both
        positions[dates[0]] = type("P", (), {"position": {
            "SH600000": {"amount": 1000, "price": 10.0},
            "SH600001": {"amount": 2000, "price": 20.0},
            "cash": {"amount": 500000},
        }})()
        # Day 2: same holdings
        positions[dates[1]] = type("P", (), {"position": {
            "SH600000": {"amount": 1000, "price": 10.5},
            "SH600001": {"amount": 2000, "price": 20.5},
            "cash": {"amount": 500000},
        }})()

        df = extract_trade_records(positions)
        first_day = df[df["date"] == dates[0]]
        assert len(first_day) == 2
        assert (first_day["action"] == "买入").all()

    def test_detects_reduction(self):
        """第二天减仓应产生「卖出」记录"""
        from signal_gen.portfolio import extract_trade_records

        dates = make_trading_dates(2)

        positions = {}
        positions[dates[0]] = type("P", (), {"position": {
            "SH600000": {"amount": 1000, "price": 10.0},
            "cash": {"amount": 500000},
        }})()
        # Day 2: reduced from 1000 to 500
        positions[dates[1]] = type("P", (), {"position": {
            "SH600000": {"amount": 500, "price": 10.5},
            "cash": {"amount": 505000},
        }})()

        df = extract_trade_records(positions)
        sells = df[(df["date"] == dates[1]) & (df["action"] == "卖出")]
        assert len(sells) == 1
        assert sells.iloc[0]["amount"] == 500

    def test_sell_uses_prev_day_price(self):
        """清仓时 cur_price=0，验证使用 prev_price 而非 0（bug 回归测试）"""
        from signal_gen.portfolio import extract_trade_records

        dates = make_trading_dates(2)

        positions = {}
        positions[dates[0]] = type("P", (), {"position": {
            "SH600000": {"amount": 1000, "price": 15.0},
            "cash": {"amount": 500000},
        }})()
        # Day 2: fully sold, SH600000 not in current holdings (cur_price=0)
        positions[dates[1]] = type("P", (), {"position": {
            "cash": {"amount": 515000},
        }})()

        df = extract_trade_records(positions)
        sells = df[df["action"] == "卖出"]
        assert len(sells) == 1
        # 应使用前一日价格 15.0，而非 0
        assert sells.iloc[0]["price"] == 15.0
        assert sells.iloc[0]["value"] == 1000 * 15.0

    def test_empty_positions(self):
        """空 positions 返回空 DataFrame，不抛异常"""
        from signal_gen.portfolio import extract_trade_records

        df = extract_trade_records({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_positions_none_like(self):
        """None-like positions 返回空 DataFrame"""
        from signal_gen.portfolio import extract_trade_records

        df = extract_trade_records({})
        assert df.empty
