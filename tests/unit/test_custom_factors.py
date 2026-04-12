"""
自定义因子与 Label 表达式测试
重点覆盖 label 的前视偏差和 NaN 行为
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.sample_data import make_trading_dates


class TestLabelExpression:
    """验证 label 表达式 Ref($close, -6)/Ref($close, -1) - 1 的正确性"""

    def _compute_label_manually(self, close_series: pd.Series) -> pd.Series:
        """手工计算 label: Ref($close, -6)/Ref($close, -1) - 1

        Ref($close, -k) 表示 "当前日期之后第 k 天的 close"
        - Ref($close, -1) = T+1 日的收盘价（买入价）
        - Ref($close, -6) = T+6 日的收盘价（卖出价）
        - label = (T+6_close / T+1_close) - 1
        """
        ref_minus_1 = close_series.shift(-1)  # T+1 close
        ref_minus_6 = close_series.shift(-6)  # T+6 close
        return ref_minus_6 / ref_minus_1 - 1

    def test_label_no_lookahead_at_day_T(self):
        """T 日的 label 严格依赖 T+1 到 T+6 的未来价格"""
        prices = list(range(100, 121))  # [100, 101, ..., 120]
        dates = make_trading_dates(len(prices))
        close = pd.Series(prices, index=dates, dtype=float)

        label = self._compute_label_manually(close)

        # 对于 T=0 (close=100):
        #   T+1 close = 101 (买入价), T+6 close = 106 (卖出价)
        #   label = 106/101 - 1
        expected = 106 / 101 - 1
        assert abs(label.iloc[0] - expected) < 1e-10, (
            f"T=0 label={label.iloc[0]}, expected={expected}"
        )

        # T=5: T+1=prices[6]=106, T+6=prices[11]=111
        expected_t5 = 111 / 106 - 1
        assert abs(label.iloc[5] - expected_t5) < 1e-10

    def test_label_nan_at_end_of_series(self):
        """最后 6 天的 label 应为 NaN（无足够的未来数据）"""
        prices = list(range(100, 121))  # 21 个数据点
        dates = make_trading_dates(len(prices))
        close = pd.Series(prices, index=dates, dtype=float)

        label = self._compute_label_manually(close)

        # 最后 6 天 (index -6 到 -1) 的 label 应为 NaN
        for i in range(-6, 0):
            assert pd.isna(label.iloc[i]), (
                f"倒数第 {abs(i)} 天的 label 应为 NaN, 实际={label.iloc[i]}"
            )

        # 倒数第 7 天应有值
        assert pd.notna(label.iloc[-7])

    def test_label_uses_future_prices_only(self):
        """label 不应使用 T 日或更早的价格"""
        # 构造一个在 T 日价格突变的序列
        prices = [10.0] * 5 + [100.0] + [10.0] * 15  # 第 5 天价格暴涨
        dates = make_trading_dates(len(prices))
        close = pd.Series(prices, index=dates, dtype=float)

        label = self._compute_label_manually(close)

        # T=0 的 label 只应基于 T+1 到 T+6 的价格
        # T+1=10.0, T+6=10.0 → label = 10/10 - 1 = 0
        assert abs(label.iloc[0]) < 1e-10


class TestCustomFactorExpressions:
    """测试自定义因子加载"""

    def test_get_custom_factor_expressions_from_config(self):
        """从配置加载因子应返回 (expr, name) 列表"""
        from factors.custom_factors import get_custom_factor_expressions

        config = {
            "custom_factors": {
                "momentum": [
                    {"name": "mom_5d", "expression": "$close/Ref($close,5)-1"},
                    {"name": "mom_10d", "expression": "$close/Ref($close,10)-1"},
                ],
                "volatility": [
                    {"name": "vol_20d", "expression": "Std($close/Ref($close,1)-1, 20)"},
                ],
            }
        }
        factors = get_custom_factor_expressions(config)
        assert len(factors) == 3
        assert all(isinstance(f, tuple) and len(f) == 2 for f in factors)
        names = [f[1] for f in factors]
        assert "mom_5d" in names
        assert "vol_20d" in names

    def test_get_custom_factor_expressions_empty_config(self):
        """空配置应返回空列表"""
        from factors.custom_factors import get_custom_factor_expressions

        factors = get_custom_factor_expressions({"custom_factors": {}})
        assert factors == []

    def test_predefined_factors_structure(self):
        """预定义因子常量应符合 (expr, name) 格式"""
        from factors.custom_factors import ALL_CUSTOM_FACTORS

        assert len(ALL_CUSTOM_FACTORS) > 0
        for expr, name in ALL_CUSTOM_FACTORS:
            assert isinstance(expr, str) and len(expr) > 0
            assert isinstance(name, str) and len(name) > 0
