"""
Issue 2 回归测试：停牌检测逻辑
验证 check_suspend_days 能正确检测 volume=0、amount=0、NaN 等场景
"""

import numpy as np
import pandas as pd
import pytest


def _make_stock_df(dates, volumes, amounts=None):
    """构造简单的股票行情 DataFrame"""
    df = pd.DataFrame({"date": dates, "volume": volumes})
    if amounts is not None:
        df["amount"] = amounts
    return df


class TestCheckSuspendDays:
    """停牌检测单测"""

    def _checker(self):
        """构造 DataHealthChecker 但不依赖真实 config"""
        from data.health_check import DataHealthChecker
        from unittest.mock import patch

        fake_config = {
            "raw_csv_dir": "/tmp/fake",
            "health_check": {
                "max_price_change_pct": 20.0,
                "min_trading_days_pct": 80.0,
            },
        }
        with patch.object(DataHealthChecker, "__init__", lambda self, *a, **kw: None):
            checker = DataHealthChecker.__new__(DataHealthChecker)
            checker.config = fake_config
            checker.max_price_change = 20.0
            checker.min_trading_pct = 80.0
        return checker

    def test_volume_zero_detected(self):
        """volume=0 应被检测为停牌"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        volumes = [100000, 0, 200000, 0, 300000]
        amounts = [1e8, 0, 2e8, 0, 3e8]
        df = _make_stock_df(dates, volumes, amounts)

        result = checker.check_suspend_days(df, "SH600000")
        assert len(result) == 2

    def test_amount_zero_detected(self):
        """volume>0 但 amount=0 也应被检测为停牌"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        volumes = [100000, 50000, 200000]
        amounts = [1e8, 0, 2e8]
        df = _make_stock_df(dates, volumes, amounts)

        result = checker.check_suspend_days(df, "SH600001")
        assert len(result) == 1, "amount=0 应被检测为停牌"

    def test_nan_volume_detected(self):
        """volume=NaN 应被检测为停牌"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        volumes = [100000, np.nan, 200000]
        amounts = [1e8, np.nan, 2e8]
        df = _make_stock_df(dates, volumes, amounts)

        result = checker.check_suspend_days(df, "SH600002")
        assert len(result) == 1

    def test_nan_amount_only_detected(self):
        """volume 正常但 amount=NaN 应被检测为停牌"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        volumes = [100000, 50000, 200000]
        amounts = [1e8, np.nan, 2e8]
        df = _make_stock_df(dates, volumes, amounts)

        result = checker.check_suspend_days(df, "SH600003")
        assert len(result) == 1, "amount=NaN 应被检测为停牌"

    def test_no_amount_column_fallback(self):
        """无 amount 列时仅用 volume 检测"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        volumes = [100000, 0, 200000]
        df = _make_stock_df(dates, volumes)

        result = checker.check_suspend_days(df, "SH600004")
        assert len(result) == 1

    def test_all_normal_no_suspend(self):
        """全部正常数据应无停牌"""
        checker = self._checker()
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        volumes = [100000, 200000, 300000, 400000, 500000]
        amounts = [1e8, 2e8, 3e8, 4e8, 5e8]
        df = _make_stock_df(dates, volumes, amounts)

        result = checker.check_suspend_days(df, "SH600005")
        assert len(result) == 0

    def test_empty_df(self):
        """空 DataFrame 应返回空列表"""
        checker = self._checker()
        df = pd.DataFrame(columns=["date", "volume", "amount"])
        result = checker.check_suspend_days(df, "SH600006")
        assert result == []
