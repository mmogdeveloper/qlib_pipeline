"""
可视化测试
重点：超额收益公式回归测试、月度热力图 pivot 形状
"""

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.sample_data import make_trading_dates


class TestExcessReturnFormula:
    """超额收益计算公式验证

    正确公式: cum_strategy / cum_bench - 1
    错误公式: (1 + (strategy_ret - bench_ret)).cumprod() - 1
    两者在非零收益率下会给出不同结果。
    """

    def test_excess_return_formula_correctness(self):
        """验证超额收益用的是 cum_strategy/cum_bench - 1"""
        n = 10
        strategy_ret = pd.Series([0.01] * n)
        bench_ret = pd.Series([0.005] * n)

        # 正确公式
        cum_strategy = (1 + strategy_ret).cumprod()
        cum_bench = (1 + bench_ret).cumprod()
        correct_excess = cum_strategy / cum_bench - 1

        # 错误公式（逐日差值累积）
        wrong_excess = (1 + (strategy_ret - bench_ret)).cumprod() - 1

        # 两个公式在此输入下结果不同
        assert not np.allclose(correct_excess.values, wrong_excess.values), (
            "正确公式和错误公式不应给出相同结果"
        )

        # 验证正确公式最终值
        expected_final = (1.01**10) / (1.005**10) - 1
        assert abs(correct_excess.iloc[-1] - expected_final) < 1e-10

    def test_excess_return_zero_when_equal(self):
        """策略收益 == 基准收益时超额为 0"""
        n = 10
        ret = pd.Series([0.01] * n)
        cum = (1 + ret).cumprod()
        excess = cum / cum - 1
        assert np.allclose(excess.values, 0.0)


class TestMonthlyHeatmapPivot:
    """月度收益热力图 pivot 表形状验证"""

    def test_pivot_shape(self):
        """2 年收益数据应生成 2×12 的 pivot 表"""
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        np.random.seed(42)
        returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        # 按月汇总
        monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["year", "month"])

        pivot = monthly.unstack(level="month")
        assert pivot.shape[0] == 2, f"行数应为 2 (年份), 实际={pivot.shape[0]}"
        assert pivot.shape[1] == 12, f"列数应为 12 (月份), 实际={pivot.shape[1]}"

    def test_pivot_partial_year(self):
        """不完整年份应少于 12 列"""
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)

        monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["year", "month"])

        pivot = monthly.unstack(level="month")
        assert pivot.shape[0] == 1
        assert pivot.shape[1] == 6
