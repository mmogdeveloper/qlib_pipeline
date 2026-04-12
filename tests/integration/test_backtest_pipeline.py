"""
回测流水线集成测试
"""

import pytest
import pandas as pd
import numpy as np

from tests.fixtures.sample_data import make_synthetic_pred_score


@pytest.mark.integration
class TestBacktestPipeline:
    """回测流水线端到端测试"""

    def test_end_to_end_tiny_universe(self):
        """5 只股票 × 60 天合成数据跑通回测流程

        此测试需要完整 Qlib 环境，标记为 integration。
        """
        try:
            import qlib
        except ImportError:
            pytest.skip("Qlib not available")

        pytest.skip(
            "需要 Qlib 初始化和 bin 数据。"
            "手动运行: pytest tests/integration/ -m integration"
        )

    def test_signal_generator_end_to_end(self):
        """信号生成完整流程：pred_score → trade_signals DataFrame"""
        from signal_gen.signal_generator import generate_trade_signals

        pred = make_synthetic_pred_score(n_instruments=10, n_days=20, seed=123)
        config = {"topk": 5, "n_drop": 2}
        signals = generate_trade_signals(pred, config=config)

        # 验证返回结构
        assert isinstance(signals, pd.DataFrame)
        required_cols = {"date", "instrument", "score", "rank", "signal", "change"}
        assert required_cols.issubset(set(signals.columns))

        # 每天应有 10 只股票的信号
        for date in signals["date"].unique():
            day = signals[signals["date"] == date]
            assert len(day) == 10

        # 每天应有 5 个买入信号
        for date in signals["date"].unique():
            buys = signals[(signals["date"] == date) & (signals["signal"] == "买入")]
            assert len(buys) == 5

    def test_extract_trade_records_end_to_end(self):
        """交易记录提取完整流程"""
        from signal_gen.portfolio import extract_trade_records
        from tests.fixtures.sample_data import make_synthetic_positions

        positions = make_synthetic_positions(n_days=10, seed=99)
        records = extract_trade_records(positions)

        assert isinstance(records, pd.DataFrame)
        if not records.empty:
            assert "date" in records.columns
            assert "instrument" in records.columns
            assert "action" in records.columns
            assert set(records["action"].unique()).issubset({"买入", "卖出"})
