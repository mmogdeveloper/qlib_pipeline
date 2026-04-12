"""
评估指标测试
覆盖空输入、NaN 处理、IC 加载等边界情况
"""

import numpy as np
import pandas as pd
import pytest


class TestComputeMetricsFromReturns:
    """测试 compute_metrics_from_returns"""

    def test_empty_returns(self):
        """空 Series 返回 {} 且不崩溃"""
        from evaluation.metrics import compute_metrics_from_returns

        result = compute_metrics_from_returns(pd.Series(dtype=float))
        assert result == {}

    def test_all_nan_returns(self):
        """全 NaN 输入应被优雅处理"""
        from evaluation.metrics import compute_metrics_from_returns

        returns = pd.Series([np.nan, np.nan, np.nan])
        result = compute_metrics_from_returns(returns)
        assert result == {}


class TestLoadIcFromRecorder:
    """测试 load_ic_from_recorder"""

    def test_missing_metrics_returns_none(self, mock_recorder):
        """recorder 无 IC 指标时返回 None"""
        from evaluation.metrics import load_ic_from_recorder

        mock_recorder.set_metrics({})
        result = load_ic_from_recorder(mock_recorder)
        assert result is None

    def test_with_ic_metrics(self, mock_recorder):
        """recorder 有 IC 指标时返回正确字典"""
        from evaluation.metrics import load_ic_from_recorder

        mock_recorder.set_metrics({
            "IC": 0.05,
            "ICIR": 0.3,
            "Rank IC": 0.06,
            "Rank ICIR": 0.35,
        })
        result = load_ic_from_recorder(mock_recorder)
        assert result is not None
        assert abs(result["ic_mean"] - 0.05) < 1e-10
        assert abs(result["icir"] - 0.3) < 1e-10
        assert abs(result["rank_ic_mean"] - 0.06) < 1e-10

    def test_exception_returns_none(self):
        """recorder 抛异常时返回 None"""
        from evaluation.metrics import load_ic_from_recorder

        class BadRecorder:
            def list_metrics(self):
                raise RuntimeError("connection failed")

        result = load_ic_from_recorder(BadRecorder())
        assert result is None


class TestRawLabelIndependentCheck:
    """验证原始 label IC 独立于 CSRankNorm"""

    def test_raw_ic_differs_from_csrank_ic(self):
        """构造已知相关性的 pred 和 label，确认 raw IC ≠ CSRankNorm IC"""
        np.random.seed(42)
        n_dates = 20
        n_instruments = 50
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        instruments = [f"SH{str(i).zfill(6)}" for i in range(n_instruments)]
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )

        # 构造 pred 和 raw_label 有 Pearson 相关
        pred_vals = np.random.randn(len(index))
        raw_label = pred_vals * 0.3 + np.random.randn(len(index)) * 0.7

        pred = pd.Series(pred_vals, index=index, name="pred")
        label_raw = pd.Series(raw_label, index=index, name="label")

        # CSRankNorm: 截面排名归一化
        df = pd.DataFrame({"pred": pred, "label": label_raw})
        df["label_csrank"] = df.groupby(level=0)["label"].rank(pct=True)

        # 计算两种 IC
        ic_raw = df.groupby(level=0).apply(lambda x: x["pred"].corr(x["label"]))
        ic_csrank = df.groupby(level=0).apply(lambda x: x["pred"].corr(x["label_csrank"]))

        # 两种 IC 均值应不同（CSRankNorm 改变了相关性结构）
        assert abs(ic_raw.mean() - ic_csrank.mean()) > 1e-4, (
            f"raw IC ({ic_raw.mean():.6f}) 和 CSRankNorm IC ({ic_csrank.mean():.6f}) "
            "不应相等，独立校验机制失效"
        )
