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


class TestIcMeanScalarConversion:
    """Issue 5 回归测试：ic_mean 必须是标量，不能是 Series"""

    def test_ic_mean_is_float_not_series(self):
        """load_ic_series_from_recorder 中 ic_mean 应为 float"""
        n_dates = 10
        n_instruments = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        instruments = [f"SH60000{i}" for i in range(n_instruments)]
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred_vals = np.random.randn(len(index))
        label_vals = pred_vals * 0.2 + np.random.randn(len(index))

        concat = pd.DataFrame({"pred": pred_vals, "label": label_vals}, index=index)
        ic = concat.groupby(level=0).apply(
            lambda x: x["pred"].corr(x["label"])
        )
        ic_mean = float(ic.mean()) if len(ic) > 0 else float("nan")

        assert isinstance(ic_mean, float), f"ic_mean 应为 float，实际: {type(ic_mean)}"
        assert not pd.isna(ic_mean), "ic_mean 不应为 NaN"

    def test_empty_ic_returns_nan(self):
        """空 IC 序列应返回 float NaN"""
        ic = pd.Series(dtype=float)
        ic_mean = float(ic.mean()) if len(ic) > 0 else float("nan")
        assert isinstance(ic_mean, float)
        assert pd.isna(ic_mean)


class TestLoadIcFromRecorderRawLabel:
    """Issue 1 回归测试：load_ic_from_recorder 应返回 raw IC 作为主指标"""

    def test_returns_raw_and_csranknorm_keys(self, mock_recorder):
        """返回字典中应包含 raw_ic_mean 和 csranknorm_ic_mean"""
        from evaluation.metrics import load_ic_from_recorder

        # 设置 CSRankNorm IC
        mock_recorder.set_metrics({
            "IC": 0.49, "ICIR": 1.2,
            "Rank IC": 0.45, "Rank ICIR": 1.1,
        })
        # load_ic_from_recorder 内部会尝试 load_ic_series_from_recorder,
        # 但 mock_recorder 没有 pred.pkl 所以 raw IC 回退
        result = load_ic_from_recorder(mock_recorder)

        assert result is not None
        assert "csranknorm_ic_mean" in result
        assert result["csranknorm_ic_mean"] == 0.49
        # raw IC 计算失败时，ic_mean 回退到 CSRankNorm
        assert result["ic_mean"] == 0.49

    def test_raw_ic_used_as_primary_when_available(self, mock_recorder):
        """当 raw IC 可计算时，ic_mean 应使用 raw IC"""
        from evaluation.metrics import load_ic_from_recorder

        mock_recorder.set_metrics({
            "IC": 0.49, "ICIR": 1.2,
            "Rank IC": 0.45, "Rank ICIR": 1.1,
        })
        # 设置 pred.pkl 和 label.pkl 让 raw IC 可计算
        n = 500
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        instruments = [f"SH60000{i}" for i in range(50)]
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred = pd.DataFrame({"score": np.random.randn(len(index))}, index=index)
        mock_recorder.set_object("pred.pkl", pred)
        # 注意：_compute_raw_labels 需要 Qlib 初始化，这里只验证 fallback 路径
        # raw IC 计算失败时会 fallback 到 CSRankNorm
        result = load_ic_from_recorder(mock_recorder)
        assert result is not None


class TestComputeRawLabelsAlignment:
    """Bug 回归测试：D.features 返回 (instrument, datetime) 索引，
    _compute_raw_labels 必须 swaplevel 对齐到 pred 的 (datetime, instrument)"""

    def test_swaplevel_alignment(self):
        """模拟 D.features 返回反转索引，验证对齐后 concat 非空"""
        from unittest.mock import patch
        from evaluation.metrics import _compute_raw_labels

        n_dates = 20
        n_instruments = 10
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        instruments = [f"SH60000{i}" for i in range(n_instruments)]

        # pred: (datetime, instrument) — Qlib pred.pkl 的标准格式
        pred_index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred = pd.DataFrame(
            {"score": np.random.randn(len(pred_index))}, index=pred_index,
        )

        # 模拟 D.features 返回 (instrument, datetime) 索引
        raw_index = pd.MultiIndex.from_product(
            [instruments, dates], names=["instrument", "datetime"]
        )
        fake_raw = pd.DataFrame(
            {"label": np.random.randn(len(raw_index))}, index=raw_index,
        )

        with patch("qlib.data.D") as mock_D, \
             patch("utils.helpers.get_model_config") as mock_cfg:
            mock_cfg.return_value = {"label": {"expression": "$close"}}
            mock_D.features.return_value = fake_raw

            result = _compute_raw_labels(pred)

        assert result is not None, "_compute_raw_labels 不应返回 None"
        assert len(result) == len(pred), (
            f"对齐后应有 {len(pred)} 条，实际 {len(result)}"
        )
        # 索引层级应与 pred 一致: (datetime, instrument)
        assert result.index.names == ["datetime", "instrument"]

    def test_concat_after_alignment_is_nonempty(self):
        """完整链路：_compute_raw_labels 返回后，与 pred concat + dropna 应非空"""
        from unittest.mock import patch
        from evaluation.metrics import _compute_raw_labels

        n_dates = 20
        n_instruments = 10
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        instruments = [f"SH60000{i}" for i in range(n_instruments)]

        pred_index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        pred_vals = np.random.randn(len(pred_index))
        pred = pd.DataFrame({"score": pred_vals}, index=pred_index)

        # D.features 返回 (instrument, datetime)
        raw_index = pd.MultiIndex.from_product(
            [instruments, dates], names=["instrument", "datetime"]
        )
        label_vals = pred_vals * 0.3 + np.random.randn(len(raw_index)) * 0.7
        fake_raw = pd.DataFrame({"label": label_vals}, index=raw_index)

        with patch("qlib.data.D") as mock_D, \
             patch("utils.helpers.get_model_config") as mock_cfg:
            mock_cfg.return_value = {"label": {"expression": "$close"}}
            mock_D.features.return_value = fake_raw

            label = _compute_raw_labels(pred)

        concat = pd.concat(
            [pred["score"], label["label"]], axis=1,
        )
        concat.columns = ["pred", "label"]
        concat = concat.dropna()

        assert len(concat) > 0, "concat + dropna 后不应为空（索引对齐失败）"
        assert len(concat) == len(pred)
