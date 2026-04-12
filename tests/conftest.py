"""
共享 fixtures
所有 fixture 不依赖真实 Qlib .bin 数据或 AKShare 网络请求
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保项目根目录在 sys.path 中，方便 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.sample_data import (
    make_synthetic_ohlcv,
    make_synthetic_pred_score,
    make_synthetic_returns,
    make_synthetic_positions,
    make_trading_dates,
    make_instruments,
)


@pytest.fixture(autouse=True)
def fixed_seed():
    """固定随机种子保证可复现"""
    np.random.seed(42)
    yield


@pytest.fixture
def synthetic_ohlcv():
    """5 只股票 × 30 个交易日的合成 OHLCV DataFrame"""
    return make_synthetic_ohlcv(n_instruments=5, n_days=30, seed=42)


@pytest.fixture
def synthetic_pred_score():
    """5 只股票 × 10 个交易日的预测分数"""
    return make_synthetic_pred_score(n_instruments=5, n_days=10, seed=42)


@pytest.fixture
def synthetic_returns():
    """252 个交易日的日收益率序列"""
    return make_synthetic_returns(n_days=252, seed=42)


@pytest.fixture
def synthetic_positions():
    """5 天的模拟持仓"""
    return make_synthetic_positions(n_days=5, seed=42)


@pytest.fixture
def tmp_config_dir(tmp_path):
    """在 tmp_path 下创建完整的 config/ YAML 目录"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "strategy_config.yaml").write_text(
        """strategy:
  type: "topk_dropout"
  topk: 3
  n_drop: 1
  weighting: "equal"
  rebalance:
    frequency: "day"
  trading_rules:
    limit_up_filter: true
    limit_down_filter: true
    st_filter: true
    suspend_filter: true
  cost:
    buy_commission: 0.0003
    sell_commission: 0.0003
    stamp_tax: 0.001
    buy_slippage: 0.0002
    sell_slippage: 0.0002
  initial_cash: 10000000
""",
        encoding="utf-8",
    )

    (config_dir / "backtest_config.yaml").write_text(
        """backtest:
  start_date: "2024-01-02"
  end_date: "2024-02-28"
  benchmark: "SH000300"
  account: 10000000
  exchange_kwargs:
    limit_threshold: 0.099
    deal_price: "close"
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 5
""",
        encoding="utf-8",
    )

    (config_dir / "factor_config.yaml").write_text(
        """factor:
  use_alpha158: true
  custom_factors:
    momentum:
      - name: "mom_5d"
        expression: "$close/Ref($close,5)-1"
  preprocessing:
    fillna:
      method: "ffill_then_median"
    winsorize:
      method: "mad"
      n_sigma: 3
    normalize:
      method: "zscore"
""",
        encoding="utf-8",
    )

    (config_dir / "data_config.yaml").write_text(
        """data:
  start_date: "2015-01-01"
  end_date: "2026-04-12"
  raw_csv_dir: "/tmp/test_raw_csv"
  qlib_data_dir: "/tmp/test_qlib_data"
  benchmark_index: "000300"
  split:
    train:
      start: "2015-01-01"
      end: "2020-12-31"
    valid:
      start: "2021-01-01"
      end: "2022-12-31"
    test:
      start: "2023-01-01"
      end: "2026-04-12"
""",
        encoding="utf-8",
    )

    (config_dir / "model_config.yaml").write_text(
        """model:
  default: "lgbm"
  label:
    expression: "Ref($close, -6)/Ref($close, -1) - 1"
    name: "Ref($close, -6)/Ref($close, -1) - 1"
  lgbm:
    class: "LGBModel"
    module_path: "qlib.contrib.model.gbdt"
    kwargs:
      loss: "mse"
      n_estimators: 10
      num_leaves: 31
  save_dir: "models/saved"
""",
        encoding="utf-8",
    )

    return config_dir


@pytest.fixture
def mock_recorder():
    """Mock 的 Qlib Recorder"""

    class FakeRecorder:
        def __init__(self):
            self._objects = {}
            self._metrics = {}

        def load_object(self, name):
            return self._objects.get(name)

        def list_metrics(self):
            return self._metrics

        def set_object(self, name, obj):
            self._objects[name] = obj

        def set_metrics(self, metrics):
            self._metrics = metrics

    return FakeRecorder()
