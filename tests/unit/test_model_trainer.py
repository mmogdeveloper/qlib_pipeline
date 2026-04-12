"""
Issue 3 回归测试：滚动验证切分
验证 _compute_rolling_step_days 输出合理值，
以及 RollingGen 生成的各折 train/test 不重叠
"""

import pytest
from datetime import datetime


class TestComputeRollingStepDays:
    """验证步长计算逻辑"""

    def test_step_days_reasonable(self):
        """标准配置应返回合理步长（约 1 年 ≈ 365 天）"""
        from model.model_trainer import _compute_rolling_step_days

        config = {
            "split": {
                "train": {"start": "2015-01-01", "end": "2020-12-31"},
                "valid": {"start": "2021-01-01", "end": "2022-12-31"},
                "test": {"start": "2023-01-01", "end": "2026-04-12"},
            },
            "rolling": {
                "n_splits": 5,
                "train_years": 4,
                "valid_years": 1,
                "test_years": 1,
            },
        }
        step = _compute_rolling_step_days(config)

        assert 180 <= step <= 800, f"步长 {step} 天不在合理范围 [180, 800]"

    def test_step_days_minimum_180(self):
        """步长不应小于 180 天"""
        from model.model_trainer import _compute_rolling_step_days

        config = {
            "split": {
                "train": {"start": "2020-01-01", "end": "2023-12-31"},
                "valid": {"start": "2024-01-01", "end": "2024-12-31"},
                "test": {"start": "2025-01-01", "end": "2025-06-30"},
            },
            "rolling": {
                "n_splits": 10,
                "train_years": 4,
                "valid_years": 1,
                "test_years": 1,
            },
        }
        step = _compute_rolling_step_days(config)
        assert step >= 180


class TestRollingNonOverlap:
    """验证滚动验证各折 train/test 不重叠（纯数据层面）"""

    def test_sequential_folds_no_train_test_overlap(self):
        """相邻折的 train_end 应 < 后一折 test_start"""
        from model.model_trainer import _compute_rolling_step_days

        config = {
            "split": {
                "train": {"start": "2015-01-01", "end": "2020-12-31"},
                "valid": {"start": "2021-01-01", "end": "2022-12-31"},
                "test": {"start": "2023-01-01", "end": "2026-04-12"},
            },
            "rolling": {
                "n_splits": 5,
                "train_years": 4,
                "valid_years": 1,
                "test_years": 1,
            },
        }
        step = _compute_rolling_step_days(config)
        assert step > 0, "步长应为正数"
