"""
因子预处理器配置测试
只校验配置字典结构，不实例化真正的 Qlib Processor
"""

import pytest


class TestGetInferProcessors:
    """测试推理阶段预处理器配置"""

    def test_with_mad_returns_robust_zscore(self):
        """winsorize.method=mad 时返回 RobustZScoreNorm"""
        from factors.preprocessor import get_infer_processors

        config = {
            "preprocessing": {
                "winsorize": {"method": "mad", "n_sigma": 3},
                "normalize": {"method": "zscore"},
                "fillna": {"method": "ffill_then_median"},
            }
        }
        processors = get_infer_processors(config)

        classes = [p["class"] for p in processors]
        assert "RobustZScoreNorm" in classes
        # MAD 模式下不应同时出现 ZScoreNorm
        assert "ZScoreNorm" not in classes

    def test_fallback_zscore(self):
        """无 MAD 且 normalize.method=zscore 时返回 ZScoreNorm"""
        from factors.preprocessor import get_infer_processors

        config = {
            "preprocessing": {
                "winsorize": {"method": "sigma"},  # not mad
                "normalize": {"method": "zscore"},
                "fillna": {"method": "median"},
            }
        }
        processors = get_infer_processors(config)

        classes = [p["class"] for p in processors]
        assert "ZScoreNorm" in classes
        assert "RobustZScoreNorm" not in classes

    def test_fillna_processor_included(self):
        """ffill_then_median 应产生 Fillna processor"""
        from factors.preprocessor import get_infer_processors

        config = {
            "preprocessing": {
                "winsorize": {},
                "normalize": {},
                "fillna": {"method": "ffill_then_median"},
            }
        }
        processors = get_infer_processors(config)
        classes = [p["class"] for p in processors]
        assert "Fillna" in classes

    def test_processor_module_path(self):
        """所有 processor 的 module_path 应指向 qlib"""
        from factors.preprocessor import get_infer_processors

        config = {
            "preprocessing": {
                "winsorize": {"method": "mad"},
                "fillna": {"method": "ffill_then_median"},
            }
        }
        processors = get_infer_processors(config)
        for p in processors:
            assert "qlib" in p["module_path"]


class TestGetLearnProcessors:
    """测试训练阶段预处理器配置"""

    def test_has_dropna_and_csrank(self):
        """必然包含 DropnaLabel 和 CSRankNorm"""
        from factors.preprocessor import get_learn_processors

        processors = get_learn_processors()
        classes = [p["class"] for p in processors]
        assert "DropnaLabel" in classes
        assert "CSRankNorm" in classes

    def test_csrank_targets_label(self):
        """CSRankNorm 应作用于 label 字段"""
        from factors.preprocessor import get_learn_processors

        processors = get_learn_processors()
        csrank = [p for p in processors if p["class"] == "CSRankNorm"][0]
        assert csrank.get("kwargs", {}).get("fields_group") == "label"

    def test_order_dropna_before_csrank(self):
        """DropnaLabel 应在 CSRankNorm 之前"""
        from factors.preprocessor import get_learn_processors

        processors = get_learn_processors()
        classes = [p["class"] for p in processors]
        dropna_idx = classes.index("DropnaLabel")
        csrank_idx = classes.index("CSRankNorm")
        assert dropna_idx < csrank_idx, "DropnaLabel 必须在 CSRankNorm 之前"
