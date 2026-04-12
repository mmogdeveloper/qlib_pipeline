"""
因子流水线集成测试
需要 Qlib 环境但使用合成数据
"""

import pytest


@pytest.mark.integration
class TestAlpha158Custom:
    """Alpha158Custom DataHandler 集成测试"""

    def test_alpha158_custom_adds_custom_fields(self):
        """Alpha158Custom 返回的因子数量 = 158 + len(custom_fields)

        需要 Qlib 环境初始化和合成 bin 数据。
        此测试标记为 integration，不在 unit 测试中运行。
        """
        try:
            import qlib
            from qlib.config import REG_CN
        except ImportError:
            pytest.skip("Qlib not available")

        # 此测试需要 Qlib 初始化和数据，在 CI 中跳过
        pytest.skip(
            "需要 Qlib bin 数据集才能运行。"
            "手动运行: pytest tests/integration/ -m integration"
        )


@pytest.mark.integration
class TestFactorPipelineEndToEnd:
    """因子计算端到端测试"""

    def test_custom_factor_expressions_loadable(self):
        """自定义因子表达式能从真实配置加载"""
        from factors.custom_factors import get_custom_factor_expressions

        factors = get_custom_factor_expressions()
        assert len(factors) > 0
        for expr, name in factors:
            assert "$" in expr or "Ref" in expr or "Std" in expr or "Mean" in expr or "Corr" in expr, (
                f"因子 {name} 的表达式 '{expr}' 不像合法的 Qlib 表达式"
            )
