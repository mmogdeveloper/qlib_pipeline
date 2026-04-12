"""
自定义因子
使用 Qlib Expression 引擎定义动量、波动率、换手率、量价背离、流动性因子
"""

from typing import List, Tuple, Optional

from loguru import logger

from utils.helpers import get_factor_config


def get_custom_factor_expressions(config: Optional[dict] = None) -> List[Tuple[str, str]]:
    """从配置文件加载自定义因子表达式

    Returns:
        [(expression, name), ...] 格式的因子列表
    """
    config = config or get_factor_config()
    factors = []

    custom = config.get("custom_factors", {})
    for category, items in custom.items():
        for item in items:
            name = item["name"]
            expr = item["expression"]
            factors.append((expr, name))
            logger.debug(f"加载自定义因子: {name} = {expr}")

    logger.info(f"加载了 {len(factors)} 个自定义因子")
    return factors


def get_all_factor_fields(config: Optional[dict] = None) -> List[Tuple[str, str]]:
    """获取全部因子字段（用于 DataHandler 配置）

    Returns:
        [(expression, name), ...] 格式列表
    """
    return get_custom_factor_expressions(config)


# ── 预定义因子类别 ──────────────────────────────────────────

MOMENTUM_FACTORS = [
    ("$close/Ref($close,5)-1", "mom_5d"),
    ("$close/Ref($close,10)-1", "mom_10d"),
    ("$close/Ref($close,20)-1", "mom_20d"),
    ("$close/Ref($close,60)-1", "mom_60d"),
]

VOLATILITY_FACTORS = [
    ("Std($close/Ref($close,1)-1, 20)", "vol_20d"),
    ("Std($close/Ref($close,1)-1, 5)", "vol_5d"),
]

TURNOVER_FACTORS = [
    ("Mean($turnover, 5)", "turnover_5d_mean"),
    ("$turnover/Mean($turnover,20)", "turnover_ratio"),
]

PRICE_VOLUME_FACTORS = [
    ("Corr($close, $volume, 20)", "corr_close_vol_20d"),
    ("Corr($close, $volume, 10)", "corr_close_vol_10d"),
]

LIQUIDITY_FACTORS = [
    ("Mean($amount/$volume, 10)", "vwap_ratio"),
]

# 全部自定义因子
ALL_CUSTOM_FACTORS = (
    MOMENTUM_FACTORS
    + VOLATILITY_FACTORS
    + TURNOVER_FACTORS
    + PRICE_VOLUME_FACTORS
    + LIQUIDITY_FACTORS
)
