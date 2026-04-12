"""
Qlib Alpha158 因子集 + 自定义因子配置

提供两种方案：
1. get_alpha158_handler_config()     — 纯 Alpha158 (158因子)
2. get_alpha158_with_custom_config() — Alpha158Custom (158 + N 自定义因子)
   通过继承 Alpha158 并覆盖 _get_fields() 实现真正的因子注入
"""

from typing import Dict, Any, List, Tuple, Optional

from loguru import logger

from utils.helpers import get_factor_config
from factors.preprocessor import get_infer_processors, get_learn_processors


def get_alpha158_handler_config(
    instruments: str = "csi300",
    train_period: Tuple[str, str] = ("2015-01-01", "2020-12-31"),
    valid_period: Tuple[str, str] = ("2021-01-01", "2022-12-31"),
    test_period: Tuple[str, str] = ("2023-01-01", "2026-04-12"),
    label_expr: str = "Ref($close, -6)/Ref($open, -1) - 1",
    factor_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """纯 Alpha158 DatasetH 配置（158个因子）

    实例化后 Qlib 内部执行：
    init_instance_by_config → DatasetH → Alpha158.__init__()
    → Alpha158._get_fields() 返回158个表达式
    → D.features() 计算因子值
    """
    logger.info("构建 Alpha158 DataHandler 配置（纯 158 因子）")
    f_config = factor_config or get_factor_config()

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": instruments,
                    "start_time": train_period[0],
                    "end_time": test_period[1],
                    "fit_start_time": train_period[0],
                    "fit_end_time": train_period[1],
                    "infer_processors": get_infer_processors(f_config),
                    "learn_processors": get_learn_processors(f_config),
                    "label": [label_expr],
                },
            },
            "segments": {
                "train": train_period,
                "valid": valid_period,
                "test": test_period,
            },
        },
    }


def get_alpha158_with_custom_config(
    instruments: str = "csi300",
    train_period: Tuple[str, str] = ("2015-01-01", "2020-12-31"),
    valid_period: Tuple[str, str] = ("2021-01-01", "2022-12-31"),
    test_period: Tuple[str, str] = ("2023-01-01", "2026-04-12"),
    label_expr: str = "Ref($close, -6)/Ref($open, -1) - 1",
    custom_fields: Optional[List[Tuple[str, str]]] = None,
    factor_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Alpha158 + 自定义因子配置

    使用 factors.custom_handler.Alpha158Custom，
    它继承 Alpha158 并覆盖 _get_fields() 追加自定义因子。
    自定义因子和 Alpha158 因子一起被 D.features() 计算，
    经过同一套 infer_processors 预处理，真正参与模型训练。

    Args:
        custom_fields: [(expression, name), ...] 自定义因子列表
    """
    f_config = factor_config or get_factor_config()

    if not custom_fields:
        logger.info("无自定义因子，回退到纯 Alpha158")
        return get_alpha158_handler_config(
            instruments, train_period, valid_period, test_period,
            label_expr, factor_config,
        )

    logger.info(f"构建 Alpha158Custom 配置: 158 + {len(custom_fields)} 自定义因子")

    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                # 使用自定义 Handler 而非原版 Alpha158
                "class": "Alpha158Custom",
                "module_path": "factors.custom_handler",
                "kwargs": {
                    "instruments": instruments,
                    "start_time": train_period[0],
                    "end_time": test_period[1],
                    "fit_start_time": train_period[0],
                    "fit_end_time": train_period[1],
                    "infer_processors": get_infer_processors(f_config),
                    "learn_processors": get_learn_processors(f_config),
                    "label": [label_expr],
                    # 自定义因子通过 kwargs 传入 Alpha158Custom
                    "custom_fields": custom_fields,
                },
            },
            "segments": {
                "train": train_period,
                "valid": valid_period,
                "test": test_period,
            },
        },
    }
