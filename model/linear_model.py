"""
Ridge 回归模型配置
基于 Qlib 内置的 LinearModel
"""

from typing import Dict, Any, Optional

from loguru import logger

from utils.helpers import get_model_config


def get_linear_model_config(config: Optional[dict] = None) -> Dict[str, Any]:
    """返回 Ridge 回归模型的 Qlib 配置

    Returns:
        模型配置字典
    """
    config = config or get_model_config()
    linear_cfg = config["linear"]

    model_config = {
        "class": linear_cfg["class"],
        "module_path": linear_cfg["module_path"],
        "kwargs": linear_cfg["kwargs"],
    }

    logger.info(f"Linear 模型配置: estimator={linear_cfg['kwargs']['estimator']}, "
                f"alpha={linear_cfg['kwargs']['alpha']}")
    return model_config
