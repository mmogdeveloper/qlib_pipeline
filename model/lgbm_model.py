"""
LightGBM 模型配置
基于 Qlib 内置的 LGBModel
"""

from typing import Dict, Any, Optional

from loguru import logger

from utils.helpers import get_model_config


def get_lgbm_model_config(config: Optional[dict] = None) -> Dict[str, Any]:
    """返回 LightGBM 模型的 Qlib 配置

    Returns:
        模型配置字典，可直接用于 qlib workflow
    """
    config = config or get_model_config()
    lgbm_cfg = config["lgbm"]

    model_config = {
        "class": lgbm_cfg["class"],
        "module_path": lgbm_cfg["module_path"],
        "kwargs": lgbm_cfg["kwargs"],
    }

    logger.info(f"LightGBM 模型配置: n_estimators={lgbm_cfg['kwargs']['n_estimators']}, "
                f"lr={lgbm_cfg['kwargs']['learning_rate']}, "
                f"max_depth={lgbm_cfg['kwargs']['max_depth']}")
    return model_config
