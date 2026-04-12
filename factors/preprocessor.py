"""
因子预处理流水线
缺失值填充、去极值、标准化、行业中性化
使用 Qlib 内置的 Processor 类
"""

from typing import Optional, Dict, Any, List

from loguru import logger

from utils.helpers import get_factor_config


def get_infer_processors(config: Optional[dict] = None) -> List[Dict[str, Any]]:
    """根据配置生成 Qlib 推理阶段的预处理器列表

    这些 processor 会在 DataHandler.setup_data() 时被
    qlib.utils.init_instance_by_config() 实例化为真正的 Processor 对象

    Qlib processor 所在包: qlib.data.dataset.processor

    Returns:
        Qlib processor 配置列表（符合 init_instance_by_config 格式）
    """
    config = config or get_factor_config()
    preproc = config.get("preprocessing", {})
    processors = []

    # 1. 去极值 + 标准化
    winsorize = preproc.get("winsorize", {})
    normalize = preproc.get("normalize", {})
    if winsorize.get("method") == "mad":
        # RobustZScoreNorm: 基于 MAD (Median Absolute Deviation) 的
        # 稳健标准化，同时完成去极值 (clip_outlier) 和归一化
        processors.append({
            "class": "RobustZScoreNorm",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            },
        })
        logger.info("预处理: MAD去极值 + RobustZScore标准化")
    elif normalize.get("method") == "zscore":
        # 标准 Z-score（无 MAD 去极值时使用）
        processors.append({
            "class": "ZScoreNorm",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "fields_group": "feature",
            },
        })
        logger.info("预处理: Z-Score标准化")

    # 2. 缺失值填充
    fillna = preproc.get("fillna", {})
    if fillna.get("method") in ("ffill_then_median", "median"):
        processors.append({
            "class": "Fillna",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "fields_group": "feature",
            },
        })
        logger.info("预处理: 缺失值填充")

    return processors


def get_learn_processors(config: Optional[dict] = None) -> List[Dict[str, Any]]:
    """根据配置生成 Qlib 训练阶段的预处理器列表

    learn_processors 仅对训练/验证集标签生效，
    用于标签的预处理（去空值、截面归一化等）

    Returns:
        Qlib processor 配置列表
    """
    processors = [
        # 删除标签为 NaN 的行
        {
            "class": "DropnaLabel",
            "module_path": "qlib.data.dataset.processor",
        },
        # 截面排名归一化（让标签在截面上均匀分布 [0, 1]）
        {
            "class": "CSRankNorm",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {"fields_group": "label"},
        },
    ]
    logger.info("学习预处理: DropnaLabel + CSRankNorm(label)")
    return processors


def build_handler_config(
    instruments: str,
    start_time: str,
    end_time: str,
    fit_start_time: str,
    fit_end_time: str,
    label_expr: str,
    feature_fields: list = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """构建完整的 DataHandler 配置

    返回的字典可被 qlib.utils.init_instance_by_config() 实例化为
    qlib.contrib.data.handler.Alpha158 对象

    Args:
        instruments: instruments 名称
        start_time: 数据起始时间
        end_time: 数据结束时间
        fit_start_time: fit 起始时间（用于标准化参数计算）
        fit_end_time: fit 结束时间
        label_expr: 标签表达式
        feature_fields: 额外特征字段 [(expr, name), ...]
        config: 因子配置

    Returns:
        DataHandler 配置字典
    """
    config = config or get_factor_config()

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
            "infer_processors": get_infer_processors(config),
            "learn_processors": get_learn_processors(config),
            "label": [label_expr],
        },
    }

    return handler_config
