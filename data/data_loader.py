"""
统一数据加载接口
初始化 Qlib 并提供统一的数据访问方法
"""

import threading
from pathlib import Path
from typing import Optional, Dict, Any

import qlib
from qlib.config import REG_CN
from loguru import logger

from utils.helpers import get_data_config, expand_path


class DataLoader:
    """Qlib 数据加载器，统一管理数据初始化和访问"""

    _initialized = False
    _lock = threading.Lock()

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_data_config()
        self.qlib_dir = str(expand_path(self.config["qlib_data_dir"]))

    def init_qlib(self, **kwargs) -> None:
        """初始化 Qlib

        线程安全，只会初始化一次，重复调用自动跳过
        """
        with DataLoader._lock:
            if DataLoader._initialized:
                logger.debug("Qlib 已初始化，跳过")
                return

            logger.info(f"初始化 Qlib，数据目录: {self.qlib_dir}")
            qlib.init(
                provider_uri=self.qlib_dir,
                region=REG_CN,
                **kwargs,
            )
            DataLoader._initialized = True
            logger.info("Qlib 初始化完成")

    def get_dataset_config(self) -> Dict[str, Any]:
        """返回数据集划分配置（训练/验证/测试区间）"""
        split = self.config["split"]
        return {
            "train": (split["train"]["start"], split["train"]["end"]),
            "valid": (split["valid"]["start"], split["valid"]["end"]),
            "test": (split["test"]["start"], split["test"]["end"]),
        }

    def get_instruments(self) -> str:
        """返回 instruments 名称"""
        return "csi300"

    def get_benchmark(self) -> str:
        """返回基准代码（从配置读取）"""
        index_code = self.config.get("benchmark_index", "000300")
        return f"SH{index_code}"


# 全局单例
_loader: Optional[DataLoader] = None


def get_data_loader(config: Optional[dict] = None) -> DataLoader:
    """获取全局 DataLoader 实例"""
    global _loader
    if _loader is None:
        _loader = DataLoader(config)
    return _loader
