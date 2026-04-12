"""
自定义 DataHandler：Alpha158 + 自定义因子

继承 Qlib 的 Alpha158，覆盖 _get_fields() 方法来追加自定义因子。
这是 Qlib 官方推荐的扩展方式。

Alpha158 继承链：
  Alpha158 → DataHandlerLP → DataHandler
  Alpha158._get_fields() 返回 feature/label 的字段定义
  DataHandlerLP 将这些字段传给 qlib.data.D.features() 计算

通过覆盖 _get_fields()，可以在 Alpha158 的 158 个因子基础上
追加任意自定义因子表达式。
"""

from typing import List, Tuple

from loguru import logger

from qlib.contrib.data.handler import Alpha158


class Alpha158Custom(Alpha158):
    """Alpha158 + 自定义因子的 DataHandler

    用法：
        在 dataset_config 中将 class 替换为 Alpha158Custom，
        并通过 kwargs["custom_fields"] 传入自定义因子。

    示例::

        handler_config = {
            "class": "Alpha158Custom",
            "module_path": "factors.custom_handler",
            "kwargs": {
                "instruments": "csi300",
                "start_time": "2015-01-01",
                "end_time": "2026-04-12",
                ...
                "custom_fields": [
                    ("$close/Ref($close,5)-1", "mom_5d"),
                    ("Std($close/Ref($close,1)-1, 20)", "vol_20d"),
                ],
            },
        }
    """

    def __init__(self, custom_fields: List[Tuple[str, str]] = None, **kwargs):
        """
        Args:
            custom_fields: [(expression, name), ...] 自定义因子列表
            **kwargs: Alpha158 的其他参数 (instruments, start_time, ...)
        """
        self._custom_fields = custom_fields or []
        if self._custom_fields:
            logger.info(f"Alpha158Custom: 加载 {len(self._custom_fields)} 个自定义因子")
        super().__init__(**kwargs)

    def _get_fields(self):
        """覆盖 Alpha158._get_fields()，追加自定义因子

        Returns:
            (feature_fields, label_fields) 元组
            feature_fields: [(expression, name), ...]
            label_fields: [(expression, name), ...]
        """
        # 获取 Alpha158 原始的 158 个因子 + label 定义
        fields, names = super()._get_fields()

        # 追加自定义因子到 feature 字段
        if self._custom_fields:
            for expr, name in self._custom_fields:
                fields.append(expr)
                names.append(name)
            logger.info(
                f"因子总数: Alpha158({len(fields) - len(self._custom_fields)}) "
                f"+ 自定义({len(self._custom_fields)}) = {len(fields)}"
            )

        return fields, names
