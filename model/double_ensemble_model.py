"""
DoubleEnsemble 模型配置

使用 Qlib 内置 DEnsembleModel，基于 LightGBM 的双重集成。
无需额外安装 PyTorch，与现有 LightGBM 依赖完全兼容。

两大核心机制 (论文: https://arxiv.org/abs/2010.01265):
  - 样本重加权 (SR): 对预测误差大的样本动态增加权重，减少标注噪声影响
  - 特征选择 (FS): 每轮自动筛选信息量最高的因子子集，提高泛化性

与单 LightGBM 相比的优势:
  - 不需要调整特征权重，FS 机制自动处理高相关因子组
  - 对 A 股噪声标签（未来收益率方差大）更鲁棒
  - 无超参数搜索负担，默认配置即可使用
"""

from typing import Dict, Any, Optional

from loguru import logger


def get_double_ensemble_config(config: Optional[dict] = None) -> Dict[str, Any]:
    """返回 DoubleEnsemble 模型的 Qlib 配置

    model_config.yaml 中可以通过 double_ensemble 节点覆盖默认参数:

        double_ensemble:
          num_models: 6
          enable_sr: true
          enable_fs: true
          epochs: 100
    """
    from utils.helpers import get_model_config
    config = config or get_model_config()
    de_cfg = config.get("double_ensemble", {})

    num_models = de_cfg.get("num_models", 6)
    enable_sr = de_cfg.get("enable_sr", True)
    enable_fs = de_cfg.get("enable_fs", True)
    bins_fs = de_cfg.get("bins_fs", 5)

    model_config = {
        "class": "DEnsembleModel",
        "module_path": "qlib.contrib.model.double_ensemble",
        "kwargs": {
            "base_model": de_cfg.get("base_model", "gbm"),
            "num_models": num_models,
            "enable_sr": enable_sr,
            "enable_fs": enable_fs,
            "alpha1": de_cfg.get("alpha1", 1.0),
            "alpha2": de_cfg.get("alpha2", 1.0),
            "bins_sr": de_cfg.get("bins_sr", 10),
            "bins_fs": bins_fs,
            "epochs": de_cfg.get("epochs", 100),
            "sample_ratios": de_cfg.get(
                "sample_ratios", [0.8, 0.7, 0.6, 0.5, 0.4][:bins_fs]
            ),
        },
    }

    logger.info(
        f"DoubleEnsemble 配置: n_models={num_models}, SR={enable_sr}, FS={enable_fs}"
    )
    return model_config
