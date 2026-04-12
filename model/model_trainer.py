"""
模型训练器

Qlib API 调用链：
1. init_instance_by_config(dataset_config)
   → DatasetH → Alpha158Custom handler → _get_fields() → D.features()
   → 一次性计算 158 + N 个因子
2. init_instance_by_config(model_config) → LGBModel / LinearModel / MLPModel
3. model.fit(dataset) → 训练
4. SignalRecord(model, dataset).generate() → 生成 pred.pkl
5. SigAnaRecord().generate() → 计算 IC/ICIR
"""

import pickle
from typing import Dict, Any, Optional

from loguru import logger

from utils.helpers import get_model_config, get_data_config, ensure_dir, PROJECT_ROOT
from model.lgbm_model import get_lgbm_model_config
from model.linear_model import get_linear_model_config
from model.mlp_model import get_mlp_model_config
from factors.alpha158 import get_alpha158_handler_config, get_alpha158_with_custom_config
from factors.custom_factors import get_custom_factor_expressions


MODEL_REGISTRY = {
    "lgbm": get_lgbm_model_config,
    "linear": get_linear_model_config,
    "mlp": get_mlp_model_config,
}


def get_model_config_by_name(model_name: str, config: Optional[dict] = None) -> Dict[str, Any]:
    """根据名称获取可被 init_instance_by_config() 实例化的模型配置"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型: {model_name}, 可选: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](config)


def build_workflow_config(
    model_name: str = "lgbm",
    use_custom_factors: bool = True,
    model_config: Optional[dict] = None,
    data_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """构建 Qlib workflow 配置"""
    m_config = model_config or get_model_config()
    d_config = data_config or get_data_config()

    split = d_config["split"]
    train_period = (split["train"]["start"], split["train"]["end"])
    valid_period = (split["valid"]["start"], split["valid"]["end"])
    test_period = (split["test"]["start"], split["test"]["end"])
    label_expr = m_config["label"]["expression"]

    if use_custom_factors:
        custom_fields = get_custom_factor_expressions()
        dataset_config = get_alpha158_with_custom_config(
            instruments="csi300",
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            label_expr=label_expr,
            custom_fields=custom_fields,
        )
    else:
        dataset_config = get_alpha158_handler_config(
            instruments="csi300",
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            label_expr=label_expr,
        )

    model_cfg = get_model_config_by_name(model_name, model_config)
    ensure_dir(PROJECT_ROOT / m_config.get("save_dir", "models/saved"))

    logger.info(f"Workflow 配置: 模型={model_name}, 自定义因子={use_custom_factors}")
    return {"model": model_cfg, "dataset": dataset_config}


def train_and_predict(
    model_name: str = "lgbm",
    use_custom_factors: bool = True,
) -> Any:
    """训练模型并生成预测

    Returns:
        Qlib Recorder 对象
    """
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

    workflow = build_workflow_config(model_name, use_custom_factors)

    # 1. 实例化 DatasetH
    # 如果 use_custom_factors=True，handler 是 Alpha158Custom：
    #   Alpha158Custom.__init__() → super().__init__() → _get_fields()
    #   → 返回 158 + N 个因子表达式 → D.features() 一次性计算全部因子
    # 自定义因子在此步骤就真正计算并进入 DatasetH，与 Alpha158 因子一起
    # 经过 infer_processors (RobustZScoreNorm, Fillna) 预处理
    logger.info("初始化数据集（因子计算中，可能需要几分钟）...")
    dataset = init_instance_by_config(workflow["dataset"])
    logger.info("数据集初始化完成")

    # 2. 实例化模型
    logger.info(f"初始化模型: {model_name}")
    model = init_instance_by_config(workflow["model"])

    # 3. 训练 + 记录
    with R.start(experiment_name=f"qlib_pipeline_{model_name}"):
        R.log_params(model=model_name, custom_factors=use_custom_factors)

        # model.fit(dataset): Qlib 模型从 dataset 获取 train/valid 数据训练
        logger.info("开始训练...")
        model.fit(dataset)
        logger.info("训练完成")

        # 保存模型
        model_save_dir = ensure_dir(PROJECT_ROOT / "models" / "saved")
        model_path = model_save_dir / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"模型已保存: {model_path}")

        recorder = R.get_recorder()

        # SignalRecord: model.predict(dataset) → pred.pkl
        sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
        sr.generate()
        logger.info("pred.pkl 已生成")

        # SigAnaRecord: 从 pred.pkl + label 计算 IC/ICIR 等
        sar = SigAnaRecord(ana_long_short=False, ann_scaler=252, recorder=recorder)
        sar.generate()
        logger.info("IC/ICIR 分析已完成")

        logger.info(f"实验 ID: {recorder.info['id']}")

    return recorder
