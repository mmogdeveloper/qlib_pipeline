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


from datetime import datetime
from dateutil.relativedelta import relativedelta


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


def _generate_rolling_splits(data_config: dict) -> list:
    """根据配置生成滚动时序切分方案

    Returns:
        [{"train": (start, end), "valid": (start, end), "test": (start, end)}, ...]
    """
    rolling_cfg = data_config.get("rolling", {})
    n_splits = rolling_cfg.get("n_splits", 5)
    train_years = rolling_cfg.get("train_years", 4)
    valid_years = rolling_cfg.get("valid_years", 1)
    test_years = rolling_cfg.get("test_years", 1)

    total_start = datetime.strptime(data_config["split"]["train"]["start"], "%Y-%m-%d")
    total_end = datetime.strptime(data_config["split"]["test"]["end"], "%Y-%m-%d")

    window_years = train_years + valid_years + test_years
    total_years = (total_end - total_start).days / 365.25
    step_years = max(1, (total_years - window_years) / max(n_splits - 1, 1))

    splits = []
    for i in range(n_splits):
        offset = relativedelta(months=int(step_years * 12 * i))
        train_start = total_start + offset
        train_end = train_start + relativedelta(years=train_years) - relativedelta(days=1)
        valid_start = train_end + relativedelta(days=1)
        valid_end = valid_start + relativedelta(years=valid_years) - relativedelta(days=1)
        test_start = valid_end + relativedelta(days=1)
        test_end = test_start + relativedelta(years=test_years) - relativedelta(days=1)

        if test_end > total_end:
            test_end = total_end
        if test_start > total_end:
            break

        splits.append({
            "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            "valid": (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
            "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
        })

    return splits


def train_and_predict_rolling(
    model_name: str = "lgbm",
    use_custom_factors: bool = True,
) -> dict:
    """滚动时序验证：多折训练+预测，返回拼接后的 pred 和各折 IC

    Returns:
        {"all_preds": DataFrame, "fold_metrics": [...], "recorder": last_recorder}
    """
    import pandas as pd
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

    d_config = get_data_config()
    m_config = get_model_config()

    splits = _generate_rolling_splits(d_config)
    logger.info(f"滚动验证: {len(splits)} 折")
    for i, s in enumerate(splits):
        logger.info(f"  折{i+1}: train={s['train']}, valid={s['valid']}, test={s['test']}")

    all_preds = []
    fold_metrics = []
    last_recorder = None

    for i, split in enumerate(splits):
        logger.info(f"{'='*60}")
        logger.info(f"滚动验证 折{i+1}/{len(splits)}")
        logger.info(f"{'='*60}")

        label_expr = m_config["label"]["expression"]
        custom_fields = get_custom_factor_expressions() if use_custom_factors else None

        if custom_fields:
            dataset_config = get_alpha158_with_custom_config(
                instruments="csi300",
                train_period=split["train"],
                valid_period=split["valid"],
                test_period=split["test"],
                label_expr=label_expr,
                custom_fields=custom_fields,
            )
        else:
            dataset_config = get_alpha158_handler_config(
                instruments="csi300",
                train_period=split["train"],
                valid_period=split["valid"],
                test_period=split["test"],
                label_expr=label_expr,
            )

        model_cfg = get_model_config_by_name(model_name, m_config)

        logger.info(f"折{i+1}: 初始化数据集...")
        dataset = init_instance_by_config(dataset_config)

        logger.info(f"折{i+1}: 训练模型...")
        model = init_instance_by_config(model_cfg)

        with R.start(experiment_name=f"qlib_pipeline_{model_name}_rolling"):
            R.log_params(
                model=model_name, fold=i+1, n_folds=len(splits),
                train=str(split["train"]), valid=str(split["valid"]),
                test=str(split["test"]),
            )
            model.fit(dataset)

            recorder = R.get_recorder()
            sr = SignalRecord(model=model, dataset=dataset, recorder=recorder)
            sr.generate()

            sar = SigAnaRecord(ana_long_short=False, ann_scaler=252, recorder=recorder)
            sar.generate()

            pred = recorder.load_object("pred.pkl")
            test_start = pd.Timestamp(split["test"][0])
            test_end = pd.Timestamp(split["test"][1])
            pred_test = pred.loc[
                (pred.index.get_level_values(0) >= test_start) &
                (pred.index.get_level_values(0) <= test_end)
            ]
            all_preds.append(pred_test)

            ic_metrics = recorder.list_metrics()
            fold_result = {
                "fold": i + 1,
                "test_period": f"{split['test'][0]}~{split['test'][1]}",
                "ic_mean": ic_metrics.get("IC", 0),
                "icir": ic_metrics.get("ICIR", 0),
                "rank_ic": ic_metrics.get("Rank IC", 0),
                "rank_icir": ic_metrics.get("Rank ICIR", 0),
                "n_samples": len(pred_test),
            }
            fold_metrics.append(fold_result)
            logger.info(
                f"折{i+1} 结果: IC={fold_result['ic_mean']:.4f}, "
                f"ICIR={fold_result['icir']:.4f}, 样本={fold_result['n_samples']}"
            )
            last_recorder = recorder

    # 拼接所有折的 test 预测，去重（重叠区间保留后一折）
    combined = pd.concat(all_preds)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # 将拼接后的 pred 保存到最后一个 recorder
    if last_recorder is not None:
        last_recorder.save_objects(**{"pred.pkl": combined})
        logger.info(f"滚动验证完成: 合并 {len(combined)} 条预测")

    # 输出各折 IC 汇总
    logger.info("")
    logger.info("=" * 70)
    logger.info("滚动验证各折 IC 汇总:")
    logger.info("=" * 70)
    logger.info(f"{'折':>4s} {'测试区间':<24s} {'IC':>8s} {'ICIR':>8s} {'RankIC':>8s} {'样本':>8s}")
    logger.info("-" * 70)
    for fm in fold_metrics:
        logger.info(
            f"{fm['fold']:>4d} {fm['test_period']:<24s} "
            f"{fm['ic_mean']:>8.4f} {fm['icir']:>8.4f} "
            f"{fm['rank_ic']:>8.4f} {fm['n_samples']:>8d}"
        )
    avg_ic = sum(f["ic_mean"] for f in fold_metrics) / len(fold_metrics) if fold_metrics else 0
    avg_icir = sum(f["icir"] for f in fold_metrics) / len(fold_metrics) if fold_metrics else 0
    logger.info("-" * 70)
    logger.info(f"{'平均':>4s} {'':24s} {avg_ic:>8.4f} {avg_icir:>8.4f}")
    logger.info("")

    return {
        "all_preds": combined,
        "fold_metrics": fold_metrics,
        "recorder": last_recorder,
    }
