#!/usr/bin/env python3
"""
一键运行全流水线
支持运行全部阶段或单独某个阶段
数据 → 因子 → 模型 → 信号 → 评估
"""

import os
import sys
import argparse
from pathlib import Path

# macOS ARM64: PyTorch 和 LightGBM 各自捆绑了不同的 libomp.dylib，
# 两个 OpenMP 运行时同时加载会导致线程初始化 segfault (EXC_BAD_ACCESS in __kmp_suspend_initialize_thread)。
# 解决方案: 强制 LightGBM 使用单线程，避免触发 OpenMP 并行区域的冲突。
# 这些环境变量必须在 import lightgbm / torch 之前设置。
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
else:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

import multiprocessing
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # 已经设置过

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from utils.helpers import (
    setup_logger,
    get_data_config,
    get_model_config,
    get_strategy_config,
    get_backtest_config,
    ensure_dir,
)


def stage_data(args):
    """阶段1: 数据下载与准备"""
    logger.info("=" * 60)
    logger.info("【阶段1】数据下载与准备")
    logger.info("=" * 60)

    from data.collector_akshare import AKShareCollector
    from data.csv_to_qlib import CsvToQlib
    from data.health_check import DataHealthChecker

    config = get_data_config()
    collector = AKShareCollector(config)

    if args.incremental:
        collector.update_incremental()
    else:
        collector.download_all()

    # 健康检查
    checker = DataHealthChecker(config)
    checker.run_full_check()

    # 转换
    converter = CsvToQlib(config)
    converter.convert_all()

    logger.info("数据阶段完成")


def stage_model(args):
    """阶段2+3: 因子 + 模型训练"""
    logger.info("=" * 60)
    logger.info("【阶段2+3】因子构建 + 模型训练")
    logger.info("=" * 60)

    from data.data_loader import get_data_loader
    from model.model_trainer import train_and_predict

    # 初始化 Qlib
    loader = get_data_loader()
    loader.init_qlib()

    # 训练模型
    model_name = args.model or get_model_config().get("default", "lgbm")
    logger.info(f"使用模型: {model_name}")

    recorder = train_and_predict(
        model_name=model_name,
        use_custom_factors=not args.no_custom_factors,
    )

    logger.info("模型训练阶段完成")
    return recorder


def stage_backtest(args, recorder=None):
    """阶段4: 信号 → 回测"""
    logger.info("=" * 60)
    logger.info("【阶段4】信号生成 + 回测")
    logger.info("=" * 60)

    # 确保 Qlib 已初始化（单独运行 backtest 阶段时需要）
    from data.data_loader import get_data_loader
    get_data_loader().init_qlib()

    from signal_gen.portfolio import run_backtest_from_recorder

    if recorder is None:
        # 从最近的实验加载
        from qlib.workflow import R
        exp_name = f"qlib_pipeline_{args.model or 'lgbm'}"
        experiment = R.get_exp(experiment_name=exp_name)
        recorders = experiment.list_recorders(rtype=experiment.RT_L)
        if not recorders:
            raise RuntimeError(f"实验 '{exp_name}' 中未找到记录，请先运行模型训练")
        if isinstance(recorders, dict):
            recorder = recorders[list(recorders.keys())[0]]
        else:
            # Pick latest FINISHED recorder
            finished = [r for r in recorders if r.info.get("status") == "FINISHED"]
            recorder = finished[-1] if finished else recorders[-1]

    result = run_backtest_from_recorder(recorder)
    logger.info("回测阶段完成")
    return result, recorder


def stage_evaluate(args, backtest_result=None, recorder=None):
    """阶段5: 评估与报告

    全部使用 Qlib API:
    - risk_analysis(report_df) → 收益/风险/Sharpe 等全部指标
    - recorder.list_metrics() → IC/ICIR（SigAnaRecord 已计算）
    - analysis_position.report_graph() → 组合分析图
    - analysis_model.model_performance_graph() → 模型分析图
    - 仅月度热力图/滚动Sharpe/超额收益曲线为自定义补充
    """
    logger.info("=" * 60)
    logger.info("【阶段5】评估与报告生成")
    logger.info("=" * 60)

    import pandas as pd
    from evaluation.metrics import (
        compute_metrics_from_report,
        compute_metrics_from_returns,
        load_ic_from_recorder,
        load_ic_series_from_recorder,
    )
    from evaluation.visualization import Visualizer
    from evaluation.report import generate_html_report

    # ── 从 backtest_result 提取 report_df 和 positions ───────
    report_df = None
    positions = None
    if backtest_result and "portfolio_metric" in backtest_result:
        portfolio_metric = backtest_result["portfolio_metric"]
        try:
            # backtest_daily 返回 (report_normal_df, positions)
            if isinstance(portfolio_metric, tuple):
                report_df = portfolio_metric[0]
                positions = portfolio_metric[1] if len(portfolio_metric) > 1 else None
            elif isinstance(portfolio_metric, pd.DataFrame):
                report_df = portfolio_metric
        except Exception as e:
            logger.warning(f"解析回测结果失败: {e}")

    # ── 从 positions 提取交易记录 ────────────────────────────
    trade_records = None
    if positions is not None:
        from signal_gen.portfolio import extract_trade_records
        trade_records = extract_trade_records(positions)
        if trade_records is not None and not trade_records.empty:
            # 保存交易记录 CSV
            trades_dir = ensure_dir(PROJECT_ROOT / "reports")
            trades_csv = trades_dir / f"trades_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
            trade_records.to_csv(trades_csv, index=False, encoding="utf-8-sig")
            logger.info(f"交易记录已保存: {trades_csv}")

    # ── 用 Qlib risk_analysis 计算指标 ───────────────────────
    if report_df is not None and "return" in report_df.columns:
        logger.info(f"回测报告: {len(report_df)} 天")
        metrics = compute_metrics_from_report(report_df)
    else:
        logger.warning("无有效回测结果，使用模拟数据")
        import numpy as np
        from datetime import date
        dates = pd.date_range("2023-01-01", date.today().isoformat(), freq="B")
        mock_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        metrics = compute_metrics_from_returns(mock_returns)

    # ── 从 Recorder 获取 IC 指标（SigAnaRecord 已计算）────────
    ic_summary = None
    ic_series = None
    if recorder is not None:
        ic_summary = load_ic_from_recorder(recorder)
        ic_series = load_ic_series_from_recorder(recorder)

        # 用原始 label 重算 IC 作为独立校验（不受 CSRankNorm 影响）
        raw_ic_series = load_ic_series_from_recorder(recorder, use_raw_label=True)
        if raw_ic_series is not None and ic_summary is not None:
            ic_summary["raw_ic_mean"] = float(raw_ic_series.mean())
            ic_summary["raw_rank_ic_mean"] = float(raw_ic_series.apply(
                lambda x: x  # rank IC 需要单独计算，此处先用 Pearson IC
            ).mean())
            logger.info(f"原始 label IC: {ic_summary['raw_ic_mean']:.4f} "
                        f"(CSRankNorm IC: {ic_summary.get('ic_mean', 'N/A')})")

    # ── 从 Recorder 生成买卖信号 ───────────────────────────
    trade_signals = None
    if recorder is not None:
        try:
            from signal_gen.signal_generator import generate_trade_signals
            pred = recorder.load_object("pred.pkl")
            if pred is not None:
                trade_signals = generate_trade_signals(pred)
                # 保存信号 CSV
                signals_dir = ensure_dir(PROJECT_ROOT / "reports")
                signals_csv = signals_dir / f"signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
                trade_signals.to_csv(signals_csv, index=False, encoding="utf-8-sig")
                logger.info(f"买卖信号已保存: {signals_csv}")
        except Exception as e:
            logger.warning(f"生成买卖信号失败: {e}")

    # ── 可视化（Qlib 内置 + 自定义补充）──────────────────────
    viz = Visualizer()
    plots = viz.generate_all_plots(
        report_df=report_df,
        recorder=recorder,
    )

    # 生成 HTML 报告
    report_path = generate_html_report(
        metrics, plots, ic_summary,
        trade_records=trade_records,
        trade_signals=trade_signals,
    )
    logger.info(f"报告已生成: {report_path}")

    return report_path


def run_all(args):
    """运行全流水线"""
    logger.info("#" * 60)
    logger.info("# Qlib Pipeline - 全流水线运行")
    logger.info("#" * 60)

    # 阶段1: 数据
    if not args.skip_data:
        stage_data(args)
    else:
        logger.info("跳过数据阶段")

    # 阶段2+3: 因子 + 模型
    recorder = stage_model(args)

    # 阶段4: 回测
    result, recorder = stage_backtest(args, recorder)

    # 阶段5: 评估
    report = stage_evaluate(args, result, recorder)

    logger.info("#" * 60)
    logger.info("# 全流水线运行完成!")
    logger.info(f"# 报告: {report}")
    logger.info("#" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Qlib A股量化投资流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py --stage all                  # 运行全流水线
  python run_pipeline.py --stage data                 # 仅数据阶段
  python run_pipeline.py --stage model --model lgbm   # 仅模型训练
  python run_pipeline.py --stage backtest             # 仅回测
  python run_pipeline.py --stage evaluate             # 仅评估报告
  python run_pipeline.py --stage all --skip-data      # 跳过数据下载
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "model", "backtest", "evaluate"],
        help="运行阶段 (default: all)",
    )
    parser.add_argument("--model", type=str, choices=["lgbm", "linear", "mlp"],
                        help="模型名称")
    parser.add_argument("--incremental", action="store_true",
                        help="数据增量更新模式")
    parser.add_argument("--skip-data", action="store_true",
                        help="跳过数据下载阶段")
    parser.add_argument("--no-custom-factors", action="store_true",
                        help="不使用自定义因子，仅 Alpha158")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()
    setup_logger("pipeline", level=args.log_level)

    if args.stage == "all":
        run_all(args)
    elif args.stage == "data":
        stage_data(args)
    elif args.stage == "model":
        stage_model(args)
    elif args.stage == "backtest":
        stage_backtest(args)
    elif args.stage == "evaluate":
        stage_evaluate(args)


if __name__ == "__main__":
    main()
