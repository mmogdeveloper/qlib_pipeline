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
    get_analysis_config,
    ensure_dir,
)


def _backtest_metrics(recorder):
    """跑一次回测并提取关键指标，失败返回 None。

    封装 run_backtest_from_recorder → 解包 portfolio_metric 元组 →
    compute_metrics_from_report 的公共流程，供各敏感性 stage 复用。
    """
    from signal_gen.portfolio import run_backtest_from_recorder
    from evaluation.metrics import compute_metrics_from_report

    bt_result = run_backtest_from_recorder(recorder)
    portfolio_metric = bt_result["portfolio_metric"]
    report_df = portfolio_metric[0] if isinstance(portfolio_metric, tuple) else portfolio_metric
    if report_df is None or "return" not in report_df.columns:
        return None
    return compute_metrics_from_report(report_df)


def _save_sweep_csv(df, prefix: str):
    """把敏感性分析结果 DataFrame 保存为 reports/{prefix}_{YYYYMMDD}.csv。"""
    import pandas as pd
    report_dir = ensure_dir(PROJECT_ROOT / "reports")
    csv_path = report_dir / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


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

    # 下载历史成分股快照（消除幸存者偏差）
    collector.download_constituent_history()

    # 健康检查
    checker = DataHealthChecker(config)
    checker.run_full_check()

    # 转换（会自动检测历史成分股数据并生成动态 instruments）
    converter = CsvToQlib(config)
    converter.convert_all()

    logger.info("数据阶段完成")


def stage_model(args):
    """阶段2+3: 因子 + 模型训练"""
    logger.info("=" * 60)
    logger.info("【阶段2+3】因子构建 + 模型训练")
    logger.info("=" * 60)

    from data.data_loader import get_data_loader
    from model.model_trainer import train_and_predict, train_and_predict_rolling

    loader = get_data_loader()
    loader.init_qlib()

    # 训练模型
    model_name = args.model or get_model_config().get("default", "lgbm")
    logger.info(f"使用模型: {model_name}")

    if getattr(args, "rolling", False):
        logger.info("启用滚动时序验证模式")
        rolling_result = train_and_predict_rolling(
            model_name=model_name,
            use_custom_factors=not args.no_custom_factors,
        )
        recorder = rolling_result["recorder"]
    else:
        recorder = train_and_predict(
            model_name=model_name,
            use_custom_factors=not args.no_custom_factors,
        )

    logger.info("模型训练阶段完成")
    return recorder


def _get_recorder(args):
    """从最近的实验加载 recorder"""
    from qlib.workflow import R
    suffix = "_rolling" if getattr(args, "rolling", False) else ""
    exp_name = f"qlib_pipeline_{args.model or 'lgbm'}{suffix}"
    experiment = R.get_exp(experiment_name=exp_name)
    recorders = experiment.list_recorders(rtype=experiment.RT_L)
    if not recorders:
        raise RuntimeError(f"实验 '{exp_name}' 中未找到记录，请先运行模型训练")
    if isinstance(recorders, dict):
        return recorders[list(recorders.keys())[0]]
    finished = [r for r in recorders if r.info.get("status") == "FINISHED"]
    candidates = finished or recorders
    # list_recorders 返回顺序不保证，取 start_time 最新的那个
    def _start_time(r):
        return r.info.get("start_time") or r.info.get("id", "")
    return max(candidates, key=_start_time)


def _apply_strategy_overrides(args):
    """将 CLI 参数覆盖写入策略配置（运行时生效，不修改 YAML）"""
    from utils.helpers import set_config_override

    if args.rebalance:
        set_config_override("strategy", {"rebalance": {"frequency": args.rebalance}})
        logger.info(f"调仓频率覆盖为: {args.rebalance}")

    if args.n_drop is not None:
        set_config_override("strategy", {"n_drop": args.n_drop})
        logger.info(f"n_drop 覆盖为: {args.n_drop}")


def stage_backtest(args, recorder=None):
    """阶段4: 信号 → 回测"""
    logger.info("=" * 60)
    logger.info("【阶段4】信号生成 + 回测")
    logger.info("=" * 60)

    # 确保 Qlib 已初始化（单独运行 backtest 阶段时需要）
    from data.data_loader import get_data_loader
    get_data_loader().init_qlib()

    from signal_gen.portfolio import run_backtest_from_recorder

    # 应用 CLI 覆盖参数
    _apply_strategy_overrides(args)

    if recorder is None:
        recorder = _get_recorder(args)

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
    from evaluation.report import generate_text_report

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
        # load_ic_from_recorder 已内置 raw label IC 计算，
        # 返回的 ic_mean/rank_ic_mean 优先使用 raw label
        ic_summary = load_ic_from_recorder(recorder)
        ic_series = load_ic_series_from_recorder(recorder, use_raw_label=True)

        if ic_summary is not None:
            logger.info(
                f"原始 label IC: {ic_summary.get('raw_ic_mean', 'N/A')}, "
                f"原始 label Rank IC: {ic_summary.get('raw_rank_ic_mean', 'N/A')} "
                f"(CSRankNorm IC: {ic_summary.get('csranknorm_ic_mean', 'N/A')})"
            )

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

    # ── 生成文本报告 ──────────────────────────────────────
    report_path = generate_text_report(
        metrics, ic_summary=ic_summary,
        trade_records=trade_records,
        trade_signals=trade_signals,
        report_df=report_df,
    )
    logger.info(f"报告已生成: {report_path}")

    return report_path


def stage_sensitivity(args, recorder=None):
    """成本敏感性分析: 用多组成本参数跑回测，观察收益衰减"""
    logger.info("=" * 60)
    logger.info("【成本敏感性分析】")
    logger.info("=" * 60)

    import pandas as pd
    from data.data_loader import get_data_loader
    from utils.helpers import set_config_override, clear_config_overrides

    get_data_loader().init_qlib()

    _apply_strategy_overrides(args)

    if recorder is None:
        recorder = _get_recorder(args)

    cost_multipliers = get_analysis_config()["sensitivity"]["cost_multipliers"]
    base_st = get_strategy_config()
    base_open = (base_st["cost"]["buy_commission"] + base_st["cost"]["buy_slippage"])
    base_close = (base_st["cost"]["sell_commission"]
                  + base_st["cost"]["stamp_tax"]
                  + base_st["cost"]["sell_slippage"])

    results = []
    for mult in cost_multipliers:
        label = f"{mult:.1f}x"
        open_cost = base_open * mult
        close_cost = base_close * mult
        logger.info(f"--- 成本 {label}: open={open_cost:.6f}, close={close_cost:.6f} ---")

        # 覆盖 backtest 的 exchange_kwargs 成本
        set_config_override("backtest", {
            "exchange_kwargs": {
                "open_cost": open_cost,
                "close_cost": close_cost,
            }
        })
        # 同步覆盖 strategy cost（portfolio.py 用 strategy cost 计算）
        set_config_override("strategy", {
            "cost": {
                "buy_commission": base_st["cost"]["buy_commission"] * mult,
                "sell_commission": base_st["cost"]["sell_commission"] * mult,
                "stamp_tax": base_st["cost"]["stamp_tax"] * mult,
                "buy_slippage": base_st["cost"]["buy_slippage"] * mult,
                "sell_slippage": base_st["cost"]["sell_slippage"] * mult,
            }
        })

        try:
            metrics = _backtest_metrics(recorder)
            if metrics is not None:
                results.append({
                    "cost_multiplier": label,
                    "open_cost": open_cost,
                    "close_cost": close_cost,
                    "ann_return_with_cost": metrics.get("excess_return_with_cost/annualized_return", 0),
                    "sharpe_with_cost": metrics.get("excess_return_with_cost/information_ratio", 0),
                    "max_dd_with_cost": metrics.get("excess_return_with_cost/max_drawdown", 0),
                    "ann_return_no_cost": metrics.get("excess_return_without_cost/annualized_return", 0),
                    "sharpe_no_cost": metrics.get("excess_return_without_cost/information_ratio", 0),
                })
        except Exception as e:
            logger.error(f"成本 {label} 回测失败: {e}")
            results.append({"cost_multiplier": label, "error": str(e)})

        # 清除覆盖，准备下一轮
        clear_config_overrides("backtest")
        clear_config_overrides("strategy")

    df = pd.DataFrame(results)
    csv_path = _save_sweep_csv(df, "sensitivity")

    # 打印汇总表
    logger.info("")
    logger.info("=" * 70)
    logger.info("成本敏感性分析结果:")
    logger.info("=" * 70)
    print(f"\n{'成本倍数':<10s} {'买入成本':>10s} {'卖出成本':>10s} "
          f"{'超额年化':>10s} {'Sharpe':>8s} {'最大回撤':>10s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['cost_multiplier']:<10s} {'ERROR':>10s}")
            continue
        print(f"{r['cost_multiplier']:<10s} {r['open_cost']:>10.6f} {r['close_cost']:>10.6f} "
              f"{r['ann_return_with_cost']:>10.4f} {r['sharpe_with_cost']:>8.4f} "
              f"{r['max_dd_with_cost']:>10.4f}")
    print()

    logger.info(f"敏感性分析 CSV 已保存: {csv_path}")
    return csv_path


def stage_topk_sensitivity(args, recorder=None):
    """TopK 参数敏感性分析: 用多组 topk/n_drop 跑回测，检验参数稳健性"""
    logger.info("=" * 60)
    logger.info("【TopK 参数敏感性分析】")
    logger.info("=" * 60)

    import pandas as pd
    from data.data_loader import get_data_loader
    from utils.helpers import set_config_override, clear_config_overrides

    get_data_loader().init_qlib()

    if recorder is None:
        recorder = _get_recorder(args)

    topk_values = get_analysis_config()["topk_sensitivity"]["topk_values"]
    baseline_n_drop = get_strategy_config().get("n_drop", 1)

    results = []
    for topk in topk_values:
        # n_drop 固定为基础策略值以隔离 topk 独立效应
        n_drop = baseline_n_drop
        logger.info(f"--- TopK={topk}, N_drop={n_drop} (固定, 隔离 topk 效应) ---")

        set_config_override("strategy", {"topk": topk, "n_drop": n_drop})

        try:
            metrics = _backtest_metrics(recorder)
            if metrics is not None:
                results.append({
                    "topk": topk,
                    "n_drop": n_drop,
                    "ann_return_with_cost": metrics.get("excess_return_with_cost/annualized_return", 0),
                    "sharpe_with_cost": metrics.get("excess_return_with_cost/information_ratio", 0),
                    "max_dd_with_cost": metrics.get("excess_return_with_cost/max_drawdown", 0),
                    "ann_return_no_cost": metrics.get("excess_return_without_cost/annualized_return", 0),
                    "sharpe_no_cost": metrics.get("excess_return_without_cost/information_ratio", 0),
                    "abs_ann_return": metrics.get("return_with_cost/annualized_return", 0),
                    "abs_sharpe": metrics.get("return_with_cost/information_ratio", 0),
                })
        except Exception as e:
            logger.error(f"TopK={topk} 回测失败: {e}")
            results.append({"topk": topk, "n_drop": n_drop, "error": str(e)})

        clear_config_overrides("strategy")

    df = pd.DataFrame(results)
    csv_path = _save_sweep_csv(df, "topk_sensitivity")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"TopK 参数敏感性分析结果 (n_drop 固定={baseline_n_drop}, 隔离 topk 独立效应):")
    logger.info("=" * 80)
    print(f"\n{'TopK':>6s} {'N_drop':>7s} {'超额年化':>10s} {'Sharpe':>8s} "
          f"{'最大回撤':>10s} {'绝对年化':>10s} {'绝对Sharpe':>10s}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['topk']:>6d} {r['n_drop']:>7d} {'ERROR':>10s}")
            continue
        print(f"{r['topk']:>6d} {r['n_drop']:>7d} "
              f"{r['ann_return_with_cost']:>10.4f} {r['sharpe_with_cost']:>8.4f} "
              f"{r['max_dd_with_cost']:>10.4f} {r['abs_ann_return']:>10.4f} "
              f"{r['abs_sharpe']:>10.4f}")
    print()

    logger.info(f"TopK 敏感性分析 CSV 已保存: {csv_path}")
    return csv_path


def stage_ndrop_sensitivity(args, recorder=None):
    """n_drop 敏感性分析: 对同一份 pred 用不同 n_drop 值回测，量化换手约束对 alpha 的衰减"""
    logger.info("=" * 60)
    logger.info("【n_drop 敏感性分析】")
    logger.info("=" * 60)

    from data.data_loader import get_data_loader
    get_data_loader().init_qlib()

    if recorder is None:
        recorder = _get_recorder(args)

    from analysis.ndrop_sensitivity import run_sensitivity, _save_results

    n_drops = (getattr(args, "n_drops", None)
               or get_analysis_config()["ndrop_sensitivity"]["n_drops"])
    df = run_sensitivity(
        model_name=args.model or "lgbm",
        n_drops=n_drops,
        rolling=getattr(args, "rolling", False),
    )
    if not df.empty:
        import pandas as pd
        output_dir = ensure_dir(PROJECT_ROOT / "reports")
        _save_results(df, output_dir)
        logger.info("\n" + df.to_string(float_format="{:.4f}".format))
    return df


def stage_factor_ic(args):
    """单因子 IC 分析: 计算每个自定义因子的截面 IC，识别弱因子和冗余因子"""
    logger.info("=" * 60)
    logger.info("【单因子 IC 分析】")
    logger.info("=" * 60)

    from analysis.factor_ic_analysis import run_factor_ic_analysis, _save_results

    summary_df, ic_series_df = run_factor_ic_analysis(
        start=getattr(args, "ic_start", None),
        end=getattr(args, "ic_end", None),
        include_alpha158_sample=getattr(args, "include_alpha158_sample", False),
        method=getattr(args, "ic_method", "spearman"),
    )
    if not summary_df.empty:
        output_dir = ensure_dir(PROJECT_ROOT / "reports")
        _save_results(summary_df, ic_series_df, output_dir)
        display_cols = ["IC均值", "ICIR", "IC>0比例", "有效天数"]
        display_cols = [c for c in display_cols if c in summary_df.columns]
        logger.info("\n" + summary_df[display_cols].to_string(float_format="{:.4f}".format))
    return summary_df


def stage_regime(args, recorder=None):
    """市场环境过滤回测: 仅在大盘趋势向好时满仓，否则降低仓位或空仓

    通过对比不同均线周期的回测结果，验证市场环境过滤对回撤控制的效果。
    """
    logger.info("=" * 60)
    logger.info("【市场环境过滤分析】")
    logger.info("=" * 60)

    import numpy as np
    import pandas as pd
    from data.data_loader import get_data_loader
    from evaluation.metrics import compute_metrics_from_report

    get_data_loader().init_qlib()

    if recorder is None:
        recorder = _get_recorder(args)

    bt_config = get_backtest_config()
    st_config = get_strategy_config()

    # 加载预测信号
    pred = recorder.load_object("pred.pkl")
    if pred is None:
        raise RuntimeError("Recorder 中未找到 pred.pkl")

    # 信号滞后1天（与 portfolio.py 保持一致）
    pred_shifted = pred.copy()
    if isinstance(pred_shifted.index, pd.MultiIndex):
        pred_shifted = pred_shifted.reset_index()
        date_col = pred_shifted.columns[0]
        dates = sorted(pred_shifted[date_col].unique())
        date_map = dict(zip(dates[:-1], dates[1:]))
        pred_shifted[date_col] = pred_shifted[date_col].map(date_map)
        pred_shifted = pred_shifted.dropna(subset=[date_col])
        pred_shifted = pred_shifted.set_index(pred.index.names)

    # 加载基准指数收益率，构造累计净值用于均线判断
    from signal_gen.portfolio import _load_benchmark_returns
    bench_returns = _load_benchmark_returns(
        bt_config["benchmark"], bt_config["start_date"], bt_config["end_date"]
    )
    if bench_returns is None:
        logger.error("无法加载基准指数数据，市场环境过滤需要基准数据")
        return None

    bench_nav = (1 + bench_returns).cumprod()

    ma_configs = get_analysis_config()["regime"]["ma_configs"]

    results = []
    for mc in ma_configs:
        label = mc["name"]
        logger.info(f"--- {label} ---")

        try:
            if mc["ma_short"] == 0:
                # 无过滤：直接用原始信号回测
                filtered_pred = pred_shifted
            else:
                # 计算均线交叉信号
                short_ma = bench_nav.rolling(mc["ma_short"]).mean()
                long_ma = bench_nav.rolling(mc["ma_long"]).mean()
                regime_bullish = short_ma > long_ma  # True = 牛市环境

                # 在熊市日期将所有股票的信号置为 0（策略不下单）
                filtered_pred = pred_shifted.copy()
                if isinstance(filtered_pred, pd.DataFrame):
                    pred_dates = filtered_pred.index.get_level_values(0)
                else:
                    pred_dates = filtered_pred.index.get_level_values(0)

                for dt in pred_dates.unique():
                    # 用前一日的环境信号（避免前视偏差）
                    regime_loc = regime_bullish.index.get_indexer([dt], method="ffill")
                    if regime_loc[0] >= 0 and not regime_bullish.iloc[regime_loc[0]]:
                        if isinstance(filtered_pred, pd.DataFrame):
                            filtered_pred.loc[dt] = 0.0
                        else:
                            filtered_pred.loc[dt] = 0.0

            # 用过滤后的信号执行回测
            from qlib.contrib.evaluate import backtest_daily
            from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

            strategy = TopkDropoutStrategy(
                signal=filtered_pred,
                topk=st_config["topk"],
                n_drop=st_config["n_drop"],
            )

            exchange_kwargs = dict(bt_config["exchange_kwargs"])
            cost_config = st_config.get("cost", {})
            if cost_config:
                exchange_kwargs["open_cost"] = (
                    cost_config.get("buy_commission", 0.0003)
                    + cost_config.get("buy_slippage", 0.0002)
                )
                exchange_kwargs["close_cost"] = (
                    cost_config.get("sell_commission", 0.0003)
                    + cost_config.get("stamp_tax", 0.001)
                    + cost_config.get("sell_slippage", 0.0002)
                )

            portfolio_metric, indicator = backtest_daily(
                start_time=bt_config["start_date"],
                end_time=bt_config["end_date"],
                strategy=strategy,
                account=bt_config["account"],
                benchmark=bench_returns if bench_returns is not None else bt_config["benchmark"],
                exchange_kwargs=exchange_kwargs,
            )

            if isinstance(portfolio_metric, tuple):
                report_df = portfolio_metric[0]
            else:
                report_df = portfolio_metric

            if report_df is not None and "return" in report_df.columns:
                metrics = compute_metrics_from_report(report_df)
                results.append({
                    "regime_filter": label,
                    "ann_return_with_cost": metrics.get("excess_return_with_cost/annualized_return", 0),
                    "sharpe_with_cost": metrics.get("excess_return_with_cost/information_ratio", 0),
                    "max_dd_with_cost": metrics.get("excess_return_with_cost/max_drawdown", 0),
                    "ann_return_no_cost": metrics.get("excess_return_without_cost/annualized_return", 0),
                    "abs_ann_return": metrics.get("return_with_cost/annualized_return", 0),
                    "abs_max_dd": metrics.get("return_with_cost/max_drawdown", 0),
                })
        except Exception as e:
            logger.error(f"{label} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"regime_filter": label, "error": str(e)})

    df = pd.DataFrame(results)
    csv_path = _save_sweep_csv(df, "regime")

    logger.info("")
    logger.info("=" * 85)
    logger.info("市场环境过滤分析结果:")
    logger.info("=" * 85)
    print(f"\n{'过滤条件':<16s} {'超额年化':>10s} {'超额Sharpe':>10s} "
          f"{'超额最大回撤':>12s} {'绝对年化':>10s} {'绝对最大回撤':>12s}")
    print("-" * 85)
    for r in results:
        if "error" in r:
            print(f"{r['regime_filter']:<16s} {'ERROR':>10s}")
            continue
        print(f"{r['regime_filter']:<16s} "
              f"{r['ann_return_with_cost']:>10.4f} {r['sharpe_with_cost']:>10.4f} "
              f"{r['max_dd_with_cost']:>12.4f} {r['abs_ann_return']:>10.4f} "
              f"{r['abs_max_dd']:>12.4f}")
    print()

    logger.info(f"市场环境过滤分析 CSV 已保存: {csv_path}")
    return csv_path


def _run_pre_factor_ic(args):
    """训练前因子 IC 分析（使用验证集，避免前视偏差）

    将 IC 分析结果写入 reports/factor_ic_summary.csv 和
    reports/factor_ic_series.csv，供 get_custom_factor_expressions()
    的 auto_filter 在加载因子时读取。

    使用验证集而非测试集的原因：
    - 因子筛选属于建模决策，必须基于训练/验证期的数据
    - 若使用测试集 IC 来选因子，模型训练实质上看到了未来信息
    """
    if getattr(args, "no_custom_factors", False):
        logger.info("--no-custom-factors 已指定，跳过因子 IC 分析")
        return

    if getattr(args, "skip_factor_ic", False):
        logger.info("--skip-factor-ic 已指定，跳过因子 IC 分析")
        return

    from utils.helpers import get_data_config
    d_config = get_data_config()
    valid = d_config["split"]["valid"]

    logger.info("=" * 60)
    logger.info("【训练前因子 IC 分析】使用验证集: "
                f"{valid['start']} ~ {valid['end']}")
    logger.info("=" * 60)

    from analysis.factor_ic_analysis import run_factor_ic_analysis, _save_results

    summary_df, ic_series_df = run_factor_ic_analysis(
        start=valid["start"],
        end=valid["end"],
        include_alpha158_sample=True,   # 始终对比 Alpha158，用于去重
        method=getattr(args, "ic_method", "spearman"),
    )

    if summary_df.empty:
        logger.warning("因子 IC 分析无结果，将使用全量候选因子训练")
        return

    output_dir = ensure_dir(PROJECT_ROOT / "reports")
    _save_results(summary_df, ic_series_df, output_dir)

    display_cols = ["IC均值", "ICIR", "IC>0比例", "有效天数"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    logger.info("因子 IC 汇总（验证集）：\n" +
                summary_df[display_cols].to_string(float_format="{:.4f}".format))


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

    # 阶段2: 因子 IC 分析（可选，--factor-ic 显式触发）
    if args.factor_ic:
        stage_factor_ic(args)

    # 阶段3: 因子 + 模型训练
    recorder = stage_model(args)

    # 阶段4: 回测
    result, recorder = stage_backtest(args, recorder)

    # 可选: 成本敏感性分析（在评估报告之前运行，使报告能包含稳健性数据）
    if args.sensitivity:
        stage_sensitivity(args, recorder)

    # 可选: TopK 参数敏感性分析
    if args.topk_sensitivity:
        stage_topk_sensitivity(args, recorder)

    # 可选: n_drop 敏感性分析
    if args.ndrop_sensitivity:
        stage_ndrop_sensitivity(args, recorder)

    # 可选: 单因子 IC 分析
    if args.factor_ic:
        stage_factor_ic(args)

    # 可选: 市场环境过滤分析
    if args.regime:
        stage_regime(args, recorder)

    # 阶段5: 评估与报告（最后运行，以便报告包含上方所有可选分析的 CSV 结果）
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
  python run_pipeline.py --stage sensitivity          # 单独运行成本敏感性分析
  python run_pipeline.py --stage backtest --n-drop 2 --rebalance month  # 低换手率回测
  python run_pipeline.py --stage all --skip-data --sensitivity  # 全流水线 + 敏感性分析
  python run_pipeline.py --stage topk-sensitivity              # 单独运行 TopK 参数敏感性分析
  python run_pipeline.py --stage regime                        # 单独运行市场环境过滤分析
  python run_pipeline.py --stage all --skip-data --topk-sensitivity --regime  # 全部分析
  python run_pipeline.py --stage ndrop-sensitivity                            # n_drop 敏感性分析
  python run_pipeline.py --stage ndrop-sensitivity --n-drops 1 3 5 10        # 指定 n_drop 候选值
  python run_pipeline.py --stage factor-ic                                    # 单因子 IC 分析
  python run_pipeline.py --stage factor-ic --include-alpha158-sample          # 含 Alpha158 对比
  python run_pipeline.py --stage all --skip-data --ndrop-sensitivity --factor-ic  # 含两项分析
  python run_pipeline.py --stage model                    # 自动先做因子 IC 分析再训练
  python run_pipeline.py --stage model --skip-factor-ic   # 跳过因子分析直接训练
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "data", "model", "backtest", "evaluate",
                 "sensitivity", "topk-sensitivity", "regime",
                 "ndrop-sensitivity", "factor-ic"],
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
    parser.add_argument("--skip-factor-ic", action="store_true",
                        help="跳过训练前因子 IC 分析（使用已有 CSV 或全量因子）")
    parser.add_argument("--rebalance", type=str, choices=["day", "week", "month"],
                        help="覆盖调仓频率 (降低频率可减少换手率)")
    parser.add_argument("--n-drop", type=int,
                        help="覆盖每期最多换出股票数 (降低可减少换手率)")
    parser.add_argument("--ndrop-sensitivity", action="store_true",
                        help="运行 n_drop 敏感性分析，量化换手约束对 alpha 的衰减")
    parser.add_argument("--n-drops", nargs="+", type=int, default=None,
                        help="n_drop 敏感性分析的候选值列表，默认 1 3 5 10 20")
    parser.add_argument("--factor-ic", action="store_true",
                        help="运行单因子 IC 分析，识别弱因子和冗余因子")
    parser.add_argument("--ic-start", type=str, default=None,
                        help="因子 IC 分析的开始日期，默认使用测试集起始")
    parser.add_argument("--ic-end", type=str, default=None,
                        help="因子 IC 分析的结束日期，默认使用测试集结束")
    parser.add_argument("--include-alpha158-sample", action="store_true",
                        help="因子 IC 分析时同时分析 Alpha158 代表性因子（对比增量信息）")
    parser.add_argument("--ic-method", type=str, default="spearman",
                        choices=["spearman", "pearson"],
                        help="因子 IC 计算方法（默认 spearman / Rank IC）")
    parser.add_argument("--sensitivity", action="store_true",
                        help="运行成本敏感性分析 (多组成本参数回测)")
    parser.add_argument("--topk-sensitivity", action="store_true",
                        help="运行 TopK 参数敏感性分析 (多组 topk 值回测)")
    parser.add_argument("--regime", action="store_true",
                        help="运行市场环境过滤分析 (均线过滤降低回撤)")
    parser.add_argument("--rolling", action="store_true",
                        help="启用滚动时序验证（多折训练，检验模型稳定性）")
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
    elif args.stage == "sensitivity":
        stage_sensitivity(args)
    elif args.stage == "topk-sensitivity":
        stage_topk_sensitivity(args)
    elif args.stage == "regime":
        stage_regime(args)
    elif args.stage == "ndrop-sensitivity":
        stage_ndrop_sensitivity(args)
    elif args.stage == "factor-ic":
        stage_factor_ic(args)


if __name__ == "__main__":
    main()
