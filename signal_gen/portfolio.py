"""
组合构建与回测执行
使用 Qlib 内置的 TopkDropoutStrategy 和 backtest 引擎

核心依赖 Qlib API:
- qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy
- qlib.contrib.evaluate.backtest_daily (高层封装)
- qlib.backtest (底层回测引擎)
"""

from typing import Dict, Any, Optional

import pandas as pd
from loguru import logger

from utils.helpers import get_strategy_config, get_backtest_config


def build_strategy_config(config: Optional[dict] = None) -> Dict[str, Any]:
    """构建 TopkDropoutStrategy 的配置字典

    TopkDropoutStrategy 是 Qlib 提供的选股策略：
    - 每期选分数最高的 topk 只股票
    - 每期最多换出 n_drop 只（避免频繁调仓）

    Returns:
        可被 init_instance_by_config() 实例化的策略配置
    """
    config = config or get_strategy_config()

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": None,  # 运行时由 pred_score 注入
            "topk": config["topk"],
            "n_drop": config["n_drop"],
        },
    }

    logger.info(f"策略配置: TopkDropout, K={config['topk']}, N_drop={config['n_drop']}")
    return strategy_config


def run_backtest(pred_score: pd.DataFrame, config: Optional[dict] = None) -> Dict:
    """使用 Qlib backtest_daily 执行回测

    backtest_daily 是 Qlib 提供的高层回测接口，内部流程：
    1. 创建 Exchange（交易所模拟器，处理涨跌停、成本等）
    2. 创建 SimulatorExecutor（逐日执行器）
    3. 每日调用 strategy.generate_trade_decision() 获取交易决策
    4. executor 执行交易并更新账户

    Args:
        pred_score: 模型预测分数
            - DataFrame: columns=['score'], MultiIndex=(datetime, instrument)
            - 或 Series: MultiIndex=(datetime, instrument)
        config: 回测配置覆盖

    Returns:
        {"portfolio_metric": DataFrame, "indicator": dict}
        portfolio_metric: 逐日组合指标（收益率、净值等）
        indicator: 汇总指标
    """
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    bt_config = config or get_backtest_config()
    st_config = get_strategy_config()

    logger.info("开始执行回测...")
    logger.info(f"  回测区间: {bt_config['start_date']} ~ {bt_config['end_date']}")
    logger.info(f"  初始资金: {bt_config['account']:,.0f}")
    logger.info(f"  基准: {bt_config['benchmark']}")

    # 实例化 TopkDropoutStrategy
    strategy = TopkDropoutStrategy(
        signal=pred_score,
        topk=st_config["topk"],
        n_drop=st_config["n_drop"],
    )

    # 执行回测
    # exchange_kwargs 控制交易规则：涨跌停阈值、成交价、交易成本等
    portfolio_metric, indicator = backtest_daily(
        start_time=bt_config["start_date"],
        end_time=bt_config["end_date"],
        strategy=strategy,
        account=bt_config["account"],
        benchmark=bt_config["benchmark"],
        exchange_kwargs=bt_config["exchange_kwargs"],
    )

    logger.info("回测执行完成")
    return {
        "portfolio_metric": portfolio_metric,
        "indicator": indicator,
    }


def extract_trade_records(positions: dict) -> pd.DataFrame:
    """从 backtest_daily 返回的 positions 中提取交易记录

    通过比较相邻交易日的持仓快照，推导出每日买入/卖出操作。

    Args:
        positions: {date: Position} 字典，backtest_daily 返回的 positions_normal

    Returns:
        DataFrame with columns:
            date, instrument, action(买入/卖出), amount, price, value
        如果无交易记录返回空 DataFrame
    """
    if not positions:
        logger.warning("positions 为空，无法提取交易记录")
        return pd.DataFrame()

    dates = sorted(positions.keys())
    records = []

    for i, date in enumerate(dates):
        pos = positions[date]

        # Position 对象可用 dict 方式遍历持仓股票
        current_holdings = {}
        try:
            if hasattr(pos, "position"):
                # Position.position 是 dict: {instrument: {amount, price, ...}}
                for inst, info in pos.position.items():
                    if inst == "cash":
                        continue
                    if isinstance(info, dict):
                        current_holdings[inst] = {
                            "amount": info.get("amount", 0),
                            "price": info.get("price", 0),
                        }
            elif hasattr(pos, "keys"):
                for inst in pos.keys():
                    if inst == "cash":
                        continue
                    info = pos[inst]
                    if isinstance(info, dict):
                        current_holdings[inst] = {
                            "amount": info.get("amount", 0),
                            "price": info.get("price", 0),
                        }
        except Exception as e:
            logger.debug(f"解析 {date} 持仓失败: {e}")
            continue

        if i == 0:
            # 第一天：所有持仓视为买入
            for inst, info in current_holdings.items():
                if info["amount"] > 0:
                    records.append({
                        "date": date,
                        "instrument": inst,
                        "action": "买入",
                        "amount": info["amount"],
                        "price": info["price"],
                        "value": info["amount"] * info["price"],
                    })
        else:
            prev_pos = positions[dates[i - 1]]
            prev_holdings = {}
            try:
                if hasattr(prev_pos, "position"):
                    for inst, info in prev_pos.position.items():
                        if inst == "cash":
                            continue
                        if isinstance(info, dict):
                            prev_holdings[inst] = info.get("amount", 0)
                elif hasattr(prev_pos, "keys"):
                    for inst in prev_pos.keys():
                        if inst == "cash":
                            continue
                        info = prev_pos[inst]
                        if isinstance(info, dict):
                            prev_holdings[inst] = info.get("amount", 0)
            except Exception:
                pass

            all_instruments = set(current_holdings.keys()) | set(prev_holdings.keys())
            for inst in all_instruments:
                cur_amt = current_holdings.get(inst, {}).get("amount", 0)
                prev_amt = prev_holdings.get(inst, 0)
                cur_price = current_holdings.get(inst, {}).get("price", 0)
                diff = cur_amt - prev_amt

                if abs(diff) < 1e-6:
                    continue

                if diff > 0:
                    records.append({
                        "date": date,
                        "instrument": inst,
                        "action": "买入",
                        "amount": diff,
                        "price": cur_price,
                        "value": diff * cur_price,
                    })
                else:
                    # 卖出时用前一日价格估算（实际成交价可能不同）
                    sell_price = cur_price if cur_price > 0 else 0
                    records.append({
                        "date": date,
                        "instrument": inst,
                        "action": "卖出",
                        "amount": abs(diff),
                        "price": sell_price,
                        "value": abs(diff) * sell_price,
                    })

    if not records:
        logger.warning("未提取到任何交易记录")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "action", "instrument"]).reset_index(drop=True)
    logger.info(f"提取交易记录: {len(df)} 条 (买入 {(df['action']=='买入').sum()}, 卖出 {(df['action']=='卖出').sum()})")
    return df


def _load_benchmark_returns(benchmark_csv: str, start_date: str, end_date: str) -> pd.Series:
    """从 CSV 文件加载基准收益率序列"""
    from pathlib import Path
    from utils.helpers import expand_path, get_data_config

    config = get_data_config()
    csv_path = expand_path(config["raw_csv_dir"]) / "index" / "SH000300.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date").set_index("date")
    returns = df["close"].pct_change().fillna(0)
    returns.index.name = "datetime"
    return returns


def run_backtest_from_recorder(recorder) -> Dict:
    """从 Qlib Recorder 中加载预测信号并执行回测

    Recorder 是 Qlib 的实验管理对象，由 model_trainer 中的
    R.start() + SignalRecord.generate() 写入预测信号 pred.pkl。

    Args:
        recorder: Qlib Recorder 对象

    Returns:
        回测结果字典
    """
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    bt_config = get_backtest_config()
    st_config = get_strategy_config()

    # 从 recorder 加载预测信号
    # pred.pkl 格式: DataFrame, MultiIndex=(datetime, instrument), columns=['score']
    pred = recorder.load_object("pred.pkl")
    if pred is None:
        raise RuntimeError("Recorder 中未找到 pred.pkl，请确认模型训练已完成")

    logger.info(f"从 Recorder 加载预测信号: {pred.shape}")
    logger.info("开始回测...")

    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=st_config["topk"],
        n_drop=st_config["n_drop"],
    )

    # 加载基准收益率，作为 Series 传入避免 Qlib 内部查找失败
    bench_returns = _load_benchmark_returns(
        bt_config["benchmark"], bt_config["start_date"], bt_config["end_date"]
    )
    benchmark = bench_returns if bench_returns is not None else bt_config["benchmark"]

    portfolio_metric, indicator = backtest_daily(
        start_time=bt_config["start_date"],
        end_time=bt_config["end_date"],
        strategy=strategy,
        account=bt_config["account"],
        benchmark=benchmark,
        exchange_kwargs=bt_config["exchange_kwargs"],
    )

    logger.info("回测完成")
    return {
        "portfolio_metric": portfolio_metric,
        "indicator": indicator,
    }
