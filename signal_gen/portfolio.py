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


def _apply_rebalance_frequency(
    pred: pd.DataFrame, frequency: str, weekday: int = 4
) -> pd.DataFrame:
    """按调仓频率对预测分数做"前向填充"

    backtest_daily 是按日推进的, 想实现 week/month 调仓, 只需让非调仓日
    的分数等于上一个调仓日的分数 — 这样 TopkDropoutStrategy 在非调仓日
    看到的排名与上一日完全一致, 不会触发换仓。

    Args:
        pred: MultiIndex=(datetime, instrument), 含 score 列
        frequency: "day" / "week" / "month"
        weekday: 周内调仓日 (0=Mon, 4=Fri), 仅 frequency=week 生效

    Returns:
        前向填充后的 pred (索引不变)
    """
    if frequency == "day" or pred is None or pred.empty:
        return pred
    if not isinstance(pred.index, pd.MultiIndex):
        return pred

    dates = pd.DatetimeIndex(sorted(pred.index.get_level_values(0).unique()))
    if frequency == "week":
        # 每周中第一个 >= weekday 的交易日为调仓日
        # 用 ISO 周分组, 取每组的"目标星期或之后第一个"
        df_d = pd.DataFrame({"d": dates})
        df_d["wk"] = df_d["d"].dt.isocalendar().week
        df_d["yr"] = df_d["d"].dt.isocalendar().year
        df_d["dow"] = df_d["d"].dt.dayofweek

        def pick(group):
            ge = group[group["dow"] >= weekday]
            return ge.iloc[0]["d"] if not ge.empty else group.iloc[-1]["d"]

        rebal_dates = (
            df_d.groupby(["yr", "wk"], sort=True).apply(pick).tolist()
        )
    elif frequency == "month":
        df_d = pd.DataFrame({"d": dates})
        df_d["ym"] = df_d["d"].dt.to_period("M")
        rebal_dates = df_d.groupby("ym")["d"].first().tolist()
    else:
        return pred

    rebal_set = set(pd.Timestamp(d) for d in rebal_dates)
    logger.info(
        f"调仓频率={frequency}: {len(dates)} 个交易日 → {len(rebal_set)} 个调仓日"
    )

    # 把非调仓日的分数替换为最近一次调仓日的分数
    # 实现: 先按 (instrument) 分组, 把非调仓日的 score 置 NaN, 再 ffill
    pred_sorted = pred.sort_index()
    score_col = pred_sorted.columns[0]

    date_idx = pred_sorted.index.get_level_values(0)
    is_rebal = pd.Series([pd.Timestamp(d) in rebal_set for d in date_idx],
                         index=pred_sorted.index)
    masked = pred_sorted[score_col].where(is_rebal)
    # 按股票 ffill
    filled = masked.groupby(level=1).ffill()
    # 头部仍为 NaN 的交易日 (首个调仓日之前) 直接保留原值, 不影响后续
    filled = filled.fillna(pred_sorted[score_col])

    out = pred_sorted.copy()
    out[score_col] = filled
    return out


def _align_to_pred(df: pd.DataFrame, pred_index: pd.MultiIndex) -> pd.DataFrame:
    """将 D.features() 返回的 DataFrame 对齐到 pred 的索引层级顺序

    D.features() 返回 (instrument, datetime)，pred 是 (datetime, instrument)。
    此函数统一 swaplevel 并 reindex 到 pred 的索引。
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df
    if list(df.index.names) != list(pred_index.names):
        df = df.swaplevel(0, 1).sort_index()
    return df


def _filter_untradable_stocks(pred_score: pd.DataFrame) -> pd.DataFrame:
    """过滤不可交易的股票（停牌、ST、次新）

    通过 Qlib D.features() 查询当日 volume 和近期涨跌幅来判断：
    - 停牌: volume == 0
    - ST: 近10个交易日最大|涨跌幅| <= 5.5%（ST股涨跌停5%，留0.5%容差）
    - 次新: 股票在数据中的首个交易日距今不满60个交易日

    Args:
        pred_score: MultiIndex=(datetime, instrument) 的预测分数

    Returns:
        过滤后的 pred_score
    """
    from qlib.data import D

    st_config = get_strategy_config()
    trading_rules = st_config.get("trading_rules", {})

    if not isinstance(pred_score.index, pd.MultiIndex):
        return pred_score

    instruments = pred_score.index.get_level_values(1).unique().tolist()
    dates = pred_score.index.get_level_values(0)
    start_date = str(dates.min())[:10]
    end_date = str(dates.max())[:10]

    total_before = len(pred_score)
    mask = pd.Series(True, index=pred_score.index)

    # 停牌过滤: 以 D.calendar 为权威交易日历，交叉验证股票数据是否缺失
    # AKShare 在停牌日不写入数据（直接跳过），所以 D.features 对停牌日返回 NaN，
    # pred 中通常也不会产生停牌日的条目。此过滤的作用：
    #   1. 准确统计实际停牌事件数（用于诊断），
    #   2. 防御未来数据源引入 carry-forward 时误购停牌股。
    if trading_rules.get("suspend_filter", False):
        try:
            close_df = D.features(
                instruments, ["$close"], start_time=start_date, end_time=end_date
            )
            if close_df is None or close_df.empty:
                logger.warning("停牌过滤: D.features($close) 返回空数据，跳过")
            else:
                if isinstance(close_df.index, pd.MultiIndex):
                    if list(close_df.index.names) != list(pred_score.index.names):
                        close_df = close_df.swaplevel(0, 1)

                # 从 close_df 日期集合推断交易日历，避免额外的 D.calendar 调用
                close_date_strs = close_df.index.get_level_values(0).astype(str).str[:10]
                close_inst_strs = close_df.index.get_level_values(1).astype(str)
                cal_set = set(close_date_strs.unique())
                n_missing = len(cal_set) * len(instruments) - len(close_df)
                logger.info(
                    f"停牌检测: {len(cal_set)}个交易日 × {len(instruments)}只股票 = "
                    f"{len(cal_set) * len(instruments)}条理论记录，"
                    f"实际有数据 {len(close_df)}条，推算停牌/缺失事件 {n_missing}次"
                )

                # 向量化判断：pred 中有但 close_df 无数据且属于有效交易日的条目 → 停牌
                close_mi = pd.MultiIndex.from_arrays([close_date_strs, close_inst_strs])
                pred_date_strs = pred_score.index.get_level_values(0).astype(str).str[:10]
                pred_mi = pd.MultiIndex.from_arrays([
                    pred_date_strs,
                    pred_score.index.get_level_values(1).astype(str),
                ])
                suspended_arr = pred_date_strs.isin(cal_set).to_numpy() & ~pred_mi.isin(close_mi)
                n_suspended = int(suspended_arr.sum())
                if n_suspended > 0:
                    mask.loc[pred_score.index[suspended_arr]] = False
                logger.info(
                    f"停牌过滤: pred中移除 {n_suspended} 条 "
                    f"(AKShare数据源已自动排除停牌日，此值接近0属正常)"
                )
        except Exception as e:
            logger.warning(f"停牌过滤失败: {e}")

    # 一字涨停板过滤: T 日 open 相对于 T-1 日 close 涨幅 > 9.5% → 集合竞价已封死涨停，无法买入
    # pred 到此处已经 shift 1 天，pred 日期 T 对应的交易行为是"T 日开盘买入"。
    # 检查表达式: $open / Ref($close,1) - 1 > 0.095（一字涨停板场景）
    if trading_rules.get("limit_up_filter", False):
        try:
            price_df = D.features(
                instruments,
                ["$open", "Ref($close,1)"],
                start_time=start_date,
                end_time=end_date,
            )
            if price_df is not None and not price_df.empty:
                price_df = _align_to_pred(price_df, pred_score.index)
                price_df.columns = ["open", "prev_close"]
                valid = price_df["prev_close"] > 0
                gap = pd.Series(0.0, index=price_df.index)
                gap.loc[valid] = price_df.loc[valid, "open"] / price_df.loc[valid, "prev_close"] - 1
                is_limit_up_open = gap > 0.095
                common_idx = mask.index.intersection(is_limit_up_open.index)
                n_limit_up = int(is_limit_up_open.loc[common_idx].sum())
                mask.loc[common_idx] = mask.loc[common_idx] & ~is_limit_up_open.loc[common_idx]
                logger.info(f"一字涨停板过滤: 移除 {n_limit_up} 条记录（开盘即封板，无法买入）")
        except Exception as e:
            logger.warning(f"一字板过滤失败: {e}")

    # ST 过滤: 近10日最大|日收益率| <= 5.5%（ST股涨跌停5%）
    if trading_rules.get("st_filter", False):
        try:
            close_df = D.features(
                instruments, ["$close", "Ref($close,1)"],
                start_time=start_date, end_time=end_date,
            )
            close_df = _align_to_pred(close_df, pred_score.index)
            close_df.columns = ["close", "prev_close"]
            daily_ret = (close_df["close"] / close_df["prev_close"] - 1).abs()
            max_ret_10d = daily_ret.groupby(level=1).rolling(10, min_periods=5).max()
            max_ret_10d = max_ret_10d.droplevel(0)
            is_st = max_ret_10d <= 0.055
            common_idx = mask.index.intersection(is_st.index)
            mask.loc[common_idx] = mask.loc[common_idx] & ~is_st.loc[common_idx]
            n_st = is_st.loc[common_idx].sum()
            logger.info(f"ST 过滤（涨跌幅推断）: 移除 {n_st} 条记录")
        except Exception as e:
            logger.warning(f"ST 过滤失败: {e}")

    # 次新股过滤: 上市不满 N 个交易日
    ipo_config = trading_rules.get("ipo_filter", {})
    if ipo_config.get("enabled", False):
        try:
            ipo_days = ipo_config.get("days", 60)
            all_calendar = D.calendar(start_time="2005-01-01", end_time=end_date)
            cal_index = {d: i for i, d in enumerate(all_calendar)}

            vol_all = D.features(
                instruments, ["$volume"], start_time="2005-01-01", end_time=end_date,
            )
            vol_all = _align_to_pred(vol_all, pred_score.index)
            traded = vol_all[vol_all.iloc[:, 0] > 0]
            idx_df = traded.index.to_frame(index=False)
            first_trade = idx_df.groupby("instrument")["datetime"].min()

            ipo_mask = pd.Series(False, index=pred_score.index)
            for inst, listing_dt in first_trade.items():
                if listing_dt is pd.NaT:
                    continue
                listing_idx = cal_index.get(listing_dt, 0)
                cutoff_idx = listing_idx + ipo_days
                if cutoff_idx < len(all_calendar):
                    cutoff_date = all_calendar[cutoff_idx]
                else:
                    cutoff_date = all_calendar[-1]
                inst_mask = (pred_score.index.get_level_values(1) == inst) & \
                            (pred_score.index.get_level_values(0) < cutoff_date)
                ipo_mask = ipo_mask | inst_mask

            mask = mask & ~ipo_mask
            n_ipo = ipo_mask.sum()
            logger.info(f"次新股过滤（{ipo_days}日）: 移除 {n_ipo} 条记录")
        except Exception as e:
            logger.warning(f"次新股过滤失败: {e}")

    filtered = pred_score.loc[mask]
    total_after = len(filtered)
    logger.info(f"不可交易标的过滤: {total_before} → {total_after} 条 (移除 {total_before - total_after})")
    return filtered


def _build_strategy(st_config: dict, signal):
    """根据 weighting 配置实例化合适的策略

    - equal: TopkDropoutStrategy (Qlib 内置, 等权)
    - score_weighted: TopkScoreWeightedStrategy (本项目自定义, WeightStrategyBase)
    """
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    weighting = st_config.get("weighting", "equal")
    topk = st_config["topk"]

    if weighting == "score_weighted":
        from signal_gen.strategies import TopkScoreWeightedStrategy
        logger.info(f"策略: TopkScoreWeighted (topk={topk})")
        return TopkScoreWeightedStrategy(signal=signal, topk=topk)

    n_drop = st_config["n_drop"]
    logger.info(f"策略: TopkDropout (topk={topk}, n_drop={n_drop}, equal weight)")
    return TopkDropoutStrategy(signal=signal, topk=topk, n_drop=n_drop)


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

    # 将 signal 滞后1天，避免前视偏差:
    # 模型在 T 日用 T 日收盘数据生成 signal[T]，
    # shift 后 signal[T] 变成 signal[T+1]，即 T+1 日才使用该信号交易
    pred_score_shifted = pred_score.copy()
    if isinstance(pred_score_shifted.index, pd.MultiIndex):
        # MultiIndex=(datetime, instrument): 对 datetime 层 shift
        pred_score_shifted = pred_score_shifted.reset_index()
        date_col = pred_score_shifted.columns[0]  # datetime
        dates = sorted(pred_score_shifted[date_col].unique())
        date_map = dict(zip(dates[:-1], dates[1:]))  # T → T+1
        pred_score_shifted[date_col] = pred_score_shifted[date_col].map(date_map)
        pred_score_shifted = pred_score_shifted.dropna(subset=[date_col])
        pred_score_shifted = pred_score_shifted.set_index(pred_score.index.names)
    logger.info("signal 已滞后1天 (shift), 消除前视偏差")

    # 过滤不可交易标的（停牌、ST、次新）
    pred_score_shifted = _filter_untradable_stocks(pred_score_shifted)

    # 调仓频率: week/month 通过分数前向填充实现
    rebal = st_config.get("rebalance", {})
    pred_score_shifted = _apply_rebalance_frequency(
        pred_score_shifted,
        rebal.get("frequency", "day"),
        rebal.get("weekday", 4),
    )

    # 实例化策略 (按 weighting 选择 equal=Topk Dropout / score_weighted=自定义)
    strategy = _build_strategy(st_config, pred_score_shifted)

    # 合并 exchange_kwargs
    exchange_kwargs = dict(bt_config["exchange_kwargs"])
    trading_rules = st_config.get("trading_rules", {})
    if trading_rules.get("limit_up_filter"):
        exchange_kwargs.setdefault("limit_threshold", 0.099)

    # 用 strategy_config 的成本覆盖 backtest_config（单一事实源）
    cost_config = st_config.get("cost", {})
    if cost_config:
        open_cost = cost_config.get("buy_commission", 0.0003) + cost_config.get("buy_slippage", 0.0002)
        close_cost = (cost_config.get("sell_commission", 0.0003)
                      + cost_config.get("stamp_tax", 0.001)
                      + cost_config.get("sell_slippage", 0.0002))
        exchange_kwargs["open_cost"] = open_cost
        exchange_kwargs["close_cost"] = close_cost
        logger.info(f"交易成本(来自strategy_config): 买入={open_cost:.4f}, 卖出={close_cost:.4f}")

    # 执行回测
    portfolio_metric, indicator = backtest_daily(
        start_time=bt_config["start_date"],
        end_time=bt_config["end_date"],
        strategy=strategy,
        account=bt_config["account"],
        benchmark=bt_config["benchmark"],
        exchange_kwargs=exchange_kwargs,
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
                            prev_holdings[inst] = {
                                "amount": info.get("amount", 0),
                                "price": info.get("price", 0),
                            }
                elif hasattr(prev_pos, "keys"):
                    for inst in prev_pos.keys():
                        if inst == "cash":
                            continue
                        info = prev_pos[inst]
                        if isinstance(info, dict):
                            prev_holdings[inst] = {
                                "amount": info.get("amount", 0),
                                "price": info.get("price", 0),
                            }
            except Exception:
                pass

            all_instruments = set(current_holdings.keys()) | set(prev_holdings.keys())
            for inst in all_instruments:
                cur_amt = current_holdings.get(inst, {}).get("amount", 0)
                prev_amt = prev_holdings.get(inst, {}).get("amount", 0)
                cur_price = current_holdings.get(inst, {}).get("price", 0)
                prev_price = prev_holdings.get(inst, {}).get("price", 0)
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
                    # 卖出/清仓: 用前一日价格估算（清仓时 cur_price=0）
                    sell_price = prev_price if prev_price > 0 else cur_price
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

    # 诊断: 打印 pred 日期范围，帮助排查空仓问题
    if isinstance(pred.index, pd.MultiIndex):
        pred_dates = pred.index.get_level_values(0)
        logger.info(f"pred 日期范围: {str(pred_dates.min())[:10]} ~ {str(pred_dates.max())[:10]}, "
                    f"共 {pred_dates.nunique()} 个交易日, {pred.index.get_level_values(1).nunique()} 只股票")
    logger.info("开始回测...")

    # 将 signal 滞后1天，避免前视偏差
    pred_shifted = pred.copy()
    if isinstance(pred_shifted.index, pd.MultiIndex):
        pred_shifted = pred_shifted.reset_index()
        date_col = pred_shifted.columns[0]
        dates = sorted(pred_shifted[date_col].unique())
        date_map = dict(zip(dates[:-1], dates[1:]))
        pred_shifted[date_col] = pred_shifted[date_col].map(date_map)
        pred_shifted = pred_shifted.dropna(subset=[date_col])
        pred_shifted = pred_shifted.set_index(pred.index.names)
    logger.info("signal 已滞后1天 (shift), 消除前视偏差")

    # 诊断: 检测 pred 与回测窗口的重叠情况
    if isinstance(pred_shifted.index, pd.MultiIndex):
        shifted_dates = pred_shifted.index.get_level_values(0)
        bt_start = pd.Timestamp(bt_config["start_date"])
        bt_end = pd.Timestamp(bt_config["end_date"])
        overlap = ((shifted_dates >= bt_start) & (shifted_dates <= bt_end)).sum()
        if overlap == 0:
            logger.warning(
                f"⚠ pred shift 后无日期落入回测窗口 [{bt_config['start_date']} ~ {bt_config['end_date']}]! "
                f"shift 后日期范围: {str(shifted_dates.min())[:10]} ~ {str(shifted_dates.max())[:10]}. "
                "投资组合将全程空仓，请检查 recorder 是否来自正确的模型。"
            )
        else:
            logger.debug(f"pred shift 后 {overlap} 条记录落入回测窗口")

    # 过滤不可交易标的（停牌、ST、次新）
    pred_shifted = _filter_untradable_stocks(pred_shifted)

    rebal = st_config.get("rebalance", {})
    pred_shifted = _apply_rebalance_frequency(
        pred_shifted,
        rebal.get("frequency", "day"),
        rebal.get("weekday", 4),
    )

    strategy = _build_strategy(st_config, pred_shifted)

    # 加载基准收益率，作为 Series 传入避免 Qlib 内部查找失败
    bench_returns = _load_benchmark_returns(
        bt_config["benchmark"], bt_config["start_date"], bt_config["end_date"]
    )
    benchmark = bench_returns if bench_returns is not None else bt_config["benchmark"]

    # 合并交易成本: strategy_config 为单一事实源
    exchange_kwargs = dict(bt_config["exchange_kwargs"])
    cost_config = st_config.get("cost", {})
    if cost_config:
        open_cost = cost_config.get("buy_commission", 0.0003) + cost_config.get("buy_slippage", 0.0002)
        close_cost = (cost_config.get("sell_commission", 0.0003)
                      + cost_config.get("stamp_tax", 0.001)
                      + cost_config.get("sell_slippage", 0.0002))
        exchange_kwargs["open_cost"] = open_cost
        exchange_kwargs["close_cost"] = close_cost

    portfolio_metric, indicator = backtest_daily(
        start_time=bt_config["start_date"],
        end_time=bt_config["end_date"],
        strategy=strategy,
        account=bt_config["account"],
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )

    logger.info("回测完成")
    return {
        "portfolio_metric": portfolio_metric,
        "indicator": indicator,
    }
