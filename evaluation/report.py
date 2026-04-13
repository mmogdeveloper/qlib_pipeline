"""
文本报告生成器
汇总指标、IC、买卖信号、交易记录，输出纯文本报告
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from utils.helpers import (
    ensure_dir,
    PROJECT_ROOT,
    get_strategy_config,
    get_backtest_config,
)


def _format_strategy_logic() -> list:
    """从配置生成策略执行逻辑文本"""
    lines = []
    try:
        st = get_strategy_config()
        bt = get_backtest_config()
    except Exception as e:
        lines.append(f"  (无法加载策略配置: {e})")
        return lines

    topk = st.get("topk", 30)
    n_drop = st.get("n_drop", 0)
    st_type = st.get("type", "topk_dropout")
    weighting = st.get("weighting", "equal")
    rb = st.get("rebalance", {}) or {}
    freq = rb.get("frequency", "day")
    weekday = rb.get("weekday", None)
    rules = st.get("trading_rules", {}) or {}
    ipo = rules.get("ipo_filter", {}) or {}
    cost = st.get("cost", {}) or {}
    init_cash = st.get("initial_cash", bt.get("account", 0))

    ex = bt.get("exchange_kwargs", {}) or {}
    deal_price = ex.get("deal_price", "open")
    limit_thr = ex.get("limit_threshold", 0.099)
    trade_unit = ex.get("trade_unit", 100)
    min_cost = ex.get("min_cost", 5)
    benchmark = bt.get("benchmark", "")
    start = bt.get("start_date", "")
    end = bt.get("end_date", "")

    wd_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
    freq_desc = {"day": "每日", "week": f"每周({wd_map.get(weekday,'周五')})", "month": "每月"}.get(freq, freq)

    lines.append("  ▸ 选股方式:")
    lines.append(f"      策略类型          {st_type}")
    lines.append(f"      持仓数量 (topk)   {topk}")
    if "dropout" in str(st_type):
        lines.append(f"      最大换出 (n_drop) {n_drop}  (每期最多替换 {n_drop} 只股票, 降低换手率)")
    lines.append(f"      权重分配          {weighting} (等权)" if weighting == "equal" else f"      权重分配          {weighting}")

    lines.append("")
    lines.append("  ▸ 交易执行:")
    lines.append(f"      调仓频率          {freq_desc}")
    lines.append(f"      成交价格          T+1 {deal_price} (次日开盘价撮合)")
    lines.append(f"      最小交易单位      {trade_unit} 股 (1手)")
    lines.append(f"      涨跌停阈值        ±{limit_thr*100:.1f}%")

    lines.append("")
    lines.append("  ▸ 过滤规则 (避免不可交易标的):")
    if rules.get("limit_up_filter", True):
        lines.append("      · 涨停不买入 (avoid limit_up)")
    if rules.get("limit_down_filter", True):
        lines.append("      · 跌停不卖出 (avoid limit_down)")
    if rules.get("suspend_filter", True):
        lines.append("      · 停牌标的跳过")
    if ipo.get("enabled", False):
        lines.append(f"      · 次新股过滤: 上市后 {ipo.get('days', 60)} 日内不交易")

    lines.append("")
    lines.append("  ▸ 交易成本 (双边):")
    open_cost = cost.get("buy_commission", 0) + cost.get("buy_slippage", 0)
    close_cost = (
        cost.get("sell_commission", 0)
        + cost.get("stamp_tax", 0)
        + cost.get("sell_slippage", 0)
    )
    lines.append(
        f"      买入成本          {open_cost*10000:.1f}bp "
        f"(佣金{cost.get('buy_commission',0)*10000:.1f}bp + 滑点{cost.get('buy_slippage',0)*10000:.1f}bp)"
    )
    lines.append(
        f"      卖出成本          {close_cost*10000:.1f}bp "
        f"(佣金{cost.get('sell_commission',0)*10000:.1f}bp + 印花税{cost.get('stamp_tax',0)*10000:.1f}bp "
        f"+ 滑点{cost.get('sell_slippage',0)*10000:.1f}bp)"
    )
    lines.append(f"      最低佣金          {min_cost} 元/笔")

    lines.append("")
    lines.append("  ▸ 回测设置:")
    lines.append(f"      初始资金          {init_cash:,.0f} 元")
    lines.append(f"      回测区间          {start} ~ {end}")
    lines.append(f"      基准指数          {benchmark}")

    lines.append("")
    lines.append("  ▸ 信号 → 下单流程:")
    lines.append("      1) 模型每日输出全市场股票预测分数")
    lines.append(f"      2) 分数排序: 取 Top{topk} 为目标持仓")
    if "dropout" in str(st_type):
        lines.append(f"      3) 与上期持仓对比, 至多换出 {n_drop} 只 (卖出分数最低的)")
        lines.append(f"      4) 换入新进 Top{topk} 的股票, 使用 T+1 开盘价成交")
    else:
        lines.append(f"      3) 全仓切换至新 Top{topk}, T+1 开盘价成交")
    lines.append("      5) 应用涨跌停/停牌/次新股过滤, 跳过不可交易标的")
    lines.append("      6) 扣除双边交易成本后更新净值")

    return lines


def generate_text_report(
    metrics: Dict[str, float],
    ic_summary: Optional[Dict[str, float]] = None,
    output_dir: Optional[str] = None,
    trade_records=None,
    trade_signals=None,
) -> str:
    """生成纯文本回测报告

    Args:
        metrics: 指标字典
        ic_summary: IC 汇总指标
        output_dir: 输出目录
        trade_records: 交易记录 DataFrame
        trade_signals: 买卖信号 DataFrame

    Returns:
        报告文件路径
    """
    logger.info("生成文本报告...")

    if output_dir:
        out_dir = ensure_dir(Path(output_dir))
    else:
        out_dir = ensure_dir(PROJECT_ROOT / "reports")

    today = datetime.now().strftime("%Y%m%d")
    report_path = out_dir / f"report_{today}.txt"

    lines = []
    sep = "=" * 60

    # ── 标题 ──
    lines.append(sep)
    lines.append("  Qlib 量化回测报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # ── 核心指标 ──
    lines.append("")
    lines.append("【核心指标】")
    lines.append("-" * 40)
    # 按类别分组输出，优先展示含成本超额收益
    _pct_keys = {"annualized_return", "max_drawdown", "mean", "std"}
    for name, val in sorted(metrics.items()):
        if isinstance(val, float):
            metric_short = name.split("/")[-1] if "/" in name else name
            if metric_short in _pct_keys:
                formatted = f"{val:.6f}"
            else:
                formatted = f"{val:.4f}"
        else:
            formatted = str(val)
        lines.append(f"  {name:<45s} {formatted}")

    # ── IC 指标 ──
    if ic_summary:
        lines.append("")
        lines.append("【因子 IC 指标】")
        lines.append("-" * 40)

        # 区分 CSRankNorm IC 和原始 IC
        norm_keys = ["ic_mean", "icir", "rank_ic_mean", "rank_icir"]
        raw_keys = ["raw_ic_mean", "raw_rank_ic_mean"]
        has_raw = any(k in ic_summary for k in raw_keys)

        if has_raw:
            lines.append("  -- CSRankNorm label 口径（训练用，数值偏高） --")
        for name in norm_keys:
            if name in ic_summary and ic_summary[name] is not None:
                lines.append(f"  {name:<20s} {ic_summary[name]:.4f}")

        if has_raw:
            lines.append("")
            lines.append("  -- 原始收益率 label 口径（真实预测力） --")
            for name in raw_keys:
                if name in ic_summary and ic_summary[name] is not None:
                    lines.append(f"  {name:<20s} {ic_summary[name]:.4f}")
            lines.append("")
            lines.append("  ⚠ CSRankNorm 会放大 IC，评估模型真实预测力请看原始口径")

        # 输出其他自定义 IC 指标
        shown = set(norm_keys + raw_keys)
        for name, val in ic_summary.items():
            if name not in shown and val is not None:
                lines.append(f"  {name:<20s} {val:.4f}")

    # ── 策略执行逻辑 ──
    lines.append("")
    lines.append("【策略执行逻辑】")
    lines.append("-" * 40)
    lines.extend(_format_strategy_logic())

    # ── 买卖信号 ──
    if trade_signals is not None and not trade_signals.empty:
        latest_date = trade_signals["date"].max()
        latest = trade_signals[trade_signals["date"] == latest_date].copy()
        latest = latest.sort_values("rank")

        n_buy = (latest["signal"] == "买入").sum()
        n_sell = (latest["signal"] == "卖出").sum()
        n_new_buy = (latest["change"] == "新买入").sum()
        n_new_sell = (latest["change"] == "新卖出").sum()
        latest_date_str = str(latest_date)[:10]

        lines.append("")
        lines.append("【买卖信号】")
        lines.append("-" * 40)
        lines.append(f"  最新信号日期: {latest_date_str}")
        lines.append(f"  买入: {n_buy}  卖出: {n_sell}  新买入: {n_new_buy}  新卖出: {n_new_sell}")
        lines.append("")

        display = latest[latest["signal"].isin(["买入", "卖出"])]
        if not display.empty:
            lines.append(f"  {'标的':<12s} {'分数':>8s} {'排名':>6s} {'信号':<6s} {'变动'}")
            lines.append("  " + "-" * 50)
            for _, row in display.iterrows():
                change_str = row["change"] if row["change"] else ""
                lines.append(
                    f"  {row['instrument']:<12s} {row['score']:>8.4f} {int(row['rank']):>6d} "
                    f"{row['signal']:<6s} {change_str}"
                )

        # 信号变动历史
        changes = trade_signals[trade_signals["change"].isin(["新买入", "新卖出"])].copy()
        changes = changes.sort_values("date", ascending=False).head(50)
        if not changes.empty:
            lines.append("")
            lines.append("  信号变动历史（最近50条）:")
            lines.append(f"  {'日期':<12s} {'标的':<12s} {'分数':>8s} {'变动'}")
            lines.append("  " + "-" * 44)
            for _, row in changes.iterrows():
                date_str = str(row["date"])[:10]
                lines.append(
                    f"  {date_str:<12s} {row['instrument']:<12s} {row['score']:>8.4f} {row['change']}"
                )

    # ── 交易记录 ──
    if trade_records is not None and not trade_records.empty:
        total_buys = len(trade_records[trade_records["action"] == "买入"])
        total_sells = len(trade_records[trade_records["action"] == "卖出"])
        total_buy_value = trade_records.loc[trade_records["action"] == "买入", "value"].sum()
        total_sell_value = trade_records.loc[trade_records["action"] == "卖出", "value"].sum()
        unique_instruments = trade_records["instrument"].nunique()
        trade_days = trade_records["date"].nunique()

        lines.append("")
        lines.append("【交易记录】")
        lines.append("-" * 40)
        lines.append(f"  交易日数: {trade_days}  涉及标的: {unique_instruments}")
        lines.append(f"  买入: {total_buys} 笔 / {total_buy_value:,.0f} 元")
        lines.append(f"  卖出: {total_sells} 笔 / {total_sell_value:,.0f} 元")
        lines.append("")
        lines.append(f"  {'日期':<12s} {'标的':<12s} {'方向':<6s} {'数量':>8s} {'价格':>8s} {'金额':>12s}")
        lines.append("  " + "-" * 60)
        for _, row in trade_records.iterrows():
            date_str = str(row["date"])[:10]
            lines.append(
                f"  {date_str:<12s} {row['instrument']:<12s} {row['action']:<6s} "
                f"{row['amount']:>8,.0f} {row['price']:>8.3f} {row['value']:>12,.2f}"
            )

    # ── 尾部 ──
    lines.append("")
    lines.append(sep)
    lines.append("  Qlib Pipeline - A股量化投资流水线 | 数据源: AKShare")
    lines.append(sep)

    text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 同时输出到日志
    logger.info(f"文本报告已生成: {report_path}")
    print("\n" + text)

    return str(report_path)
