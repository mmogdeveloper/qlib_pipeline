"""
文本报告生成器
汇总指标、IC、买卖信号、交易记录，输出纯文本报告
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from utils.helpers import (
    ensure_dir,
    PROJECT_ROOT,
    get_strategy_config,
    get_backtest_config,
)


def _find_latest_csv(pattern: str) -> Optional[Path]:
    """在 reports/ 下找最新的匹配 CSV"""
    files = sorted((PROJECT_ROOT / "reports").glob(pattern))
    return files[-1] if files else None


def _format_monthly_returns(report_df: pd.DataFrame) -> list:
    """生成月度/年度收益热力图表 (纯文本)"""
    lines = []
    if report_df is None or "return" not in report_df.columns:
        return lines

    col = "return"
    if "return_wo_cost" in report_df.columns:
        col = "return_wo_cost"
    if "excess_return_with_cost" in report_df.columns:
        col = "excess_return_with_cost"

    df = report_df[[col]].copy()
    df.index = pd.to_datetime(df.index)
    monthly = (1 + df[col]).resample("ME").prod() - 1

    if monthly.empty:
        return lines

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    }).pivot(index="year", columns="month", values="ret")

    lines.append(f"  指标口径: {col}")
    lines.append("  " + "年份   " + "  ".join(f"{m:>6d}月" for m in range(1, 13)) + "   年度")
    lines.append("  " + "-" * 100)
    for year, row in pivot.iterrows():
        year_ret = (1 + row.dropna()).prod() - 1
        cells = []
        for m in range(1, 13):
            v = row.get(m)
            if pd.isna(v):
                cells.append(f"{'—':>7s}")
            else:
                cells.append(f"{v*100:>+6.2f}%")
        lines.append(f"  {year}  " + "  ".join(cells) + f"  {year_ret*100:>+6.2f}%")
    return lines


def _format_robustness_summary() -> list:
    """读取 sensitivity / topk / regime CSV, 汇总稳健性结论"""
    lines = []
    goals = {"topk稳健性": None, "市场环境过滤": None, "回撤控制": None}

    # ── TopK 敏感性 ──
    topk_csv = _find_latest_csv("topk_sensitivity_*.csv")
    if topk_csv:
        try:
            df = pd.read_csv(topk_csv)
            lines.append(f"  ▸ TopK 参数敏感性  (来源: {topk_csv.name})")
            cols = [c for c in ["topk", "n_drop", "ann_return_with_cost",
                                "sharpe_with_cost", "max_dd_with_cost"] if c in df.columns]
            if cols:
                lines.append("    " + "  ".join(f"{c:>20s}" for c in cols))
                for _, row in df.iterrows():
                    cells = []
                    for c in cols:
                        v = row[c]
                        cells.append(f"{v:>20.4f}" if isinstance(v, (int, float)) else f"{str(v):>20s}")
                    lines.append("    " + "  ".join(cells))

                if "sharpe_with_cost" in df.columns:
                    sharpes = df["sharpe_with_cost"].dropna()
                    if len(sharpes) >= 2:
                        std, mean = sharpes.std(), sharpes.mean()
                        cv = abs(std / mean) if mean else float("inf")
                        verdict = "稳健" if cv < 0.3 else ("较稳健" if cv < 0.5 else "敏感")
                        goals["topk稳健性"] = (verdict, f"Sharpe CV={cv:.2f} (mean={mean:.2f})")
                        lines.append(f"    → Sharpe 离散系数 CV={cv:.2f}  判定: {verdict}")
            lines.append("")
        except Exception as e:
            lines.append(f"  (读取 {topk_csv.name} 失败: {e})")

    # ── Regime 过滤 ──
    regime_csv = _find_latest_csv("regime_*.csv")
    if regime_csv:
        try:
            df = pd.read_csv(regime_csv)
            lines.append(f"  ▸ 市场环境过滤  (来源: {regime_csv.name})")
            cols = [c for c in ["regime_filter", "ann_return_with_cost",
                                "sharpe_with_cost", "max_dd_with_cost"] if c in df.columns]
            if cols:
                lines.append("    " + "  ".join(f"{c:>22s}" for c in cols))
                for _, row in df.iterrows():
                    cells = []
                    for c in cols:
                        v = row[c]
                        cells.append(f"{v:>22.4f}" if isinstance(v, (int, float)) else f"{str(v):>22s}")
                    lines.append("    " + "  ".join(cells))

                if {"regime_filter", "max_dd_with_cost"}.issubset(df.columns):
                    base = df[df["regime_filter"].astype(str).str.contains(
                        "none|off|baseline|False|无过滤|基准", case=False, na=False
                    )]
                    filt = df[~df.index.isin(base.index)]
                    if not base.empty and not filt.empty:
                        dd_base = base["max_dd_with_cost"].iloc[0]
                        dd_best = filt["max_dd_with_cost"].max()  # 回撤是负数, max 即最浅
                        improve = dd_best - dd_base
                        regime_verdict = "有效" if improve > 0.01 else ("微弱" if improve > 0 else "无效")
                        goals["市场环境过滤"] = (regime_verdict, f"最大回撤改善 {improve*100:+.2f}pct")
                        lines.append(f"    → 回撤改善 {improve*100:+.2f}pct  判定: {regime_verdict}")
                    if not base.empty:
                        dd_base = base["max_dd_with_cost"].iloc[0]
                        if dd_base > -0.10:
                            dd_verdict, dd_detail = "优秀", f"最大回撤 {dd_base*100:.2f}% (<10%)"
                        elif dd_base > -0.15:
                            dd_verdict, dd_detail = "良好", f"最大回撤 {dd_base*100:.2f}% (<15%)"
                        elif dd_base > -0.20:
                            dd_verdict, dd_detail = "一般", f"最大回撤 {dd_base*100:.2f}% (<20%)"
                        else:
                            dd_verdict, dd_detail = "较弱", f"最大回撤 {dd_base*100:.2f}% (>20%)"
                        goals["回撤控制"] = (dd_verdict, dd_detail)
            lines.append("")
        except Exception as e:
            lines.append(f"  (读取 {regime_csv.name} 失败: {e})")

    # ── 成本敏感性 ──
    cost_csv = _find_latest_csv("sensitivity_*.csv")
    if cost_csv:
        try:
            df = pd.read_csv(cost_csv)
            lines.append(f"  ▸ 交易成本敏感性  (来源: {cost_csv.name})")
            lines.append(f"    共 {len(df)} 组成本参数, 详见 CSV")
            lines.append("")
        except Exception:
            pass

    if not lines:
        lines.append("  (未发现 sensitivity / topk / regime CSV, 建议运行:")
        lines.append("     python run_pipeline.py --stage all --skip-data --topk-sensitivity --regime)")
        return lines

    # ── 目标达成评分 ──
    lines.append("  ▸ 策略目标达成评分:")
    for goal, res in goals.items():
        if res is None:
            lines.append(f"    [ ? ] {goal:<14s}  未评估")
        else:
            verdict, detail = res
            mark = {
                "稳健": "✔", "较稳健": "✔", "有效": "✔", "优秀": "✔", "良好": "✔",
                "敏感": "✗", "无效": "✗", "较弱": "✗",
                "微弱": "△", "一般": "△",
            }.get(verdict, "?")
            lines.append(f"    [ {mark} ] {goal:<14s}  {verdict:<6s}  {detail}")

    return lines


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
    report_df: Optional[pd.DataFrame] = None,
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
    _rename = {
        "return_with_cost/information_ratio": "return_with_cost/sharpe_ratio",
        "return_without_cost/information_ratio": "return_without_cost/sharpe_ratio",
    }
    for name, val in sorted(metrics.items()):
        if isinstance(val, float):
            metric_short = name.split("/")[-1] if "/" in name else name
            if metric_short in _pct_keys:
                formatted = f"{val:.6f}"
            else:
                formatted = f"{val:.4f}"
        else:
            formatted = str(val)
        display_name = _rename.get(name, name)
        lines.append(f"  {display_name:<45s} {formatted}")

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

    # ── 稳健性汇总 ──
    lines.append("")
    lines.append("【稳健性汇总 (TopK / Regime / 成本)】")
    lines.append("-" * 40)
    lines.extend(_format_robustness_summary())

    # ── 月度收益热力表 ──
    monthly_lines = _format_monthly_returns(report_df) if report_df is not None else []
    if monthly_lines:
        lines.append("")
        lines.append("【月度 / 年度收益】")
        lines.append("-" * 40)
        lines.extend(monthly_lines)

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
