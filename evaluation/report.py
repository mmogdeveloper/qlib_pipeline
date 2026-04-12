"""
文本报告生成器
汇总指标、IC、买卖信号、交易记录，输出纯文本报告
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from utils.helpers import ensure_dir, PROJECT_ROOT


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
    for name, val in metrics.items():
        if isinstance(val, float):
            if "率" in name or "收益" in name or "CAGR" in name or "回撤" in name:
                formatted = f"{val:.4%}"
            else:
                formatted = f"{val:.4f}"
        else:
            formatted = str(val)
        lines.append(f"  {name:<20s} {formatted}")

    # ── IC 指标 ──
    if ic_summary:
        lines.append("")
        lines.append("【因子 IC 指标】")
        lines.append("-" * 40)
        for name, val in ic_summary.items():
            lines.append(f"  {name:<20s} {val:.4f}")

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
