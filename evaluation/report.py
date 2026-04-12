"""
HTML 报告生成器
汇总指标表格和图表，生成完整的回测报告
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from utils.helpers import ensure_dir, PROJECT_ROOT


def _img_to_relative_path(img_path: str, report_dir: Path) -> str:
    """将图片路径转为相对于报告目录的路径"""
    try:
        return str(Path(img_path).relative_to(report_dir))
    except ValueError:
        # 不在同一目录树下，复制到 figures/ 子目录
        import shutil
        figures_dir = ensure_dir(report_dir / "figures")
        dest = figures_dir / Path(img_path).name
        shutil.copy2(img_path, dest)
        return str(dest.relative_to(report_dir))


def generate_html_report(
    metrics: Dict[str, float],
    plots: Dict[str, str],
    ic_summary: Optional[Dict[str, float]] = None,
    output_dir: Optional[str] = None,
    trade_records=None,
    trade_signals=None,
) -> str:
    """生成 HTML 回测报告

    Args:
        metrics: 指标字典
        plots: {图表名: 图片路径} 字典
        ic_summary: IC 汇总指标
        output_dir: 输出目录
        trade_records: 交易记录 DataFrame (columns: date, instrument, action, amount, price, value)
        trade_signals: 买卖信号 DataFrame (columns: date, instrument, score, rank, signal, change)

    Returns:
        报告文件路径
    """
    logger.info("生成 HTML 报告...")

    if output_dir:
        out_dir = ensure_dir(Path(output_dir))
    else:
        out_dir = ensure_dir(PROJECT_ROOT / "reports")

    today = datetime.now().strftime("%Y%m%d")
    report_path = out_dir / f"report_{today}.html"

    # 构建指标表格
    metrics_rows = ""
    for name, val in metrics.items():
        if isinstance(val, float):
            if "率" in name or "收益" in name or "CAGR" in name or "回撤" in name:
                formatted = f"{val:.4%}"
            else:
                formatted = f"{val:.4f}"
        else:
            formatted = str(val)
        metrics_rows += f"<tr><td>{name}</td><td>{formatted}</td></tr>\n"

    # IC 指标
    ic_rows = ""
    if ic_summary:
        for name, val in ic_summary.items():
            ic_rows += f"<tr><td>{name}</td><td>{val:.4f}</td></tr>\n"

    # 图表: 使用相对路径引用，避免 base64 嵌入导致 HTML 文件过大
    images_html = ""
    for name, path in plots.items():
        try:
            rel_path = _img_to_relative_path(path, out_dir)
            images_html += f"""
            <div class="chart">
                <img src="{rel_path}" alt="{name}">
            </div>
            """
        except Exception as e:
            logger.warning(f"图表 {name} 引用失败: {e}")

    # 交易记录表格
    trades_html = ""
    if trade_records is not None and not trade_records.empty:
        import pandas as pd
        trades_rows = ""
        for _, row in trade_records.iterrows():
            date_str = str(row["date"])[:10] if hasattr(row["date"], "strftime") else str(row["date"])[:10]
            action_class = "buy" if row["action"] == "买入" else "sell"
            trades_rows += f"""<tr class="{action_class}">
                <td>{date_str}</td>
                <td>{row['instrument']}</td>
                <td>{row['action']}</td>
                <td>{row['amount']:,.0f}</td>
                <td>{row['price']:.3f}</td>
                <td>{row['value']:,.2f}</td>
            </tr>\n"""

        # 汇总统计
        total_buys = len(trade_records[trade_records["action"] == "买入"])
        total_sells = len(trade_records[trade_records["action"] == "卖出"])
        total_buy_value = trade_records.loc[trade_records["action"] == "买入", "value"].sum()
        total_sell_value = trade_records.loc[trade_records["action"] == "卖出", "value"].sum()
        unique_instruments = trade_records["instrument"].nunique()
        trade_days = trade_records["date"].nunique()

        trades_html = f"""
        <h2>交易记录</h2>
        <div class="trade-summary">
            <span>交易日数: <b>{trade_days}</b></span>
            <span>涉及标的: <b>{unique_instruments}</b></span>
            <span>买入: <b>{total_buys}</b> 笔 / {total_buy_value:,.0f} 元</span>
            <span>卖出: <b>{total_sells}</b> 笔 / {total_sell_value:,.0f} 元</span>
        </div>
        <div class="table-scroll">
        <table>
            <thead>
                <tr><th>日期</th><th>标的</th><th>方向</th><th>数量</th><th>价格</th><th>金额</th></tr>
            </thead>
            <tbody>
                {trades_rows}
            </tbody>
        </table>
        </div>
        """

    # 买卖信号表格（仅展示最近一个交易日 + 有变动的信号）
    signals_html = ""
    if trade_signals is not None and not trade_signals.empty:
        import pandas as pd
        # 最新交易日的全部信号
        latest_date = trade_signals["date"].max()
        latest_signals = trade_signals[trade_signals["date"] == latest_date].copy()
        latest_signals = latest_signals.sort_values("rank")

        # 汇总
        n_buy = (latest_signals["signal"] == "买入").sum()
        n_sell = (latest_signals["signal"] == "卖出").sum()
        n_new_buy = (latest_signals["change"] == "新买入").sum()
        n_new_sell = (latest_signals["change"] == "新卖出").sum()
        latest_date_str = str(latest_date)[:10]

        # 只展示有信号的（买入/卖出），不展示「观望」
        display_signals = latest_signals[latest_signals["signal"].isin(["买入", "卖出"])]

        signal_rows = ""
        for _, row in display_signals.iterrows():
            sig_class = "buy" if row["signal"] == "买入" else "sell"
            change_str = f' <b>({row["change"]})</b>' if row["change"] else ""
            signal_rows += f"""<tr class="{sig_class}">
                <td>{row['instrument']}</td>
                <td>{row['score']:.4f}</td>
                <td>{int(row['rank'])}</td>
                <td>{row['signal']}{change_str}</td>
            </tr>\n"""

        # 信号变动历史（新买入/新卖出）
        changes = trade_signals[trade_signals["change"].isin(["新买入", "新卖出"])].copy()
        changes = changes.sort_values("date", ascending=False).head(100)
        change_rows = ""
        for _, row in changes.iterrows():
            date_str = str(row["date"])[:10]
            chg_class = "buy" if row["change"] == "新买入" else "sell"
            change_rows += f"""<tr class="{chg_class}">
                <td>{date_str}</td>
                <td>{row['instrument']}</td>
                <td>{row['score']:.4f}</td>
                <td>{row['change']}</td>
            </tr>\n"""

        signals_html = f"""
        <h2>买卖信号</h2>
        <div class="signal-summary">
            <span>最新信号日期: <b>{latest_date_str}</b></span>
            <span>买入标的: <b>{n_buy}</b></span>
            <span>卖出标的: <b>{n_sell}</b></span>
            <span>新买入: <b style="color:#e53935">{n_new_buy}</b></span>
            <span>新卖出: <b style="color:#43a047">{n_new_sell}</b></span>
        </div>
        <h3>最新持仓信号 ({latest_date_str})</h3>
        <div class="table-scroll">
        <table>
            <thead>
                <tr><th>标的</th><th>预测分数</th><th>排名</th><th>信号</th></tr>
            </thead>
            <tbody>
                {signal_rows}
            </tbody>
        </table>
        </div>
        """
        if change_rows:
            signals_html += f"""
        <h3>信号变动历史（最近100条）</h3>
        <div class="table-scroll">
        <table>
            <thead>
                <tr><th>日期</th><th>标的</th><th>预测分数</th><th>变动</th></tr>
            </thead>
            <tbody>
                {change_rows}
            </tbody>
        </table>
        </div>
        """

    # HTML 模板
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qlib 回测报告 - {today}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 20px 40px;
            background: #f5f5f5; color: #333;
        }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        h2 {{ color: #2196F3; margin-top: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        table {{
            border-collapse: collapse; width: 100%;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px 16px; text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #2196F3; color: white; }}
        tr:hover {{ background: #f0f7ff; }}
        .chart {{
            background: white; padding: 15px; margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            text-align: center;
        }}
        .chart img {{ max-width: 100%; height: auto; }}
        .trade-summary {{
            display: flex; gap: 24px; flex-wrap: wrap;
            margin: 12px 0; padding: 12px 16px;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        .trade-summary span {{ color: #555; }}
        .trade-summary b {{ color: #1a1a1a; }}
        .table-scroll {{ max-height: 600px; overflow-y: auto; margin: 15px 0; }}
        tr.buy td:nth-child(3) {{ color: #e53935; font-weight: bold; }}
        tr.sell td:nth-child(3) {{ color: #43a047; font-weight: bold; }}
        .signal-summary {{
            display: flex; gap: 24px; flex-wrap: wrap;
            margin: 12px 0; padding: 12px 16px;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        .signal-summary span {{ color: #555; }}
        .signal-summary b {{ color: #1a1a1a; }}
        .footer {{
            text-align: center; color: #999; margin-top: 40px;
            padding: 20px; border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Qlib 量化回测报告</h1>
        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>核心指标</h2>
        <table>
            <thead>
                <tr><th>指标</th><th>数值</th></tr>
            </thead>
            <tbody>
                {metrics_rows}
            </tbody>
        </table>

        {"<h2>因子 IC 指标</h2><table><thead><tr><th>指标</th><th>数值</th></tr></thead><tbody>" + ic_rows + "</tbody></table>" if ic_rows else ""}

        {signals_html}

        {trades_html}

        <h2>图表分析</h2>
        {images_html}

        <div class="footer">
            <p>Qlib Pipeline - A股量化投资流水线 | 数据源: AKShare</p>
        </div>
    </div>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML 报告已生成: {report_path}")
    return str(report_path)
