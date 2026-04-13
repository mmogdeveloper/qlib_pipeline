"""拆解 2024-09 异常月份的 alpha vs beta 来源（基于 rolling OOS pred）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from data.data_loader import get_data_loader
from signal_gen.portfolio import run_backtest_from_recorder
from qlib.workflow import R

get_data_loader().init_qlib()

exp = R.get_exp(experiment_name="qlib_pipeline_lgbm_rolling")
recs = exp.list_recorders(rtype=exp.RT_L)
if isinstance(recs, dict):
    recs = list(recs.values())
recorder = max(recs, key=lambda x: x.info.get("start_time") or "")
print(f"recorder: {recorder.info['id'][:8]} {recorder.info.get('start_time')}")

bt = run_backtest_from_recorder(recorder)
pm = bt["portfolio_metric"]
report_df = pm[0] if isinstance(pm, tuple) else pm

print(f"\nreport_df columns: {list(report_df.columns)}")
print(f"date range: {report_df.index.min()} ~ {report_df.index.max()}")

# 月度聚合
report_df = report_df.copy()
report_df["month"] = pd.to_datetime(report_df.index).strftime("%Y-%m")

monthly = report_df.groupby("month").agg(
    strategy=("return", lambda s: (1 + s).prod() - 1),
    bench=("bench", lambda s: (1 + s).prod() - 1),
    excess_with_cost=("excess_return_with_cost", lambda s: (1 + s).prod() - 1),
    excess_no_cost=("excess_return_without_cost", lambda s: (1 + s).prod() - 1),
    days=("return", "count"),
)
monthly["alpha_share"] = monthly["excess_with_cost"] / monthly["strategy"]

print("\n========== 月度收益拆解 (Rolling OOS) ==========")
print(monthly.to_string())

print("\n========== TOP 5 强势月份 (按 strategy return) ==========")
top = monthly.nlargest(5, "strategy")
print(top.to_string())

print("\n========== 2024-09 详情 ==========")
sept = monthly.loc["2024-09"]
print(sept)
print(f"\nstrategy 收益: {sept['strategy']:.2%}")
print(f"benchmark 收益: {sept['bench']:.2%}")
print(f"超额(含成本): {sept['excess_with_cost']:.2%}")
print(f"alpha 占比: {sept['alpha_share']:.1%}")
print(f"  解读: alpha {sept['excess_with_cost']:.2%} + beta {sept['bench']:.2%} ≈ strategy {sept['strategy']:.2%}")

# 保存月度表
out = Path(__file__).resolve().parents[1] / "reports" / "monthly_decomp_rolling.csv"
monthly.to_csv(out, encoding="utf-8-sig")
print(f"\n已保存: {out}")
