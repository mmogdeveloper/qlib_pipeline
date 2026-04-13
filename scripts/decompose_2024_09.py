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

# ──────────────────────────────────────────────────────────────
# 日频 OLS 回归：r_strat = alpha + beta * r_bench + ε
# 正确归因 beta 爆发：简单的"超额/总收益"比值会低估 beta 作用
# ──────────────────────────────────────────────────────────────
import numpy as np
from scipy import stats

print("\n" + "=" * 54)
print("日频 OLS beta 回归（alpha vs beta 严谨拆解）")
print("=" * 54)

# 全程基线：建立正常市场环境下的策略 beta
r_s_all = report_df["return"].values
r_b_all = report_df["bench"].values
mask_all = np.isfinite(r_s_all) & np.isfinite(r_b_all)
beta_all, alpha_daily_all, r_val_all, p_all, _ = stats.linregress(
    r_b_all[mask_all], r_s_all[mask_all]
)
alpha_ann_all = alpha_daily_all * 252
print(f"\n[全程基线 2023-01 ~ 今]")
print(f"  Beta          = {beta_all:.4f}  (1.0=随市，>1=杠杆暴露)")
print(f"  日均Alpha     = {alpha_daily_all*1e4:.2f} bp")
print(f"  年化Alpha     = {alpha_ann_all:.2%}")
print(f"  R²            = {r_val_all**2:.4f}  (beta可解释的收益占比)")

# 2024-09 单月回归
sept_mask = report_df["month"] == "2024-09"
if sept_mask.sum() < 5:
    print("\n[2024-09] 样本不足，跳过回归")
else:
    r_s_09 = report_df.loc[sept_mask, "return"].values
    r_b_09 = report_df.loc[sept_mask, "bench"].values
    valid_09 = np.isfinite(r_s_09) & np.isfinite(r_b_09)
    beta_09, alpha_daily_09, r_val_09, p_09, _ = stats.linregress(
        r_b_09[valid_09], r_s_09[valid_09]
    )
    alpha_ann_09 = alpha_daily_09 * 252

    # 用基线 beta 反推"如果 beta 正常，该月策略收益应是多少"
    bench_sept_total = (1 + r_b_09).prod() - 1
    # 基线 beta 对应的策略 beta 贡献
    beta_contrib_baseline = beta_all * bench_sept_total
    # 实际 beta 贡献
    beta_contrib_actual = beta_09 * bench_sept_total
    # alpha 贡献（实际总收益 - 实际 beta 贡献）
    strat_sept_total = (1 + r_s_09).prod() - 1
    alpha_contrib = strat_sept_total - beta_contrib_actual

    print(f"\n[2024-09 单月回归]  样本={valid_09.sum()} 个交易日")
    print(f"  Beta          = {beta_09:.4f}  (基线={beta_all:.4f}, 超出 {beta_09 - beta_all:+.4f})")
    print(f"  日均Alpha     = {alpha_daily_09*1e4:.2f} bp")
    print(f"  年化Alpha     = {alpha_ann_09:.2%}")
    print(f"  R²            = {r_val_09**2:.4f}")
    print(f"  p值           = {p_09:.4f}  ({'显著' if p_09 < 0.05 else '不显著，样本量不足'})")

    print(f"\n[2024-09 收益拆解]")
    print(f"  基准(沪深300) = {bench_sept_total:+.2%}")
    print(f"  策略总收益    = {strat_sept_total:+.2%}")
    print(f"  Beta贡献      = β({beta_09:.2f}) × 基准({bench_sept_total:.2%}) = {beta_contrib_actual:+.2%}")
    print(f"  Alpha贡献     = 总收益 - Beta贡献 = {alpha_contrib:+.2%}")

    # 结论
    alpha_pct = abs(alpha_contrib) / abs(strat_sept_total) if strat_sept_total != 0 else 0
    beta_pct = abs(beta_contrib_actual) / abs(strat_sept_total) if strat_sept_total != 0 else 0
    print(f"\n  → Alpha占比={alpha_pct:.0%}，Beta占比={beta_pct:.0%}")
    if beta_pct > 0.7:
        print(f"  ✗ 判定：2024-09 主要由 Beta 爆发驱动，非模型选股能力")
        print(f"    建议：剔除该月后重算年化超额，才是策略真实 alpha 水平")
    elif alpha_pct > 0.5:
        print(f"  ✓ 判定：2024-09 仍有显著 alpha 贡献，模型选股有效")
    else:
        print(f"  ~ 判定：2024-09 alpha/beta 混合，需结合持仓行业分布进一步分析")

# 剔除 2024-09 后重算全程指标（净化后的真实 alpha 水平）
non_sept = ~(report_df["month"].isin(["2024-09"]))
r_s_excl = report_df.loc[non_sept, "return"].values
r_b_excl = report_df.loc[non_sept, "bench"].values
valid_excl = np.isfinite(r_s_excl) & np.isfinite(r_b_excl)
beta_excl, alpha_daily_excl, _, _, _ = stats.linregress(
    r_b_excl[valid_excl], r_s_excl[valid_excl]
)

# 年化超额（剔除2024-09）
from qlib.contrib.evaluate import risk_analysis
excess_excl = (
    report_df.loc[non_sept, "return"]
    - report_df.loc[non_sept, "bench"]
    - report_df.loc[non_sept, "cost"]
)
try:
    ra_excl = risk_analysis(excess_excl)["risk"]
    ann_excess_excl = ra_excl.get("annualized_return", float("nan"))
    ir_excl = ra_excl.get("information_ratio", float("nan"))
    print(f"\n[剔除2024-09后 净化指标]")
    print(f"  年化超额收益  = {ann_excess_excl:.2%}  (含成本，原始={sept['excess_with_cost']:.2%}/月无法直接比)")
    print(f"  IR            = {ir_excl:.4f}")
    print(f"  年化Alpha(OLS)= {alpha_daily_excl*252:.2%}")
except Exception:
    print(f"\n[剔除2024-09后] 年化Alpha(OLS) = {alpha_daily_excl*252:.2%}")

# 保存回归结果
reg_out = Path(__file__).resolve().parents[1] / "reports" / "beta_decomp_2024_09.csv"
reg_rows = [
    {"区间": "全程基线", "Beta": beta_all, "年化Alpha": alpha_ann_all, "R2": r_val_all**2},
]
if sept_mask.sum() >= 5:
    reg_rows.append(
        {"区间": "2024-09", "Beta": beta_09, "年化Alpha": alpha_ann_09, "R2": r_val_09**2}
    )
reg_rows.append(
    {"区间": "剔除2024-09", "Beta": beta_excl, "年化Alpha": alpha_daily_excl * 252, "R2": float("nan")}
)
pd.DataFrame(reg_rows).to_csv(reg_out, index=False, encoding="utf-8-sig")
print(f"\n回归结果已保存: {reg_out}")
