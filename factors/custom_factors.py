"""
自定义因子
使用 Qlib Expression 引擎定义动量、波动率、换手率、量价背离、流动性因子

get_custom_factor_expressions() 是唯一对外接口：
  - auto_filter.enabled=true 时，自动读取 IC 分析结果，过滤弱因子和冗余因子
  - auto_filter.enabled=false 或 CSV 不存在时，返回全量候选因子
"""

from typing import List, Tuple, Optional

from loguru import logger

from utils.helpers import get_factor_config, PROJECT_ROOT


def get_custom_factor_expressions(config: Optional[dict] = None) -> List[Tuple[str, str]]:
    """从配置文件加载自定义因子表达式，并按 IC 分析结果动态过滤

    过滤流程（仅 auto_filter.enabled=true 且 CSV 存在时生效）：
      1. ICIR 阈值过滤：|ICIR| < icir_min_abs 的因子丢弃
      2. 去重过滤：IC 时序相关系数 > corr_max 的因子对，移除 ICIR 绝对值较小的

    Returns:
        [(expression, name), ...] 格式的因子列表（仅剩通过过滤的因子）
    """
    config = config or get_factor_config()
    candidates = _load_candidates(config)

    af = config.get("auto_filter", {})
    if not af.get("enabled", False):
        logger.info(f"auto_filter 未启用，返回全量 {len(candidates)} 个候选因子")
        return candidates

    summary, ic_series = _load_filter_data(af)

    if summary is None:
        if af.get("fallback_on_missing", True):
            logger.warning("auto_filter: IC 数据文件不存在，回退到全量因子")
            return candidates
        raise FileNotFoundError(
            f"auto_filter 需要先运行: python analysis/factor_ic_analysis.py --include-alpha158-sample"
        )

    filtered = _apply_icir_filter(candidates, summary, af.get("icir_min_abs", 0.1))
    filtered = _apply_dedup_filter(filtered, ic_series, summary, af.get("corr_max", 0.90))

    # 第三道过滤：LightGBM feature importance（gain 口径）
    fi_cfg = af.get("feature_importance_filter", {})
    if fi_cfg.get("enabled", False):
        filtered = _apply_feature_importance_filter(
            filtered,
            fi_cfg.get("importance_path", "reports/feature_importance_lgbm.csv"),
            fi_cfg.get("min_gain_pct", 0.5),
            fi_cfg.get("fallback_on_missing", True),
        )

    removed = [n for _, n in candidates if n not in {n2 for _, n2 in filtered}]
    if removed:
        logger.info(f"auto_filter 移除 {len(removed)} 个因子: {removed}")
    logger.info(f"auto_filter 后保留 {len(filtered)} 个自定义因子: {[n for _, n in filtered]}")
    return filtered


# ── 内部函数 ────────────────────────────────────────────────

def _load_candidates(config: dict) -> List[Tuple[str, str]]:
    """从 factor_config.yaml 读取全量候选因子"""
    factors = []
    for category, items in config.get("custom_factors", {}).items():
        for item in items:
            factors.append((item["expression"], item["name"]))
            logger.debug(f"候选因子: {item['name']} = {item['expression']}")
    logger.info(f"候选自定义因子共 {len(factors)} 个")
    return factors


def _load_filter_data(af: dict):
    """读取 IC 汇总 CSV 和 IC 时序 CSV，返回 (summary_df, ic_series_df)"""
    import pandas as pd

    summary_path = PROJECT_ROOT / af.get("ic_summary_path", "reports/factor_ic_summary.csv")
    series_path  = PROJECT_ROOT / af.get("ic_series_path",  "reports/factor_ic_series.csv")

    if not summary_path.exists():
        logger.warning(f"IC 汇总文件不存在: {summary_path}")
        return None, None

    summary = pd.read_csv(summary_path, index_col=0, encoding="utf-8-sig")

    ic_series = None
    if series_path.exists():
        ic_series = pd.read_csv(series_path, index_col=0, encoding="utf-8-sig")
    else:
        logger.warning(f"IC 时序文件不存在: {series_path}，跳过去重过滤")

    return summary, ic_series


def _apply_icir_filter(
    candidates: List[Tuple[str, str]],
    summary,
    icir_min_abs: float,
) -> List[Tuple[str, str]]:
    """第一道过滤：|ICIR| < icir_min_abs 的因子丢弃"""
    if "ICIR" not in summary.columns:
        logger.warning("IC 汇总表缺少 ICIR 列，跳过 ICIR 过滤")
        return candidates

    kept = []
    for expr, name in candidates:
        if name not in summary.index:
            # 不在 IC 分析结果中（可能是新增因子，未分析），保留
            logger.debug(f"  {name}: 不在 IC 分析结果中，保留")
            kept.append((expr, name))
            continue

        icir = summary.loc[name, "ICIR"]
        if abs(icir) >= icir_min_abs:
            kept.append((expr, name))
            logger.debug(f"  {name}: ICIR={icir:.4f} ≥ {icir_min_abs}，保留")
        else:
            logger.info(f"  {name}: ICIR={icir:.4f} < {icir_min_abs}，移除")

    return kept


def _apply_dedup_filter(
    candidates: List[Tuple[str, str]],
    ic_series,
    summary,
    corr_max: float,
) -> List[Tuple[str, str]]:
    """第二道过滤：IC 时序相关系数 > corr_max 的因子对，保留 |ICIR| 更高的

    同时对比 ic_series 中的 Alpha158 参考因子（列名以 A158_ 开头），
    若自定义因子与任意 Alpha158 因子相关系数 > corr_max，说明该因子在
    Alpha158 中已有等价表示，直接移除（Alpha158 本身始终保留）。
    """
    if ic_series is None or len(candidates) <= 1:
        return candidates

    names = [n for _, n in candidates]
    available = [n for n in names if n in ic_series.columns]
    # Alpha158 参考列（由 --include-alpha158-sample 生成）
    alpha158_cols = [c for c in ic_series.columns if c.startswith("A158_")]

    if len(available) == 0:
        return candidates

    all_cols = list(set(available + alpha158_cols))
    series_sub = ic_series[[c for c in all_cols if c in ic_series.columns]].dropna(how="all")
    corr_mat = series_sub.corr(method="pearson")

    def _icir_abs(name):
        if "ICIR" in summary.columns and name in summary.index:
            return abs(summary.loc[name, "ICIR"])
        return 0.0

    keep_set = set(available)

    # 第一步：对比 Alpha158 参考因子，相关性过高则直接移除自定义因子
    for name in list(available):
        if name not in keep_set:
            continue
        for a158 in alpha158_cols:
            if a158 not in corr_mat.columns or name not in corr_mat.index:
                continue
            c = corr_mat.loc[name, a158]
            if abs(c) > corr_max:
                keep_set.discard(name)
                logger.info(
                    f"  去重(vs Alpha158): {name} 与 {a158} 的 IC 相关={c:.3f} > {corr_max}，"
                    f"Alpha158 已有等价因子，移除 {name}"
                )
                break

    # 第二步：剩余自定义因子之间互相去重，保留 |ICIR| 更高的
    remaining = [n for n in available if n in keep_set]
    sorted_by_icir = sorted(remaining, key=_icir_abs, reverse=True)

    for i, name_a in enumerate(sorted_by_icir):
        if name_a not in keep_set:
            continue
        for name_b in sorted_by_icir[i + 1:]:
            if name_b not in keep_set:
                continue
            if name_a not in corr_mat.index or name_b not in corr_mat.columns:
                continue
            c = corr_mat.loc[name_a, name_b]
            if abs(c) > corr_max:
                keep_set.discard(name_b)
                logger.info(
                    f"  去重(自定义): {name_b} 与 {name_a} 的 IC 相关={c:.3f} > {corr_max}，"
                    f"移除 {name_b}（ICIR={_icir_abs(name_b):.4f}）"
                )

    result = [(expr, name) for expr, name in candidates
              if name not in available or name in keep_set]
    return result


def _apply_feature_importance_filter(
    candidates: List[Tuple[str, str]],
    importance_path: str,
    min_gain_pct: float,
    fallback_on_missing: bool,
) -> List[Tuple[str, str]]:
    """第三道过滤：读取上一次训练输出的 feature importance CSV

    仅过滤自定义因子（Alpha158 的因子不在 candidates 里，永远不受影响）。
    gain_pct < min_gain_pct 的自定义因子被视为"模型实际不使用"，直接移除。

    注意：该过滤依赖上一次训练的结果（迭代精简），
    首次训练前文件不存在时由 fallback_on_missing 控制行为。
    """
    import pandas as pd

    imp_path = PROJECT_ROOT / importance_path
    if not imp_path.exists():
        if fallback_on_missing:
            logger.warning(
                f"feature_importance_filter: {imp_path} 不存在，跳过此道过滤"
                "（首次训练后自动生成，下次运行时生效）"
            )
            return candidates
        raise FileNotFoundError(
            f"feature_importance_filter 需要先完成一次训练以生成: {imp_path}"
        )

    try:
        imp = pd.read_csv(imp_path, encoding="utf-8-sig")
        if "feature" not in imp.columns or "gain_pct" not in imp.columns:
            logger.warning("feature_importance_filter: CSV 缺少 feature/gain_pct 列，跳过")
            return candidates

        gain_map = dict(zip(imp["feature"], imp["gain_pct"]))
    except Exception as e:
        logger.warning(f"feature_importance_filter: 读取 CSV 失败，跳过: {e}")
        return candidates

    kept = []
    for expr, name in candidates:
        pct = gain_map.get(name)
        if pct is None:
            # 该因子不在上次训练的特征集中（新增因子），保留以便本次训练评估
            logger.debug(f"  {name}: 不在 importance CSV 中（新增），保留")
            kept.append((expr, name))
        elif pct >= min_gain_pct:
            kept.append((expr, name))
            logger.debug(f"  {name}: gain_pct={pct:.2f}% ≥ {min_gain_pct}%，保留")
        else:
            logger.info(f"  {name}: gain_pct={pct:.2f}% < {min_gain_pct}%，移除（模型实际不使用）")

    return kept


def get_all_factor_fields(config: Optional[dict] = None) -> List[Tuple[str, str]]:
    return get_custom_factor_expressions(config)


# ── 预定义因子类别（仅供参考，实际因子从 factor_config.yaml 读取）──

MOMENTUM_FACTORS = [
    ("$close/Ref($close,5)-1", "mom_5d"),
    ("$close/Ref($close,10)-1", "mom_10d"),
    ("$close/Ref($close,20)-1", "mom_20d"),
    ("$close/Ref($close,60)-1", "mom_60d"),
]

VOLATILITY_FACTORS = [
    ("Std($close/Ref($close,1)-1, 20)", "vol_20d"),
    ("Std($close/Ref($close,1)-1, 5)", "vol_5d"),
]

TURNOVER_FACTORS = [
    ("Mean($turnover, 5)", "turnover_5d_mean"),
    ("$turnover/Mean($turnover,20)", "turnover_ratio"),
]

PRICE_VOLUME_FACTORS = [
    ("Corr($close, $volume, 20)", "corr_close_vol_20d"),
    ("Corr($close, $volume, 10)", "corr_close_vol_10d"),
]

LIQUIDITY_FACTORS = [
    ("Mean($amount/$volume, 10)", "vwap_ratio"),
]

ALL_CUSTOM_FACTORS = (
    MOMENTUM_FACTORS
    + VOLATILITY_FACTORS
    + TURNOVER_FACTORS
    + PRICE_VOLUME_FACTORS
    + LIQUIDITY_FACTORS
)
