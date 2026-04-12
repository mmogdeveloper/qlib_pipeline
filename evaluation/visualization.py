"""
可视化模块

Qlib 内置图表（自动调用）：
- analysis_model.model_performance_graph: IC/累计IC/分组收益
- analysis_position.report_graph: 净值/回撤/换手率

自定义补充图表（Qlib 不提供）：
- 月度收益热力图
- 滚动 Sharpe
- 超额收益曲线
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from loguru import logger

from utils.helpers import ensure_dir, PROJECT_ROOT

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


class Visualizer:
    """可视化绘图器"""

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = ensure_dir(Path(output_dir))
        else:
            self.output_dir = ensure_dir(PROJECT_ROOT / "reports" / "figures")

    def _save_fig(self, fig, name: str) -> str:
        path = self.output_dir / f"{name}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"图表已保存: {path}")
        finally:
            plt.close(fig)
        return str(path)

    # ── Qlib 内置图表 ────────────────────────────────────────

    def generate_qlib_model_plots(self, recorder) -> dict:
        """调用 Qlib analysis_model 生成模型分析图

        qlib.contrib.report.analysis_model.model_performance_graph()
        生成: IC 时序、累计 IC、分组收益等图表

        Args:
            recorder: Qlib Recorder 对象
        """
        plots = {}
        try:
            from qlib.contrib.report import analysis_model

            pred = recorder.load_object("pred.pkl")
            label = recorder.load_object("label.pkl")
            if pred is None or label is None:
                logger.warning("pred.pkl 或 label.pkl 不存在，跳过模型分析图")
                return plots

            # 拼接为 Qlib 期望的 (score, label) DataFrame
            pred_label = pd.concat(
                [pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred,
                 label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label],
                axis=1,
            )
            pred_label.columns = ["score", "label"]
            pred_label = pred_label.dropna()

            fig_list = analysis_model.model_performance_graph(pred_label)
            for i, fig in enumerate(fig_list):
                plots[f"qlib_model_{i}"] = self._save_fig(fig, f"qlib_model_analysis_{i}")
            logger.info(f"Qlib 模型分析图: {len(fig_list)} 张")

        except ImportError:
            logger.debug("qlib.contrib.report 不可用，跳过")
        except Exception as e:
            logger.warning(f"Qlib 模型分析图生成失败: {e}")
        return plots

    def generate_qlib_position_plots(self, report_df: pd.DataFrame) -> dict:
        """调用 Qlib analysis_position 生成组合分析图

        qlib.contrib.report.analysis_position.report_graph()
        生成: 累计收益 vs 基准、回撤、换手率等图表

        Args:
            report_df: backtest_daily 返回的 report DataFrame
        """
        plots = {}
        try:
            from qlib.contrib.report import analysis_position

            fig_list = analysis_position.report_graph(report_df, show_notebook=False)
            for i, fig in enumerate(fig_list):
                plots[f"qlib_position_{i}"] = self._save_fig(fig, f"qlib_position_analysis_{i}")
            logger.info(f"Qlib 组合分析图: {len(fig_list)} 张")

        except ImportError:
            logger.debug("qlib.contrib.report 不可用，跳过")
        except Exception as e:
            logger.warning(f"Qlib 组合分析图生成失败: {e}")
        return plots

    # ── 自定义补充图表（Qlib 不提供的）──────────────────────────

    def plot_monthly_heatmap(self, returns: pd.Series) -> str:
        """月度收益热力图（Qlib 不提供此图表）"""
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot_table(
            values="return", index="year", columns="month", aggfunc="first"
        )
        pivot.columns = [f"{m}月" for m in pivot.columns]

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title("月度收益热力图", fontsize=14)
        ax.set_ylabel("年份")
        return self._save_fig(fig, "monthly_heatmap")

    def plot_rolling_sharpe(self, returns: pd.Series, window: int = 60) -> str:
        """滚动 Sharpe 比率（Qlib 不提供此图表）"""
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1, color="orange")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"滚动{window}日 Sharpe 比率", fontsize=12)
        ax.set_ylabel("Sharpe")
        fig.autofmt_xdate()
        return self._save_fig(fig, "rolling_sharpe")

    def plot_excess_return(self, returns: pd.Series, benchmark_returns: pd.Series) -> str:
        """累计超额收益曲线（Qlib 不提供独立的超额收益图）"""
        excess = returns - benchmark_returns
        cum_excess = (1 + excess).cumprod() - 1

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(cum_excess.index, cum_excess.values, color="green", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(cum_excess.index, 0, cum_excess.values,
                        where=cum_excess.values >= 0, alpha=0.3, color="green")
        ax.fill_between(cum_excess.index, 0, cum_excess.values,
                        where=cum_excess.values < 0, alpha=0.3, color="red")
        ax.set_title("累计超额收益", fontsize=14)
        ax.set_ylabel("超额收益")
        fig.autofmt_xdate()
        return self._save_fig(fig, "excess_return")

    # ── 汇总入口 ────────────────────────────────────────────

    def generate_all_plots(
        self,
        report_df: Optional[pd.DataFrame] = None,
        recorder=None,
        returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        ic_series: Optional[pd.Series] = None,
    ) -> dict:
        """生成全部图表

        优先使用 Qlib 内置图表，仅补充 Qlib 不提供的自定义图表。

        Args:
            report_df: backtest_daily 返回的 report DataFrame
            recorder: Qlib Recorder 对象
            returns: 策略收益率（report_df 不可用时的回退）
            benchmark_returns: 基准收益率（回退用）
            ic_series: IC 时间序列（回退用）
        """
        logger.info("生成可视化图表...")
        plots = {}

        # 1. Qlib 组合分析图（净值、回撤、换手率）
        if report_df is not None:
            plots.update(self.generate_qlib_position_plots(report_df))

        # 2. Qlib 模型分析图（IC、分组收益）
        if recorder is not None:
            plots.update(self.generate_qlib_model_plots(recorder))

        # 3. 自定义补充图表（Qlib 不提供的）
        ret = returns
        bench = benchmark_returns
        if ret is None and report_df is not None and "return" in report_df.columns:
            ret = report_df["return"]
            bench = report_df.get("bench")

        if ret is not None:
            plots["monthly_heatmap"] = self.plot_monthly_heatmap(ret)
            plots["rolling_sharpe"] = self.plot_rolling_sharpe(ret)
            if bench is not None:
                plots["excess_return"] = self.plot_excess_return(ret, bench)

        logger.info(f"共生成 {len(plots)} 张图表")
        return plots
