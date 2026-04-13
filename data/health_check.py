"""
数据健康检查
检测缺失交易日、异常价格跳变、停牌标记等
"""

from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from loguru import logger

from utils.helpers import get_data_config, expand_path


class DataHealthChecker:
    """数据健康检查器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_data_config()
        self.raw_dir = expand_path(self.config["raw_csv_dir"])
        self.stock_dir = self.raw_dir / "stocks"
        self.meta_dir = self.raw_dir / "meta"
        self.max_price_change = self.config["health_check"]["max_price_change_pct"]
        self.min_trading_pct = self.config["health_check"]["min_trading_days_pct"]

    def load_trade_dates(self) -> pd.DatetimeIndex:
        """加载交易日历"""
        path = self.meta_dir / "trade_dates.csv"
        if not path.exists():
            logger.warning("交易日历文件不存在")
            return pd.DatetimeIndex([])
        df = pd.read_csv(path, parse_dates=["trade_date"])
        return pd.DatetimeIndex(df["trade_date"])

    def check_missing_days(self, df: pd.DataFrame, symbol: str, trade_dates: pd.DatetimeIndex) -> List[str]:
        """检测缺失交易日

        Args:
            df: 股票行情 DataFrame
            symbol: 股票代码
            trade_dates: 完整交易日历

        Returns:
            缺失日期列表
        """
        if df.empty or trade_dates.empty:
            return []

        stock_dates = pd.DatetimeIndex(df["date"])
        # 取交叉范围
        start = max(stock_dates.min(), trade_dates.min())
        end = min(stock_dates.max(), trade_dates.max())
        expected = trade_dates[(trade_dates >= start) & (trade_dates <= end)]
        missing = expected.difference(stock_dates)

        if len(missing) > 0:
            missing_pct = len(missing) / len(expected) * 100
            if missing_pct > (100 - self.min_trading_pct):
                logger.warning(
                    f"[{symbol}] 缺失 {len(missing)} 个交易日 "
                    f"({missing_pct:.1f}%)，部分日期: {missing[:5].strftime('%Y-%m-%d').tolist()}"
                )
        return missing.strftime("%Y-%m-%d").tolist()

    def check_price_anomaly(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """检测异常价格跳变（超过阈值）

        Returns:
            异常记录列表 [{date, pct_change}, ...]
        """
        if df.empty or len(df) < 2:
            return []

        df = df.sort_values("date").reset_index(drop=True)
        pct = df["close"].pct_change() * 100
        anomalies = []

        for idx in pct.index:
            if pd.notna(pct[idx]) and abs(pct[idx]) > self.max_price_change:
                date_str = df.loc[idx, "date"]
                if hasattr(date_str, "strftime"):
                    date_str = date_str.strftime("%Y-%m-%d")
                anomalies.append({
                    "date": date_str,
                    "pct_change": round(pct[idx], 2),
                })

        if anomalies:
            logger.warning(f"[{symbol}] 发现 {len(anomalies)} 个异常价格跳变")
        return anomalies

    def check_suspend_days(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """检测停牌日（volume==0 或 amount==0 或对应值缺失）

        A 股停牌判定：成交量为 0 或成交额为 0 均视为停牌。
        部分数据源停牌日 volume/amount 可能为 NaN 而非 0。

        Returns:
            停牌日期列表
        """
        if df.empty:
            return []
        vol_suspended = (df["volume"] == 0) | df["volume"].isna()
        if "amount" in df.columns:
            amt_suspended = (df["amount"] == 0) | df["amount"].isna()
            is_suspended = vol_suspended | amt_suspended
        else:
            is_suspended = vol_suspended
        suspend = df[is_suspended]
        dates = []
        for _, row in suspend.iterrows():
            d = row["date"]
            if hasattr(d, "strftime"):
                d = d.strftime("%Y-%m-%d")
            dates.append(d)

        if dates:
            logger.info(f"[{symbol}] 停牌 {len(dates)} 天")
        return dates

    def run_full_check(self) -> Dict:
        """对所有股票执行完整健康检查

        Returns:
            检查结果汇总字典
        """
        logger.info("=" * 60)
        logger.info("开始数据健康检查")
        logger.info("=" * 60)

        trade_dates = self.load_trade_dates()
        csv_files = list(self.stock_dir.glob("*.csv"))

        if not csv_files:
            logger.error(f"未找到数据文件: {self.stock_dir}")
            return {}

        results = {
            "total_stocks": len(csv_files),
            "stocks_with_missing_days": 0,
            "stocks_with_anomalies": 0,
            "stocks_with_suspend": 0,
            "details": {},
        }

        for csv_file in csv_files:
            symbol = csv_file.stem
            try:
                df = pd.read_csv(csv_file, parse_dates=["date"])
                detail = {}

                # 1. 缺失交易日
                missing = self.check_missing_days(df, symbol, trade_dates)
                if missing:
                    detail["missing_days"] = len(missing)
                    results["stocks_with_missing_days"] += 1

                # 2. 异常价格
                anomalies = self.check_price_anomaly(df, symbol)
                if anomalies:
                    detail["price_anomalies"] = anomalies
                    results["stocks_with_anomalies"] += 1

                # 3. 停牌日
                suspend = self.check_suspend_days(df, symbol)
                if suspend:
                    detail["suspend_days"] = len(suspend)
                    results["stocks_with_suspend"] += 1

                if detail:
                    results["details"][symbol] = detail

            except Exception as e:
                logger.error(f"[{symbol}] 健康检查失败: {e}")

        # 汇总
        logger.info("=" * 60)
        logger.info("健康检查完成:")
        logger.info(f"  总股票数: {results['total_stocks']}")
        logger.info(f"  有缺失交易日: {results['stocks_with_missing_days']}")
        logger.info(f"  有异常价格跳变: {results['stocks_with_anomalies']}")
        logger.info(f"  有停牌记录: {results['stocks_with_suspend']}")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":
    from utils.helpers import setup_logger
    setup_logger("health_check")
    checker = DataHealthChecker()
    checker.run_full_check()
