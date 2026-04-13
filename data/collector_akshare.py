"""
AKShare 数据下载器
从 AKShare 下载沪深300成分股日频行情、基准指数、交易日历等数据
支持全量下载和增量更新，带多线程并发、限速、重试机制
"""

import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import akshare as ak
import pandas as pd
from loguru import logger
from tqdm import tqdm

from utils.helpers import (
    get_data_config,
    expand_path,
    ensure_dir,
    stock_code_to_qlib,
    today_str,
)


class AKShareCollector:
    """AKShare 数据采集器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_data_config()
        self.raw_dir = ensure_dir(expand_path(self.config["raw_csv_dir"]))
        self.start_date = self.config["start_date"].replace("-", "")
        self.end_date = self.config["end_date"].replace("-", "")
        self.max_workers = self.config["download"]["max_workers"]
        self.rate_limit = self.config["download"]["rate_limit_sleep"]
        self.max_retries = self.config["download"]["max_retries"]
        self.retry_base_delay = self.config["download"]["retry_base_delay"]

        # 子目录
        self.stock_dir = ensure_dir(self.raw_dir / "stocks")
        self.index_dir = ensure_dir(self.raw_dir / "index")
        self.meta_dir = ensure_dir(self.raw_dir / "meta")

        # 文件写入锁（增量更新 CSV 合并时使用）
        self._file_lock = threading.Lock()
        # 全局 API 限速锁，确保线程间请求间隔
        self._api_lock = threading.Lock()
        self._last_api_call = 0.0

    # ── 全局限速 ────────────────────────────────────────────
    def _throttle(self):
        """确保所有线程的 API 调用间隔不低于 rate_limit 秒"""
        with self._api_lock:
            now = time.time()
            elapsed = now - self._last_api_call
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
            self._last_api_call = time.time()

    # ── 重试装饰器 ──────────────────────────────────────────
    def _retry(self, func, *args, **kwargs):
        """带指数退避的重试机制"""
        for attempt in range(self.max_retries):
            try:
                self._throttle()  # 每次尝试前都全局限速
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"重试{self.max_retries}次后仍失败: {e}")
                    raise
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(f"第{attempt+1}次失败，{delay:.1f}秒后重试: {e}")
                time.sleep(delay)

    # ── 获取沪深300成分股列表 ──────────────────────────────────
    def get_csi300_stocks(self) -> pd.DataFrame:
        """获取当前沪深300成分股列表

        Returns:
            DataFrame: 包含 code, name 列
        """
        logger.info("获取沪深300成分股列表...")
        df = self._retry(ak.index_stock_cons_csindex, symbol=self.config["benchmark_index"])
        # 标准化列名：兼容 AKShare 不同版本的返回格式
        col_rename = {
            "成分券代码": "code", "成分券名称": "name",
            "constituent_code": "code", "constituent_name": "name",
        }
        df = df.rename(columns=col_rename)
        if "code" not in df.columns or "name" not in df.columns:
            # 最后兜底：假定前两列为 code, name
            cols = df.columns.tolist()
            logger.warning(f"AKShare 返回列名未识别: {cols}，按位置取前两列")
            df = df.rename(columns={cols[0]: "code", cols[1]: "name"})
        df = df[["code", "name"]]

        df["code"] = df["code"].astype(str).str.zfill(6)
        logger.info(f"获取到 {len(df)} 只成分股")

        # 保存成分股列表
        df.to_csv(self.meta_dir / "csi300_stocks.csv", index=False)
        return df

    # ── 获取交易日历 ────────────────────────────────────────
    def get_trade_dates(self) -> pd.DataFrame:
        """获取交易日历"""
        logger.info("获取交易日历...")
        df = self._retry(ak.tool_trade_date_hist_sina)
        df.columns = ["trade_date"]
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 筛选日期范围
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 保存
        df.to_csv(self.meta_dir / "trade_dates.csv", index=False)
        logger.info(f"交易日历: {len(df)} 个交易日 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
        return df

    # ── 6位代码 → Sina 格式 ────────────────────────────────────
    @staticmethod
    def _to_sina_symbol(code: str) -> str:
        """6位纯数字代码 → Sina 格式 (sz000001 / sh600000)"""
        code = str(code).zfill(6)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        return f"{prefix}{code}"

    # ── 下载单只股票行情 ─────────────────────────────────────
    def _download_single_stock(
        self, code: str, start_date: str, end_date: str, save: bool = True
    ) -> Optional[pd.DataFrame]:
        """下载单只股票的日频后复权行情（使用 Sina 数据源）

        Args:
            code: 6位股票代码
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            save: 是否直接保存到 CSV，增量更新时传 False 由调用方合并后保存

        Returns:
            DataFrame 或 None（下载失败时）
        """
        try:
            sina_symbol = self._to_sina_symbol(code)

            # 分别请求后复权和不复权数据，用于计算真实复权因子
            df = self._retry(
                ak.stock_zh_a_daily,
                symbol=sina_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq",
            )
            if df is None or df.empty:
                logger.warning(f"[{code}] 无数据")
                return None

            df_raw = self._retry(
                ak.stock_zh_a_daily,
                symbol=sina_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )

            # stock_zh_a_daily 返回英文列名: date, open, high, low, close, volume, amount, turnover, outstanding_share
            # 确保必要字段存在
            required = ["date", "open", "high", "low", "close", "volume", "amount"]
            missing_fields = [col for col in required if col not in df.columns]
            if missing_fields:
                logger.warning(f"[{code}] 缺少关键字段: {missing_fields}，跳过")
                return None

            # 处理 turnover (换手率) 字段：缺失填 NaN
            if "turnover" not in df.columns:
                df["turnover"] = float("nan")

            # 计算复权因子: factor = 后复权收盘价 / 不复权收盘价
            df["date"] = pd.to_datetime(df["date"])
            df_raw["date"] = pd.to_datetime(df_raw["date"])
            raw_close = df_raw.set_index("date")["close"]
            df = df.set_index("date")
            df["factor"] = df["close"] / df.index.map(raw_close)
            df = df.reset_index()

            # 向量化计算 VWAP
            valid = (df["volume"] > 0) & (df["amount"] > 0)
            df["vwap"] = df["close"].copy()
            df.loc[valid, "vwap"] = df.loc[valid, "amount"] / df.loc[valid, "volume"]

            # 只保留需要的列
            df = df[["date", "open", "high", "low", "close", "volume", "amount",
                      "turnover", "factor", "vwap"]]
            df = df.sort_values("date").reset_index(drop=True)

            if save:
                qlib_code = stock_code_to_qlib(code)
                save_path = self.stock_dir / f"{qlib_code}.csv"
                df.to_csv(save_path, index=False)
            return df

        except Exception as e:
            logger.error(f"[{code}] 下载失败: {e}")
            return None

    # ── 下载基准指数 ─────────────────────────────────────────
    def download_benchmark(self) -> pd.DataFrame:
        """下载沪深300指数日线数据"""
        logger.info("下载沪深300基准指数...")
        symbol = self.config["benchmark_symbol"]
        df = self._retry(ak.stock_zh_index_daily, symbol=symbol)

        df["date"] = pd.to_datetime(df["date"])

        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df = df.sort_values("date").reset_index(drop=True)

        save_path = self.index_dir / f"{self.config['benchmark_symbol'].upper()}.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"基准指数已保存: {save_path} ({len(df)} 条)")
        return df

    # ── 全量下载 ─────────────────────────────────────────────
    def download_all(self) -> None:
        """全量下载所有沪深300成分股的历史数据"""
        logger.info("=" * 60)
        logger.info("开始全量下载...")
        logger.info(f"数据区间: {self.start_date} ~ {self.end_date}")
        logger.info(f"并发线程: {self.max_workers}")
        logger.info("=" * 60)

        # 1. 交易日历
        self.get_trade_dates()

        # 2. 成分股列表
        stocks = self.get_csi300_stocks()
        codes = stocks["code"].tolist()

        # 3. 基准指数
        self.download_benchmark()

        # 4. 并发下载个股数据
        logger.info(f"开始下载 {len(codes)} 只股票数据...")
        success, failed = 0, 0
        failed_codes = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._download_single_stock, code, self.start_date, self.end_date
                ): code
                for code in codes
            }
            with tqdm(total=len(codes), desc="下载股票数据") as pbar:
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            success += 1
                        else:
                            failed += 1
                            failed_codes.append(code)
                    except Exception as e:
                        failed += 1
                        failed_codes.append(code)
                        logger.error(f"[{code}] 异常: {e}")
                    pbar.update(1)

        # 5. 保存下载状态
        status = {
            "last_update": today_str(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total": len(codes),
            "success": success,
            "failed": failed,
            "failed_codes": failed_codes,
        }
        status_path = self.meta_dir / "download_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        logger.info(f"全量下载完成: 成功 {success}, 失败 {failed}")
        if failed_codes:
            logger.warning(f"失败股票: {failed_codes}")

    # ── 增量更新 ─────────────────────────────────────────────
    def update_incremental(self) -> None:
        """基于上次更新日期进行增量下载"""
        status_path = self.meta_dir / "download_status.json"
        if not status_path.exists():
            logger.warning("未找到下载状态文件，执行全量下载")
            self.download_all()
            return

        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)

        last_date = status.get("last_update", self.start_date)
        # 从上次更新日期的下一天开始
        start = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        if start > end:
            logger.info("数据已是最新，无需更新")
            return

        logger.info(f"增量更新: {start} ~ {end}")

        # 更新交易日历
        self.get_trade_dates()

        # 获取最新成分股列表
        stocks = self.get_csi300_stocks()
        codes = stocks["code"].tolist()

        # 更新基准指数
        self.download_benchmark()

        # 并发增量下载
        success, failed = 0, 0
        failed_codes = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_single_stock, code, start, end, save=False): code
                for code in codes
            }
            with tqdm(total=len(codes), desc="增量更新") as pbar:
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            # 合并新旧数据后保存（线程安全）
                            qlib_code = stock_code_to_qlib(code)
                            existing_path = self.stock_dir / f"{qlib_code}.csv"
                            with self._file_lock:
                                if existing_path.exists():
                                    old_df = pd.read_csv(existing_path, parse_dates=["date"])
                                    merged = pd.concat([old_df, result]).drop_duplicates(
                                        subset=["date"], keep="last"
                                    )
                                    merged = merged.sort_values("date").reset_index(drop=True)
                                    merged.to_csv(existing_path, index=False)
                                else:
                                    result.to_csv(existing_path, index=False)
                            success += 1
                        else:
                            failed += 1
                            failed_codes.append(code)
                    except Exception as e:
                        failed += 1
                        failed_codes.append(code)
                    pbar.update(1)

        # 更新状态
        status.update({
            "last_update": end,
            "end_date": end,
        })
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        logger.info(f"增量更新完成: 成功 {success}, 失败 {failed}")

    # ── 下载历史成分股快照 ─────────────────────────────────────
    def download_constituent_history(self) -> dict:
        """下载沪深300历史成分股快照（半年度调整节点）

        CSI300 每年6月和12月第二个周五进行成分股调整。
        通过 index_stock_cons_weight_csindex 查询各调整日的成分股列表，
        生成时间维度的成分股变动记录，用于消除幸存者偏差。

        Returns:
            {date_str: [code1, code2, ...]} 历史成分股快照字典
        """
        from datetime import datetime as dt

        logger.info("下载沪深300历史成分股快照...")

        start = dt.strptime(self.start_date, "%Y%m%d")
        end = dt.strptime(self.end_date, "%Y%m%d")

        # 生成半年度调整查询日期（6月、12月各月中）
        rebalance_dates = []
        year = start.year
        while year <= end.year:
            for month in [6, 12]:
                date_str = f"{year}{month:02d}15"
                if self.start_date <= date_str <= self.end_date:
                    rebalance_dates.append(date_str)
            year += 1
        # 补充首尾日期，确保覆盖整个回测区间
        if rebalance_dates and rebalance_dates[0] > self.start_date:
            rebalance_dates.insert(0, self.start_date)
        if not rebalance_dates:
            rebalance_dates = [self.start_date]

        snapshots = {}
        for date in rebalance_dates:
            try:
                df = self._retry(
                    ak.index_stock_cons_weight_csindex,
                    symbol=self.config["benchmark_index"],
                    date=date,
                )
                if df is None or df.empty:
                    continue
                col_rename = {
                    "成分券代码": "code", "constituent_code": "code",
                    "成份券代码": "code",
                }
                df = df.rename(columns=col_rename)
                if "code" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "code"})
                codes = df["code"].astype(str).str.zfill(6).tolist()
                snapshots[date] = codes
                logger.info(f"  {date}: {len(codes)} 只成分股")
            except Exception as e:
                logger.warning(f"  {date}: 获取失败 ({e})")

        save_path = self.meta_dir / "csi300_history.json"
        if snapshots:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(snapshots, f, ensure_ascii=False, indent=2)
            logger.info(f"历史成分股快照已保存: {save_path} ({len(snapshots)} 个快照)")
        else:
            logger.warning(
                "未获取到历史成分股数据，回测将存在幸存者偏差。"
                "请确认 AKShare 版本支持 index_stock_cons_weight_csindex 的 date 参数"
            )

        return snapshots

    # ── 获取所有已下载股票列表 ─────────────────────────────────
    def get_downloaded_stocks(self) -> List[str]:
        """返回已下载数据的股票代码列表（Qlib格式）"""
        csv_files = list(self.stock_dir.glob("*.csv"))
        return [f.stem for f in csv_files]


if __name__ == "__main__":
    from utils.helpers import setup_logger
    setup_logger("collector")
    collector = AKShareCollector()
    collector.download_all()
