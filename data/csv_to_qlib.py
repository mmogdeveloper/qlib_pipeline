"""
CSV → Qlib .bin 转换器
将 AKShare 下载的 CSV 数据转换为 Qlib 所需的 .bin 二进制格式
生成 instruments、calendars 等元数据文件
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from utils.helpers import (
    get_data_config,
    expand_path,
    ensure_dir,
    stock_code_to_qlib,
)


class CsvToQlib:
    """CSV 转 Qlib .bin 格式转换器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_data_config()
        self.raw_dir = expand_path(self.config["raw_csv_dir"])
        self.qlib_dir = expand_path(self.config["qlib_data_dir"])
        self.stock_dir = self.raw_dir / "stocks"
        self.meta_dir = self.raw_dir / "meta"

        # Qlib 数据子目录
        self.features_dir = ensure_dir(self.qlib_dir / "features")
        self.instruments_dir = ensure_dir(self.qlib_dir / "instruments")
        self.calendars_dir = ensure_dir(self.qlib_dir / "calendars")

    def normalize_csv(self) -> Path:
        """将所有股票 CSV 规范化为 Qlib dump_bin 所需格式

        Qlib 期望的 CSV 格式:
        - 文件名: <SYMBOL>.csv (如 SH600000.csv)
        - 列: date, open, high, low, close, volume, amount, factor, vwap, turnover
        - date 格式: YYYY-MM-DD

        Returns:
            规范化 CSV 输出目录路径
        """
        logger.info("规范化 CSV 数据为 Qlib 格式...")
        norm_dir = ensure_dir(self.raw_dir / "normalized")

        csv_files = list(self.stock_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"未找到股票 CSV 文件: {self.stock_dir}")

        count = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, parse_dates=["date"])
                if df.empty:
                    logger.debug(f"[{csv_file.stem}] 数据为空，跳过")
                    continue

                # 确保字段完整
                fields = ["date", "open", "high", "low", "close", "volume",
                          "amount", "factor", "vwap", "turnover"]
                # 关键字段不可缺失
                critical_fields = ["date", "open", "high", "low", "close", "volume", "amount"]
                missing_critical = [f for f in critical_fields if f not in df.columns]
                if missing_critical:
                    logger.warning(f"[{csv_file.stem}] 缺少关键字段 {missing_critical}，跳过此股票")
                    continue

                # 补全可选字段
                if "factor" not in df.columns:
                    df["factor"] = 1.0
                if "vwap" not in df.columns:
                    df["vwap"] = df.apply(
                        lambda r: r["amount"] / r["volume"]
                        if (pd.notna(r["volume"]) and r["volume"] > 0
                            and pd.notna(r["amount"]) and r["amount"] > 0)
                        else r["close"],
                        axis=1,
                    )
                if "turnover" not in df.columns:
                    df["turnover"] = 0.0

                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                df = df[fields].sort_values("date")

                # 数值类型确保
                for col in fields[1:]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                out_path = norm_dir / csv_file.name
                df.to_csv(out_path, index=False)
                count += 1
            except Exception as e:
                logger.error(f"规范化 {csv_file.name} 失败: {e}")

        logger.info(f"规范化完成: {count} 只股票")
        return norm_dir

    def generate_instruments(self) -> None:
        """生成 instruments 文件，优先使用历史成分股消除幸存者偏差

        如果存在 csi300_history.json（由 download_constituent_history 生成），
        则为每只股票生成其实际在指数中的日期范围，避免幸存者偏差。
        否则回退到静态模式（全部数据范围），并输出警告。
        """
        history_path = self.meta_dir / "csi300_history.json"
        if history_path.exists():
            self._generate_dynamic_instruments(history_path)
        else:
            logger.warning(
                "⚠ 未找到历史成分股数据 (csi300_history.json)，"
                "使用静态成分股列表，回测存在幸存者偏差！"
                "请在数据阶段运行 download_constituent_history() 获取历史数据"
            )
            self._generate_static_instruments()

    def _generate_static_instruments(self) -> None:
        """静态 instruments：用每只股票的完整数据范围（有幸存者偏差）"""
        logger.info("生成静态 instruments 文件...")
        csv_files = list(self.stock_dir.glob("*.csv"))
        records = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, parse_dates=["date"])
                if df.empty:
                    continue
                symbol = csv_file.stem
                start = df["date"].min().strftime("%Y-%m-%d")
                end = df["date"].max().strftime("%Y-%m-%d")
                records.append(f"{symbol}\t{start}\t{end}")
            except Exception as e:
                logger.error(f"处理 {csv_file.name} 失败: {e}")

        instruments_path = self.instruments_dir / "csi300.txt"
        with open(instruments_path, "w") as f:
            f.write("\n".join(sorted(records)))
        logger.info(f"静态 instruments 已生成: {instruments_path} ({len(records)} 只股票)")

        all_path = self.instruments_dir / "all.txt"
        with open(all_path, "w") as f:
            f.write("\n".join(sorted(records)))

    def _generate_dynamic_instruments(self, history_path) -> None:
        """动态 instruments：根据历史成分股快照生成准确的在指数日期范围

        逻辑：
        1. 读取各调整日的成分股快照
        2. 对每只股票，找出它首次和末次出现在快照中的日期
        3. 结合该股票实际数据的起止日期，取交集
        4. 如果股票在中间被移出又移入，合并为多段记录
        """
        import json

        logger.info("生成动态 instruments 文件（基于历史成分股）...")

        with open(history_path, "r", encoding="utf-8") as f:
            snapshots = json.load(f)

        if not snapshots:
            logger.warning("历史成分股快照为空，回退到静态模式")
            self._generate_static_instruments()
            return

        sorted_dates = sorted(snapshots.keys())

        # 读取每只股票的实际数据范围
        stock_data_range = {}
        for csv_file in self.stock_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, usecols=["date"], parse_dates=["date"])
                if df.empty:
                    continue
                symbol = csv_file.stem
                stock_data_range[symbol] = (
                    df["date"].min().strftime("%Y-%m-%d"),
                    df["date"].max().strftime("%Y-%m-%d"),
                )
            except Exception:
                pass

        # 为每只股票构建在指数中的日期段
        # 每个快照代表一个半年度区间：[snapshot_date, next_snapshot_date)
        stock_segments = {}  # {qlib_code: [(start, end), ...]}

        for i, snap_date in enumerate(sorted_dates):
            codes = snapshots[snap_date]
            # 区间起点：快照日期
            seg_start = f"{snap_date[:4]}-{snap_date[4:6]}-{snap_date[6:]}"
            # 区间终点：下一个快照日期前一天，或数据末尾
            if i + 1 < len(sorted_dates):
                nd = sorted_dates[i + 1]
                seg_end = f"{nd[:4]}-{nd[4:6]}-{nd[6:]}"
            else:
                seg_end = self.config["end_date"]

            for code in codes:
                qlib_code = stock_code_to_qlib(code)
                if qlib_code not in stock_segments:
                    stock_segments[qlib_code] = []

                # 尝试合并相邻区间
                if (stock_segments[qlib_code]
                        and stock_segments[qlib_code][-1][1] >= seg_start):
                    prev_start, _ = stock_segments[qlib_code][-1]
                    stock_segments[qlib_code][-1] = (prev_start, seg_end)
                else:
                    stock_segments[qlib_code].append((seg_start, seg_end))

        # 生成 instruments 记录，与实际数据范围取交集
        records = []
        for qlib_code, segments in sorted(stock_segments.items()):
            if qlib_code not in stock_data_range:
                continue
            data_start, data_end = stock_data_range[qlib_code]
            for seg_start, seg_end in segments:
                # 取交集
                actual_start = max(seg_start, data_start)
                actual_end = min(seg_end, data_end)
                if actual_start <= actual_end:
                    records.append(f"{qlib_code}\t{actual_start}\t{actual_end}")

        instruments_path = self.instruments_dir / "csi300.txt"
        with open(instruments_path, "w") as f:
            f.write("\n".join(records))

        n_stocks = len(set(r.split("\t")[0] for r in records))
        logger.info(
            f"动态 instruments 已生成: {instruments_path} "
            f"({n_stocks} 只股票, {len(records)} 条记录)"
        )

        # all.txt 也使用动态数据
        all_path = self.instruments_dir / "all.txt"
        with open(all_path, "w") as f:
            f.write("\n".join(records))

    def generate_calendars(self) -> None:
        """生成 calendars/day.txt

        格式: 每行一个交易日 YYYY-MM-DD
        """
        logger.info("生成 calendars 文件...")
        trade_dates_path = self.meta_dir / "trade_dates.csv"

        if trade_dates_path.exists():
            df = pd.read_csv(trade_dates_path, parse_dates=["trade_date"])
            dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
        else:
            # 从所有股票 CSV 中提取交易日
            logger.warning("交易日历文件不存在，从股票数据中提取")
            all_dates = set()
            for csv_file in self.stock_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file, parse_dates=["date"])
                    all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
                except Exception:
                    pass
            dates = sorted(all_dates)

        cal_path = self.calendars_dir / "day.txt"
        with open(cal_path, "w") as f:
            f.write("\n".join(dates))
        logger.info(f"calendars 文件已生成: {cal_path} ({len(dates)} 个交易日)")

    def dump_bin(self) -> None:
        """将 CSV 转为 Qlib .bin 二进制格式

        Qlib 二进制格式:
        - features/<INSTRUMENT>/<field>.day.bin: 按日历偏移存储的 float32 数组
        - calendars/day.txt: 交易日列表
        - instruments/csi300.txt: 股票列表及起止日期
        """
        import struct
        import numpy as np

        logger.info("开始转换 CSV → Qlib .bin 格式...")

        # 先规范化 CSV
        norm_dir = self.normalize_csv()

        # 读取日历建立日期→偏移索引
        cal_path = self.calendars_dir / "day.txt"
        with open(cal_path) as f:
            calendar_dates = [line.strip() for line in f if line.strip()]
        date_to_idx = {d: i for i, d in enumerate(calendar_dates)}
        cal_len = len(calendar_dates)

        fields = ["open", "high", "low", "close", "volume", "amount", "factor", "vwap", "turnover"]
        csv_files = list(norm_dir.glob("*.csv"))
        count = 0

        for csv_file in csv_files:
            symbol = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                # 找到该股票在日历中的起止索引
                dates_in_csv = [str(d)[:10] for d in df["date"]]
                valid_dates = [d for d in dates_in_csv if d in date_to_idx]
                if not valid_dates:
                    continue
                start_idx = date_to_idx[min(valid_dates)]
                end_idx = date_to_idx[max(valid_dates)]
                bin_len = end_idx - start_idx + 1

                feat_dir = ensure_dir(self.features_dir / symbol)

                for field in fields:
                    if field not in df.columns:
                        continue
                    # 只存储 [start_idx, end_idx] 范围的数据
                    arr = np.full(bin_len, np.nan, dtype=np.float32)
                    for _, row in df.iterrows():
                        date_str = str(row["date"])[:10]
                        if date_str in date_to_idx:
                            idx = date_to_idx[date_str] - start_idx
                            if 0 <= idx < bin_len:
                                arr[idx] = float(row[field])

                    # factor 停牌日前向填充（复权因子在停牌期间不变）
                    if field == "factor" and np.isnan(arr).any():
                        last = np.nan
                        for i in range(len(arr)):
                            if not np.isnan(arr[i]):
                                last = arr[i]
                            elif not np.isnan(last):
                                arr[i] = last
                        # 反向填充最前面的 NaN
                        valid_vals = arr[~np.isnan(arr)]
                        if len(valid_vals) > 0:
                            fill = valid_vals[0]
                            for i in range(len(arr)):
                                if np.isnan(arr[i]):
                                    arr[i] = fill
                                else:
                                    break
                        arr = np.where(np.isnan(arr), np.float32(1.0), arr)

                    # Qlib binary 格式要求第一个 float32 为该股票的日历起始索引（ref_start_index）
                    # 格式: [float32(start_idx)] [float32 values...]
                    header = np.array([float(start_idx)], dtype=np.float32)
                    bin_data = np.concatenate([header, arr])
                    bin_path = feat_dir / f"{field}.day.bin"
                    bin_data.tofile(str(bin_path))

                count += 1
            except Exception as e:
                logger.error(f"dump_bin {csv_file.name} 失败: {e}")

        logger.info(f"dump_bin 转换完成: {count} 只股票")

    def convert_all(self) -> None:
        """完整转换流程: 规范化 → 生成元数据 → dump_bin"""
        logger.info("=" * 60)
        logger.info("开始 CSV → Qlib 数据转换")
        logger.info("=" * 60)

        # 1. 生成 instruments
        self.generate_instruments()

        # 2. 生成 calendars
        self.generate_calendars()

        # 3. dump_bin 转换
        self.dump_bin()

        logger.info("数据转换全部完成!")
        logger.info(f"Qlib 数据目录: {self.qlib_dir}")


if __name__ == "__main__":
    from utils.helpers import setup_logger
    setup_logger("csv_to_qlib")
    converter = CsvToQlib()
    converter.convert_all()
