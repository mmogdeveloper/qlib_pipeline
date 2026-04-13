#!/usr/bin/env python3
"""
首次全量数据下载脚本
下载沪深300成分股历史行情、交易日历、基准指数
并转换为 Qlib .bin 格式
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from utils.helpers import setup_logger, get_data_config
from data.collector_akshare import AKShareCollector
from data.csv_to_qlib import CsvToQlib
from data.health_check import DataHealthChecker


def main():
    parser = argparse.ArgumentParser(description="首次全量数据下载")
    parser.add_argument("--start-date", type=str, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, help="并发线程数")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，直接转换")
    parser.add_argument("--skip-convert", action="store_true", help="跳过转换")
    parser.add_argument("--skip-check", action="store_true", help="跳过健康检查")
    args = parser.parse_args()

    setup_logger("init_data")
    logger.info("=" * 60)
    logger.info("Qlib Pipeline - 首次全量数据初始化")
    logger.info("=" * 60)

    # 加载配置并应用命令行覆盖
    config = get_data_config()
    if args.start_date:
        config["start_date"] = args.start_date
    if args.end_date:
        config["end_date"] = args.end_date
    if args.workers:
        config["download"]["max_workers"] = args.workers

    # 阶段1: 下载数据
    if not args.skip_download:
        logger.info("[阶段1/4] 全量下载数据...")
        collector = AKShareCollector(config)
        collector.download_all()

        logger.info("[阶段2/4] 下载历史成分股快照（消除幸存者偏差）...")
        snapshots = collector.download_constituent_history()
        if snapshots:
            dates = sorted(snapshots.keys())
            sizes = [len(snapshots[d]) for d in dates]
            if len(set(sizes)) == 1:
                logger.warning(
                    "⚠ 历史成分股快照中所有日期返回相同数量的股票，"
                    "AKShare 的 index_stock_cons_weight_csindex 可能忽略了 date 参数，"
                    "幸存者偏差可能仍然存在！请手动检查 csi300_history.json 确认不同日期的成分股确实不同。"
                )
            else:
                logger.info(f"历史成分股快照验证通过: {len(dates)} 个时间节点，成分股数量有变化")
        else:
            logger.warning("⚠ 历史成分股快照获取失败，instruments 将回退到静态模式（幸存者偏差存在）")
    else:
        logger.info("[阶段1/4] 跳过下载")
        logger.info("[阶段2/4] 跳过历史成分股快照下载")

    # 阶段3: 数据健康检查
    if not args.skip_check:
        logger.info("[阶段3/4] 数据健康检查...")
        checker = DataHealthChecker(config)
        checker.run_full_check()
    else:
        logger.info("[阶段3/4] 跳过健康检查")

    # 阶段4: CSV → Qlib .bin 转换
    if not args.skip_convert:
        logger.info("[阶段4/4] CSV → Qlib .bin 转换（自动检测历史成分股数据）...")
        converter = CsvToQlib(config)
        converter.convert_all()
    else:
        logger.info("[阶段4/4] 跳过转换")

    logger.info("=" * 60)
    logger.info("数据初始化全部完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
