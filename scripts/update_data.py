#!/usr/bin/env python3
"""
每日增量数据更新脚本
自动检测最后更新日期，从该日期起增量拉取新数据
日志输出到 logs/data_update_YYYYMMDD.log
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from utils.helpers import setup_logger, get_data_config
from data.collector_akshare import AKShareCollector
from data.csv_to_qlib import CsvToQlib
from data.health_check import DataHealthChecker


def main():
    parser = argparse.ArgumentParser(description="每日增量数据更新")
    parser.add_argument("--skip-convert", action="store_true", help="跳过 bin 转换")
    parser.add_argument("--skip-check", action="store_true", help="跳过健康检查")
    args = parser.parse_args()

    setup_logger("data_update")
    logger.info("=" * 60)
    logger.info("Qlib Pipeline - 每日增量数据更新")
    logger.info("=" * 60)

    config = get_data_config()

    # 1. 增量下载
    logger.info("[1/3] 增量下载数据...")
    collector = AKShareCollector(config)
    collector.update_incremental()

    # 2. 健康检查
    if not args.skip_check:
        logger.info("[2/3] 数据健康检查...")
        checker = DataHealthChecker(config)
        checker.run_full_check()
    else:
        logger.info("[2/3] 跳过健康检查")

    # 3. 重新转换 bin
    if not args.skip_convert:
        logger.info("[3/3] 重新生成 Qlib .bin 数据...")
        converter = CsvToQlib(config)
        converter.convert_all()
    else:
        logger.info("[3/3] 跳过转换")

    logger.info("=" * 60)
    logger.info("每日更新完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
