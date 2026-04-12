"""
工具函数模块
提供配置加载、路径管理、日志初始化等通用功能
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import yaml
from loguru import logger


# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """加载 YAML 配置文件

    Args:
        config_name: 配置文件名（不含路径），如 'data_config.yaml'

    Returns:
        配置字典
    """
    config_path = PROJECT_ROOT / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_data_config() -> Dict[str, Any]:
    return load_config("data_config.yaml")["data"]


def get_factor_config() -> Dict[str, Any]:
    return load_config("factor_config.yaml")["factor"]


def get_model_config() -> Dict[str, Any]:
    return load_config("model_config.yaml")["model"]


def get_strategy_config() -> Dict[str, Any]:
    return load_config("strategy_config.yaml")["strategy"]


def get_backtest_config() -> Dict[str, Any]:
    return load_config("backtest_config.yaml")["backtest"]


def expand_path(path_str: str) -> Path:
    """展开路径中的 ~ 和环境变量"""
    return Path(os.path.expanduser(os.path.expandvars(path_str)))


def ensure_dir(path: Path) -> Path:
    """确保目录存在，不存在则创建"""
    path = expand_path(str(path)) if isinstance(path, str) else path
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(
    log_name: str = "pipeline",
    log_dir: Optional[str] = None,
    level: str = "INFO",
) -> None:
    """初始化 loguru 日志

    Args:
        log_name: 日志名前缀
        log_dir: 日志目录，默认为项目 logs/ 目录
        level: 日志级别
    """
    # 移除默认 handler
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )

    # 文件输出
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"
    else:
        log_dir = expand_path(log_dir)
    ensure_dir(log_dir)

    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{log_name}_{today}.log"
    logger.add(
        str(log_file),
        level=level,
        rotation="100 MB",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )
    logger.info(f"日志文件: {log_file}")


def today_str() -> str:
    """返回当天日期字符串 YYYYMMDD"""
    return datetime.now().strftime("%Y%m%d")


def date_to_str(dt) -> str:
    """日期对象转字符串 YYYY-MM-DD"""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d")


def str_to_date(s: str) -> datetime:
    """字符串转日期对象，支持 YYYY-MM-DD 和 YYYYMMDD"""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {s}")


def stock_code_to_qlib(code: str) -> str:
    """股票代码转 Qlib 格式

    AKShare 6位纯数字 → Qlib SH600000 / SZ000001 格式
    """
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def qlib_code_to_akshare(qlib_code: str) -> str:
    """Qlib 代码转 AKShare 6位数字格式"""
    return qlib_code[2:]  # 去掉 SH/SZ 前缀


def merge_config_with_args(config: Dict, args: Dict) -> Dict:
    """用命令行参数覆盖 YAML 配置（仅覆盖非 None 的参数）"""
    merged = config.copy()
    for key, value in args.items():
        if value is not None:
            merged[key] = value
    return merged
