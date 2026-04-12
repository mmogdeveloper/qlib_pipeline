"""
合成测试数据生成器
生成不依赖真实 Qlib / AKShare 数据的合成 OHLCV、pred_score、returns 等
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def make_trading_dates(n_days: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    """生成模拟交易日序列（跳过周末）"""
    dates = []
    current = pd.Timestamp(start)
    while len(dates) < n_days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return pd.DatetimeIndex(dates)


def make_instruments(n: int = 5) -> list:
    """生成模拟股票代码"""
    return [f"SH60000{i}" for i in range(n)]


def make_synthetic_ohlcv(
    n_instruments: int = 5,
    n_days: int = 30,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成 OHLCV DataFrame，MultiIndex: (datetime, instrument)"""
    rng = np.random.RandomState(seed)
    dates = make_trading_dates(n_days)
    instruments = make_instruments(n_instruments)

    rows = []
    for inst in instruments:
        price = start_price
        for dt in dates:
            ret = rng.normal(0, 0.02)
            close = price * (1 + ret)
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            open_ = price * (1 + rng.normal(0, 0.005))
            volume = rng.randint(100000, 1000000)
            amount = close * volume
            rows.append({
                "datetime": dt,
                "instrument": inst,
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "amount": round(amount, 2),
            })
            price = close

    df = pd.DataFrame(rows)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df


def make_synthetic_pred_score(
    n_instruments: int = 5,
    n_days: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """生成符合 Qlib 格式的预测分数 DataFrame，MultiIndex: (datetime, instrument)"""
    rng = np.random.RandomState(seed)
    dates = make_trading_dates(n_days)
    instruments = make_instruments(n_instruments)

    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    scores = rng.randn(len(index))
    return pd.DataFrame({"score": scores}, index=index)


def make_synthetic_returns(
    n_days: int = 252,
    mean_daily: float = 0.0005,
    std_daily: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """生成日收益率序列"""
    rng = np.random.RandomState(seed)
    dates = make_trading_dates(n_days)
    returns = rng.normal(mean_daily, std_daily, n_days)
    return pd.Series(returns, index=dates, name="return")


def make_synthetic_positions(
    n_days: int = 5,
    instruments: list = None,
    seed: int = 42,
) -> dict:
    """模拟 backtest_daily 返回的 positions dict

    返回 {date: MockPosition} 字典，MockPosition 有 .position 属性
    """
    rng = np.random.RandomState(seed)
    if instruments is None:
        instruments = make_instruments(5)

    dates = make_trading_dates(n_days)
    positions = {}

    for i, dt in enumerate(dates):
        pos_data = {}
        # 模拟每天持有一部分股票
        n_hold = max(2, len(instruments) - 1)
        held = rng.choice(instruments, size=n_hold, replace=False)
        for inst in held:
            pos_data[inst] = {
                "amount": rng.randint(100, 1000) * 100,
                "price": round(rng.uniform(10, 50), 2),
            }
        pos_data["cash"] = {"amount": rng.uniform(100000, 500000)}

        positions[dt] = type("MockPosition", (), {"position": pos_data})()

    return positions


def make_close_series(prices: list, n_instruments: int = 1) -> pd.DataFrame:
    """从给定收盘价列表创建简单 DataFrame，用于 label 计算验证"""
    dates = make_trading_dates(len(prices))
    instruments = make_instruments(n_instruments)
    rows = []
    for inst in instruments:
        for dt, price in zip(dates, prices):
            rows.append({"datetime": dt, "instrument": inst, "close": price})
    df = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    return df
