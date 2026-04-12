# 测试过程中发现的问题

## 审查通过（无 bug）

在编写测试的过程中对源码进行了仔细审查，**未发现明确的 bug**。以下是关键逻辑的审查结论：

### 1. Signal Shift（前视偏差防护）✅
`portfolio.py:79-88` 的 shift 逻辑正确：
- `date_map = dict(zip(dates[:-1], dates[1:]))` 将 T 映射到 T+1
- 最后一天数据因无映射目标被 `dropna` 丢弃
- shift 后第一个可用日期 = 原第二个交易日，符合 T+1 才使用信号的设计

### 2. 清仓时 sell_price 处理 ✅
`portfolio.py:249` 正确使用了 `prev_price`：
```python
sell_price = prev_price if prev_price > 0 else cur_price
```
清仓时 `cur_price=0`（不在当前持仓中），使用前一日价格估算，逻辑正确。

### 3. 成本计算 ✅
`portfolio.py:114-119` 和 `strategy_config.yaml` 中的成本配置一致：
- `open_cost = buy_commission(0.0003) + buy_slippage(0.0002) = 0.0005`
- `close_cost = sell_commission(0.0003) + stamp_tax(0.001) + sell_slippage(0.0002) = 0.0015`
- `backtest_config.yaml` 中的默认值与计算结果一致

### 4. Label 表达式 ✅
`Ref($close, -6)/Ref($close, -1) - 1` 正确表示未来 5 日持有收益（T+1 买入, T+6 卖出）

## 潜在风险（非 bug，但值得注意）

### 1. `stock_code_to_qlib` 对北交所代码的处理
北交所股票代码以 `8` 或 `4` 开头（如 `430047`、`830946`），当前逻辑会将它们映射为 `SZ` 前缀。
如果需要支持北交所，需要额外处理。当前仅覆盖沪深市场，不算 bug。

### 2. Signal shift 在 `run_backtest` 和 `run_backtest_from_recorder` 中重复
同样的 shift 逻辑在两个函数中各写了一遍（`portfolio.py:79-88` 和 `portfolio.py:316-323`），
如果未来修改一处忘记修改另一处可能引入不一致。建议提取为公共函数。
