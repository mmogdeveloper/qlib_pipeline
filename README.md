# Qlib学习交流群
1091248039

# Qlib Pipeline - A股量化投资流水线

基于 Microsoft Qlib 框架构建的完整 A 股量化投资流水线，数据源使用 AKShare 免费接口，覆盖「数据 → 因子 → 模型 → 信号 → 评估」五个阶段。

## 项目结构

```
qlib_pipeline/
├── config/                     # YAML 配置文件
│   ├── data_config.yaml        # 数据配置（区间、路径、下载参数）
│   ├── factor_config.yaml      # 因子配置（Alpha158 + 自定义因子）
│   ├── model_config.yaml       # 模型配置（LightGBM/Ridge/MLP 超参数）
│   ├── strategy_config.yaml    # 策略配置（TopK、交易成本、A股规则）
│   └── backtest_config.yaml    # 回测配置（区间、基准、指标）
├── data/                       # 数据模块
│   ├── collector_akshare.py    # AKShare 数据下载器（全量/增量）
│   ├── csv_to_qlib.py          # CSV → Qlib .bin 格式转换
│   ├── data_loader.py          # 统一数据加载接口
│   └── health_check.py         # 数据健康检查
├── factors/                    # 因子模块
│   ├── alpha158.py             # Qlib Alpha158 因子集配置
│   ├── custom_factors.py       # 自定义因子（动量/波动率/换手率等）
│   └── preprocessor.py         # 因子预处理（去极值/标准化/中性化）
├── model/                      # 模型模块
│   ├── lgbm_model.py           # LightGBM 模型配置
│   ├── linear_model.py         # Ridge 回归模型配置
│   ├── mlp_model.py            # MLP 模型（PyTorch）
│   └── model_trainer.py        # 统一训练/保存/加载接口
├── signal/                     # 信号模块
│   ├── signal_generator.py     # 信号生成器
│   └── portfolio.py            # 组合构建与回测执行
├── evaluation/                 # 评估模块
│   ├── metrics.py              # 评估指标计算
│   ├── visualization.py        # 可视化图表
│   └── report.py               # HTML 报告生成
├── scripts/                    # 脚本
│   ├── init_data.py            # 首次全量下载
│   └── update_data.py          # 每日增量更新
├── utils/
│   └── helpers.py              # 工具函数
├── logs/                       # 日志目录
├── reports/                    # 报告输出目录
├── run_pipeline.py             # 一键运行全流水线
├── clear_result.py             # 清空 logs/ 和 reports/ 目录
├── diag.py                     # MLflow 实验记录诊断脚本
├── requirements.txt
└── README.md
```

## 安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 PyTorch（可选，仅 MLP 模型需要）

```bash
# CPU 版本
pip install torch

# GPU 版本（根据 CUDA 版本选择）
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 快速开始

### 首次全量数据初始化

```bash
python scripts/init_data.py
```

这将执行：
1. 从 AKShare 下载沪深300成分股的历史日频行情（2015年至今）
2. 下载交易日历和沪深300基准指数
3. 数据健康检查（缺失日、异常价格、停牌标记）
4. 转换为 Qlib `.bin` 二进制格式

可选参数：
```bash
python scripts/init_data.py --start-date 2018-01-01  # 自定义起始日期
python scripts/init_data.py --workers 4              # 调整并发数
python scripts/init_data.py --skip-download          # 跳过下载，仅转换
```

### 每日增量更新

```bash
python scripts/update_data.py
```

自动检测上次更新日期，增量拉取新数据。建议设置 cron 任务每日收盘后运行：

```bash
# crontab -e
30 16 * * 1-5 cd /path/to/qlib_pipeline && python scripts/update_data.py
```

### 运行全流水线

```bash
python run_pipeline.py --stage all
```

### 单独运行某个阶段

```bash
# 仅数据阶段
python run_pipeline.py --stage data

# 仅模型训练（默认 LightGBM）
python run_pipeline.py --stage model --model lgbm

# 使用 Ridge 回归
python run_pipeline.py --stage model --model linear

# 仅回测
python run_pipeline.py --stage backtest

# 仅生成评估报告
python run_pipeline.py --stage evaluate

# 跳过数据下载，直接训练+回测+评估
python run_pipeline.py --stage all --skip-data

# 不使用自定义因子
python run_pipeline.py --stage all --skip-data --no-custom-factors
```

### 鲁棒性分析

```bash
# 成本敏感性分析（不同买卖成本组合）
python run_pipeline.py --stage sensitivity

# TopK 参数敏感性分析（不同持仓数 / 换出数组合）
python run_pipeline.py --stage topk-sensitivity

# 市场环境过滤分析（牛熊过滤对收益的影响）
python run_pipeline.py --stage regime

# 填充报告"策略目标达成评分"（topk 稳健性 + 市场环境过滤 + 回撤控制）
python run_pipeline.py --stage all --skip-data --topk-sensitivity --regime

# 全流水线 + 全部分析
python run_pipeline.py --stage all --skip-data --sensitivity --topk-sensitivity --regime
```

分析结果以 CSV 形式写入 `reports/` 目录（文件名带日期后缀）。

### 辅助脚本

```bash
python clear_result.py    # 清空 logs/ 和 reports/ 目录
python diag.py            # 诊断 MLflow 实验记录状态
```

## 配置说明

所有参数通过 `config/` 目录下的 YAML 文件管理，无需修改代码。

### 数据配置 (`data_config.yaml`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `start_date` | 数据起始日期 | 2015-01-01 |
| `end_date` | 数据结束日期 | 2026-04-12 |
| `download.max_workers` | 下载并发数 | 8 |
| `download.rate_limit_sleep` | 请求间隔(秒) | 0.3 |

### 模型配置 (`model_config.yaml`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `default` | 默认模型 | lgbm |
| `label.expression` | 预测目标 | 未来5日收益率 |
| `lgbm.kwargs.n_estimators` | LGB 树数量 | 1000 |
| `lgbm.kwargs.learning_rate` | 学习率 | 0.0421 |

### 策略配置 (`strategy_config.yaml`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `topk` | 每期持仓数 | 30 |
| `n_drop` | 每期最多换出数 | 3 |
| `rebalance.frequency` | 调仓频率 | week |
| `cost.buy_commission` | 买入佣金 | 0.0003 |
| `cost.sell_commission` | 卖出佣金 | 0.0003 |
| `cost.stamp_tax` | 印花税（仅卖出） | 0.001 |
| `cost.buy_slippage` / `sell_slippage` | 双向滑点 | 0.0002 |

## 五个阶段详解

### 阶段1：数据

- **数据源**: AKShare（免费、无需 Token）
- **覆盖**: 沪深300成分股日频后复权行情
- **字段**: open, high, low, close, volume, amount, factor, vwap, turnover
- **特性**: 多线程并发下载、指数退避重试、增量更新

### 阶段2：因子

- **基准因子集**: Qlib Alpha158（158个量价因子）
- **自定义因子**: 动量、波动率、换手率、量价背离、流动性（量相关因子统一使用换手率而非成交量，避免股本变动带来的尺度漂移）
- **预处理**: MAD去极值 → Z-score标准化 → 缺失值填充

### 阶段3：模型

- **LightGBM（默认）**: 梯度提升决策树，支持 early stopping；macOS 下强制单线程以规避 OpenMP 冲突
- **Ridge 回归**: 带正则化的线性模型
- **MLP（可选）**: 双隐藏层全连接网络 (256-128)
- **预测目标**: 未来5日收益率
- **滚动验证**: 支持多折训练配置（见 `model_config.yaml`）

### 阶段4：信号与组合

- **选股**: TopkDropout 策略（每期选30只，最多换3只）
- **权重**: 等权配置 / 分数加权
- **A股规则**: 涨跌停限制、ST 过滤、次新股过滤、停牌处理（停牌基于交易日历缺失判断）
- **成本**: 买入≈万5(佣金万3+滑点万2)、卖出≈万15(佣金万3+印花税千1+滑点万2)
- **可交易性过滤**: 信号生成阶段剔除当日不可交易的股票，避免无效持仓

### 阶段5：评估

- **收益**: CAGR、累计收益、年化超额收益
- **风险**: 年化波动率、最大回撤、下行波动率
- **风险调整**: Sharpe、Sortino、Calmar、Information Ratio
- **因子有效性**: IC、Rank IC、ICIR、IC胜率（同时输出 CSRankNorm 处理前后的原始 IC 以便交叉校验）
- **可视化**: 净值曲线、超额收益、月度热力图、回撤水下图、IC时序图
- **报告**: 纯文本格式（`.txt`），包含全部指标、每期选股明细与交易执行逻辑

## 运行测试

### 安装测试依赖

```bash
pip install -r requirements-dev.txt
```

### 运行单元测试

```bash
make test              # 只跑 unit 测试（< 10 秒，无需网络和 Qlib 数据）
```

### 运行全部测试

```bash
make test-all          # unit + integration
```

### 生成覆盖率报告

```bash
make coverage          # 生成覆盖率报告（目标 unit 覆盖率 ≥ 70%）
```

### 测试结构

```
tests/
├── conftest.py                    # 共享 fixtures（合成 OHLCV、pred_score 等）
├── unit/                          # 单元测试（快速，无外部依赖）
│   ├── test_helpers.py            # 工具函数（代码转换、日期解析、配置加载）
│   ├── test_custom_factors.py     # Label 表达式前视偏差、因子加载
│   ├── test_preprocessor.py       # 预处理器配置结构
│   ├── test_signal_generator.py   # TopK 信号生成、格式校验
│   ├── test_portfolio.py          # Signal shift、成本一致性、交易记录
│   ├── test_metrics.py            # 评估指标边界情况、IC 独立校验
│   └── test_visualization.py      # 超额收益公式回归测试
├── integration/                   # 集成测试（需要 Qlib 环境）
│   ├── test_factor_pipeline.py    # 因子流水线端到端
│   └── test_backtest_pipeline.py  # 回测流水线端到端
└── fixtures/
    └── sample_data.py             # 合成数据生成器
```

### 自定义标记

- `@pytest.mark.slow` — 慢速测试
- `@pytest.mark.integration` — 需要 Qlib 环境的集成测试
- `@pytest.mark.requires_qlib_data` — 需要真实 Qlib bin 数据

```bash
pytest tests/ -m "not integration"   # 跳过集成测试
pytest tests/ -m "not slow"          # 跳过慢速测试
```

## 常见问题

### Q: AKShare 下载被限速怎么办？

调大 `config/data_config.yaml` 中的 `rate_limit_sleep`（默认 0.3 秒），或减少 `max_workers`。

### Q: 部分股票数据缺失？

这是正常的。退市股、长期停牌股可能无法获取完整数据。健康检查会标记这些情况。可以查看 `logs/` 目录下的日志了解详情。

### Q: 如何使用 MLP 模型？

先安装 PyTorch: `pip install torch`，然后运行：
```bash
python run_pipeline.py --stage model --model mlp
```

### Q: 如何修改回测区间？

编辑 `config/backtest_config.yaml` 中的 `start_date` 和 `end_date`。

### Q: 数据存储在哪里？

- 原始 CSV: `~/.qlib/raw_data/akshare_csv/`
- Qlib .bin: `~/.qlib/qlib_data/cn_data_akshare/`

### Q: 如何添加新的自定义因子？

编辑 `config/factor_config.yaml`，在 `custom_factors` 下添加新因子：
```yaml
custom_factors:
  my_category:
    - name: "my_factor"
      expression: "Mean($close, 10)/Mean($close, 30)-1"
      description: "自定义因子描述"
```

## 依赖版本

- Python 3.10+
- pyqlib >= 0.9.0
- akshare >= 1.12.0
- lightgbm >= 3.3.0
- torch >= 2.0.0（可选）
