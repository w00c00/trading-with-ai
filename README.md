# Trading with AI

一个受 NOFX 架构启发的 AI 自动交易系统骨架：AI 做决策，策略做信号，交易所层通过 CCXT 适配多交易所，默认纸交易，支持 Docker 部署。

> 风险提示：自动交易有亏损风险。本项目默认不会实盘下单，只有设置 `LIVE_TRADING_ENABLED=true` 且选择非 `paper` 交易所时才会提交真实订单。

## 能力

- AI Provider：OpenRouter、OpenAI、MiniMax、小米 MiMo、自定义 OpenAI-compatible endpoint
- 交易所：`paper`、Binance、Bybit、OKX、Bitget、KuCoin、Gate、Hyperliquid 等 CCXT 支持的交易所
- 内置策略：趋势动量、RSI 均值回归、波动突破、策略投票集成
- 风控：AI 置信度阈值、最大持仓额度、默认 dry-run
- 方糖 Server酱推送：成交成功后推送成本、利润、成交详情、持仓信息
- Web UI 配置：可在控制台填写 AI、交易所、方糖参数，敏感字段加密保存
- 多策略多币种：可同时选择多个币种和策略，设置运行轮次与持续时长
- 策略模拟：使用历史 K 线模拟策略收益，并在内置图表显示价格/权益曲线
- 策略上传：可从 UI 载入 JSON 模板、上传规则策略，重启后自动加载
- 实时 Dashboard：后台运行后实时展示开仓、平仓、错误和轮次状态
- 网格机器人：支持现货/合约市场类型，内置基础网格策略，也可用 AI 指导生成网格参数
- API：FastAPI `/run-once` 单轮决策和可选执行
- 部署：Docker / docker compose

## 快速启动

```bash
cp .env.example .env
docker compose up --build
```

打开：

- 实时看板：http://127.0.0.1:3471/dashboard
- 单策略运行：http://127.0.0.1:3471/trade
- 多策略与回测：http://127.0.0.1:3471/batch
- 网格机器人：http://127.0.0.1:3471/grid
- 策略中心：http://127.0.0.1:3471/strategy-center
- 配置：http://127.0.0.1:3471/config
- 健康检查：http://127.0.0.1:3471/health
- API 文档：http://127.0.0.1:3471/docs

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 3471
```

## 策略应该在哪里运行

- 单策略运行：适合一个策略、一个主标的。上传后的 NOFX/自定义策略会出现在策略下拉菜单里，可以单次决策，也可以启动循环运行。
- 多策略与回测：适合多个币种、多个策略一起运行，或选择历史时间段做回测。
- 策略中心：只负责导入和管理策略，不负责执行。
- 实时看板：查看后台运行后的开仓、平仓、错误、轮次和余额。

## 单轮决策

可以在控制台页面直接运行，也可以用 API。默认是纸交易：

```bash
curl -X POST http://127.0.0.1:3471/run-once \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC/USDT","strategy":"strategy_ensemble","execute":true}'
```

策略交易和批量运行都支持市场类型：

- `spot`：现货
- `swap`：永续合约
- `future`：交割合约

## 多策略多币种与回测

控制台的“多策略与回测”支持：

- 多币种：用逗号分隔，例如 `BTC/USDT, ETH/USDT, SOL/USDT`
- 策略开关：勾选一个或多个内置策略
- 轮次：按组合重复运行多少轮
- 运行时长：多轮运行时把轮次分布到指定秒数内
- 历史回测：不调用 AI、不下单，只用历史 K 线评估策略，并绘制价格和权益曲线
- 时间段回测：可选择模拟开始/结束时间，把内置或上传策略套到指定时间段验证
- 开始执行：创建后台运行任务，并在实时 Dashboard 显示开仓和平仓

## 自定义策略

控制台的“策略模板与上传”支持 JSON 规则策略，不执行上传代码。点击“载入模板”后可以编辑并上传，支持指标：

- `rsi`
- `sma_cross`
- `price_vs_sma`
- `breakout`

上传成功后策略会立即出现在单策略运行的策略下拉菜单，以及多策略与回测的策略开关里，并保存到 `.config/strategies/*.json`。
上传后的自定义策略也可以在“多策略与回测”里勾选，并用指定时间段做模拟测试。

## 网格机器人

控制台的“网格机器人”支持：

- 市场类型：`spot`、`swap`、`future`
- 基础策略：中性区间、趋势跟随、波动自适应、AI 自适应
- 手动上下边界，或留空由策略自动生成
- 网格间距：等差间距、等比间距
- 网格数量、投入金额、杠杆、轮询间隔
- 独立网格币种输入，不依赖单次交易标的
- AI 策略说明：选择 `AI 自适应` 时，AI 会基于行情和说明建议网格区间
- 触发后执行订单：未开启 `LIVE_TRADING_ENABLED=true` 时，非 `paper` 交易所保持 dry-run
- 每 2 小时交易总结推送：打开开关后通过方糖 Server酱推送网格触发、买卖次数、订单状态和未触发网格

合约网格会通过 CCXT 使用对应交易所的 `defaultType`，并在交易所支持时尝试设置杠杆。不同交易所的合约 symbol 和杠杆规则不完全一致，实盘前建议先用测试网或小金额验证。

## AI 配置

可以在控制台的“运行配置”里填写，也可以在 `.env` 中选择 provider：

```bash
AI_PROVIDER=openrouter
AI_MODEL=openai/gpt-4o-mini
AI_API_KEY=your-openrouter-key
```

Provider 默认 base URL：

- OpenRouter: `https://openrouter.ai/api/v1`
- OpenAI: `https://api.openai.com/v1`
- MiniMax: `https://api.minimaxi.com/v1`
- MiMo: `https://api.mimo-v2.com/v1`

也可以使用任何 OpenAI-compatible 服务：

```bash
AI_PROVIDER=custom_openai_compatible
AI_BASE_URL=https://your-provider.example/v1
AI_MODEL=your-model
AI_API_KEY=your-key
```

## UI 加密配置

控制台保存的配置不会明文写进 `.env`，而是加密保存到：

- `.config/settings.enc`
- `.config/secret.key`

密钥字段再次打开页面时只显示已配置状态，输入框留空表示不修改原值。Docker 重建后如果需要保留 UI 配置，可以给 `.config` 挂载 volume。

## 实盘交易

先确认交易所支持、API key 权限、sandbox 状态和订单单位，再打开：

```bash
DEFAULT_EXCHANGE=binance
EXCHANGE_API_KEY=<your-exchange-api-key>
EXCHANGE_SECRET=<your-exchange-secret>
EXCHANGE_SANDBOX=true
LIVE_TRADING_ENABLED=true
```

建议先在交易所测试网或小金额运行。

## 方糖 Server酱推送

在 `.env` 填入自己的 SendKey：

```bash
SERVERCHAN_SENDKEY=your-serverchan-sendkey
NOTIFY_TRADE_SUCCESS=true
NOTIFY_DRY_RUN=false
```

成交成功后会推送：

- 成交方向、数量、均价、订单 ID
- 成本/成交额
- 当前持仓数量、平均成本、持仓成本
- 卖出时的已实现利润和收益率
- 原始成交详情 JSON

利润基于程序运行期间记录的平均成本计算。如果你在接入本程序前已经有历史持仓，第一版不会自动读取历史成本，卖出利润可能显示为“缺少成本基础”。

## 关于 NOFX

NOFX 是 AGPL-3.0 开源项目，公开 README 描述了“Web Dashboard / API Server / Strategy Engine / MCP AI Client Layer / Exchange Connectors”的架构。本项目没有直接复制 NOFX 源码，而是按相似模块边界实现了一个轻量版本，交易所连接使用 CCXT 适配层完成。
