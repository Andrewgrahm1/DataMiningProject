# Stock / Trading Terms and Definitions

| Term | Definition |
|------|------------|
| Bar | A single period of price and volume data at a fixed time frame (e.g. one minute or one day), typically with open, high, low, close, and volume. |
| Basis points (bps) | One hundredth of a percent (1 bps = 0.01%). Used for spreads, fees, and reporting (e.g. 10 bps = 0.10%). |
| Book value | Valuing positions at cost basis (average price paid) when no current market price is used. |
| Broker | A party that executes orders on behalf of a trader (e.g. buys and sells securities). |
| Backtest | Running a trading strategy over historical price data to see how it would have performed. |
| Backtest result | The outcome of a backtest: final portfolio state and, optionally, the series of equity values over time. |
| CAT (Consolidated Audit Trail) | A FINRA fee on equity trades (buys and sells): $0.00003 per share, rounded up to the nearest cent. |
| Clock | The notion of “current time” used by a trading system (e.g. bar time in backtests, real time in live trading). |
| Cost basis | Total cash spent (buys) or received (sells) for a position. Average price per share = cost basis ÷ quantity. |
| Data feed | A source of market data (e.g. bars or ticks) over time, often keyed by symbol and timestamp. |
| Entry | The price (or bar) at which a position is opened. |
| Equity | Total account value: cash plus the market value of all positions. |
| Equity curve | A time series of portfolio equity over time, used for performance and drawdown analysis. |
| Fill | An executed trade: symbol, side (buy/sell), price, quantity, time, and optionally fees. |
| Fee model | A rule or function that determines the fee charged for a given trade. |
| Imputed bar / forward propagation | A bar inserted when data is missing: same prices as the previous bar, with volume (and trade count) set to zero, to keep a continuous time sequence. |
| Mark-to-market | Valuing positions at current market prices (e.g. last close) instead of cost. |
| Multi-index (symbol, timestamp) | A common way to store bar data: one row per symbol and timestamp, with columns such as open, high, low, close, volume, trade count, VWAP. |
| OHLC / OHLCV | Open (price at bar start), high (highest price in the bar), low (lowest), close (price at bar end), and volume (shares traded). Valid bars satisfy low ≤ open, close ≤ high. |
| Order | A request to trade: symbol, side (buy/sell), quantity, and order type (e.g. market, limit, stop, stop-limit), with optional limit/stop prices. |
| Order side | Whether the order is a buy or a sell. |
| Order type | Market (execute at current price), limit (execute at specified price or better), stop (trigger when price reaches a level), or stop-limit (trigger then execute as limit). |
| Portfolio | Cash plus the set of positions (holdings) by symbol; often also tracks trade history. |
| Position | A holding in a single symbol: quantity (shares) and cost basis; average price = cost basis ÷ quantity. |
| RTH (Regular Trading Hours) | The main exchange session (e.g. NYSE 9:30 AM–4:00 PM Eastern). Outside RTH is pre-market or after-hours. |
| Slippage (in basis points) | The difference between expected and actual fill price, often modeled in basis points (e.g. buys fill slightly higher, sells slightly lower). |
| Stop loss (SL) | A price (or percentage below entry) at which a long position is closed to limit loss. |
| Strategy | A set of rules that, given market data and portfolio state, decide what orders to submit. |
| TAF (Transaction Assessment Fee) | A FINRA fee on equity sells: $0.000166 per share, max $8.30 per trade, rounded up to the nearest cent. |
| Take profit (TP) | A price (or percentage above entry) at which a long position is closed to lock in profit. |
| Time frame | The length of each bar (e.g. 1 minute, 1 day). |
| Trade count | Number of individual trades (transactions) in a bar. |
| Trade history | The list of all fills (executed trades) in an account or backtest. |
| VWAP (Volume-Weighted Average Price) | The average price during a period weighted by volume; often used as a benchmark for execution quality. |
