# Stock Scanner Results

## Description

This report summarizes the output of the **scanner** (`experiments/scanner.py`), which runs a grid search over **take-profit** and **stop-loss** levels on minute OHLCV bar data. For each symbol, the scanner simulates entries at the start of each bar and exits when price hits the take-profit (TP) or stop-loss (SL) threshold, or at end of day. Only trades that exit at take-profit are counted toward revenue; cost is applied per trade. The results help compare which (TP, SL) combinations are most profitable and how often they trigger across the date range.

---

## The Trading Model Simulated

The trading model I am simulating here is one that would be given minutely stock data on one symbol and predict whether our take profit percent would be hit before the stop loss or end of the trading day was reached.

Note that this simulation assumes the model is 100% accurate so that we can tell what type of profit is possible. It does not indicate what profit you would actually get as any model used is not going to be 100% accurate.

---

## Assumptions

Here are some of the assumptions used in this simulation:

- It is assumed that every time you buy a stock, you buy the same dollar amount of it.
  - This is to simplify the calculations.
- It assumes that the model used to decide when to buy is 100% accurate. 
  - This is to show what is possible, not what will actually happen.
- Costs are estimated at 0.4% per trade.
  - This is done to simplify calculations as the actual slippage and taxes can be difficult to calculate. 0.4% should be a good conservative number.
- Buying stock is done at the high price of the next bar.
  - Since we need the information in a bar to decide to buy, we aren't going to get the stock at that price. So, this simulation assumes that we will get it at the high price of the next bar. This is a conservative estimate.

---

## Inputs From This Run

The scanner was run with the following inputs (from `scanner_results_2026-03-07_11-11-36.txt`):


| Parameter              | Value                                                                    | What it means / Why I chose this                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **symbols**            | SPY, AAPL, AMD, PLTR, SOFI, RIVN, TQQQ, SOXL, MARA, NU, NOK (11 symbols) | These are the symbols that we scanned through.I had AI suggest symbols to look at that were high volume (because we want to minimize slippage), relatively low prices (because we have limited capital and might need to buy whole shares), and had a high Beta (because we need the price to move a lot to have opportunities to make a profit). |
| **take_profit_values** | 0.6%, 0.7%, 0.8%, 0.9%, 1%, 2%, 3%, 4%, 5%, 6%, 7% (12 levels)           | These are the take profit values we looped through.Anything less than 0.6% and we will probably lose money after accounting for slippage and taxes. Anything over 7% didn't seem to matter much when I was experimenting with the ranges. Most of the time, the maximum profit was found around 7%.                                              |
| **stop_loss_values**   | 0.5%, 1%, 1.5%, 2%, 2.5%, 3% (6 levels)                                  | These are the stop loss values we looped through.Profit will always rise with an increased stop loss, because this experiment assumes a model with 100% accuracy to show the maximum profit possible. But, that's not realistic. Capping it at a 3% loss seems reasonable for real world trading.                                                |
| **start_date**         | 2022-01-01                                                               | This is the starting date for the trading data we used.I went back to 2022 because that was a relatively bad year for stocks. I don't want to just look at good data.                                                                                                                                                                            |
| **end_date**           | 2025-12-31                                                               | This is the end date for the trading data we used.I don't want to use data too close to now to preserve this most recent data for back testing. If I had used the same data for back testing and choosing our take profit and stop loss percents, I would have added in look-ahead bias.                                                         |
| **cost_per_trade**     | 0.004 (0.4%)                                                             | This is an estimate of how much each trade would cost as a percent.It can be difficult to accurately estimate this. However, high volume stocks normally have a slippage of 0.1% and taxes could be estimated at 0.1%. I chose 0.4% to be conservative.                                                                                          |
| **n_jobs**             | 4                                                                        | This is the number of concurrent processes used.Doing grid search can be labor intensive. This let me get through it quicker.                                                                                                                                                                                                                     |


---

## Table and Column Reference

Use the sections below to document what each table in the raw scanner output means and what each column represents. Fill in or keep as reference when summarizing the run.

### Input parameters

The scanner prints the run configuration at the start of the output:


| Parameter                 | Meaning                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| `symbols`                 | List of tickers scanned.                                             |
| `take_profit_values`      | Take-profit levels (decimal, e.g. 0.01 = 1%) used in the grid.       |
| `stop_loss_values`        | Stop-loss levels (decimal) used in the grid.                         |
| `start_date` / `end_date` | Date range of the bar data.                                          |
| `cost_per_trade`          | Fee per trade (decimal, e.g. 0.004 = 0.4%) applied to count as cost. |
| `n_jobs`                  | Number of parallel workers for the grid. Used to speed up execution. |


---

### Data stats (per symbol)

For each symbol, the scanner prints a one-row **Data stats** table describing the cleaned bar dataset used for that symbol:


| Column          | Meaning                                                                         |
| --------------- | ------------------------------------------------------------------------------- |
| `start_date`    | First date in the cleaned data.                                                 |
| `end_date`      | Last date in the cleaned data.                                                  |
| `total_rows`    | Number of bars (rows) in the dataset.                                           |
| `lowest_price`  | Minimum low across all bars.                                                    |
| `highest_price` | Maximum high across all bars.                                                   |
| `average_price` | Mean of close prices.                                                           |
| `pct_imputed`   | Percentage of bars that are imputed (forward-propagated fill for missing data). |


---

### Full results grid (per symbol)

For each symbol, a table lists every (take_profit, stop_loss) combination with aggregated metrics:


| Column                    | Meaning                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `take_profit`             | TP level (decimal); exit when price reaches entry × (1 + take_profit).                                                                                                                                                                                                                                                                                   |
| `stop_loss`               | SL level (decimal); exit when price reaches entry × (1 − stop_loss).                                                                                                                                                                                                                                                                                     |
| `revenue`                 | Sum of PnL from trades that exited at take-profit (before cost) as a percent. (0.1 is 10%)This is done as a percent assuming that every trade that was made was done with the exact same amount of money. To calculate the real dollar amount, multiply this number by the number of trades and by the amount you would have invested in each trade. |
| `cost`                    | cost_per_trade × number of TP-exit trades as a percent. (0.1 is 10%)This is done as a percent assuming that every trade that was made was done with the exact same amount of money. To calculate the real dollar amount, multiply this number by the number of trades and by the amount you would have invested in each trade.                       |
| `profit`                  | revenue − cost. (0.1 is 10%)This is done as a percent assuming that every trade that was made was done with the exact same amount of money. To calculate the real dollar amount, multiply this number by the number of trades and by the amount you would have invested in each trade.                                                               |
| `percent_trades`          | Fraction of all simulated entry opportunities that exited at TP. (0.1 is 10%)                                                                                                                                                                                                                                                                            |
| `percent_days`            | Fraction of trading days that had at least one TP exit. (0.1 is 10%)                                                                                                                                                                                                                                                                                     |
| `avg_trade_duration_bars` | Average number of bars from entry to TP exit.Since the bars are in minutes. This can be thought of as the number of minutes between buying and selling.                                                                                                                                                                                                 |
| `imputed_count`           | Number of TP-exit trades where the entry bar was imputed.                                                                                                                                                                                                                                                                                                |


---

### Top 10 by profit (per symbol)

After the full grid, the scanner prints the **Top 10 (take_profit, stop_loss) rows by profit** for that symbol. The columns are the same as in the full results grid; only the subset and ordering differ.

---

## Results

### What Are We Looking For

First and foremost we are looking for the symbol, take profit and stop loss that will give us the highest profit. But, the highest profit may come with some cons. Below is a chart that speaks to the pros and cons represented in each field of the table.


| Field                     | What to Look For                                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `symbol`                  | The ticker symbol of the stock. Use this to identify which stock the result applies to. High-performing symbols may indicate stocks with better grid strategy fit in this sample period. |
| `take_profit`             | The take profit level (%) used in this simulation. Look for values that are not too high to be rarely hit, nor too low to cap upside.                                                    |
| `stop_loss`               | The stop loss level (%) used. Tighter (lower) values limit drawdowns but may realize losses prematurely; looser (higher) values may risk bigger losses.                                  |
| `revenue`                 | Total percent gain from trades that hit take profit, before subtracting cost. Higher is better, but compare with costs and imputed count.                                                |
| `cost`                    | Percent lost to transaction costs based on trade count. Lower is usually better, but due to the limitations of this simulation this cannot be minimized or maximized.                    |
| `profit`                  | We would like this to be the biggest possible ultimately.                                                                                                                                |
| `percent_trades`          | I would prefer that we are able to trade a reasonable amount of times. If it only shows up 1% of the time then we are sitting around doing nothing a lot.                                 |
| `percent_days`            | If all the trading occurs on a small percentage of the days maybe we are just hitting an anomaly in the data.                                                                            |
| `avg_trade_duration_bars` | Average time (in bars/minutes) for trades to reach TP. Faster durations can recycle capital more quickly, but may also mean TP/SL thresholds are too tight.                              |
| `imputed_count`           | Number of trades where the entry bar was a forward-propagated (imputed) bar. Lower is better, as imputed bars may not reflect real trading opportunities.                                |


### Best rows (top 10 per symbol, sorted by profit)

These are the top rows of the full table presented in the results text file.


| symbol | take_profit | stop_loss | revenue  | cost    | profit   | percent_trades | percent_days | avg_trade_duration_bars | imputed_count |
| ------ | ----------- | --------- | -------- | ------- | -------- | -------------- | ------------ | ----------------------- | ------------- |
| MARA   | 0.030       | 0.030     | 2374.890 | 316.652 | 2058.238 | 0.204473       | 0.868263     | 91.608959               | 0             |
| MARA   | 0.020       | 0.030     | 2533.420 | 506.684 | 2026.736 | 0.327183       | 0.968064     | 69.092926               | 0             |
| SOXL   | 0.020       | 0.030     | 2443.080 | 488.616 | 1954.464 | 0.315516       | 0.956088     | 87.322028               | 0             |
| MARA   | 0.030       | 0.025     | 2253.210 | 300.428 | 1952.782 | 0.193997       | 0.868263     | 88.699069               | 0             |
| MARA   | 0.020       | 0.025     | 2423.840 | 484.768 | 1939.072 | 0.313031       | 0.968064     | 66.413979               | 0             |
| SOXL   | 0.030       | 0.030     | 2214.300 | 295.240 | 1919.060 | 0.190647       | 0.816367     | 115.159301              | 0             |
| SOXL   | 0.020       | 0.025     | 2348.600 | 469.720 | 1878.880 | 0.303314       | 0.956088     | 84.929924               | 0             |
| MARA   | 0.040       | 0.030     | 2043.760 | 204.376 | 1839.384 | 0.131973       | 0.694611     | 108.969820              | 0             |
| SOXL   | 0.030       | 0.025     | 2117.550 | 282.340 | 1835.210 | 0.182317       | 0.816367     | 112.978168              | 0             |
| MARA   | 0.020       | 0.020     | 2250.820 | 450.164 | 1800.656 | 0.290686       | 0.968064     | 62.889320               | 0             |
| MARA   | 0.030       | 0.020     | 2070.300 | 276.040 | 1794.260 | 0.178249       | 0.868263     | 85.014578               | 0             |
| SOXL   | 0.020       | 0.020     | 2196.740 | 439.348 | 1757.392 | 0.283702       | 0.956088     | 81.581680               | 0             |
| MARA   | 0.040       | 0.025     | 1935.640 | 193.564 | 1742.076 | 0.124991       | 0.694611     | 106.528135              | 0             |
| SOXL   | 0.030       | 0.020     | 1955.010 | 260.668 | 1694.342 | 0.168322       | 0.816367     | 109.500821              | 0             |
| SOXL   | 0.040       | 0.030     | 1789.720 | 178.972 | 1610.748 | 0.115568       | 0.617764     | 134.478198              | 0             |
| MARA   | 0.040       | 0.020     | 1776.280 | 177.628 | 1598.652 | 0.114701       | 0.694611     | 103.605986              | 0             |
| MARA   | 0.020       | 0.015     | 1978.200 | 395.640 | 1582.560 | 0.255478       | 0.968064     | 58.256900               | 0             |
| SOXL   | 0.020       | 0.015     | 1954.640 | 390.928 | 1563.712 | 0.252436       | 0.956088     | 76.989799               | 0             |
| SOXL   | 0.040       | 0.025     | 1697.680 | 169.768 | 1527.912 | 0.109625       | 0.617764     | 132.116229              | 0             |
| SOXL   | 0.030       | 0.015     | 1712.280 | 228.304 | 1483.976 | 0.147424       | 0.816367     | 105.010898              | 0             |
| RIVN   | 0.020       | 0.030     | 1789.080 | 357.816 | 1431.264 | 0.231054       | 0.888224     | 100.417388              | 0             |
| RIVN   | 0.020       | 0.025     | 1751.400 | 350.280 | 1401.120 | 0.226188       | 0.888224     | 99.271623               | 0             |
| RIVN   | 0.020       | 0.020     | 1682.980 | 336.596 | 1346.384 | 0.217352       | 0.888224     | 97.331329               | 0             |
| RIVN   | 0.020       | 0.015     | 1558.380 | 311.676 | 1246.704 | 0.201260       | 0.888224     | 94.382898               | 0             |
| TQQQ   | 0.020       | 0.030     | 1531.140 | 306.228 | 1224.912 | 0.197742       | 0.730539     | 117.273313              | 0             |
| RIVN   | 0.030       | 0.030     | 1401.060 | 186.808 | 1214.252 | 0.120628       | 0.636727     | 125.082074              | 0             |
| TQQQ   | 0.020       | 0.025     | 1496.160 | 299.232 | 1196.928 | 0.193224       | 0.730539     | 116.257392              | 0             |
| RIVN   | 0.030       | 0.025     | 1364.730 | 181.964 | 1182.766 | 0.117500       | 0.636727     | 124.230485              | 0             |
| SOFI   | 0.020       | 0.030     | 1450.080 | 290.016 | 1160.064 | 0.187273       | 0.802395     | 111.314410              | 173           |
| TQQQ   | 0.020       | 0.020     | 1430.660 | 286.132 | 1144.528 | 0.184765       | 0.730539     | 114.292229              | 0             |
| SOFI   | 0.020       | 0.025     | 1423.220 | 284.644 | 1138.576 | 0.183804       | 0.802395     | 110.433988              | 173           |
| RIVN   | 0.030       | 0.020     | 1302.540 | 173.672 | 1128.868 | 0.112146       | 0.636727     | 122.605486              | 0             |
| SOFI   | 0.020       | 0.020     | 1370.120 | 274.024 | 1096.096 | 0.176947       | 0.802395     | 108.443012              | 173           |
| RIVN   | 0.010       | 0.030     | 1762.970 | 705.188 | 1057.782 | 0.455364       | 0.998004     | 62.234145               | 0             |
| TQQQ   | 0.020       | 0.015     | 1308.820 | 261.764 | 1047.056 | 0.169030       | 0.730539     | 111.253434              | 0             |
| RIVN   | 0.020       | 0.010     | 1308.400 | 261.680 | 1046.720 | 0.168976       | 0.888224     | 89.062412               | 0             |
| RIVN   | 0.010       | 0.025     | 1736.320 | 694.528 | 1041.792 | 0.448481       | 0.998004     | 60.938087               | 0             |


### Parameter Selection

Based on the results, I would stick with these parameters:


| symbol | recommended_take_profit | recommended_stop_loss |
| ------ | ----------------------- | --------------------- |
| MARA   | 2% - 4%                 | 2% - 3%               |
| SOXL   | 2% - 4%                 | 2% - 3%               |
| RIVN   | 2% - 4%                 | 2% - 3%               |


Ideally, we would find rows where the stop loss is less than the take profit. There are some rows that show this is possible.
