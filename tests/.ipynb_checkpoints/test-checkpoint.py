from datetime import datetime, timedelta, timezone
import os

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(
    api_key=os.getenv("APCA_API_KEY_ID"),
    secret_key=os.getenv("APCA_API_SECRET_KEY"),
)

end = datetime.now(timezone.utc) - timedelta(minutes=20)
start = end - timedelta(days=5)

request = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Minute,
    start=start,
    end=end,
)

bars = client.get_stock_bars(request)
df = bars.df
print(df)
print(df.columns)

df_save = df.reset_index()
df_save.to_csv("alpaca_sample_data.csv", index=False)