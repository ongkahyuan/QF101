import yfinance as yf
from datetime import datetime

ticker = yf.Ticker('AAPL')

history = ticker.history(period='1d', start='2022-7-1', end='2022-8-1')


date_time = datetime(2022,7,4).strftime("%Y-%m-%d")
print(date_time)
print(history.loc['2022-07-01'])
print(history.loc[datetime(2022,7,1)])