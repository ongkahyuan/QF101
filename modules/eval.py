import modules.get_data as getData
import pandas as pd
# import model
import yfinance as yf
from heapq import heappush, heappop

class Eval: 
    def __init__(self, df,startDate, endDate):
        self.ticker = yf.Ticker('AAPL')
        self.startDate = startDate
        self.endDate = endDate
        self.df = df
        self.dataGetter = getData.GetData(df, startDate, endDate)

    def markToMarket(self):
        # everyday, I can hold at most 5 long positions and 5 short positions
        long = []
        short = []
    
        day = self.startDate
        while day < self.endDate:
            options = self.dataGetter.getAllCurrentPrice(day)
            pass

            day += 1


        # just return the variance from market price??
        return 0
    
    def maximumLoss(self, modelCallPrice, modelPutPrice, marketcallPrice, marketPutPrice, start, expire):
        # first identify whether i will buy or sell the option
        # then figure out minLoss, maxLoss, minProfit, maxProfit given the movement of the stockprice 
        pass
if __name__ == "__main__":
    df = pd.read_csv("./trimmed.csv")
    # date = dt.datetime(2021, 2, 20)
    
    gd = Eval(df)
    # print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    print(gd.getSpecificCurrentPrice("2022-07-22", "2022-07-04", 70))
    pass