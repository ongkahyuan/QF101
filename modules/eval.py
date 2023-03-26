# import modules.get_data as getData
import get_data as getData
import pandas as pd
# import model
import yfinance as yf
from heapq import heappush, heappop
from datetime import datetime, timedelta
import model


class Eval:
    def __init__(self, df, startDate: datetime, endDate: datetime):
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

        underpricing = 0
        overpricing = 0
        contracts = 0
        while day < self.endDate:
            options = self.dataGetter.getAllCurrentPrice(day)

            # todo: vectorise this so that parallel computation can happen
            for index, row in options.iterrows():
                euro_c, euro_p, amer_c, amer_p = model.model(
                    row.S, row.K, row.tau, row.sigma)
                underpricing += max(row.c_ask-amer_c, 0) + \
                    max(row.p_ask-amer_p, 0)
                overpricing += max(amer_c - row.c_ask, 0) + \
                    max(amer_p-row.p_ask, 0)
                contracts += 1
            day += timedelta(days=1)

        # just return the variance from market price??
        return "Overpricing per contract: ", overpricing/contracts, "Underpricing per contract: ", underpricing/contracts

    def maximumLoss(self, modelCallPrice, modelPutPrice, marketcallPrice, marketPutPrice, start, expire):
        # first identify whether i will buy or sell the option
        # then figure out minLoss, maxLoss, minProfit, maxProfit given the movement of the stockprice
        pass


if __name__ == "__main__":
    df = pd.read_csv(
        "./trimmed.csv", parse_dates=[" [EXPIRE_DATE]", " [QUOTE_DATE]"])
    # date = dt.datetime(2021, 2, 20)

    evalObj = Eval(df, datetime(2022, 7, 1), datetime(2022, 8, 1))

    evalObj.markToMarket()

    # print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    pass
