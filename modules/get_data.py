import pandas as pd

class GetData:
    def __init__(self, data: pd.DataFrame, startDate: str, endDate: str) -> None:

        data[" [QUOTE_DATE]"] = pd.to_datetime(data[" [QUOTE_DATE]"]) #datetime conversion from pandas
        data[" [EXPIRE_DATE]"] = pd.to_datetime(data[" [EXPIRE_DATE]"]) #added

        start = pd.to_datetime(startDate)
        end = pd.to_datetime(endDate)

        self.data = data[(data[" [QUOTE_DATE]"]>= start) & (data[" [EXPIRE_DATE]"]<= end)] #datetime comparison


    def getModelParams(self, frames: pd.DataFrame):
        def genTau(row):
            return row[" [DTE]"]/365
        def genSigma(row):
            return (row[" [C_VEGA]"] + row[" [P_VEGA]"])/2

        frames.rename({" [UNDERLYING_LAST]": 'S',
                       " [STRIKE]": 'K',
                       " [C_BID]": 'c_bid',
                       " [C_ASK]": 'c_ask',
                       " [P_BID]": 'p_bid',
                       " [P_ASK]": 'p_ask', 
                       " [EXPIRE_DATE]": 'expire_date'
                       }, axis=1, inplace=True)
        frames['tau'] = frames.apply(lambda row: genTau(row), axis= 1)
        frames['sigma'] = frames.apply(lambda row: genSigma(row), axis= 1)

        col_list = ['S', 'K', 'tau', 'sigma', 'c_bid', 'c_ask', 'p_bid', 'p_ask', 'expire_date']
        return frames[col_list]

    def getSpecificCurrentPrice(self, expDate: str, quoteDate: str, strikePrice: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        
        self.data[" [QUOTE_DATE]"] = pd.to_datetime(self.data[" [QUOTE_DATE]"]) #added
        self.data[" [EXPIRE_DATE]"] = pd.to_datetime(self.data[" [EXPIRE_DATE]"]) #added


        current = self.data.loc[
            (self.data[" [QUOTE_DATE]"] == quoteDate) & 
            (self.data[" [EXPIRE_DATE]"] == expDate) & 
            (self.data[" [STRIKE]"] == strikePrice)]
        current = self.getModelParams(current)

        others = self.data.loc[(self.data[" [QUOTE_DATE]"] == quoteDate)]

        others = others.drop(others[
            (others[" [QUOTE_DATE]"] == quoteDate) & 
            (others[" [EXPIRE_DATE]"] == expDate) & 
            (others[" [STRIKE]"] == strikePrice)].index)
        others = self.getModelParams(others)
        # print(current, "\n", others)
        return current, others

    def getAllCurrentPrice(self, quoteDate: str)-> pd.DataFrame:

        self.data[" [QUOTE_DATE]"] = pd.to_datetime(self.data[" [QUOTE_DATE]"]) #added

        res = self.data.loc[(self.data[" [QUOTE_DATE]"] == quoteDate)]
        res = self.getModelParams(res)
        print(res)
        return res

if __name__ == "__main__":
    df = pd.read_csv("./trimmed.csv")
    # date = dt.datetime(2021, 2, 20)
    
    gd = GetData(df, pd.to_datetime("2022-07-01"), pd.to_datetime("2022-08-01")) #added
    # print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    print(gd.getSpecificCurrentPrice(pd.to_datetime("2022-07-22"), pd.to_datetime("2022-07-04"), 70)) #added
