import pandas as pd
from datetime import datetime


class GetData:
    def __init__(self, data: pd.DataFrame, startDate: str, endDate: str) -> None:
        # apply functions
        def genTau(row):
            return row[" [DTE]"]/365

        # date conversion
        data[" [QUOTE_DATE]"] = pd.to_datetime(
            data[" [QUOTE_DATE]"])  # datetime conversion from pandas
        data[" [EXPIRE_DATE]"] = pd.to_datetime(
            data[" [EXPIRE_DATE]"])  # added

        start = pd.to_datetime(startDate)
        end = pd.to_datetime(endDate)

        # filter by date
        data = data[(data[" [QUOTE_DATE]"] >= start) & (
            data[" [EXPIRE_DATE]"] <= end)].copy(deep=True)

        # rename columns
        data.rename({" [UNDERLYING_LAST]": 'S',
                     " [STRIKE]": 'K',
                     " [C_BID]": 'c_bid',
                     " [C_ASK]": 'c_ask',
                     " [P_BID]": 'p_bid',
                     " [P_ASK]": 'p_ask',
                     " [C_IV]": 'c_vega',
                     " [P_IV]": 'p_vega',
                     " [EXPIRE_DATE]": 'expire_date',
                     " [QUOTE_DATE]": 'quote_date'
                     }, axis=1, inplace=True)

        # pre-compute tau and sigma
        data['tau'] = data.apply(lambda row: genTau(row), axis=1)
        data = data[data['tau'] > 0]

        # convert columns to numeric
        col_list = ['S', 'K', 'tau', 'c_bid', 'c_ask', 'p_bid',
                    'p_ask', 'c_vega', 'p_vega', 'expire_date', 'quote_date']
        for col in col_list[:-2]:
            if data[col].dtype == str:
                data[col] = data[col].str.strip()
            # frames[col] = pd.to_numeric(frames[col], errors='coerce').copy(deep=True)
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # set values
        self.data = data
        self.startDate = startDate
        self.endDate = endDate

    def getModelParams(self, frames: pd.DataFrame):
        col_list = ['S', 'K', 'tau', 'c_bid', 'c_ask', 'p_bid',
                    'p_ask', 'c_vega', 'p_vega', 'expire_date', 'quote_date']
        return frames[col_list].copy(deep=True)

    def getSpecificCurrentPrice(self, expDate: str, quoteDate: str, strikePrice: float):
        exp = pd.to_datetime(expDate)
        quote = pd.to_datetime(quoteDate)

        current = self.data.loc[
            (self.data["quote_date"] == quote) &
            (self.data["expire_date"] == exp) &
            (self.data["K"] == strikePrice)]
        current = self.getModelParams(current)

        others = self.data.loc[(self.data["quote_date"] == quote)]
        others = others.drop(others[
            (others["quote_date"] == quoteDate) &
            (others["expire_date"] == expDate) &
            (others["K"] == strikePrice)].index)
        others = self.getModelParams(others)
        # print(current, "\n", others)
        return current, others

    def getAllCurrentPrice(self, quoteDate: str) -> pd.DataFrame:
        quote = pd.to_datetime(quoteDate)
        res = self.data.loc[(self.data["quote_date"] == quote)]
        res = self.getModelParams(res)
        # print(res)
        return res


if __name__ == "__main__":
    df = pd.read_csv("./trimmed.csv", low_memory=False)
    # date = dt.datetime(2021, 2, 20)

    gd = GetData(df, pd.to_datetime("2022-07-01"),
                 pd.to_datetime("2022-08-01"))  # added
    print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    # print(gd.getSpecificCurrentPrice(pd.to_datetime("2022-07-22"),pd.to_datetime("2022-07-04"), 70))  # added
