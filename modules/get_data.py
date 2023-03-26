import pandas as pd
from datetime import datetime


class GetData:
    def __init__(self, data: pd.DataFrame, startDate: datetime, endDate: datetime) -> None:
        self.data = data[(data[" [QUOTE_DATE]"] >= startDate)
                         & (data[" [EXPIRE_DATE]"] <= endDate)]
        self.startDate = startDate
        self.endDate = endDate

    def getModelParams(self, frames: pd.DataFrame):
        def genTau(row):
            return row[" [DTE]"] / 365

        def genSigma(row):
            return (row[" [C_VEGA]"] + row[" [P_VEGA]"]) / 2

        frames.rename(
            {
                " [UNDERLYING_LAST]": "S",
                " [STRIKE]": "K",
                " [C_BID]": "c_bid",
                " [C_ASK]": "c_ask",
                " [P_BID]": "p_bid",
                " [P_ASK]": "p_ask",
                " [EXPIRE_DATE]": "expire_date",
            },
            axis=1,
            inplace=True,
        )
        frames["tau"] = frames.apply(lambda row: genTau(row), axis=1)
        frames["sigma"] = frames.apply(lambda row: genSigma(row), axis=1)

        col_list = [
            "S",
            "K",
            "tau",
            "sigma",
            "c_bid",
            "c_ask",
            "p_bid",
            "p_ask",
            "expire_date",
        ]

        for col in col_list[:-1]:
            if frames[col].dtype == str:
                frames[col] = frames[col].str.strip()
            frames[col] = pd.to_numeric(frames[col], errors='coerce')

        return frames[col_list]

    def getSpecificCurrentPrice(self, expDate: datetime, quoteDate: datetime, strikePrice: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        current = self.data.loc[(self.data[" [QUOTE_DATE]"] == quoteDate) & (
            self.data[" [EXPIRE_DATE]"] == expDate) & (self.data[" [STRIKE]"] == strikePrice)]
        current = self.getModelParams(current)

        others = self.data.loc[(self.data[" [QUOTE_DATE]"] == quoteDate)]
        others = others.drop(
            others[
                (others[" [QUOTE_DATE]"] == quoteDate)
                & (others[" [EXPIRE_DATE]"] == expDate)
                & (others[" [STRIKE]"] == strikePrice)
            ].index
        )
        others = self.getModelParams(others)
        # print(current, "\n", others)
        return current, others

    def getAllCurrentPrice(self, quoteDate: datetime) -> pd.DataFrame:
        res = self.data.loc[(self.data[" [QUOTE_DATE]"] == quoteDate)]
        res = self.getModelParams(res)
        # print(res)
        return res


if __name__ == "__main__":
    df = pd.read_csv(
        "./trimmed.csv", parse_dates=[" [EXPIRE_DATE]", " [QUOTE_DATE]"])
    # date = datetime(2021, 2, 20)

    gd = GetData(df, datetime(2022, 7, 1), datetime(2022, 8, 1))
    gd.getAllCurrentPrice(datetime(2022, 7, 1))
    # gd.getAllCurrentPrice("2022-07-04")
    # print(gd.getSpecificCurrentPrice(datetime(
    #     2022, 7, 1), datetime(2022, 7, 1), 70.0))
