from __future__ import annotations
from typing import List

import pandas as pd


class ModelData:
    default_parameters_col_names = ['S', 'K', 'tau', 'c_bid', 'c_ask', 'p_bid',
                    'p_ask', 'c_vega', 'p_vega', 'expire_date', 'quote_date']
    def __init__(self, raw_data: pd.DataFrame, start_date: str, end_date: str, parameters_col_names:List[str] = None) -> None:
        """
        start_date:             First date of data contained
        end_date:               Last date of data contained
        raw_data:               Raw pandas df that was given in init
        data:                   Cleaned pandas df
        parameters_col_names:   List of columns which contain model parameters data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = raw_data
        self.data = self._getCleanedRawDF()
        self.parameters_col_names = list(ModelData.default_parameters_col_names) if parameters_col_names is None else list(parameters_col_names)

    # Cleans a given data_frame; Requires self.start_date and self.end_date
    def _getCleanedRawDF(self) -> pd.DataFrame:
        raw_data = self.raw_data
        # date conversion - datetime conversion from pandas
        raw_data[" [QUOTE_DATE]"] = pd.to_datetime(raw_data[" [QUOTE_DATE]"])
        raw_data[" [EXPIRE_DATE]"] = pd.to_datetime(raw_data[" [EXPIRE_DATE]"])

        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)

        # filter by date
        data = raw_data[(raw_data[" [QUOTE_DATE]"] >= start_dt) & (raw_data[" [EXPIRE_DATE]"] <= end_dt)].copy(deep=True)

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
        data['tau'] = data.apply(lambda row: row[" [DTE]"]/365, axis=1)
        data = data[data['tau'] > 0]

        # convert columns to numeric
        col_list = ['S', 'K', 'tau', 'c_bid', 'c_ask', 'p_bid',
                    'p_ask', 'c_vega', 'p_vega', 'expire_date', 'quote_date']
        for col in col_list[:-2]:
            if data[col].dtype == str:
                data[col] = data[col].str.strip()
            data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    # Returns data of parameter columns of the given df
    def getModelParamData(self, df: pd.DataFrame):
        return df[self.parameters_col_names].copy(deep=True)

    # Returns data of parameter columns for a given quote date
    def getAllCurrentPrice(self, quote_date: str) -> pd.DataFrame:
        quote_dt = pd.to_datetime(quote_date)
        res = self.data.loc[(self.data["quote_date"] == quote_dt)]
        res = self.getModelParamData(res)
        return res


    # <><><> Unused Functions
    # def getSpecificCurrentPrice(self, expDate: str, quoteDate: str, strikePrice: float):
    #     exp = pd.to_datetime(expDate)
    #     quote = pd.to_datetime(quoteDate)

    #     current = self.data.loc[
    #         (self.data["quote_date"] == quote) &
    #         (self.data["expire_date"] == exp) &
    #         (self.data["K"] == strikePrice)]
    #     current = self.getModelParamData(current)

    #     others = self.data.loc[(self.data["quote_date"] == quote)]
    #     others = others.drop(others[
    #         (others["quote_date"] == quoteDate) &
    #         (others["expire_date"] == expDate) &
    #         (others["K"] == strikePrice)].index)
    #     others = self.getModelParamData(others)
    #     # print(current, "\n", others)
    #     return current, others

if __name__ == "__main__":
    df = pd.read_csv("./trimmed.csv", low_memory=False)
    # date = dt.datetime(2021, 2, 20)

    gd = ModelData(df, pd.to_datetime("2022-07-01"),
    pd.to_datetime("2022-08-01"))  # added
    print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    # print(gd.getSpecificCurrentPrice(pd.to_datetime("2022-07-22"),pd.to_datetime("2022-07-04"), 70))  # added
