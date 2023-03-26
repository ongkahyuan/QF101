import modules.eval as eval
import modules.get_data as getData
import modules.model as model
import modules.vis as vis
import pandas as pd
import datetime #imported date time library

df = pd.read_csv("./trimmed.csv")

gd = getData.GetData(df, pd.to_datetime("2022-07-01"), pd.to_datetime("2022-08-01")) #changed string to date time format

# res = []

# for contract in data:
#     res.append(eval.eval(contract))

# vis.vis(res)