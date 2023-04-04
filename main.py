from __future__ import annotations
from typing import List
from datetime import datetime #imported date time library

import pandas as pd

import modules.eval as evaluate
import modules.get_data as getData
import modules.model as model
import modules.vis as vis


startDate = datetime(2022,7,1)
endDate = datetime(2022,8,1)
df = pd.read_csv("./trimmed.csv", low_memory=False)

gd = getData.GetData(df, startDate, endDate) #changed string to date time format

evaluator = evaluate.Eval(df,startDate,endDate)

print(evaluator.compareModeltoMarket())
# res = []

# for contract in data:
#     res.append(eval.eval(contract))

# vis.vis(res)