import modules.eval as eval
import modules.get_data as getData
import modules.model as model
import modules.vis as vis
import pandas as pd

df = pd.read_csv("./trimmed.csv")
gd = getData.GetData(df, "2022-07-01", "2022-08-01")

# res = []

# for contract in data:
#     res.append(eval.eval(contract))

# vis.vis(res)