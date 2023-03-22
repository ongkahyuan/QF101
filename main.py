import modules.eval as eval
import modules.get_data as getData
import modules.model as model
import modules.vis as vis

data = getData.getData()

res = []

for contract in data:
    res.append(eval.eval(contract))

vis.vis(res)