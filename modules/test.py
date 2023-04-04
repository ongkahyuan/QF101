from __future__ import annotations
from typing import List

import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

from tree_model import Node, BinomialTreeModel
from eval import Eval

def main():
    df = pd.read_csv("./trimmed.csv", parse_dates=[" [EXPIRE_DATE]", " [QUOTE_DATE]"], low_memory=False)
    evalObj = Eval(df, datetime(2022, 7, 1), datetime(2022, 12, 1))
    dates = [evalObj.startDate + timedelta(days=i) for i in range((evalObj.endDate-evalObj.startDate).days)]
    overpricingc,overpricingp, underpricingc, underpricingp = evalObj.compareModeltoMarket()
    plt.plot(dates, overpricingc, label="Daily Overpricing % Spread per contract (Call)")
    plt.plot(dates, overpricingp, label="Daily Overpricing % Spread per contract (Put)")
    plt.plot(dates, underpricingc, label="Daily Underpricing % Spread per contract (Call)")
    plt.plot(dates, underpricingp, label="Daily Underpricing % Spread per contract (Put)")
    
    plt.legend()
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.xlabel('xlabel', fontsize=16)
    plt.show()

    model = BinomialTreeModel()

    pass



if __name__ == "__main__":
    main()