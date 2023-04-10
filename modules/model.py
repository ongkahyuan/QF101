from __future__ import annotations
from typing import List

# Binomial Tree Option Pricing (nested list with for loop)
import math
import numpy as np
import pandas as pd
import datetime as dt
import pprint
from enum import Enum
# T-t -> tau

class OptionType(Enum):
    CALL = 0
    PUT = 1

class OptionStyle(Enum):
    EUROPEAN = 0
    AMERICAN = 1

class Model:

    def __init__(self):
        self.dividendHistory = pd.read_csv('./DivInfo.csv', parse_dates=['EffDate', 'DeclarationDate'])
        self.rfRates = pd.read_csv("./Treasury Yield 10 Years.csv", parse_dates=['Date'])

    def _calc_opt_price_model1(self, otype:OptionType, style:OptionStyle, S, K, tau, sigma, r=0.03, q=0.00, N=100):
        # Intermediate variables
        deltaT = tau/N
        u = math.exp(sigma*math.sqrt(deltaT))
        d = 1/u

        # This handles 0 sigma or very small sigma
        if u-d < 0.0001 or (math.exp((r-q)*deltaT)-d)/(u-d) > 1:
            return max(S-K, 0) if otype == OptionType.CALL else max(K-S, 0)

        # Intermediate variables (cont'd)
        p = (math.exp((r-q)*deltaT)-d)/(u-d)
        discount = math.exp(-r*deltaT)

        # Create binomial tree in the form of a 2d list
        tree = Model._generate_tree(N)

        # Populate terminal layer
        for j in range(N+1):
            tree[N][j] = max(0, S*(u**j)*(d**(N-j))-K) if otype == OptionType.CALL else max(0, K-S*(u**j)*(d**(N-j)))

        # Calculate Option Prices
        for i in range(N-1, 0-1, -1):
            for j in range(i+1):
                option_price = discount * p*tree[i+1][j+1] + (1-p)*tree[i+1][j]
                if style == OptionStyle.EUROPEAN:
                    tree[i][j] = option_price
                else: # American
                    currentStockPrice = S*(u**j)*(d**(i-j))
                    tree[i][j] = max(option_price, currentStockPrice-K) if otype == OptionType.CALL else max(option_price, K-currentStockPrice)
                
        # Return the value of the root node
        return tree[0][0]

    def _calc_opt_price_model2(self, otype:OptionType, style:OptionStyle, quote_date,  S, K, tau, sigma, r=0.03,  N=10, autoR=True):
        # obtain rolling average of risk free rate
        r = self.getRfRate(quote_date) if autoR else r

        # Intermediate variables
        dividend_dates = self._checkIfDividend(tau, quote_date)
        deltaT = tau/N
        u = math.exp(sigma*math.sqrt(deltaT))
        d = 1/u

        # This handles 0 sigma or very small sigma
        if u-d < 0.0001 or (math.exp(r*deltaT)-d)/(u-d) > 1:
            return math.exp(r*tau) * (max(S-K, 0) if otype==OptionType.CALL else max(K-S, 0))

        # Intermediate variables (cont'd)
        p = (math.exp(r*deltaT)-d)/(u-d)
        discount = math.exp(-r*tau/N)

        trials:List[int] = []
        current_date = quote_date
        for date, amount in dividend_dates + [(quote_date+dt.timedelta(tau*365), 0)]:
            trials.append(round((date-current_date).days*N/(tau*365)))

        # To ensure the last tree has no dividend correction
        dividend_dates.append((None, None))
        current_date = date
        trees = []
        current = 1
        for trial in trials:
            layer = []
            for i in range(current):
                layer.append(Model._generate_tree(trial))
            trees.append(layer)
            current *= (trial+1)

        # Initialise first node of first layer of first tree with initial stock price
        trees[0][0][0][0] = S
        for layerIndex in range(len(trees)):
            self._generateHeadAndTail(
                u, d, trees, layerIndex, dividend_dates[layerIndex][1])

        # Calculate final option prices for last layer
        for tree in trees[-1]:
            for i in range(len(tree)):
                node = tree[-1][i]
                tree[-1][i] = max(0, node-K) if otype == OptionType.CALL else max(0, K-node)

        # Backprop option payoff
        for layerInd in range(len(trees)-1, -1, -1):
            layer = trees[layerInd]
            for treeInd in range(len(layer)-1, -1, -1):
                tree = layer[treeInd]
                baseStockPrice = tree[0][0]
                for i in range(len(tree)-2, -1, -1):
                    for j in range(i+1):
                        currentStockPrice = baseStockPrice*(u**j)*(d**(i-j))
                        if otype == OptionType.CALL:
                            futurePayoff = discount * (p*tree[i+1][j+1]+(1-p)*tree[i+1][j])
                            payoff = max(futurePayoff, max(currentStockPrice-K, 0))
                        else:
                            futurePayoff = discount * (p*tree[i+1][j+1]+(1-p)*tree[i+1][j])
                            payoff = max(futurePayoff, max(K-currentStockPrice, 0))
                        tree[i][j] = payoff
                
                if layerInd != 0:
                    # put the payoff in the previous layer
                    prevLayer = trees[layerInd-1]
                    pTreeLen = len(prevLayer[0])
                    pTreeIndex = treeInd // pTreeLen
                    pNodeIndex = treeInd % pTreeLen

                    trees[layerInd-1][pTreeIndex][-1][pNodeIndex] = layer[treeInd][0][0]

        return trees[0][0][0][0]


    def calculate_option_price(self, type:OptionType, style:OptionStyle, S, K, tau, sigma, r=0.03, q=0.0, N=10, quote_date = None):
        dividend_dates = self._checkIfDividend(tau, quote_date) if quote_date is not None else []
        if not dividend_dates:
            return self._calc_opt_price_model1(type, style, S, K, tau, sigma, r, q, N)
        else:
            return self._calc_opt_price_model2(type, style, quote_date, S, K, tau, sigma, r, N)

    # HELPER FUNCTIONS
    @staticmethod
    def _generate_tree(num_layers):
        return [[0.0 for _ in range(i+1)] for i in range(num_layers+1)]

    # For model 2
    def _generateHeadAndTail(self, u, d, trees, layerIndex, div):
        layer = trees[layerIndex]
        for treeIndex, tree in enumerate(layer):
            # print(trees[layerIndex][treeIndex])
            N = len(tree[-1])
            for i in range(N):
                nextSSP = tree[0][0]*u**i*d**(N-i)
                trees[layerIndex][treeIndex][-1][i] = nextSSP
                if layerIndex < len(trees)-1:
                    trees[layerIndex+1][treeIndex *
                                        len(tree) + i][0][0] = nextSSP - div
        if layerIndex != 0:
            # set the prev layer tail option price
            pass

    def _estDivPrice(self, date: dt.datetime) -> int:
        offset = 0
        if date.month == 2:
            offset = -1
        year = int(str(date.year + offset)[-2:])
        return round(0.015*year - 0.095, 3)

    def _checkIfDividend(self, tau: float, quoteDate: dt.datetime):
        numDays = tau*365
        endDate = quoteDate + dt.timedelta(days=numDays)

        filter = (quoteDate >= self.dividendHistory['DeclarationDate']) & (
            quoteDate <= self.dividendHistory['EffDate'])
        dates = self.dividendHistory[filter]

        possible_dates = [(7, 2), (8, 5), (7, 8), (6, 11)]
        numberOfYears = math.ceil(tau)

        cutoffDate = quoteDate
        result = []
        if len(dates):
            cutoffDate = dates.iloc[0].EffDate + dt.timedelta(days=10)
            result.append((dt.datetime.combine(dates.iloc[0].EffDate.date(
            ), dt.datetime.min.time()), dates.iloc[0].CashAmount))
        for i in range(numberOfYears):
            baseYear = quoteDate.year
            for dd, mm in possible_dates:
                candidate = dt.datetime(baseYear+i, mm, dd)
                if candidate >= cutoffDate and candidate < endDate:
                    result.append((candidate, self._estDivPrice(candidate)))
        return result
    
    def getRfRate(self, date: dt.datetime, rolling_days=30):
        # print(self.rfRates)
        rates = self.rfRates.iloc[(
            self.rfRates['Date']-date).abs().argsort()[:rolling_days]].copy()
        return rates.loc[:, 'High'].mean()/100
        
    # <><> Unused Functions
    def binomial_tree_option_pricing(self, S, K, r, q, tau, sigma, N=100):
        """
        Currently overpricing the option prices slightly, might be because there is no discount factor
        """
        delta_t = tau / N
        u = np.exp(sigma * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp((r - q) * delta_t) - d) / (u - d)

        # Build the binomial tree
        stock_price = np.zeros((N+1, N+1))
        stock_price[0, 0] = S
        for i in range(1, N+1):
            stock_price[i, 0] = stock_price[i-1, 0] * u
            for j in range(1, i+1):
                stock_price[i, j] = stock_price[i-1, j-1] * d

        # Calculate option prices at expiration
        call_option = np.maximum(stock_price[N] - K, 0)
        put_option = np.maximum(K - stock_price[N], 0)

        # Work backwards through the tree to calculate option prices at each node
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                call_option[j] = (p * call_option[j+1] + (1-p) *
                                  call_option[j]) * np.exp(-r * delta_t)
                put_option[j] = (p * put_option[j+1] + (1-p) *
                                 put_option[j]) * np.exp(-r * delta_t)
                stock_price[i, j] = stock_price[i, j]

                # Check for early exercise (American options only)
                exercise_value = K - \
                    stock_price[i, j] if put_option[j] > call_option[j] else stock_price[i, j] - K
                if exercise_value > 0:
                    call_option[j] = exercise_value if call_option[j] < exercise_value else call_option[j]
                    put_option[j] = exercise_value if put_option[j] < exercise_value else put_option[j]

        # Calculate final option prices
        american_call_price = call_option[0]
        european_call_price = call_option[0] * np.exp(-r * tau)
        american_put_price = put_option[0]
        european_put_price = put_option[0] * np.exp(-r * tau)

        return (european_call_price, european_put_price, american_call_price, american_put_price)



if __name__ == '__main__':
    S = 50.0
    K = 50.0
    tau = 130/365
    # tau = 183/365
    sigma = 0.3
    r = 0.1
    q = 0.01
    mod = Model()
    # print('European Value: {0}, American Option Value: {1}'.format(
    # *mod.model('call', S, K, tau, sigma)))
    # mod._checkIfDividend(tau, dt.datetime(year=2022, month=7, day=29))
    # v1 = mod.calculate_option_price('call', S, K, tau, sigma)
    v2 = mod.calculate_option_price(OptionType.CALL, OptionStyle.AMERICAN, S, K, tau, sigma, r=0.03, q=0.0, N=40, quote_date = dt.datetime(
        year=2022, month=7, day=29))
    print( v2)
    # price = mod._estDivPrice(dt.datetime(year=2020, month=11, day=5))
    # print(price)
    print(mod.getRfRate(dt.datetime(2022, 7, 1)))
