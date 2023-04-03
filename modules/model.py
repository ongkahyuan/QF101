# Binomial Tree Option Pricing (nested list with for loop)
import math
import numpy as np
import pandas as pd
import datetime as dt
import pprint
# T-t -> tau


class Model:

    def __init__(self):
        self.dividendHistory = pd.read_csv(
            './DivInfo.csv', parse_dates=['EffDate', 'DeclarationDate'])

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

    def model(self, optionType, S, K, tau, sigma, r=0.034, q=0.017, N=100):
        if optionType != 'call' and optionType != 'put':
            return 'Invalid option type', 'Invalid option type'

        deltaT = tau/N
        u = math.exp(sigma*math.sqrt(deltaT))
        d = 1/u

        # This handles 0 sigma or very small sigma
        if u-d < 0.0001 or (math.exp((r-q)*deltaT)-d)/(u-d) > 1:
            return (max(S-K, 0), max(S-K, 0)) if optionType == "call" else (max(K-S, 0), max(K-S, 0))

        p = (math.exp((r-q)*deltaT)-d)/(u-d)

        euro = [[0.0 for j in range(i+1)] for i in range(N+1)]
        amer = [[0.0 for j in range(i+1)] for i in range(N+1)]

        for j in range(N+1):
            if optionType == 'call':
                euro[N][j] = max(0, S*(u**j)*(d**(N-j))-K)
                amer[N][j] = max(0, S*(u**j)*(d**(N-j))-K)
            else:
                euro[N][j] = max(0, K-S*(u**j)*(d**(N-j)))
                amer[N][j] = max(0, K-S*(u**j)*(d**(N-j)))

        # 3.
        discount = math.exp(-r*deltaT)
        for i in range(N-1, 0-1, -1):
            for j in range(i+1):
                if optionType == 'call':
                    europeanOptFuturePayoff = discount * \
                        (p*euro[i+1][j+1]+(1-p)*euro[i+1][j])
                    americanOptFuturePayoff = discount * \
                        (p*amer[i+1][j+1]+(1-p)*amer[i+1][j])
                else:
                    europeanOptFuturePayoff = discount * \
                        (p*euro[i+1][j+1]+(1-p)*euro[i+1][j])
                    americanOptFuturePayoff = discount * \
                        (p*amer[i+1][j+1]+(1-p)*amer[i+1][j])

                currentStockPrice = S*(u**j)*(d**(i-j))
                if optionType == 'call':
                    euro[i][j] = europeanOptFuturePayoff
                    amer[i][j] = max(americanOptFuturePayoff,
                                     max(currentStockPrice-K, 0))
                else:
                    euro[i][j] = europeanOptFuturePayoff
                    amer[i][j] = max(americanOptFuturePayoff,
                                     max(K-currentStockPrice, 0))
        # 4.
        euro_value = euro[0][0]
        amer_value = amer[0][0]
        return euro_value, amer_value

    def modelv2(self, optionType, quoteDate,  S, K, tau, sigma, r=0.034,  N=10):
        dividedDates = self.checkIfDividend(tau, quoteDate)
        deltaT = tau/N
        u = math.exp(sigma*math.sqrt(deltaT))
        d = 1/u

        # This handles 0 sigma or very small sigma
        if u-d < 0.0001 or (math.exp((r*deltaT)-d)/(u-d))> 1:
            return (max(S-K, 0)) if optionType == "call" else ( max(K-S, 0))
        
        p = (math.exp(r*deltaT)-d)/(u-d)

        if not dividedDates:
            return self.model(optionType, S, K, tau, sigma, r=r, q=0, N=N)[1]

        trials = []
        currentDate = quoteDate
        for date, amount in dividedDates + [(quoteDate+dt.timedelta(tau*365), 0)]:
            trials.append(round((date-currentDate).days*N/(tau*365)))

        # To ensure the last tree has no dividend correction
        dividedDates.append((None, None))
        currentDate = date
        trees = []
        current = 1
        for trial in trials:
            layer = []
            for i in range(current):
                layer.append(self.treeGenerator(trial))
            trees.append(layer)
            current *= (trial+1)

        pp = pprint.PrettyPrinter(indent=4)
        # Initialise first node of first layer of first tree with initial stock price
        trees[0][0][0][0] = S
        for layerIndex in range(len(trees)):
            self.generateHeadAndTail(
                u, d, trees, layerIndex, dividedDates[layerIndex][1])

        # Calculate final option prices for last layer
        for tree in trees[-1]:
            for i in range(len(tree)):
                node = tree[-1][i]
                if optionType == 'call':
                    tree[-1][i] = max(0, node-K)
                else:
                    tree[-1][i] = max(0, K-node)

        # Backprop option payoff
        discount = math.exp(-r*tau/N)
        # print(f"len tree: {len(trees)}")
        for layerInd in range(len(trees)-1, -1, -1):
            # print(f"li: {layerInd}")
            layer = trees[layerInd]
            # self.backPropOptionPrices(optionType, tau/N, r, trees, ~i)
            for treeInd in range(len(layer)-1, -1, -1):
                tree = layer[treeInd]
                # print(f"ti: {treeInd}")
                # pp.pprint(tree)
                baseStockPrice = tree[0][0]
                for i in range(len(tree)-2, -1, -1):
                    for j in range(i+1):
                        currentStockPrice = baseStockPrice*(u**j)*(d**(i-j))
                        if optionType == 'call':
                            futurePayoff = discount * \
                                (p*tree[i+1][j+1]+(1-p)*tree[i+1][j])
                            payoff = max(futurePayoff, max(
                                currentStockPrice-K, 0))
                        else:
                            futurePayoff = discount * \
                                (p*tree[i+1][j+1]+(1-p)*tree[i+1][j])
                            payoff = max(futurePayoff, max(
                                K-currentStockPrice, 0))
                        tree[i][j] = payoff
                if layerInd != 0:
                    # put the payoff in the previous layer
                    prevLayer = trees[layerInd-1]
                    pTreeLen = len(prevLayer[0])
                    pTreeIndex = treeInd // pTreeLen
                    pNodeIndex = treeInd % pTreeLen

                    trees[layerInd-1][pTreeIndex][-1][pNodeIndex] = layer[treeInd][0][0]
                    # print(
                    #     f"pTI: {pTreeIndex}, cTI: {treeInd}, nTI: {pNodeIndex}")
                    # print("PIND", pTreeIndex)
        # pp.pprint(trees)
        # print(trees[0][0][0])
        # pp.pprint(trees[0][-1])
        # pp.pprint(trees[1][0][0])
        # pp.pprint(trees[1][1][0])
        # pr = []
        # pt = []

        # for i, layer in enumerate(trees):
        #     if i == 1:
        #         for j, tree in enumerate(layer):
        #             for k in range(len(tree)):
        #                 pr.append(round(tree[-1][k]))
        #     if i == 2:
        #         for j, tree in enumerate(layer):
        #             pt.append(round(tree[0][0]))
        # print(f"CHECK: {pr, pt}")

        return trees[0][0][0][0]

    def treeGenerator(self, trials):
        return [[0.0 for j in range(i+1)] for i in range(trials+1)]

    def generateHeadAndTail(self, u, d, trees, layerIndex, div):
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

    def backPropOptionPrices(self, optionType, deltaT, r, trees, layerIndex):
        discount = math.exp(-r*deltaT)
        layer = trees[layerIndex]
        for treeIndex in range(len(layer)):
            for nodeI in range(len(layer[treeIndex])):
                for nodeJ in range(nodeI+1):
                    payoff = 0
                    if optionType == 'call':
                        payoff = discount*p

        if layerIndex != 0:
            # set the prev layer tail option price
            pass

    def estDivPrice(self, date: dt.datetime) -> int:
        year = int(str(date.year)[-2:])
        return round(0.015*year - 0.095, 3)

    def checkIfDividend(self, tau: float, quoteDate: dt.datetime):
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
                    result.append((candidate, self.estDivPrice(candidate)))
        return result


if __name__ == '__main__':
    S = 50.0
    K = 50.0
    tau = 130/365
    # tau = 183/365
    sigma = 0.4
    r = 0.1
    q = 0.01
    mod = Model()
    # print('European Value: {0}, American Option Value: {1}'.format(
    # *mod.model('call', S, K, tau, sigma)))
    # mod.checkIfDividend(tau, dt.datetime(year=2022, month=7, day=29))
    v1 = mod.model('call', S, K, tau, sigma)
    v2 = mod.modelv2('call', dt.datetime(
        year=2022, month=7, day=29), S, K, tau, sigma)
    print(v1, v2)
