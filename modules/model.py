# Binomial Tree Option Pricing (nested list with for loop)
import math
import numpy as np
import pandas as pd
import datetime as dt
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
        if not self.checkIfDividend():
            return self.model(optionType, S, K, tau, sigma, r=r, q=0, N=N)
        beforeBreak = quoteDate
        return 0

    def estDivPrice(self, date: dt.datetime) -> int:
        year = int(str(date.year)[-2:])
        return round(0.015*year - 0.095,3)

    def checkIfDividend(self, tau: float, quoteDate: dt.datetime):
        numDays = tau*365
        endDate = quoteDate + dt.timedelta(days=numDays)

        filter = (quoteDate >= self.dividendHistory['DeclarationDate']) & (quoteDate <= self.dividendHistory['EffDate'])

        dates =  self.dividendHistory[filter]
        print(dates)
        possible_dates = [(7,2), (8,5), (7,8), (6,11)]
        numberOfYears = math.ceil(tau)
        testDate = quoteDate
        result = []
        if len(dates):
            testDate = dates.iloc[0].EffDate + dt.timedelta(days=10)
            result.append((dates.iloc[0].EffDate.date(), dates.iloc[0].CashAmount))
        for i in range(numberOfYears):
            baseYear = quoteDate.year
            for dd,mm in possible_dates:
                candidate = dt.datetime(baseYear+i,mm,dd)
                if candidate >= testDate and candidate < endDate:
                    result.append((candidate, self.estDivPrice(candidate)))
        print(result)

        # Check for actual dividend dates
        # for _, dates in self.dividendHistory.iterrows():
        #     # print(dates['EffDate'], dates['DeclarationDate'])
        #     if quoteDate >= dates['DeclarationDate'] and quoteDate < dates['EffDate']:
        #         print(dates["DeclarationDate"])
        #         divDate, divPrice = dates["DeclarationDate"], dates["CashAmount"]

        # if not divDate and not divPrice:
        #     pass


if __name__ == '__main__':
    S = 50.0
    K = 50.0
    tau = 183/365
    sigma = 50
    r = 0.1
    q = 0.01
    mod = Model()
    print('European Value: {0}, American Option Value: {1}'.format(
        *mod.model('call', S, K, tau, sigma)))
    mod.checkIfDividend(tau, dt.datetime(year=2022, month=7, day=29))
