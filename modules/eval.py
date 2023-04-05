# import modules.get_data as getData
import get_data as getData
import pandas as pd
from matplotlib import pyplot as plt
# import model
import yfinance as yf
from heapq import heappush, heappop
from datetime import datetime, timedelta
import model as model
import numpy as np
import sys
import math


class Eval:
    def __init__(self, df, startDate: datetime, endDate: datetime):
        self.ticker = yf.Ticker('AAPL')
        self.priceHistory = self.ticker.history(period='1d', start=startDate.strftime(
            "%Y-%m-%d"), end=endDate.strftime("%Y-%m-%d"))
        self.startDate = startDate
        self.endDate = endDate
        self.df = df
        self.dataGetter = getData.GetData(df, startDate, endDate)

    def compareModeltoMarketDistribution(self, threshold=2):
        currentDay = self.startDate
        modelPoints = []
        marketPoints = []
        while currentDay < self.endDate:
            options = self.dataGetter.getAllCurrentPrice(currentDay)
            if len(options) == 0:
                currentDay += timedelta(days=1)
                continue
            # options['model_c'] = options.apply(lambda row: model.Model().modelv2('call',row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega']), axis=1)
            # options['model_p'] = options.apply(lambda row: model.Model().modelv2('put', row['quote_date'],row['S'], row['K'], row['tau'], row['p_vega']), axis=1)
            options['model_c'] = options.apply(lambda row: model.Model().model('call', row['S'], row['K'], row['tau'], row['c_vega'])[1], axis=1)
            options['model_p'] = options.apply(lambda row: model.Model().model('put', row['S'], row['K'], row['tau'], row['p_vega'])[1], axis=1)
            modelPoints += list(options['model_c']) + list(options['model_p'])
            marketPoints += list(options['c_ask']) + list(options['p_ask'])
            currentDay += timedelta(days=1)

        modelPoints = list(filter(lambda x: x > 5, modelPoints))
        marketPoints = list(filter(lambda x: x > 5, marketPoints))
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(modelPoints, bins=30, label="model", color="orange")
        axs[1].hist(marketPoints, bins=30, label="market")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        return

    def tradeUntilExpiry(self, spread=0.2, rebalancing=True):
        '''
        everyday i will buy the top 5 underpriced contracts (to buy, look for asking price)
        and sell the top 5 overpriced contracts (to sell, look for bidding price)
        assume: 0.2 cents spread (aka commission to the broker)

        for every contract sold, currentBalance += (optionPrice - spread)
        for every contract bought, currentBalance -= (marketAskingPrice + spread)

        for every call option SOLD, i need to know what the HIGH is for the day -> HIGH - Strike price = loss
        for every put option SOLD, i also need to know what the LOW is for the day -> Strike price - LOW = loss
        (vice versa for options BOUGHT)
        '''

        currentDay = self.startDate
        losses = {"max_loss_per_call_sold": 0, "max_loss_per_put_sold": 0,
                  "total_call_losses": 0, "total_put_losses": 0}
        profit = {"min_profit_per_call_bought": sys.maxsize,
                  "min_profit_per_put_bought": sys.maxsize, "total_call_profit": 0, "total_put_profit": 0}

        columns = list(self.dataGetter.getAllCurrentPrice(currentDay).columns)
        columns += ['underpriced', 'overpriced', 'min_profit', 'max_loss']
        callsBought = pd.DataFrame(columns=columns)
        putsBought = pd.DataFrame(columns=columns)
        callsSold = pd.DataFrame(columns=columns)
        putsSold = pd.DataFrame(columns=columns)
        modelObj = model.Model()
        currentBalance = 0
        dailyBalance = []
        # index 0 for calls, index 1 for puts
        tradeCounter = [0, 0, 0, 0]
        while currentDay <= self.endDate:
            options = self.dataGetter.getAllCurrentPrice(currentDay)

            # if there are options available, I have to consider purchasing them
            if len(options) > 0:
                stockPrice = options.S.max()
                # use more accurate data if available
                if currentDay in self.priceHistory:
                    history = self.priceHistory.loc[currentDay]
                    stockHigh, stockLow = history.High, history.Low
                else:
                    stockHigh, stockLow = stockPrice, stockPrice

                # print('Date:', currentDay, '\nStock Price:', stockPrice)
                options['model_c'] = options.apply(lambda row: model.Model().modelv2(
                    'call', row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega'],autoR=False), axis=1)
                options['model_p'] = options.apply(lambda row: model.Model().modelv2(
                    'put', row['quote_date'], row['S'], row['K'], row['tau'], row['p_vega'], autoR=False), axis=1)

                # overpriced: sell because people are willing to pay for higher price that what we think they are worth
                options['c_overpriced'] = (options.c_bid - options.model_c)
                options['p_overpriced'] = (options.p_bid - options.model_p)
                # underpriced: buy because we think that people are selling for less than what we think they are worth
                options['c_underpriced'] = (options.model_c - options.c_ask)
                options['p_underpriced'] = (options.model_p - options.p_ask)

                # give me top 5 that if the difference is greater than the spread
                top5overpricedCallOptions = options[options['c_overpriced'] > spread].sort_values(
                    'c_overpriced', ascending=False).head()
                top5overpricedPutOptions = options[options['p_overpriced'] > spread].sort_values(
                    'p_overpriced', ascending=False).head()
                top5underpricedCallOptions = options[options['c_underpriced'] > spread].sort_values(
                    'c_underpriced', ascending=False).head()
                top5underpricedPutOptions = options[options['p_underpriced'] > spread].sort_values(
                    'p_underpriced', ascending=False).head()
                # min profit assumes that we exercised the option at the worst possible time
                top5underpricedCallOptions['min_profit'] = top5underpricedCallOptions.apply(
                    lambda row: max(stockHigh-row['K'], 0), axis=1)
                top5underpricedPutOptions['min_profit'] = top5underpricedPutOptions.apply(
                    lambda row: max(row['K']-stockLow, 0), axis=1)
                # max_loss assumes that the buyer exercised the option at the best possible time
                top5overpricedCallOptions['max_loss'] = top5overpricedCallOptions.apply(
                    lambda row: max(stockHigh-row['K'], 0), axis=1)
                top5overpricedPutOptions['max_loss'] = top5overpricedPutOptions.apply(
                    lambda row: max(row['K']-stockLow, 0), axis=1)

                callsBought = pd.concat(
                    [callsBought, top5underpricedCallOptions])
                putsBought = pd.concat([putsBought, top5underpricedPutOptions])
                callsSold = pd.concat([callsSold, top5overpricedCallOptions])
                putsSold = pd.concat([putsSold, top5overpricedPutOptions])

                # #update current balance after selling and buying options
                currentBalance -= (top5underpricedCallOptions.c_ask +
                                   spread).sum()
                currentBalance -= (top5underpricedPutOptions.p_ask +
                                   spread).sum()
                currentBalance += (top5overpricedCallOptions.c_bid -
                                   spread).sum()
                currentBalance += (top5overpricedPutOptions.c_bid -
                                   spread).sum()

                # check positions PnL update worst possible profit and highest possible loss
                callsBought['min_profit'] = callsBought.apply(
                    lambda row: min(row['min_profit'], stockHigh-row['K']), axis=1)
                putsBought['min_profit'] = putsBought.apply(
                    lambda row: min(row['min_profit'], row['K']-stockLow), axis=1)
                callsSold['max_loss'] = callsSold.apply(lambda row: max(
                    stockHigh-row['K'], row['max_loss'], 0), axis=1)
                putsSold['max_loss'] = putsSold.apply(lambda row: max(
                    row['K']-stockLow, row['max_loss'], 0), axis=1)

            # add max loss and min profit close expired positions
            currentBalance -= callsSold[callsSold.expire_date <=
                                        currentDay]['max_loss'].sum()
            tradeCounter[2] += len(callsSold[callsSold.expire_date <= currentDay])
            losses['total_call_losses'] += callsSold[callsSold.expire_date <=
                                                     currentDay]['max_loss'].sum()
            losses['max_loss_per_call_sold'] = max(
                losses['max_loss_per_call_sold'], callsSold[callsSold.expire_date <= currentDay]['max_loss'].max())
            currentBalance -= putsSold[putsSold.expire_date <=
                                       currentDay]['max_loss'].sum()
            tradeCounter[3] += len(putsSold[putsSold.expire_date <= currentDay])
            losses['total_put_losses'] += putsSold[putsSold.expire_date <=
                                                   currentDay]['max_loss'].sum()
            losses['max_loss_per_put_sold'] = max(
                losses['max_loss_per_put_sold'], putsSold[putsSold.expire_date <= currentDay]['max_loss'].max())
            currentBalance += callsBought[callsBought.expire_date <=
                                          currentDay]['min_profit'].sum()
            tradeCounter[0] += len(callsBought[callsBought.expire_date <= currentDay])
            profit['total_call_profit'] += callsBought[callsBought.expire_date <=
                                                       currentDay]['min_profit'].sum()
            profit['min_profit_per_call_bought'] = min(
                profit['min_profit_per_call_bought'], callsBought[callsBought.expire_date <= currentDay]['min_profit'].min())
            currentBalance += putsBought[putsBought.expire_date <=currentDay]['min_profit'].sum()
            tradeCounter[1] += len(putsBought[putsBought.expire_date <= currentDay])
            profit['total_put_profit'] += putsBought[putsBought.expire_date <=
                                                     currentDay]['min_profit'].sum()
            profit['min_profit_per_put_bought'] = min(
                profit['min_profit_per_put_bought'], putsBought[putsBought.expire_date <= currentDay]['min_profit'].min())

            # close positions by filtering out expired options
            callsSold = callsSold[callsSold.expire_date > currentDay]
            putsSold = putsSold[putsSold.expire_date > currentDay]
            callsBought = callsBought[callsBought.expire_date > currentDay]
            putsBought = putsBought[putsBought.expire_date > currentDay]

            if rebalancing:
                # if i have more than 5 positions, remove the ones with the smallest spread
                if len(callsSold) > 5:
                    fifthBestCallOverpriced = callsSold.c_overpriced.nlargest(
                        5).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (callsSold[callsSold.c_overpriced <
                                       fifthBestCallOverpriced]['c_ask'] + spread).sum()
                    currentBalance -= (callsSold[callsSold.c_overpriced <
                                       fifthBestCallOverpriced]['max_loss']).sum()
                    tradeCounter[2] += len(callsSold[callsSold.c_overpriced <
                                           fifthBestCallOverpriced])
                    callsSold = callsSold.sort_values(
                        'c_overpriced', ascending=False).head()
                if len(putsSold) > 5:
                    fifthBestPutOverpriced = callsSold.p_overpriced.nlargest(
                        5).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (putsSold[putsSold.p_overpriced <
                                       fifthBestPutOverpriced]['p_ask'] + spread).sum()
                    currentBalance -= (putsSold[putsSold.p_overpriced <
                                       fifthBestPutOverpriced]['max_loss']).sum()
                    tradeCounter[3] += len(putsSold[putsSold.p_overpriced <
                                           fifthBestPutOverpriced])
                    putsSold = putsSold.sort_values(
                        'p_overpriced', ascending=False).head()
                if len(callsBought) > 5:
                    fifthBestCallUnderpriced = callsSold.c_underpriced.nlargest(
                        5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (callsBought[callsBought.c_underpriced <
                                       fifthBestCallUnderpriced]['c_bid'] - spread).sum()
                    currentBalance += (callsBought[callsBought.c_underpriced <
                                       fifthBestCallUnderpriced]['min_profit']).sum()
                    profit['total_call_profit'] += callsBought[callsBought.c_underpriced <
                                                               fifthBestCallUnderpriced]['min_profit'].sum()
                    profit['min_profit_per_call_bought'] = min(
                        profit['min_profit_per_call_bought'], callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['min_profit'].min())
                    tradeCounter[0] += len(
                        callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced])
                    callsBought = callsBought.sort_values(
                        'c_underpriced', ascending=False).head()
                if len(putsBought) > 5:
                    fifthBestPutUnderpriced = callsSold.p_underpriced.nlargest(
                        5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (putsBought[putsBought.p_underpriced <
                                       fifthBestPutUnderpriced]['p_bid'] - spread).sum()
                    currentBalance += (putsBought[putsBought.p_underpriced <
                                       fifthBestPutUnderpriced]['min_profit']).sum()
                    profit['total_profit_profit'] += (
                        putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['min_profit']).sum()
                    profit['min_profit_per_put_bought'] = min(
                        profit['min_profit_per_put_bought'], putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['min_profit'].min())
                    tradeCounter[1] += len(
                        putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced])
                    putsBought = putsBought.sort_values(
                        'p_underpriced', ascending=False).head()

            # print('EOD Balance:' ,currentBalance, '\n')
            dailyBalance.append(currentBalance)

            currentDay += timedelta(days=1)

        print("losses stats:")
        for stat in losses:
            print(stat, ': ', round(losses[stat], 2))
        print('profits stats:')
        for stat in profit:
            print(stat, ': ', round(profit[stat], 2))

        trades = ['calls bought', 'puts bought', 'calls sold', 'puts sold']
        for i in range(4):
            print(trades[i], ": ", tradeCounter[i])
        return dailyBalance

    def tradeUntilExercised(self, spread=0.2, rebalancing=True, threshold=1.5):
        '''
        exercised when S-K > optionprice + risk free rate
        '''

        currentDay = self.startDate
        columns = list(self.dataGetter.getAllCurrentPrice(currentDay).columns)
        columns += ['underpriced', 'overpriced']
        callsBought = pd.DataFrame(columns=columns)
        putsBought = pd.DataFrame(columns=columns)
        callsSold = pd.DataFrame(columns=columns)
        putsSold = pd.DataFrame(columns=columns)
        modelObj = model.Model()
        currentBalance = 0
        dailyBalance = []
        # index 0 for calls, index 1 for puts
        tradeCounter = [0, 0, 0, 0]
        gainsCounter = [0, 0, 0, 0]
        riskCounter = [0, 0, 0, 0]
        while currentDay <= self.endDate:
            options = self.dataGetter.getAllCurrentPrice(currentDay)

            # if there are options available, I have to consider purchasing them
            if len(options) > 0:
                stockPrice = options.S.max()
                # use more accurate data if available
                if currentDay in self.priceHistory:
                    history = self.priceHistory.loc[currentDay]
                    stockHigh, stockLow = history.High, history.Low
                else:
                    stockHigh, stockLow = stockPrice, stockPrice

                # print('Date:', currentDay, '\nStock Price:', stockPrice)
                # options['model_c'] = options.apply(lambda row: modelObj.model(
                #     'call', row['S'], row['K'], row['tau'], 0.3)[1], axis=1)
                # options['model_p'] = options.apply(lambda row: modelObj.model(
                #     'put', row['S'], row['K'], row['tau'], 0.3)[1], axis=1)
                options['model_c'] = options.apply(lambda row: modelObj.modelv2(
                    'call', row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega'], autoR=True), axis=1)
                options['model_p'] = options.apply(lambda row: modelObj.modelv2(
                    'put', row['quote_date'], row['S'], row['K'], row['tau'], row['p_vega'], autoR=True), axis=1)

                # overpriced: sell because people are willing to pay for higher price that what we think they are worth
                options['c_overpriced'] = (options.c_bid - options.model_c)
                options['p_overpriced'] = (options.p_bid - options.model_p)
                # underpriced: buy because we think that people are selling for less than what we think they are worth
                options['c_underpriced'] = (options.model_c - options.c_ask)
                options['p_underpriced'] = (options.model_p - options.p_ask)

                # give me top 5 that if the difference is greater than the spread
                top5overpricedCallOptions = options[options['c_overpriced'] > spread].sort_values('c_overpriced', ascending=False).head()
                top5overpricedPutOptions = options[options['p_overpriced'] > spread].sort_values('p_overpriced', ascending=False).head()
                top5underpricedCallOptions = options[options['c_underpriced'] > spread].sort_values('c_underpriced', ascending=False).head()
                top5underpricedPutOptions = options[options['p_underpriced'] > spread].sort_values('p_underpriced', ascending=False).head()

                callsBought = pd.concat([callsBought, top5underpricedCallOptions])
                putsBought = pd.concat([putsBought, top5underpricedPutOptions])
                callsSold = pd.concat([callsSold, top5overpricedCallOptions])
                putsSold = pd.concat([putsSold, top5overpricedPutOptions])

                # #update current balance after selling and buying options
                currentBalance -= (top5underpricedCallOptions.c_ask + spread).sum()
                currentBalance -= (top5underpricedPutOptions.p_ask +spread).sum()
                currentBalance += (top5overpricedCallOptions.c_bid -spread).sum()
                currentBalance += (top5overpricedPutOptions.c_bid -spread).sum()

            # add compute pnl if signal to close position is hit
            currentBalance -= callsSold[(stockHigh - callsSold.K) >= (callsSold.c_bid*threshold)].apply(lambda row: stockHigh - row.K, axis=1).sum()
            tradeCounter[2] += len(callsSold[(stockHigh -callsSold.K) >= (callsSold.c_bid*threshold)])
            gainsCounter[2] += callsSold[(stockHigh -callsSold.K) >= (callsSold.c_bid*threshold)]['c_bid'].sum() * threshold
            riskCounter[2] += callsSold[(stockHigh -callsSold.K) >= (callsSold.c_bid*threshold)].apply(lambda row: row.S*math.exp(0.3*math.sqrt(row.tau/10))**10 - row.K, axis=1).sum()
            callsSold = callsSold[(stockHigh - callsSold.K)< (callsSold.c_ask*threshold)]

            currentBalance -= putsSold[(putsSold.K - stockLow) >= (putsSold.p_bid*threshold)].apply(lambda row: row.K - stockLow, axis=1).sum()
            tradeCounter[3] += len(putsSold[(putsSold.K - stockLow)>= (putsSold.p_bid*threshold)])
            gainsCounter[3] += putsSold[(putsSold.K - stockLow)>= (putsSold.p_bid*threshold)].p_bid.sum() * threshold
            riskCounter[3] += putsSold[(putsSold.K - stockLow)>= (putsSold.p_bid*threshold)].apply(lambda row: - row.S*(1/math.exp(0.3*math.sqrt(row.tau/10)))**10 + row.K, axis=1).sum()
            putsSold = putsSold[(putsSold.K - stockLow) <(putsSold.p_ask*threshold)]

            currentBalance += callsBought[(stockHigh - callsBought.K) >= (callsBought.c_ask*threshold)].apply(lambda row: stockHigh - row.K, axis=1).sum()
            tradeCounter[0] += len(callsBought[(stockHigh -callsBought.K) >= (callsBought.c_ask*threshold)])
            gainsCounter[0] += callsBought[(stockHigh - callsBought.K) >= (callsBought.c_ask*threshold)].c_bid.sum() * threshold
            riskCounter[0] += callsBought[(stockHigh - callsBought.K) >= (callsBought.c_ask*threshold)].c_bid.sum()
            callsBought = callsBought[(stockHigh - callsBought.K) < (callsBought.p_ask*threshold)]

            currentBalance += putsBought[(putsBought.K - stockLow) >= (putsBought.p_ask*threshold)].apply(lambda row: row.K - stockLow, axis=1).sum()
            tradeCounter[1] += len(putsBought[(putsBought.K - stockLow)>= (putsBought.p_ask*threshold)])
            gainsCounter[1] += putsBought[(putsBought.K - stockLow)>= (putsBought.p_ask*threshold)].p_bid.sum() * threshold
            riskCounter[1] += putsBought[(putsBought.K - stockLow)>= (putsBought.p_ask*threshold)].p_bid.sum()
            putsBought = putsBought[(putsBought.K - stockLow) < (putsBought.p_ask*threshold)]

            # close positions by filtering out expired options
            tradeCounter[2] += len(callsSold[callsSold.expire_date <= currentDay])
            riskCounter[2] += callsSold[callsSold.expire_date <= currentDay].apply(lambda row: row.S*math.exp(0.3*math.sqrt(row.tau/10))**10 - row.K, axis=1).sum()
            callsSold = callsSold[callsSold.expire_date > currentDay]
            tradeCounter[3] += len(putsSold[putsSold.expire_date <= currentDay])
            riskCounter[3] += putsSold[putsSold.expire_date <= currentDay].apply(lambda row: - row.S*(1/math.exp(0.3*math.sqrt(row.tau/10)))**10 + row.K, axis=1).sum()

            putsSold = putsSold[putsSold.expire_date > currentDay]
            tradeCounter[0] += len(callsBought[callsBought.expire_date <= currentDay])
            riskCounter[0] += callsBought[callsBought.expire_date <= currentDay].c_bid.sum()
            callsBought = callsBought[callsBought.expire_date > currentDay]
            tradeCounter[1] += len(putsBought[putsBought.expire_date <= currentDay])
            riskCounter[1] += putsBought[putsBought.expire_date <= currentDay].c_bid.sum()
            putsBought = putsBought[putsBought.expire_date > currentDay]

            if rebalancing:
                max_positions = 20
                rebalancing_factor = 0.5
                # if i have more than 5 positions, remove the ones with the smallest spread
                if len(callsSold) > max_positions:
                    num_to_keep = max_positions + round((len(callsSold)-max_positions)*rebalancing_factor)
                    fifthBestCallOverpriced = callsSold.c_overpriced.nlargest(num_to_keep).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (callsSold[callsSold.c_overpriced < fifthBestCallOverpriced]['c_ask'] + spread).sum()
                    tradeCounter[2] += len(callsSold[callsSold.c_overpriced < fifthBestCallOverpriced])
                    callsSold = callsSold.sort_values('c_overpriced', ascending=False).head()
                if len(putsSold) > max_positions:
                    num_to_keep = max_positions + round((len(callsSold)-max_positions)*rebalancing_factor)
                    fifthBestPutOverpriced = putsSold.p_overpriced.nlargest(num_to_keep).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (putsSold[putsSold.p_overpriced < fifthBestPutOverpriced]['p_ask'] + spread).sum()
                    tradeCounter[3] += len(putsSold[putsSold.p_overpriced < fifthBestPutOverpriced])
                    putsSold = putsSold.sort_values('p_overpriced', ascending=False).head()
                if len(callsBought) > max_positions:
                    num_to_keep = max_positions + round((len(callsSold)-max_positions)*rebalancing_factor)
                    fifthBestCallUnderpriced = callsBought.c_underpriced.nlargest(num_to_keep).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['c_bid'] - spread).sum()
                    tradeCounter[0] += len(callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced])
                    callsBought = callsBought.sort_values('c_underpriced', ascending=False).head()
                if len(putsBought) > max_positions:
                    num_to_keep = max_positions + round((len(callsSold)-max_positions)*rebalancing_factor)
                    fifthBestPutUnderpriced = putsBought.p_underpriced.nlargest(num_to_keep).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['p_bid'] - spread).sum()
                    tradeCounter[1] += len(putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced])
                    putsBought = putsBought.sort_values('p_underpriced', ascending=False).head()

            # print('EOD Balance:' ,currentBalance, '\n')
            dailyBalance.append(currentBalance)

            currentDay += timedelta(days=1)

        trades = ['calls bought', 'puts bought', 'calls sold', 'puts sold']
        totalGain = 0
        totalPain = 0
        for i in range(4):
            print(trades[i], ": ", tradeCounter[i])
            print("sharpe: ", (gainsCounter[i]*0.97)/riskCounter[i] if riskCounter[i] else 0)
            totalGain += gainsCounter[i]
            totalPain += riskCounter[i]

        print("totalSharpe: ", totalGain/totalPain)
        return dailyBalance

    def maximumLoss(self, modelCallPrice, modelPutPrice, marketcallPrice, marketPutPrice, start, expire):
        # first identify whether i will buy or sell the option
        # then figure out minLoss, maxLoss, minProfit, maxProfit given the movement of the stockprice

        pass


if __name__ == "__main__":
    df = pd.read_csv(
        "./trimmed.csv", parse_dates=[" [EXPIRE_DATE]", " [QUOTE_DATE]"], low_memory=False)

    evalObj = Eval(df, datetime(2022, 7, 1), datetime(2022, 12, 1))
    # dates = [evalObj.startDate + timedelta(days=i) for i in range((evalObj.endDate-evalObj.startDate).days)]
    # evalObj.compareModeltoMarketDistribution()

    # plt.legend()
    # plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    # plt.xlabel('xlabel', fontsize=16)
    # plt.show()

    # plt.plot(list(overpricingctau.keys()),list(overpricingctau.values()),label="Overpricing % Spread per contract (Call)")
    # plt.plot(list(overpricingptau.keys()),list(overpricingptau.values()),label="Overpricing % Spread per contract (Put)")
    # plt.plot(list(underpricingctau.keys()),list(underpricingctau.values()),label="Underpricing % Spread per contract (Call)")
    # # plt.plot(list(underpricingptau.keys()),list(underpricingptau.values()),label="Underpricing % Spread per contract (Put)")
    # plt.legend()
    # plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    # plt.xlabel('xlabel', fontsize=10)
    # # plt.savefig('overpricing.png')
    # plt.show()

    print("Without Rebalancing")
    withoutRebalancing = evalObj.tradeUntilExercised(
        rebalancing=False, threshold=1.5)
    print('\nWith Rebalancing\n')
    withRebalancing = evalObj.tradeUntilExercised(threshold=1.5)
    dates = [evalObj.startDate + timedelta(days=i)
             for i in range((evalObj.endDate-evalObj.startDate).days+1)]

    plt.plot(dates, withoutRebalancing, label="W/O Rebalancing")
    plt.plot(dates, withRebalancing, label="W Rebalancing")

    plt.legend()
    plt.show()

    # print("Without Rebalancing")
    # withoutRebalancing = evalObj.tradeUntilExercised(rebalancing=False, threshold=1.5)
    # print('\nWith Rebalancing\n')
    # withRebalancing = evalObj.tradeUntilExercised(threshold=1.5)
    # dates = [evalObj.startDate + timedelta(days=i) for i in range((evalObj.endDate-evalObj.startDate).days+1)]

    # plt.plot(dates, withoutRebalancing, label="W/O Rebalancing")
    # plt.plot(dates, withRebalancing, label="W Rebalancing")

    # print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    pass
