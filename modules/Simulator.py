from __future__ import annotations
from model_data import ModelData
from model import Model, OptionType, OptionStyle
import pandas as pd
from matplotlib import pyplot as plt
# import model
import yfinance as yf
from heapq import heappush, heappop
from datetime import datetime, timedelta
import numpy as np
import sys
import collections

from typing import Dict


class Simulator:
    def __init__(self, df, start_date: datetime, end_date: datetime):
        self.ticker = yf.Ticker('AAPL')
        self.priceHistory = self.ticker.history(period='1d', start=start_date.strftime(
            "%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        self.start_date = start_date
        self.end_date = end_date
        self.model_data:ModelData = ModelData(df, start_date, end_date)
        self.model:Model = Model()
        self.european_option_values:Dict[str, pd.DataFrame] = self._generate_option_prices(OptionStyle.EUROPEAN)
        self.american_option_values:Dict[str, pd.DataFrame] = self._generate_option_prices(OptionStyle.AMERICAN)

    def get_option_prices(self,style:OptionStyle) -> Dict[str, pd.DataFrame]:
        target_dict = self.european_option_values if style==OptionStyle.EUROPEAN else self.american_option_values
        new_dict = {}
        for key, value in target_dict.items():
            new_dict[key] = value.copy(deep=True)
        return new_dict

    def _generate_option_prices(self, style:OptionStyle):
        option_values = {}
        currentDay = self.start_date
        while currentDay < self.end_date:
            options = self.model_data.getAllCurrentPrice(currentDay)
            if len(options) == 0:
                currentDay += timedelta(days=1)
                continue
            options['model_c'] = options.apply(lambda row: self.model.calculate_option_price(OptionType.CALL, style, row['S'], row['K'], row['tau'], 0.3, quote_date=currentDay), axis=1)
            options['model_p'] = options.apply(lambda row: self.model.calculate_option_price(OptionType.PUT, style, row['S'], row['K'], row['tau'], 0.3, quote_date=currentDay), axis=1)
            options['c_diff'] = abs(options.c_ask - options.model_c)
            options['p_diff'] = abs(options.p_ask - options.model_p)
            option_values[currentDay] = options.copy(deep=True)
            currentDay += timedelta(days=1)
        return option_values
        

    def compareModeltoMarket(self, threshold=2):
        currentDay = self.start_date
        modelPoints = []
        marketPoints = []
        maxModelToMarketDifference = 0

        # index 0 for calls, index 1 for puts
        underpricing = [0, 0]
        overpricing = [0, 0]
        underpricingCounter = [0, 0]
        overpricingCounter = [0, 0]
        columns = list(self.model_data.getAllCurrentPrice(currentDay).columns)
        columns += ['underpriced', 'overpriced', 'min_profit', 'max_loss']
        # modelmarketDf = pd.DataFrame(columns=columns)
        dailyoverpricingc = []
        dailyoverpricingp = []
        dailyunderpricingc = []
        dailyunderpricingp = []
        dailyoverpricingctau = collections.defaultdict(list)
        dailyoverpricingptau = collections.defaultdict(list)
        dailyunderpricingctau = collections.defaultdict(list)
        dailyunderpricingptau = collections.defaultdict(list)
        euro_option_prices = self.get_option_prices(OptionStyle.EUROPEAN)
        amer_option_prices = self.get_option_prices(OptionStyle.AMERICAN)
        while currentDay < self.end_date:
            options = amer_option_prices.get(currentDay)
            if options is None:
                dailyoverpricingc.append(0)
                dailyoverpricingp.append(0)
                dailyunderpricingc.append(0)
                dailyunderpricingp.append(0)
                currentDay += timedelta(days=1)
                continue
            underpricing[0] += np.maximum(options.c_ask - options.model_c, np.zeros(len(options))).sum()
            underpricing[1] += np.maximum(options.p_ask - options.model_p, np.zeros(len(options))).sum()
            overpricing[0] += np.maximum(-options.c_bid + options.model_c, np.zeros(len(options))).sum()
            overpricing[1] += np.maximum(-options.p_bid + options.model_p, np.zeros(len(options))).sum()
            dailyunderpricingcall = np.maximum((options.c_ask - options.model_c)/options.c_ask, np.zeros(len(options))).sum()
            dailyunderpricingput = np.maximum((options.p_ask - options.model_p)/options.p_ask, np.zeros(len(options))).sum()
            dailyoverpricingcall = np.maximum((-options.c_bid + options.model_c)/options.c_bid, np.zeros(len(options))).sum()
            dailyoverpricingput = np.maximum((-options.p_bid + options.model_p)/options.p_bid, np.zeros(len(options))).sum()
            dailyoverpricingc.append(dailyoverpricingcall/(len(options)))
            dailyoverpricingp.append(dailyoverpricingput/(len(options)))
            dailyunderpricingc.append(dailyunderpricingcall/(len(options)))
            dailyunderpricingp.append(dailyunderpricingput/(len(options)))

            # contracts += len(options) * 2
            # print(pd.concat([options.model_c, options.c_ask, options.c_diff], axis=1).head())
            maxModelToMarketDifference = max(
                maxModelToMarketDifference, options.c_diff.max())
            maxModelToMarketDifference = max(
                maxModelToMarketDifference, options.p_diff.max())

            underpricingCounter[0] += len(options[options.c_ask-options.model_c > threshold])
            underpricingCounter[1] += len(options[options.p_ask-options.model_p > threshold])
            overpricingCounter[0] += len(options[-options.c_bid+options.model_c > threshold])
            overpricingCounter[1] += len(options[-options.p_bid+options.model_p > threshold])
            # modelmarketDf = pd.concat([modelmarketDf,options])
            for i in options[['tau']].drop_duplicates().values:
                taudf= options.loc[(options['tau']==i[0])]
                dailyunderpricingcalltau = np.maximum((taudf.c_ask - taudf.model_c)/taudf.c_ask, np.zeros(len(taudf))).sum()
                dailyunderpricingputtau = np.maximum((taudf.p_ask - taudf.model_p)/taudf.p_ask, np.zeros(len(taudf))).sum()
                dailyoverpricingcalltau = np.maximum((-taudf.c_bid + taudf.model_c)/taudf.c_bid, np.zeros(len(taudf))).sum()
                dailyoverpricingputtau = np.maximum((-taudf.p_bid + taudf.model_p)/taudf.p_bid, np.zeros(len(taudf))).sum()
                #print((taudf.c_ask-taudf.model_c)/taudf.c_ask)

                dailyoverpricingctau[i[0]].append(dailyoverpricingcalltau/(len(taudf)))
                dailyoverpricingptau[i[0]].append(dailyoverpricingputtau/(len(taudf)))
                dailyunderpricingctau[i[0]].append(dailyunderpricingcalltau/(len(taudf)))
                dailyunderpricingptau[i[0]].append(dailyunderpricingputtau/(len(taudf)))

            # options['model_c'] = options.apply(lambda row: model.Model().modelv2('call',row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega']), axis=1)
            # options['model_p'] = options.apply(lambda row: model.Model().modelv2('put', row['quote_date'],row['S'], row['K'], row['tau'], row['p_vega']), axis=1)
            # options['model_c'] = options.apply(lambda row: model.Model().model('call', row['S'], row['K'], row['tau'], row['c_vega'])[1], axis=1)
            # options['model_p'] = options.apply(lambda row: model.Model().model('put', row['S'], row['K'], row['tau'], row['p_vega'])[1], axis=1)
            #options = amer_option_prices.get(currentDay)
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
        return dailyoverpricingc, dailyoverpricingp, dailyunderpricingc, dailyunderpricingp,dailyoverpricingctau, dailyoverpricingptau, dailyunderpricingctau, dailyunderpricingptau

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

        currentDay = self.start_date
        losses = {"max_loss_per_call_sold": 0, "max_loss_per_put_sold": 0,
                  "total_call_losses": 0, "total_put_losses": 0}
        profit = {"min_profit_per_call_bought": sys.maxsize,
                  "min_profit_per_put_bought": sys.maxsize, "total_call_profit": 0, "total_put_profit": 0}

        columns = list(self.model_data.getAllCurrentPrice(currentDay).columns)
        columns += ['underpriced', 'overpriced', 'min_profit', 'max_loss']
        callsBought = pd.DataFrame(columns=columns)
        putsBought = pd.DataFrame(columns=columns)
        callsSold = pd.DataFrame(columns=columns)
        putsSold = pd.DataFrame(columns=columns)
        currentBalance = 0
        dailyBalance = []
        # index 0 for calls, index 1 for puts
        tradeCounter = [0, 0, 0, 0]
        euro_option_prices = self.get_option_prices(OptionStyle.EUROPEAN)
        amer_option_prices = self.get_option_prices(OptionStyle.AMERICAN)
        while currentDay <= self.end_date:
            options = amer_option_prices.get(currentDay, [])

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
                # options['model_c'] = options.apply(lambda row: model.Model().modelv2(
                #     'call', row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega']), axis=1)
                # options['model_p'] = options.apply(lambda row: model.Model().modelv2(
                #     'put', row['quote_date'], row['S'], row['K'], row['tau'], row['p_vega']), axis=1)

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
            currentBalance += putsBought[putsBought.expire_date <=
                                         currentDay]['min_profit'].sum()
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
                    currentBalance -= (callsSold[callsSold.c_overpriced < fifthBestCallOverpriced]['c_ask'] + spread).sum()
                    currentBalance -= (callsSold[callsSold.c_overpriced < fifthBestCallOverpriced]['max_loss']).sum()
                    tradeCounter[2] += len(callsSold[callsSold.c_overpriced < fifthBestCallOverpriced])
                    callsSold = callsSold.sort_values('c_overpriced', ascending=False).head()
                if len(putsSold) > 5:
                    fifthBestPutOverpriced = callsSold.p_overpriced.nlargest(
                        5).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (putsSold[putsSold.p_overpriced < fifthBestPutOverpriced]['p_ask'] + spread).sum()
                    currentBalance -= (putsSold[putsSold.p_overpriced < fifthBestPutOverpriced]['max_loss']).sum()
                    tradeCounter[3] += len(putsSold[putsSold.p_overpriced < fifthBestPutOverpriced])
                    putsSold = putsSold.sort_values('p_overpriced', ascending=False).head()
                if len(callsBought) > 5:
                    fifthBestCallUnderpriced = callsSold.c_underpriced.nlargest(
                        5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['c_bid'] - spread).sum()
                    currentBalance += (callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['min_profit']).sum()
                    profit['total_call_profit'] += callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['min_profit'].sum()
                    profit['min_profit_per_call_bought'] = min(profit['min_profit_per_call_bought'], callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['min_profit'].min())
                    tradeCounter[0] += len(callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced])
                    callsBought = callsBought.sort_values('c_underpriced', ascending=False).head()
                if len(putsBought) > 5:
                    fifthBestPutUnderpriced = callsSold.p_underpriced.nlargest(5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['p_bid'] - spread).sum()
                    currentBalance += (putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['min_profit']).sum()
                    profit['total_profit_profit'] += (putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['min_profit']).sum()
                    profit['min_profit_per_put_bought'] = min(profit['min_profit_per_put_bought'], putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['min_profit'].min())
                    tradeCounter[1] += len(putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced])
                    putsBought = putsBought.sort_values('p_underpriced', ascending=False).head()

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

        currentDay = self.start_date
        columns = list(self.model_data.getAllCurrentPrice(currentDay).columns)
        columns += ['underpriced', 'overpriced']
        callsBought = pd.DataFrame(columns=columns)
        putsBought = pd.DataFrame(columns=columns)
        callsSold = pd.DataFrame(columns=columns)
        putsSold = pd.DataFrame(columns=columns)
        currentBalance = 0
        dailyBalance = []
        # index 0 for calls, index 1 for puts
        tradeCounter = [0, 0, 0, 0]
        euro_option_prices = self.get_option_prices(OptionStyle.EUROPEAN)
        amer_option_prices = self.get_option_prices(OptionStyle.AMERICAN)
        while currentDay <= self.end_date:
            options = amer_option_prices.get(currentDay, [])

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
                # options['model_c'] = options.apply(lambda row: modelObj.modelv2(
                #     'call', row['quote_date'], row['S'], row['K'], row['tau'], row['c_vega']), axis=1)
                # options['model_p'] = options.apply(lambda row: modelObj.modelv2(
                #     'put', row['quote_date'], row['S'], row['K'], row['tau'], row['p_vega']), axis=1)

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
                currentBalance -= (top5underpricedPutOptions.p_ask + spread).sum()
                currentBalance += (top5overpricedCallOptions.c_bid - spread).sum()
                currentBalance += (top5overpricedPutOptions.c_bid - spread).sum()

            # add compute pnl if signal to close position is hit
            currentBalance -= callsSold[(stockHigh - callsSold.K) >= (callsSold.c_bid*threshold)].apply(lambda row: stockHigh - row.K, axis=1).sum()
            tradeCounter[2] += len(callsSold[(stockHigh - callsSold.K) >= (callsSold.c_bid*threshold)])
            callsSold = callsSold[(stockHigh - callsSold.K) < (callsSold.c_ask*threshold)]

            currentBalance -= putsSold[(putsSold.K - stockLow) >= (putsSold.p_bid*threshold)].apply(lambda row: row.K - stockLow, axis=1).sum()
            tradeCounter[3] += len(putsSold[(putsSold.K - stockLow) >= (putsSold.p_bid*threshold)])
            putsSold = putsSold[(putsSold.K - stockLow) < (putsSold.p_ask*threshold)]

            currentBalance += callsBought[(stockHigh - callsBought.K) >= (callsBought.c_ask*threshold)].apply(lambda row: stockHigh - row.K, axis=1).sum()
            tradeCounter[0] += len(callsBought[(stockHigh - callsBought.K) >= (callsBought.c_ask*threshold)])
            callsBought = callsBought[(stockHigh - callsBought.K) < (callsBought.p_ask*threshold)]

            currentBalance += putsBought[(putsBought.K - stockLow) >= (putsBought.p_ask*threshold)].apply(lambda row: row.K - stockLow, axis=1).sum()
            tradeCounter[1] += len(putsBought[(putsBought.K - stockLow) >= (putsBought.p_ask*threshold)])
            putsBought = putsBought[(putsBought.K - stockLow) < (putsBought.p_ask*threshold)]

            # close positions by filtering out expired options
            tradeCounter[2] += len(callsSold[callsSold.expire_date <= currentDay])
            callsSold = callsSold[callsSold.expire_date > currentDay]
            tradeCounter[3] += len(putsSold[putsSold.expire_date <= currentDay])
            putsSold = putsSold[putsSold.expire_date > currentDay]
            tradeCounter[0] += len(callsBought[callsBought.expire_date <= currentDay])
            callsBought = callsBought[callsBought.expire_date > currentDay]
            tradeCounter[1] += len(putsBought[putsBought.expire_date <= currentDay])
            putsBought = putsBought[putsBought.expire_date > currentDay]

            if rebalancing:
                # if i have more than 5 positions, remove the ones with the smallest spread
                if len(callsSold) > 5:
                    fifthBestCallOverpriced = callsSold.c_overpriced.nlargest(5).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (callsSold[callsSold.c_overpriced < fifthBestCallOverpriced]['c_ask'] + spread).sum()
                    tradeCounter[2] += len(callsSold[callsSold.c_overpriced < fifthBestCallOverpriced])
                    callsSold = callsSold.sort_values('c_overpriced', ascending=False).head()
                if len(putsSold) > 5:
                    fifthBestPutOverpriced = putsSold.p_overpriced.nlargest(5).iloc[-1]
                    # buy back options sold that are worse than fifth best
                    currentBalance -= (putsSold[putsSold.p_overpriced < fifthBestPutOverpriced]['p_ask'] + spread).sum()
                    tradeCounter[3] += len(putsSold[putsSold.p_overpriced < fifthBestPutOverpriced])
                    putsSold = putsSold.sort_values('p_overpriced', ascending=False).head()
                if len(callsBought) > 5:
                    fifthBestCallUnderpriced = callsBought.c_underpriced.nlargest(5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced]['c_bid'] - spread).sum()
                    tradeCounter[0] += len(callsBought[callsBought.c_underpriced < fifthBestCallUnderpriced])
                    callsBought = callsBought.sort_values('c_underpriced', ascending=False).head()
                if len(putsBought) > 5:
                    fifthBestPutUnderpriced = putsBought.p_underpriced.nlargest(5).iloc[-1]
                    # sell back options sold that are worse than fifth best
                    currentBalance += (putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced]['p_bid'] - spread).sum()
                    tradeCounter[1] += len(putsBought[putsBought.p_underpriced < fifthBestPutUnderpriced])
                    putsBought = putsBought.sort_values('p_underpriced', ascending=False).head()

            # print('EOD Balance:' ,currentBalance, '\n')
            dailyBalance.append(currentBalance)

            currentDay += timedelta(days=1)

        trades = ['calls bought', 'puts bought', 'calls sold', 'puts sold']
        for i in range(4):
            print(trades[i], ": ", tradeCounter[i])
        return dailyBalance

    def maximumLoss(self, modelCallPrice, modelPutPrice, marketcallPrice, marketPutPrice, start, expire):
        # first identify whether i will buy or sell the option
        # then figure out minLoss, maxLoss, minProfit, maxProfit given the movement of the stockprice

        pass


if __name__ == "__main__":
    df = pd.read_csv("./trimmed.csv", parse_dates=[" [EXPIRE_DATE]", " [QUOTE_DATE]"], low_memory=False)

    binaryTreeSim = Simulator(df, datetime(2022, 7, 1), datetime(2022, 8, 1))
    dates = [binaryTreeSim.start_date + timedelta(days=i) for i in range((binaryTreeSim.end_date-binaryTreeSim.start_date).days)]
    overpricingc,overpricingp, underpricingc, underpricingp,x,y,z,t = binaryTreeSim.compareModeltoMarket()
    print(overpricingc)
    print(overpricingp)
    plt.plot(dates, overpricingc, label="Daily Overpricing % Spread per contract (Call)")
    plt.plot(dates, overpricingp, label="Daily Overpricing % Spread per contract (Put)")
    plt.plot(dates, underpricingc, label="Daily Underpricing % Spread per contract (Call)")
    plt.plot(dates, underpricingp, label="Daily Underpricing % Spread per contract (Put)")
    plt.legend()
    plt.show()

    
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
    withoutRebalancing = binaryTreeSim.tradeUntilExercised(rebalancing=False, threshold=1.5)
    print('\nWith Rebalancing\n')
    withRebalancing = binaryTreeSim.tradeUntilExercised(threshold=1.5)
    dates = [binaryTreeSim.start_date + timedelta(days=i) for i in range((binaryTreeSim.end_date-binaryTreeSim.start_date).days+1)]

    plt.plot(dates, withoutRebalancing, label="W/O Rebalancing")
    plt.plot(dates, withRebalancing, label="W Rebalancing")

    plt.legend()
    plt.show()

    # print("Without Rebalancing")
    # withoutRebalancing = binaryTreeSim.tradeUntilExercised(rebalancing=False, threshold=1.5)
    # print('\nWith Rebalancing\n')
    # withRebalancing = binaryTreeSim.tradeUntilExercised(threshold=1.5)
    # dates = [binaryTreeSim.start_date + timedelta(days=i) for i in range((binaryTreeSim.end_date-binaryTreeSim.start_date).days+1)]

    # plt.plot(dates, withoutRebalancing, label="W/O Rebalancing")
    # plt.plot(dates, withRebalancing, label="W Rebalancing")

    # print(gd.getAllCurrentPrice("2022-07-01"))
    # gd.getAllCurrentPrice("2022-07-04")
    pass
