# Binomial Tree Option Pricing (nested list with for loop)
import math
import numpy as np
# T-t -> tau

def binomial_tree_option_pricing(S, K, r, q, tau, sigma, N=100):
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
            call_option[j] = (p * call_option[j+1] + (1-p) * call_option[j]) * np.exp(-r * delta_t)
            put_option[j] = (p * put_option[j+1] + (1-p) * put_option[j]) * np.exp(-r * delta_t)
            stock_price[i, j] = stock_price[i, j] 

            # Check for early exercise (American options only)
            exercise_value = K - stock_price[i, j] if put_option[j] > call_option[j] else stock_price[i, j] - K
            if exercise_value > 0:
                call_option[j] = exercise_value if call_option[j] < exercise_value else call_option[j]
                put_option[j] = exercise_value if put_option[j] < exercise_value else put_option[j]

    # Calculate final option prices
    american_call_price = call_option[0]
    european_call_price = call_option[0] * np.exp(-r * tau)
    american_put_price = put_option[0]
    european_put_price = put_option[0] * np.exp(-r * tau)

    return (european_call_price, european_put_price, american_call_price,american_put_price)

def model(type,S,K,r,q,tau,sigma,N=100):
   if type != 'call' and type != 'put':
      return 'Invalid option type','Invalid option type'
   # 1. 
   deltaT=tau/N                           # delta T is the time step
   u=math.exp(sigma*math.sqrt(deltaT))    # u is the uptick (amount that stock goes up by in each time step)
   d=1/u                                  # d is the downtick 
   p=(math.exp((r-q)*deltaT)-d)/(u-d)     # p is risk neutral probability thingy i think (used for expectation)
   # 2.
   # ec=[[0.0 for j in range(i+1)] for i in range(N+1)] # call options tree
   # ac=[[0.0 for j in range(i+1)] for i in range(N+1)] # call options tree
   # ep=[[0.0 for j in range(i+1)] for i in range(N+1)] # put options tree
   # ap=[[0.0 for j in range(i+1)] for i in range(N+1)] # put options tree
   euro=[[0.0 for j in range(i+1)] for i in range(N+1)] 
   amer=[[0.0 for j in range(i+1)] for i in range(N+1)] 
   # every row is one time step, and every column is the computation for the number of upticks 
   # the maximum number of upticks at the Nth time step is N 

   # at the Nth time step, there are N+1 possible prices: 
   # ( uptick ** N ) (downtick ** (0)), ( uptick ** N-1 ) (downtick ** (1)) ... , ( uptick ** 0 ) (downtick ** (N))
   for j in range(N+1):
      if type == 'call':
         euro[N][j]=max(0, S*(u**j)*(d**(N-j))-K)
         amer[N][j]=max(0, S*(u**j)*(d**(N-j))-K)
      else:
         euro[N][j]=max(0, K-S*(u**j)*(d**(N-j)))
         amer[N][j]=max(0, K-S*(u**j)*(d**(N-j)))
   
   # 3.
   discount=math.exp(-r*deltaT)
   for i in range(N-1,0-1,-1):
      for j in range(i+1):
         # back propagate ->
         # payoff at t = discounted average (payoff of uptick and downtick) at t+1
         if type == 'call':
            europeanOptFuturePayoff = discount*(p*euro[i+1][j+1]+(1-p)*euro[i+1][j])
            americanOptFuturePayoff = discount*(p*amer[i+1][j+1]+(1-p)*amer[i+1][j])
         else:
            europeanOptFuturePayoff = discount*(p*euro[i+1][j+1]+(1-p)*euro[i+1][j])
            americanOptFuturePayoff = discount*(p*amer[i+1][j+1]+(1-p)*amer[i+1][j])

         currentStockPrice = S*(u**j)*(d**(i-j))
         if type == 'call':
            euro[i][j] = europeanOptFuturePayoff
            amer[i][j] = max(americanOptFuturePayoff, max(currentStockPrice-K, 0))
         else:
            euro[i][j] = europeanOptFuturePayoff
            amer[i][j] = max(americanOptFuturePayoff, max(K-currentStockPrice, 0))
   
   # 4.
   euro_value=euro[0][0]
   amer_value=amer[0][0]
   return euro_value,amer_value

if __name__=='__main__':
   S=50.0; K=50.0; tau=183/365 
   sigma=0.4; r=0.1; q=0.01
   print('European Value: {0}, American Option Value: {1}'.format(
                *model('call',S,K,r,q,tau,sigma)))