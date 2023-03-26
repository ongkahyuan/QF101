# Binomial Tree Option Pricing (nested list with for loop)
import math
# T-t -> tau
def model(S,K,tau,sigma,r=0.1,q=0.02,N=100):
   # 1. 
   deltaT=tau/N                           # delta T is the time step
   u=math.exp(sigma*math.sqrt(deltaT))    # u is the uptick (amount that stock goes up by in each time step)
   d=1/u                                  # d is the downtick 
   p=(math.exp((r-q)*deltaT)-d)/(u-d)     # p is risk neutral probability thingy i think (used for expectation)
   # 2.
   ec=[[0.0 for j in range(i+1)] for i in range(N+1)] # call options tree
   ac=[[0.0 for j in range(i+1)] for i in range(N+1)] # call options tree
   ep=[[0.0 for j in range(i+1)] for i in range(N+1)] # put options tree
   ap=[[0.0 for j in range(i+1)] for i in range(N+1)] # put options tree
   # every row is one time step, and every column is the computation for the number of upticks 
   # the maximum number of upticks at the Nth time step is N 

   # at the Nth time step, there are N+1 possible prices: 
   # ( uptick ** N ) (downtick ** (0)), ( uptick ** N-1 ) (downtick ** (1)) ... , ( uptick ** 0 ) (downtick ** (N))
   for j in range(N+1):
      ec[N][j]=max(0, S*(u**j)*(d**(N-j))-K)
      ac[N][j]=max(0, S*(u**j)*(d**(N-j))-K)
      ep[N][j]=max(0, K-S*(u**j)*(d**(N-j)))
      ap[N][j]=max(0, K-S*(u**j)*(d**(N-j)))
   
   # 3.
   discount=math.exp(-r*deltaT)
   for i in range(N-1,0-1,-1):
      for j in range(i+1):
         # back propagate ->
         # payoff at t = discounted average (payoff of uptick and downtick) at t+1 
         europeanCallOptFuturePayoff = discount*(p*ec[i+1][j+1]+(1-p)*ec[i+1][j])
         americanCallOptFuturePayoff = discount*(p*ac[i+1][j+1]+(1-p)*ac[i+1][j])
         europeanPutOptFuturePayoff = discount*(p*ep[i+1][j+1]+(1-p)*ep[i+1][j])
         americanPutOptFuturePayoff = discount*(p*ap[i+1][j+1]+(1-p)*ap[i+1][j])

         currentStockPrice = S*(u**j)*(d**(i-j))

         ec[i][j] = europeanCallOptFuturePayoff
         ac[i][j] = max(americanCallOptFuturePayoff, max(currentStockPrice-K, 0))
         ep[i][j] = europeanPutOptFuturePayoff
         ap[i][j] = max(americanPutOptFuturePayoff, max(K-currentStockPrice, 0))
   
   # 4.
   euro_c=ec[0][0]
   euro_p=ep[0][0]
   amer_c=ac[0][0]
   amer_p=ap[0][0]
   return euro_c,euro_p, amer_c, amer_p

if __name__=='__main__':
   S=50.0; K=50.0; tau=183/365 
   sigma=0.4; r=0.1; q=0.01
   print('Euro Call: {0}, Euro Put: {1} \nAmer Call: {2}, Amer Put: {3}'.format(
                *model(S,K,r,q,tau,sigma)))