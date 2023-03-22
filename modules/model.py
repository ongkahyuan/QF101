# Binomial Tree Option Pricing (nested list with for loop)
import math
# T-t -> tau
def model(S,K,r,q,tau,sigma,N=100):
   # 1.
   deltaT=tau/N
   u=math.exp(sigma*math.sqrt(deltaT))
   d=1/u
   p=(math.exp((r-q)*deltaT)-d)/(u-d)
   # 2.
   fc=[[0.0 for j in range(i+1)] for i in range(N+1)]
   fp=[[0.0 for j in range(i+1)] for i in range(N+1)]
   for j in range(N+1):
      fc[N][j]=max(0, S*(u**j)*(d**(N-j))-K)
      fp[N][j]=max(0, K-S*(u**j)*(d**(N-j)))
   # 3.
   p1=1-p
   ert=math.exp(-r*deltaT)
   for i in range(N-1,0-1,-1):
      for j in range(i+1):
         fc[i][j]=ert*(p*fc[i+1][j+1]+p1*fc[i+1][j])
         fp[i][j]=ert*(p*fp[i+1][j+1]+p1*fp[i+1][j])    
   # 4.
   c=fc[0][0]
   p=fp[0][0]
   return (c, p)

if __name__=='__main__':
   S=50.0; K=50.0; tau=183/365 
   sigma=0.4; r=0.04; q=0.01
   print('Call: {0[0]}, Put: {0[1]}'.format(
                model(S,K,r,q,tau,sigma)))