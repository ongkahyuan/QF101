# QF101 Project (SMU AY2022/2023 Semester 2)
Group Members: Amrish, Bryan, Kai Feng, Joel, Jia Wei

## get_data.py
Reads option_chain from the yFinance library. For each options contract in option_chain for a given expiry date, return sigma, tau, dividend rate, risk free rate. 
write a for loop that loops through each options contract. In each loop, there should be sigma, tau, dividend rate, risk free rate. 
```
input: 
expiry date for contract: str yyyy-mm-dd

output: 
[[sigma, tau, div rate, risk free rate] for each contract] (all float)
```

## model.py
improve the model if possible
S = underlying price
K = strike price
r = risk free rate
q = dividend rate
tau = time to expiry in years
sigma = implied volatility 
```
input: (S,K,r,q,tau,sigma,N=100)
output: (call price, put price)
```

## eval.py
```
input: date of writing option, date of expiry, model call price, model put price
output: Maybe PnL, maybe the difference between model price and market price.

```
## vis.py
Find some way to visualise the data lol

## main.py
Puts everything together