import numpy as np
import numpy.random as npr

def priceSimulation(s0, vol, r, T, increment, trials):
	#params: initial stock price, volatility of asset, risk free interest rate, time to maturity, number of time steps, number of simulation trials
	#geometric brownian motion model
    s = np.zeros((increment, trials)) 
    s[0] = s0
    dt = T/increment
    for i in range(1, increment): 
        s[i] = s[i-1] * np.exp((r - 0.5*vol**2)*dt + vol * np.sqrt(dt) * npr.standard_normal(size=trials))
    print(s)
    return s[-1]

def optionPricer(flag, strike, sT, r, T):
	#params: call or put, option strike price, simulated asset prices at maturity, risk free interest rate, time to maturity
	if flag == "call":
		Payoff = np.maximum(sT-strike,0)
	else:
		Payoff = np.maximum(strike-sT,0)
	Payoff = np.exp(-r*T) * Payoff.mean()
	return Payoff

def mcOptionPrice(flag, s0, strike, r, vol, T, increments, trials):
	return optionPricer(flag, strike, priceSimulation(s0, vol, r, T, increments, trials), r, T)


# Example call to priceSimulation
s0 = 100         # Initial stock price
vol = 0.2        # Volatility (20%)
r = 0.05         # Risk-free interest rate (5%)
T = 1            # Time to maturity (1 year)
increment = 252  # Number of time steps (daily steps for a year)
trials = 10000   # Number of simulation trials

simulated_prices = priceSimulation(s0, vol, r, T, increment, trials)


# Example call to optionPricer
flag = "call"                 # Option type: 'call' or 'put'
strike = 105                  # Strike price of the option
sT = simulated_prices         # Simulated asset prices at maturity from priceSimulation
r = 0.05                      # Risk-free interest rate (5%)
T = 1                         # Time to maturity (1 year)

option_price = optionPricer(flag, strike, sT, r, T)


# Example call to mcOptionPrice
flag = "put"                  # Option type: 'call' or 'put'
s0 = 100                     # Initial stock price
strike = 95                  # Strike price of the option
r = 0.03                     # Risk-free interest rate (3%)
vol = 0.25                   # Volatility (25%)
T = 0.5                      # Time to maturity (6 months)
increments = 126             # Number of time steps (semi-daily steps for half a year)
trials = 5000                # Number of simulation trials

option_price1 = mcOptionPrice(flag, s0, strike, r, vol, T, increments, trials)
option_price15 = mcOptionPrice(flag, s0, strike, 0, vol, T, increments, trials)

option_price2 = mcOptionPrice("call", 100, 105, r, vol, T, increments, trials)
option_price25 = mcOptionPrice("call", 100, 105, 0, vol, T, increments, trials)

print(option_price1)
print(option_price15)
print(option_price2)
print(option_price25)