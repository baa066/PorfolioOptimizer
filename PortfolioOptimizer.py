# Description: This program attempts to optimize a user's portfolio using the Efficient Frontier & Python.
# Import the python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Create the fictional portfolio
assets =  ["ANKR-USD", "RVN-USD", "UMA-USD"] #<------Set your Tickers

# Assign weights to the stocks. Weights must = 1 so 0.25 for each. 
# Example: This means if I had a total of $100 USD in the portfolio, then I would have $25 USD in each stock. 
weights = np.array([0.4,0.3,0.3]) #<------Set your weights
weights

# Get the stock starting date
stockStartDate = '2021-04-01' #<------Set your historical lookback date YYYY-MM-DD

# Get the stocks ending date aka todays date and format it in the form YYYY-MM-DD
today = datetime.today().strftime('%Y-%m-%d')

today

#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()
#Store the adjusted close price of stock into the data frame
for stock in assets:
   df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate,end=today)['Close']

df

# Create the title 'Portfolio Adj Close Price History
title = 'Portfolio Adj. Close Price History    '

# Get the stocks
my_stocks = df

# Create and plot the graph
plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5

# Loop through each stock and plot the Adj Close for each day
for c in my_stocks.columns.values:
  plt.plot( my_stocks[c],  label=c)#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price USD ($)',fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()

# Show the daily simple returns, NOTE: Formula = new_price/old_price - 1
returns = df.pct_change()
returns

# To show the annualized co-variance matrix we must multiply the co-variance matrix by the number of trading days for the current year. 
# In this case the number of trading days will be 252 for this year
cov_matrix_annual = returns.cov() * 252
cov_matrix_annual

# Now calculate and show the portfolio variance using the formula:
# Expected portfolio variance = WT * (Covariance Matrix) * W
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance

# Now calculate and show the portfolio volatility using the formula:
# Expected portfolio volatility = SQRT (WT * (Covariance Matrix) * W)
# Don’t forget the volatility (standard deviation) is just the square root of the variance.
port_volatility = np.sqrt(port_variance)
port_volatility

# Calculate the portfolio annual simple return.
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
portfolioSimpleAnnualReturn

# Show the expected annual return, volatility or risk, and variance.
percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2)*100)+'%'
print("Expected annual return : "+ percent_ret)
print('Annual volatility/standard deviation/risk : '+percent_vols)
print('Annual variance : '+percent_var)

pip install PyPortfolioOpt

# Import PyPortfolioOpt libs
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Calculate the expected returns and the annualised sample covariance matrix of daily asset returns.
mu = expected_returns.mean_historical_return(df)#returns.mean() * 252
S = risk_models.sample_cov(df) #Get the sample covariance matrix

# Optimize for maximal Sharpe ratio.
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe() #Maximize the Sharpe ratio, and get the raw weights
cleaned_weights = ef.clean_weights() 
print(cleaned_weights) #Note the weights may have some rounding error, meaning they may not add up exactly to 1 but should be close
ef.portfolio_performance(verbose=True)

pip install pulp

# How much, how many should I buy?
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights 
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000) #<------Set your portfolio value or amount to invest
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
