# import libraries
from pandas_datareader import data as wb
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy.random as npr

plt.style.use('fivethirtyeight')

# get stocks symbols in our portfolio
assets = ['COST','NVDA','LMT','SLB','KO','NK','GOOG','SNE', 'PBR', 'FB','BA','NWBO','FIVE','CVX','OCPNY','GRPN']

# how much asset do we have?
nb_assets = len(assets)

# assign weights to the portfolio
weights = npr.dirichlet(np.ones(nb_assets),size=1)

# take a look at the stocks allocation
labels = assets
sizes = weights
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels = labels, autopct = '%1.1f%%', shadow = True, 
        startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# get the portfolio start date
stockStartDate = '2015-01-01'

# get the portfolio ending date
today = dt.today().strftime('%Y-%m-%d')

# create a dataframe to store the adjusted closed price
df = pd.DataFrame()

# store adjusted closed price in the df just created
for stock in assets:
    df[stock] = wb.DataReader(stock, data_source = 'yahoo', 
                              start = stockStartDate, end = today)['Adj Close']

# visually show the portfolio 
title = 'Portfolio adjusted close price history'

# get the stocks
my_stocks = df

# create & plot the graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label = c)
    
plt.title(title)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj. Close Price USD')
plt.legend(my_stocks.columns.values, loc = 'upper left')
print (plt.show())

# show the daily simple return
returns = df.pct_change()
#print(returns)

# create & show the annualized covariance matrix
cov_matrix_annual = returns.cov()*252 # number of trading days this year
#print("Portfolio covariance matrix",cov_matrix_annual)

# compute the portfolio variance
ws= weights.reshape(16)
weights = np.array(weights)
weights = weights.flatten()
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
#print("Portfolio variance ", port_variance)

# compute the portfolio volatility
port_volatility = np.sqrt(port_variance)
#print("Portfolio volatility", port_volatility)

# compute the annual portfolio return
portSimpleAnnualReturn = np.sum(returns.mean()*weights)*252
#print("Portfolio annual expected return", portSimpleAnnualReturn)

# show the expected annual return, variance & volatility
percent_exp = str(round(portSimpleAnnualReturn,2)*100)+'%'
percent_vol = str(round(port_volatility,2)*100)+'%'
percent_var = str(round(port_variance,2)*100)+'%'
print('Expected annual return:', percent_exp)
print('Variance:', percent_var)
print('Volatility:', percent_vol)
print("")
print("")
print("")


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# compute the expected returns & the annualised sample covariance 
# matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# optimize for max sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose = True)

# get the discrete allocation of each share per stock 
# -> how much of those stock can we possess for $ 150,000

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=150000)
allocation, leftover = da.lp_portfolio()
print('Discret allocation',allocation)
print('Funds remaining:${:.2f}'.format(leftover)) 


