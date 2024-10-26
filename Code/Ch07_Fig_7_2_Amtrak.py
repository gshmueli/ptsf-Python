""" Code to create Figure 7.2 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_correlations
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv')
ridership['Month'] = pd.to_datetime(ridership['Month'], format='%Y %b')
ridership.set_index('Month', inplace=True)
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

y = ridership.copy().iloc[:24,0]

#plot_correlations(series=y, lags=11, alpha=0.05, zero_lag=False)

#print("Comments: The sktime function plot_correlations() function displays \
#the time series, the ACF and the PACF, and uses Bartlett confidence \
#intervals. We only show lags up to 11 as the PACF could not be \
#computed for a lag of 12 for a series of length 24. To reproduce the \
#plot exactly as in the book use plot_acf as below.  ")

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(x=y.values, lags=18, title="", bartlett_confint=False, ax=ax)

ax.set_xticks(range(0,19,1))
ax.set_xlabel("Lag")
ax.set_ylabel("acf")

plt.savefig('Ch07_Fig_7_2_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
