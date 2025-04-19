""" Code to create Figure 7.10 """
#import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series

#warnings.filterwarnings('ignore', category=FutureWarning)

close = pd.read_csv('ptsf-Python/Data/WalmartStock.csv', parse_dates=True, index_col='Date')
close.index = pd.DatetimeIndex(close.index).to_period('D')

style.use('ggplot')
fig, ax = plt.subplots(figsize=(6,4))
ax = plot_series(close, colors=['black'], markers=[''], y_label='Close Price ($)', ax=ax)
plt.savefig('Ch07_Fig_7_10_Walmart_Stock.pdf', format='pdf', bbox_inches='tight')
plt.show()
