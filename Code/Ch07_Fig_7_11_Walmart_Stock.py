""" Code to create Figure 7.11 """
import warnings
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.style as style
from statsmodels.graphics.tsaplots import plot_acf


warnings.filterwarnings('ignore', category=FutureWarning)

close = pd.read_csv('ptsf-Python/Data/WalmartStock.csv', parse_dates=True, \
                    date_parser=lambda x: datetime.datetime.strptime(x, '%d-%b-%y'), \
                    index_col='Date')
close.index = pd.DatetimeIndex(close.index).to_period('D')
diff_close = close.iloc[:,0].diff().dropna().to_frame()

# style.use('ggplot') ## ggplot theme for plots
fig, axs = plt.subplots(nrows=1, ncols=2)

_ = plot_acf(x=close.values, ax=axs[0], lags=10,
         title="ACF Plot for Close", bartlett_confint=False)
axs[0].set_xlabel('Lag')

_ = plot_acf(x=diff_close.values, ax=axs[1], lags=10,
         title="ACF Plot for Differenced Series", bartlett_confint=False)
axs[1].set_xlabel('Lag')

plt.subplots_adjust(wspace=0.5)
plt.savefig('Ch07_Fig_7_11_Walmart_Stock.pdf', format='pdf', bbox_inches='tight')
plt.show()
