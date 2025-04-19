""" Code to create Table 7.4 """
#import warnings
import datetime
import pandas as pd

from sktime.forecasting.ardl import ARDL

#warnings.filterwarnings('ignore', category=FutureWarning)

close = pd.read_csv('ptsf-Python/Data/WalmartStock.csv', parse_dates=True, \
                    date_parser=lambda x: datetime.datetime.strptime(x, '%d-%b-%y'), \
                    index_col='Date')
close.index = pd.DatetimeIndex(close.index).to_period('D')
diff_close = close.iloc[:,0].diff().dropna().to_frame()

ar_close = ARDL(lags=1, auto_ardl=False)
ar_close.fit(close)
print(ar_close.summary())

ar_diff = ARDL(lags=1, auto_ardl=False)
ar_diff.fit(diff_close)
print(ar_diff.summary())
