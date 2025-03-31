""" Code to create Table 6.3 """
import warnings
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ardl import ARDL

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp()
test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)
seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
seas.fit(train, X=None)
print(seas.summary())
