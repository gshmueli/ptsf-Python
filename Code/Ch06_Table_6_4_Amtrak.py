""" Code to create Table 6.4 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ardl import ARDL

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp()
test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.DataFrame(np.vstack(columns).T, index=ridership.index)
X.columns = ['const', 't', 't**2']

quad_seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
quad_seas.fit(train, X=X.iloc[:len(train),1:])
print(quad_seas.summary())
