""" Code to create Table 5.2 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## shorter name
y.index = y.index.to_period('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

hwin = AutoETS(auto=False, error="mul", trend="add", seasonal="add", sp=12, n_jobs=-1)
hwin.fit(y_train)
print(hwin.summary())
print("\n\nFinal states:\n=============")
print(hwin._fitted_forecaster.states.tail(12))
