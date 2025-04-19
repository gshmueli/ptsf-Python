""" Code to create Table 6.6 """
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ardl import ARDL

sales = pd.read_csv('ptsf-Python/Data/DepartmentStoreSales.csv')
sales['Quarter'] = sales['Quarter'].apply(lambda x: '-'.join(x.split('-')[::-1]))
sales['Quarter'] = pd.PeriodIndex(sales['Quarter'], freq='Q')
sales.set_index('Quarter', inplace=True)
sales['logSales'] = np.log(sales['Sales'])
logSales = sales['logSales'].to_frame()

train, test = temporal_train_test_split(logSales, test_size=4)

exp_seas = ARDL(lags=0, order=0, trend='ct', seasonal=True, auto_ardl=False)
exp_seas.fit(train)

print(exp_seas.summary())
