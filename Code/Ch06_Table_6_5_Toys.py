""" Code to create Table 6.5 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ardl import ARDL
from sktime.performance_metrics.forecasting import mean_absolute_error, \
    mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore', category=FutureWarning)

def accuracy(actual, predicted):
    MAE = mean_absolute_error(actual, predicted)
    ME = np.mean(actual - predicted)
    MAPE = 100 * mean_absolute_percentage_error(actual, predicted)
    RMSE = np.sqrt(mean_squared_error(actual, predicted))
    df = pd.DataFrame({'MAE': MAE, 'ME': ME, 'MAPE': MAPE, 'RMSE': RMSE}, 
                        index=['Accuracy'])
    return df

revenue = pd.read_csv('ptsf-Python/Data/ToysRUsRevenues.csv')
revenue['Quarter'] = revenue['Quarter'].apply(lambda x: '-'.join(x.split('-')[::-1]))
revenue['Quarter'] = pd.PeriodIndex(revenue['Quarter'], freq='Q')
revenue.set_index('Quarter', inplace=True)

train, test = temporal_train_test_split(revenue, test_size=2)

lin_seas = ARDL(lags=0, order=0, trend='ct', seasonal=True, auto_ardl=False)
lin_seas.fit(train)

fitted = lin_seas.predict(train.index)
pred = lin_seas.predict(test.index)

print(lin_seas.summary())
acc_train = accuracy(train, fitted).round(2)
acc_test = accuracy(test, pred).round(2)
results = pd.concat([acc_train, acc_test], keys=['Training set', 'Test set'])
print(results)


