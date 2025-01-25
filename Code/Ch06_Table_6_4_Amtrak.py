""" Code to create Table 6.4 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')

# Create dummy variables for the months and drop January (Month_1)
mat = pd.get_dummies(ridership.index.month, prefix='Month', dtype=float).drop(columns=['Month_1'])

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.concat([mat, pd.DataFrame(np.vstack(columns).T)], axis=1)
X.columns = list(mat.columns) + ['const', 't', 't**2']

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

quad_seas = sm.OLS(train.values, X[:len(train)]).fit()
print(quad_seas.summary())
