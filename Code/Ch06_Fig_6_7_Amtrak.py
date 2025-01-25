""" Code to create Figure 6.7 """

import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import plot_series
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.style as style
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

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
fitted = quad_seas.predict(X[:len(train)])
pred = quad_seas.predict(X[len(train):])

fitted.index = pd.period_range(start=train.index[0], periods=len(train), freq='M')
pred.index = pd.period_range(start=test.index[0], periods=len(test), freq='M')

style.use('ggplot')
figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

axis[0] = plot_series(ridership, fitted, pred, ax=axis[0], markers=['']*3, 
            labels=None, x_label="", y_label="Ridership")
ptsf_theme(axis[0], colors=['black','blue','blue'], idx=[0,1,2],
           lty=['-','-','--'])
axis[0] = ptsf_train_test(axis[0], train.index, test.index)

resid_train = train.squeeze() - fitted
resid_test = test.squeeze() - pred
axis[1] = plot_series(resid_train, resid_test, ax=axis[1], markers=['']*2, 
            labels=None, x_label="", y_label="Residuals")
ptsf_theme(axis[1], colors=['black']*2, idx=[0,1], lty=['-','-'])
axis[1] = ptsf_train_test(axis[1], train.index, test.index)
plt.subplots_adjust(hspace=0.4)
plt.savefig('Ch06_Fig_6_7_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
