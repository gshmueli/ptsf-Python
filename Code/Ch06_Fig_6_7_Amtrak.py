""" Code to create Figure 6.7 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')
figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.DataFrame(np.vstack(columns).T, index=ridership.index)
X.columns = ['const', 't', 't**2']

quad_seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
quad_seas.fit(train, X=X.iloc[:len(train),:])

fh_fit = ForecastingHorizon(train.index, is_relative=False)
fh = ForecastingHorizon(test.index, is_relative=False)

fitted = quad_seas.predict(fh_fit)
pred = quad_seas.predict(fh, X=X.iloc[len(train):,:])

plot_series(ridership, fitted, pred, ax=axis[0], markers=['']*3, 
            labels=None, x_label="", y_label="Ridership")
ptsf_theme(axis[0], colors=['black','blue','blue'], idx=[0,1,2],
           lty=['-','-','--'])
ax = ptsf_train_test(axis[0], train.index, test.index)

resid_train = train - fitted
resid_test = test - pred
plot_series(resid_train, resid_test, ax=axis[1], markers=['']*2, 
            labels=None, x_label="", y_label="Residuals")
ptsf_theme(axis[1], colors=['black']*2, idx=[0,1], lty=['-','-'])
ax = ptsf_train_test(axis[1], train.index, test.index)
plt.subplots_adjust(hspace=0.4)
plt.savefig('Ch06_Fig_6_7_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()