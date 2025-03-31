""" Code to create Figure 6.6 """
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
ridership.index = ridership.index.to_period('M').to_timestamp()
test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

train.index = train.index.to_period('M').to_timestamp('M')
test.index = test.index.to_period('M').to_timestamp('M')

style.use('ggplot')
figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

ardl_seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
ardl_seas.fit(train, X=None)

fh_fit = ForecastingHorizon(train.index, is_relative=False)
fh = ForecastingHorizon(test.index, is_relative=False)
seas_fitted = ardl_seas.predict(fh_fit)
seas_pred = ardl_seas.predict(fh)

plot_series(ridership, seas_fitted, seas_pred, ax=axis[0], markers=['']*3, 
            labels=None, x_label="", y_label="Ridership")
ptsf_theme(axis[0], colors=['black','blue','blue'], idx=[0,1,2],
           lty=['-','-','--'])
ax = ptsf_train_test(axis[0], train.index, test.index)

resid_train = train - seas_fitted
resid_test = test - seas_pred
plot_series(resid_train, resid_test, ax=axis[1], markers=['']*2, 
            labels=None, x_label="", y_label="Residuals")
ptsf_theme(axis[1], colors=['black']*2, idx=[0,1], lty=['-','-'])
ax = ptsf_train_test(axis[1], train.index, test.index)
plt.subplots_adjust(hspace=0.4)
plt.savefig('Ch06_Fig_6_6_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
