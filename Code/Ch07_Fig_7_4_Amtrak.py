""" Code to create Figure 7.4 """
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

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv')
ridership['Month'] = pd.to_datetime(ridership['Month'], format='%Y %b')
ridership.set_index('Month', inplace=True)
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.DataFrame(np.vstack(columns).T, index=ridership.index)
X.columns = ['const', 't', 't**2']
quad_seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
quad_seas.fit(train, X=X.iloc[:len(train),1:])

quad_seas_fitted = quad_seas.predict(train.index, X=X.iloc[:len(train),1:])
resid_train = train - quad_seas_fitted

resid_ar = ARDL(lags=1, trend='c', seasonal=False, auto_ardl=False)
resid_ar.fit(resid_train, X=None)
resid_ar_fitted =  resid_ar.predict(train.index)
fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(resid_train, resid_ar_fitted, markers=['']*2,
                    labels=None, y_label="Residuals", ax=ax)
ptsf_theme(ax, colors=['black','blue'], idx=[0,1], lty=['-']*2)
ax.set_xlim([ridership.index.min(), ridership.index.max()])
ax = ptsf_train_test(ax, train.index, test.index)
plt.savefig('Ch07_Fig_7_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
print(resid_ar.summary())

print("Residuals forecast:" )
print(resid_ar.predict(ForecastingHorizon(1,is_relative=True)))
