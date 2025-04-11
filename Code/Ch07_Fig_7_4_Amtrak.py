""" Code to create Figure 7.4 """

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import plot_series
from sktime.forecasting.ardl import ARDL
from PyFableARIMA import PyFableARIMA
from ptsf_setup import ptsf_theme, ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv')
ridership['Month'] = pd.to_datetime(ridership['Month'], format='%Y %b')
ridership.set_index('Month', inplace=True)
ridership.index = ridership.index.to_period('M')

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

quad_seas = ARDL(lags=0, trend='ctt', seasonal=True, auto_ardl=False).fit(train)

fitted = quad_seas.predict(train.index)
resid = pd.Series((train.squeeze() - fitted.squeeze()), index=train.index, name='resid')

resid_ar = PyFableARIMA(formula='resid ~ 1 + pdq(1,0,0)')
        # '1 +' is used to add a constant to the ARIMA model
resid_ar.fit(resid)
resid_ar_fitted = resid_ar.get_fitted_values()

fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(resid, resid_ar_fitted, markers=['']*2,
                    labels=None, y_label="Residuals", ax=ax)
ptsf_theme(ax, colors=['black','blue'], idx=[0,1], lty=['-']*2)
ax.set_xlim([ridership.index.min(), ridership.index.max()])
ax = ptsf_train_test(ax, train.index, test.index)
plt.savefig('Ch07_Fig_7_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()

resid_ar.PyFableARIMA_report()

print("Residuals forecast:" )
print(resid_ar.predict([1]))
