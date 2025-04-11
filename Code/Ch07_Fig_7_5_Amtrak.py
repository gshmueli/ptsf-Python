""" Code to create Figure 7.5 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from PyFableARIMA import PyFableARIMA

warnings.filterwarnings('ignore', category=FutureWarning)

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

resid_resid = (resid - resid_ar_fitted).dropna()

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(x=resid_resid.values, lags=36, title="", bartlett_confint=False, ax=ax)
ax.set_xlabel("Lag")
ax.set_ylabel("acf")
plt.savefig('Ch07_Fig_7_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
