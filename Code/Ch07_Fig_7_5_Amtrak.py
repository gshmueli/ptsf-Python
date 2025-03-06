""" Code to create Figure 7.5 """

import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from PyFableARIMA import PyFableARIMA

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
resid = pd.Series(train.values.flatten() - fitted.values, index=train.index, name='resid')

resid_ar = PyFableARIMA(formula='resid ~ 1 + pdq(1,0,0)')
        # '1 +' is used to add a constant to the ARIMA model
resid_ar.fit(resid)
resid_ar_fitted = resid_ar.get_fitted_values()

resid_of_resid = pd.Series(resid.values.flatten() - resid_ar_fitted.values, 
                           index=resid.index, name='resid_of_resid')

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(x=resid_of_resid.values, lags=36, title="", bartlett_confint=False, ax=ax)
ax.set_xlabel("Lag")
ax.set_ylabel("acf")
plt.savefig('Ch07_Fig_7_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
