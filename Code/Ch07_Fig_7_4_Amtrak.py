""" Code to create Figure 7.4 """

import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import plot_series
from r_ARIMA import r_ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.style as style
from ptsf_setup import ptsf_theme, ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

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

resid_ar = r_ARIMA(formula='resid ~ 1 + pdq(1,0,0)')
        # '1 +' is used to add a constant to the ARIMA model
        # for seasonal '+ PDQ(P,D,Q)' (the default is PDQ(0,0,0))
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

resid_ar.r_ARIMA_report()

print("Residuals forecast:" )
print(resid_ar.predict([1]))
