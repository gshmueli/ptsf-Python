""" Code to create Figure 7.3 """

import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

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
resid_train = train.squeeze().values - fitted

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(x=resid_train, lags=18, title="", bartlett_confint=False, ax=ax)
ax.set_xticks(range(0,19,1))
ax.set_xlabel("Lag")
ax.set_ylabel("acf")

plt.savefig('Ch07_Fig_7_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
