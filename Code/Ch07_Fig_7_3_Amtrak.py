""" Code to create Figure 7.3 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv')
ridership['Month'] = pd.to_datetime(ridership['Month'], format='%Y %b')
ridership.set_index('Month', inplace=True)
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.DataFrame(np.vstack(columns).T, index=ridership.index)
X.columns = ['const', 't', 't**2']

quad_seas = ARDL(lags=0, order=0, trend='c', seasonal=True, auto_ardl=False)
quad_seas.fit(train, X=X.iloc[:len(train),:])

fitted = quad_seas.predict(train.index)
resid_train = train - fitted

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(x=resid_train.values, lags=18, title="", bartlett_confint=False, ax=ax)
ax.set_xticks(range(0,19,1))
ax.set_xlabel("Lag")
ax.set_ylabel("acf")

plt.savefig('Ch07_Fig_7_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
