""" Code to create Figure 6.13 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

sales = pd.read_csv('ptsf-Python/Data/DepartmentStoreSales.csv')
sales['Quarter'] = sales['Quarter'].apply(lambda x: '-'.join(x.split('-')[::-1]))
sales['Quarter'] = pd.PeriodIndex(sales['Quarter'], freq='Q')
sales.set_index('Quarter', inplace=True)
sales['logSales'] = np.log(sales['Sales'])
logSales = sales['logSales'].to_frame()

train, test = temporal_train_test_split(logSales, test_size=4)

exp_seas = ARDL(lags=0, order=0, trend='ct', seasonal=True, auto_ardl=False)
exp_seas.fit(train)

fitted = exp_seas.predict(train.index)
resid = train - fitted # display residuals in log scale
exp_fitted = np.exp(fitted) # display fitted in original scale

style.use('ggplot')
figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

axis[0] = plot_series(sales['Sales'].iloc[:len(train)], exp_fitted, markers=['']*2,    
                      y_label='Sales', ax=axis[0])
axis[1] = plot_series(resid, colors=['black'], markers=[''], y_label='Residuals', ax=axis[1])

years = mdates.YearLocator()  # every year
quartersFmt = mdates.DateFormatter('%Y-Q1')
ptsf_theme(axis[0], colors=['black','blue'], idx=[0,1], lty=['-','--'])
for i in range(2):
    axis[i].xaxis.set_major_locator(years)
    axis[i].xaxis.set_major_formatter(quartersFmt)
plt.subplots_adjust(hspace=0.4)
plt.savefig('Ch06_Fig_6_13_Department.pdf', format='pdf', bbox_inches='tight')
plt.show()
