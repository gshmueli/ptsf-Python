""" Code to create Figure 5.3 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## use a shorter name
y_test = y.truncate(before='2001-04-01')
fh = ForecastingHorizon(y_test.index, is_relative=False)

ma_trailing = y.rolling(window=12, center=False).mean().truncate(after='2001-03-31')

forecaster = NaiveForecaster(strategy='last', sp=1)
forecaster.fit(ma_trailing)
ma_trailing_pred = forecaster.predict(fh)

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(y, ma_trailing, ma_trailing_pred, markers=['']*3,
                 x_label="Time", y_label="Ridership", ax=ax)

ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])

ptsf_train_test(ax, ma_trailing.index, y.index)

plt.savefig('Ch05_Fig_5_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
