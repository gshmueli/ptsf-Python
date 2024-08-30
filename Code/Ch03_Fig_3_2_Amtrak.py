""" Code to create Figure 3-2 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import PolynomialTrendForecaster
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')

test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

# Fit model to training period
forecaster = PolynomialTrendForecaster(degree=2)
forecaster.fit(ridership_train)
fitted_values = forecaster.predict(ForecastingHorizon(ridership_train.index, is_relative=False))

# use model to generate forecasts for test period
pred_values = forecaster.predict(ForecastingHorizon(ridership_test.index, is_relative=False))

style.use('ggplot') ## ggplot theme for plots
fig, ax = plt.subplots(figsize=(6, 4))

ax = plot_series(ridership, fitted_values, pred_values, ax=ax)

ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-', '-', '--'])
ax = ptsf_train_test(ax, ridership_train.index, ridership_test.index)
plt.savefig('Ch03_Fig_3_2_Amtrak.pdf')
plt.show()
