""" Code to create Figure 3-3 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import PolynomialTrendForecaster
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

ridership = pd.read_csv('ptsf-Python/Data/Amtrak data.csv', parse_dates=['Month'], index_col='Month')

test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

# Fit model to training period
forecaster = PolynomialTrendForecaster(degree=2)
forecaster.fit(ridership_train)
fitted_values = forecaster.predict(ForecastingHorizon(ridership_train.index, is_relative=False))

# Get forecasts for test period
pred_values = forecaster.predict(ForecastingHorizon(ridership_test.index, is_relative=False))

style.use('ggplot')
figure, axs = plt.subplots(2, 1, sharex=False, figsize=(6, 4))

# Plot 1: actuals and forecasts
axs[0] = plot_series(ridership, fitted_values, pred_values, ax=axs[0],
                        title='Actual and forecasted ridership', y_label='Ridership')
ptsf_theme(axs[0], colors=['black','blue','blue'], idx=[0,1,2], lty=['-', '-', '--'])
axs[0] = ptsf_train_test(axs[0], ridership_train.index, ridership_test.index)

# Plot 2: errors
axs[1] = plot_series(ridership_train - fitted_values, ridership_test - pred_values, ax=axs[1],
                        title='Errors', y_label='Error')
ptsf_theme(axs[1], colors=['black']*2, idx=[0,1], lty=['-', '--'])
axs[1].axhline(0, color='.4', linestyle='-') 
axs[1].axvline(x=ridership_train.index[-1], color='0.37', linewidth=1.3)
plt.subplots_adjust(hspace=0.5)

plt.savefig('Ch03_Fig_3_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
