""" Code to create Figure 3-4 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import PolynomialTrendForecaster
from ptsf_setup import ptsf_theme

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')

test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

forecaster = PolynomialTrendForecaster(degree=2)
forecaster.fit(ridership_train)
fitted_values = forecaster.predict(ForecastingHorizon(ridership_train.index, is_relative=False))
pred_values = forecaster.predict(ForecastingHorizon(ridership_test.index, is_relative=False))

style.use('ggplot')
figure, ax = plt.subplots(figsize=(6, 4))

plt.hist(ridership_test - pred_values, bins=7, alpha=0.8, color='white', edgecolor='black')
plt.xlabel('Forecast Error')
plt.ylabel('Frequency')
plt.savefig('Ch03_Fig_3_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
