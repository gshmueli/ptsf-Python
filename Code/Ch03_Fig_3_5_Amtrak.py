""" Code to create Figure 3-5 """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

from sktime.split import temporal_train_test_split
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.utils import plot_series

from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

style.use('ggplot')
figure, ax = plt.subplots(figsize=(6, 4))

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')

test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

forecaster = PolynomialTrendForecaster(degree=2, prediction_intervals=True)
forecaster.fit(ridership_train)
fitted_values = forecaster.predict(ridership_train.index)
pred_values = forecaster.predict(ridership_test.index)

pred_interval = forecaster.predict_interval(ridership_test.index, coverage=[0.95])
ax = plot_series(ridership, fitted_values, pred_values, ax=ax, pred_interval=pred_interval)

ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-', '-', '--'])
ax = ptsf_train_test(ax, ridership_train.index, ridership_test.index)
plt.savefig('Ch03_Fig_3_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
