""" Code to create Figure 2-4 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.dates import MonthLocator
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from ptsf_setup import ptsf_theme

ridership = pd.read_csv('ptsf-Python/Data/Amtrak data.csv', parse_dates=['Month'], index_col='Month')

style.use('ggplot') ## ggplot theme for plots
figure, axs = plt.subplots(2, 1, sharex=False, figsize=(6, 4)) ## 2-by-1 grid; aspect ratio 6:4

forecaster = PolynomialTrendForecaster(degree=2)
forecaster.fit(ridership)
fh = ForecastingHorizon(ridership.index, is_relative=False)
fitted_values = forecaster.predict(fh)

ridership_zoom = ridership.loc['1997-01-01':'2000-12-31']

axs[0] = plot_series(ridership, fitted_values, ax=axs[0])
axs[1] = plot_series(ridership_zoom, ax=axs[1])

axs[1].xaxis.set_major_locator(MonthLocator(bymonth=1, bymonthday=1))
plt.subplots_adjust(hspace=0.5)
ptsf_theme(axs[0], colors=['black', 'blue'], idx=[0,1])
ptsf_theme(axs[1], colors=['black'], idx=[0])
plt.savefig('Ch02_Fig_2_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
