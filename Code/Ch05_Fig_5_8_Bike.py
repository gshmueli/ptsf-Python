""" Code to create Figure 5.8  """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.detrend.mstl import MSTL
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

bike_df = pd.read_csv('ptsf-Python/Data/BikeSharingHourly.csv',parse_dates=True,index_col=1)
rides = bike_df['cnt'].to_frame()
rides = rides.truncate(before='2012-07-01').truncate(after='2012-07-31')
date_range = pd.date_range(start=rides.index.min(),
            end=rides.index.min()+pd.Timedelta(hours=24*31-1), freq='H')
rides.index = date_range
rides_train = rides.iloc[:(21*24),:] # 3 weeks

mstl = MSTL(periods=np.array([24, 7*24]), return_components=False)  
deseas = mstl.fit(rides_train).transform(rides_train)

fh = ForecastingHorizon(np.arange(1,241,1), is_relative=True) ## 10 days = 240 hourly forecasts
deseas_pred = AutoETS(auto=False, seasonal=None, n_jobs=-1).fit(deseas).predict(fh)
daily_pred = NaiveForecaster(strategy="last",sp=24).fit(mstl.seasonal_['seasonal_24']).predict(fh)
weekly_pred = NaiveForecaster(strategy="last",sp=7*24).fit(mstl.seasonal_['seasonal_168']).predict(fh)

# create the combined forecasts
pred = deseas_pred['cnt'].add(daily_pred).add(weekly_pred)

# plot the original series and the forecasted series
fig, ax = plt.subplots(figsize=(6.5,4.5))
plot_series(rides_train, pred, markers=['',''], y_label="Hourly bike rentals", ax=ax)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU)) # xticks on Sundays 00:00
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ptsf_theme(ax, colors=['black','blue'], idx=[0,1], lty=['-','--'])
plt.savefig('Ch05_Fig_5_8_Bike.pdf', format='pdf', bbox_inches='tight')
plt.show()
