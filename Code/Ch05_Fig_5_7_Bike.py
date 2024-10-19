""" Code to create Figure 5.7  """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

bike_df = pd.read_csv('ptsf-Python/Data/BikeSharingHourly.csv',parse_dates=True,index_col=1)
rides = bike_df['cnt'].to_frame()
rides = rides.truncate(before='2012-07-01').truncate(after='2012-07-31')
date_range = pd.date_range(start=rides.index.min(),
            end=rides.index.min()+pd.Timedelta(hours=24*31-1), freq='H')
rides.index = date_range
rides_train = rides.iloc[:(21*24),:] # 3 weeks
rides_zoom = rides.iloc[:(4*24)] # 4 days

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8.5,6.5))
plt.subplots_adjust(hspace=0.5)

ax[0] = plot_series(rides_train, markers=[''], x_label='Week', y_label="Hourly Bike Rentals",
                title='Weekly Pattern (3 weeks)', colors=['black'], ax=ax[0])
ax[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU)) # xticks on Sundays 00:00
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # xtick format

ax[1] = plot_series(rides_zoom, markers=[''], x_label='Day', y_label="Hourly Bike Rentals",
                title='Daily Pattern (4 days)', colors=['black'], ax=ax[1])
ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 18]))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
  
plt.savefig('Ch05_Fig_5_7_Bike.pdf', format='pdf', bbox_inches='tight')
plt.show()
