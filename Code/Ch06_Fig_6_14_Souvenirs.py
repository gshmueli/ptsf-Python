""" Code to create Figure 6-14 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series

style.use('ggplot')
figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

souvenirs = pd.read_csv("ptsf-Python/Data/SouvenirSales.csv")
souvenirs['Month'] = pd.to_datetime(souvenirs['Month'], format='%Y %b')
souvenirs.set_index('Month', inplace=True)

axis[0] = plot_series(souvenirs, colors=['black'], markers=[''],
                      y_label='Sales (Australian Dollars)', ax=axis[0])
axis[1] = plot_series(np.log(souvenirs), colors=['black'], markers=[''],
                      y_label='log(Sales)', ax=axis[1])

years = mdates.YearLocator()  # every year
monthsFmt = mdates.DateFormatter('%Y Jan')
for i in range(2):
    axis[i].xaxis.set_major_locator(years)
    axis[i].xaxis.set_major_formatter(monthsFmt)
    axis[i].tick_params(axis='x', labelsize=12)  # Increase font size for x-axis tick labels
    axis[i].tick_params(axis='y', labelsize=12)  # Increase font size for y-axis tick labels
    axis[i].set_ylabel(axis[i].get_ylabel(), fontsize=14)  # Increase font size for y-axis labels

plt.subplots_adjust(hspace=0.4)
plt.savefig('Ch06_Fig_6_14_Souvenirs.pdf', format='pdf', bbox_inches='tight')
plt.show()
