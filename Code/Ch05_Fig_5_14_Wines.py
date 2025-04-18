""" Code to create Figure 5.14 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from matplotlib.dates import YearLocator, DateFormatter

ddd = {'B':np.float64, 'C':np.float64, 'D':np.float64,
       'E':np.float64, 'F':np.float64, 'G':np.float64}
wines = pd.read_csv("ptsf-Python/Data/AustralianWines.csv",
               parse_dates=True, index_col=0, dtype=ddd, na_values=['*'])
wines = wines.asfreq('MS')
wines.columns = wines.columns.str.replace('.', ' ', regex=False)

style.use('ggplot')
fig, axs = plt.subplots(3, 2, figsize=(8, 10), sharex=True)
for i in range(6):
    y_label = '' if i != 2 else 'Thousands of liters'
    plot_series(wines.iloc[:, i], markers=[''], ax=axs[i // 2, i % 2],
                y_label=y_label, colors=['black'], title=wines.columns[i])
    axs[i // 2, i % 2].set_title(wines.columns[i] + ' Wine Sales', fontsize=10)
    axs[i // 2, i % 2].xaxis.set_major_locator(YearLocator(5))  # xticks every 5-years
    axs[i // 2, i % 2].xaxis.set_major_formatter(DateFormatter('%Y-Jan'))

plt.subplots_adjust(hspace=0.5)
plt.savefig('Ch05_Fig_5_14_Wines.pdf', format='pdf', bbox_inches='tight')
plt.show()
