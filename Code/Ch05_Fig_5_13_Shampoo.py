""" Code to create Figure 5.13 """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

shampoo = pd.read_csv('ptsf-Python/Data/ShampooSales.csv', parse_dates=True, index_col=0)
shampoo.index = shampoo.index.to_period('M')  # Set frequency to monthly (12 periods per year)
style.use('ggplot')
fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(shampoo, markers=[''], y_label='Units (in thousands)', ax=ax)
ptsf_theme(ax, colors=['black'], idx=[0])
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Jan'))

plt.savefig('Ch05_Fig_5_13_Shampoo.pdf', format='pdf', bbox_inches='tight')
plt.show()
