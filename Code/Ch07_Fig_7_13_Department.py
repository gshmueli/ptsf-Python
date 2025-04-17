""" Code to create Figure 7.13 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

warnings.filterwarnings('ignore', category=FutureWarning)

shipments = pd.read_csv('ptsf-Python/Data/ApplianceShipments.csv')
shipments['Quarter'] = shipments['Quarter'].apply(lambda x: '-'.join(x.split('-')[::-1]))
shipments['Quarter'] = pd.PeriodIndex(shipments['Quarter'], freq='Q')
shipments.set_index('Quarter', inplace=True)

shipments.sort_index(inplace=True)
shipments.index = shipments.index.to_timestamp()

style.use('ggplot')
fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(shipments, y_label='Shipments (millions $)', ax=ax)
ptsf_theme(ax, colors=['black'], idx=[0])
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q1'))
plt.savefig('Ch07_Fig_7_13_Department.pdf', format='pdf', bbox_inches='tight')
plt.show()
