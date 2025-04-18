""" Code to create Figure 5.15 """
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

def parse_quarter(string):
    """ Convert 'Season-YYYY' to a datetime object representing the quarter """
    season, y = string.split('-')
    seasons = {'Winter': '01', 'Spring': '04', 'Summer': '07', 'Fall': '10'}
    month = seasons[season]
    return pd.to_datetime(f'{y}-{month}')

sales = pd.read_csv("ptsf-Python/Data/NaturalGasSales.csv", dtype={'Date': 'object'}, index_col=0)
sales.index = pd.to_datetime(sales.index.map(parse_quarter))
sales.index = sales.index.to_period('Q')
sales['MA(4)'] = sales['Gas Sales'].rolling(window=4).mean()

style.use('ggplot')
fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(sales['Gas Sales'], sales['MA(4)'], markers=['']*2, y_label="Billions BTU", ax=ax)
ptsf_theme(ax, colors=['black','red'], lty=['-','--'], 
           idx=[0,1], labels=['Gas Sales','MA(4)'], do_legend=True)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q1'))

plt.savefig('Ch05_Fig_5_15_NaturalGas.pdf', format='pdf', bbox_inches='tight')
plt.show()
