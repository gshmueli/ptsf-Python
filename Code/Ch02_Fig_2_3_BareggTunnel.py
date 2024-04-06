""" Code to create Figure 2-3 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

baregg_tunnel = pd.read_csv('ptsf-Python/Data/BareggTunnel.csv', parse_dates=['Day'], index_col='Day')
baregg_tunnel.columns = ["Number of Vehicles"]

style.use('ggplot') ## ggplot theme for plots
fig, axs = plt.subplots(2, 1, figsize=(6, 4))
data_sets = [baregg_tunnel, baregg_tunnel['2004-02-01':'2004-05-31']]
locators = [MonthLocator(interval=6), MonthLocator()]

for ax, data_set, locator in zip(axs, data_sets, locators):
    ax = plot_series(data_set, ax=ax)    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.tick_params(axis='both', labelsize='small')  # reduce size of ticklabels
    ax.xaxis.get_label().set_fontsize(9)  # reduce size of x-axis label
    ax.yaxis.get_label().set_fontsize(9)  # reduce size of y-axis label
    for line in ax.lines:
        line.set_linewidth(0.7)  # change line width
    ptsf_theme(ax, colors=['black'], idx=[0]) ## override some sktime.plot_series() defaults
    
plt.subplots_adjust(hspace=0.6)
plt.savefig('Ch02_Fig_2_3_BareggTunnel.pdf', format='pdf', bbox_inches='tight')
plt.show()

