""" Code to create Figure 2-2 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

style.use('ggplot') ## ggplot theme for plots
fig, ax = plt.subplots(figsize=(5, 3)) ## aspect ratio 5 x 3

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ax = plot_series(ridership, ax=ax)

ptsf_theme(ax, colors=['black'], idx=[0]) ## override some sktime.plot_series() defaults
plt.savefig('Ch02_Fig_2_2_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
