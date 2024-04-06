""" Code to create Figure 3-1 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

ridership = pd.read_csv('ptsf-Python/Data/Amtrak data.csv', parse_dates=['Month'], index_col='Month')

train = ridership.truncate(after='2001-03-31')
test = ridership.truncate(before='2001-04-01')

style.use('ggplot') ## ggplot theme for plots
fig, ax = plt.subplots(figsize=(6, 4))

ax = plot_series(ridership, ax=ax)

ptsf_theme(ax, colors=['black'], idx=[0])
ax = ptsf_train_test(ax, train.index, test.index)
plt.savefig('Ch03_Fig_3_1_Amtrak.pdf')
plt.show()
