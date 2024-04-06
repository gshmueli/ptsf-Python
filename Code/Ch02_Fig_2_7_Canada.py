""" Code to create Figure 2-7 """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from ptsf_setup import ptsf_theme

workhours = pd.read_csv('ptsf-Python/Data/CanadianWorkHours.csv', parse_dates=['Year'], index_col='Year')
workhours.columns = ['Hours Per Week']

style.use('ggplot') ## ggplot theme for plots
fig, ax = plt.subplots(figsize=(6, 4)) # 6x4 inch figure to match book aspect ratio
ax = plot_series(workhours, ax=ax)
ptsf_theme(ax, colors=['black'], idx=[0])
plt.savefig('Ch02_Fig_2_7_Canada.pdf', format='pdf', bbox_inches='tight')
plt.show()
