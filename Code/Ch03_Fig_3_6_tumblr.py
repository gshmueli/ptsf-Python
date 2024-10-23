""" Code to create Figure 3-6 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils import plot_series
from sktime.forecasting.ets import AutoETS
#from ptsf_setup import *
   
tumblr = pd.read_csv('ptsf-Python/Data/Tumblr.csv', parse_dates=True, index_col=0)
tumblr.index = tumblr.index.to_period('M')
views = (tumblr['People Worldwide'] / 1e6).to_frame()

style.use('ggplot')
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 5.2))
plt.subplots_adjust(wspace=0.6)
def customize_plot(ax, title):
    ax.set_title(title)
    ax.set_ylim([0, 1000])
    ax.set_xticks([pd.Period(year, freq='M') for year in ['2010', '2015', '2020']])
    ax.set_xticklabels(['2010', '2015', '2020'])
    ax.get_legend().remove()

fh = ForecastingHorizon(np.arange(1,116), is_relative=True)
models = ["AAN","MMN","MMdN"]
for i, z in enumerate([("add", "add", False), ("mul", "mul", False), ("mul", "mul", True)]):
    forecaster = AutoETS(error=z[0], trend=z[1], damped_trend=z[2]).fit(views)
    pred = forecaster.predict(fh)
    pred_interval = forecaster.predict_interval(fh=fh, coverage=np.array([.8,.6,.4,.2]))
    plot_series(views, pred, ax=axs[i], markers=['',''], colors=['black','blue'], 
                pred_interval=pred_interval, y_label="People (in millions)")
    customize_plot(axs[i], f"ETS({models[i]})")

plt.savefig('Ch03_Fig_3_6_tumblr.pdf', format='pdf', bbox_inches='tight')
