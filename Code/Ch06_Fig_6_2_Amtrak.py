""" Code to create Figure 6.2 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster
from ptsf_setup import ptsf_theme

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')

lm = TrendForecaster()
lm.fit(ridership_train)
fitted = lm.predict(ridership_train.index)

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(ridership_train, fitted, markers=['',''],
                 labels=None, x_label="", y_label="Ridership", ax=ax)
ptsf_theme(ax, colors=['black','blue'], idx=[0,1], lty=['-']*2)
plt.savefig('Ch06_Fig_6_2_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
