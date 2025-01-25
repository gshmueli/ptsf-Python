""" Code to create Figure 6.3 and Table 6.1"""
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test
from ptsf_setup import ptsf_summary

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')

lm = TrendForecaster()
lm.fit(ridership_train)
fitted = lm.predict(ridership_train.index)
pred = lm.predict(ridership_test.index)

fig, ax = plt.subplots(figsize=(6,4))
ax = plot_series(ridership, fitted, pred, markers=['']*3,
            labels=None, x_label="", y_label="Ridership", ax=ax)
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])
ax = ptsf_train_test(ax, ridership_train.index, ridership_test.index)
plt.savefig('Ch06_Fig_6_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()

ptsf_summary(lm, ridership_train)
