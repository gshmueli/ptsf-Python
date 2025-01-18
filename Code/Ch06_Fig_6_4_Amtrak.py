""" Code to create Figure 6.4 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster
from sktime.transformations.series.boxcox import LogTransformer
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')

lm = TrendForecaster()
lm.fit(ridership_train)
lm_fitted = lm.predict(ridership_train.index)
lm_pred = lm.predict(ridership_test.index)

exp_lm = LogTransformer() * TrendForecaster()
exp_lm.fit(ridership_train)
exp_lm_fitted = exp_lm.predict(ridership_train.index)
exp_lm_pred = exp_lm.predict(ridership_test.index)

labels=['Observed','Linear Fit','Linear Forecast','Exponential Fit','Exponential Forecast']
fig, ax = plt.subplots(figsize=(6,4))
ax = plot_series(ridership, lm_fitted, lm_pred, exp_lm_fitted, exp_lm_pred,
                      markers=['']*5, labels=labels, x_label="", y_label="Ridership", ax=ax)
ptsf_theme(ax, colors=['black','green','green','orange','orange'], idx=[0,1,2,3,4],
           lty=['-','-','--','-','--'], labels=labels, do_legend=True)
ax = ptsf_train_test(ax, ridership_train.index, ridership_test.index, ylim=[1300, 3000])
plt.savefig('Ch06_Fig_6_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()


## THE FOLLOWING CODE PRODUCES THE MEAN FORECAST (RATHER THAN THE MEDIAN FORECAST)
#import numpy as np
#from sktime.forecasting.trend import PolynomialTrendForecaster
#exp_lm = PolynomialTrendForecaster(degree=1, prediction_intervals=True)
#exp_lm.fit(ridership_train.map(np.log))
#fh0 = ridership_train.index ## shorter name
#fh1 = ridership_test.index ## shorter name
#exp_lm_fitted = (exp_lm.predict(fh0) + exp_lm.predict_var(fh0).values/2).map(np.exp)
#exp_lm_pred = (exp_lm.predict(fh1) + exp_lm.predict_var(fh1).values/2).map(np.exp)
