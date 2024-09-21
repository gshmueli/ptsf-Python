""" Code to create Figure 5.5 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
diff_twice = ridership.diff(12).diff(1).dropna().copy()
diff_twice.index = diff_twice.index.to_period('M')
N_TEST = 36
diff_twice_train, diff_twice_test = \
    temporal_train_test_split(diff_twice, test_size=N_TEST)

ses = ExponentialSmoothing(smoothing_level=0.2) ## trend=seasonal=None
ses.fit(diff_twice_train)
fh_fitted = ForecastingHorizon(diff_twice_train.index, is_relative=False)
ses_fitted = ses.predict(fh_fitted)
fh = ForecastingHorizon(diff_twice_test.index, is_relative=False)
ses_pred = ses.predict(fh)

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(diff_twice, ses_fitted, ses_pred, markers=['']*3,
            x_label="", y_label="Twice-Differenced", ax=ax)

ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])

ptsf_train_test(ax, diff_twice_train.index, diff_twice_test.index)
plt.savefig('Ch05_Fig_5_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
