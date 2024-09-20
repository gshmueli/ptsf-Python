""" Code to create Figure 3-8 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp()
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

f_naive = NaiveForecaster(strategy='last', sp=1).fit(ridership_train)
fh = ForecastingHorizon(ridership_test.index, is_relative=False)
pred_naive = f_naive.predict(fh=fh)
pred_naive_roll_fwd = ridership['Ridership'].shift(1).truncate(before='2001-04-01')

style.use('ggplot')
figure, axs = plt.subplots(2, 1, sharex=False, figsize=(8, 7))

# Plot 1: actuals and forecasts
axs[0] = plot_series(ridership, pred_naive, pred_naive_roll_fwd, ax=axs[0])
ptsf_theme(axs[0], colors=['black','blue','red'], idx=[0,1,2], lty=['-', '--', '--'])
axs[0] = ptsf_train_test(axs[0], ridership_train.index, ridership_test.index)

# Plot 2: errors
naive_err_train = ridership_train['Ridership'] - ridership['Ridership'].iloc[0]
naive_roll_fwd_err_train = ridership_train['Ridership'] - ridership_train['Ridership'].shift(1)
naive_err_test = ridership_test['Ridership'] - pred_naive.iloc[:,0]
naive_roll_fwd_err_test = ridership_test['Ridership'] - pred_naive_roll_fwd

axs[1] = plot_series(naive_err_train, naive_roll_fwd_err_train, naive_err_test, naive_roll_fwd_err_test, ax=axs[1],
                    title='Errors', y_label='Error')
ptsf_theme(axs[1], colors=['blue','red']*2, idx=[0,1,2,3], lty=['-', '-','--','--'])
axs[1] = ptsf_train_test(axs[1], ridership_train.index, ridership_test.index)
plt.subplots_adjust(hspace=0.5)

plt.savefig('Ch03_Fig_3_8_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
