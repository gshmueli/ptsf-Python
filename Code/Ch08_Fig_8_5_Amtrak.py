""" Code for Figure 8.5 and Table 8.2 """
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

from sktime.utils import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error
from sklearn.exceptions import DataConversionWarning
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

from nnetar import NNetAR

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="RecursiveReductionForecaster is experimental")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

style.use('ggplot')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def accuracy_df(method, y, y_pred, y_train):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred) * 100,
        'MASE': mean_absolute_scaled_error(y_true=y, y_pred=y_pred, y_train=y_train)
    }
    return pd.DataFrame(metrics, index=[method])

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'],
                        date_parser=lambda x: pd.to_datetime(x, format='%Y %b')).set_index('Month')
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

TEST_SIZE = 36
SP = 12
train, test = temporal_train_test_split(ridership, test_size=TEST_SIZE)

nnetar = NNetAR(p=11, P=1, period=SP, n_nodes=7, n_networks=20, scale_inputs=False, auto=False)
nnetar.fit(train)
pred = nnetar.predict(test.index)
fitted = nnetar.predict(train.index[SP:])

fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(ridership, fitted, pred, markers=['']*3,
                 colors=['black','blue','blue'],
                 y_label='Ridership',ax=ax)
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])

ax = ptsf_train_test(ax, train.index, test.index)
plt.savefig('Ch08_Fig_8_5_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(pd.concat([accuracy_df('Train', train[SP:], fitted, train[SP:]),
    accuracy_df('Test', test, pred, train)]).round(3))
