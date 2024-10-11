""" Code to create Table 5-1 """
#from IPython.display import display, Markdown
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ets import AutoETS
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error

warnings.filterwarnings('ignore', category=FutureWarning)

def accuracy_df(method, y, y_pred, y_train):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred),
        'MASE': mean_absolute_scaled_error(y_true=y, y_pred=y_pred, y_train=y_train)
    }
    return pd.DataFrame(metrics, index=[method])


ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
diff_twice = ridership.diff(12).diff(1).dropna().copy()
diff_twice.index = diff_twice.index.to_period('M')
N_TEST = 36
diff_twice_train, diff_twice_test = \
    temporal_train_test_split(diff_twice, test_size=N_TEST)

ses = ExponentialSmoothing(smoothing_level=0.2) ## trend=seasonal=None
ses.fit(diff_twice_train)
ses_fitted = ses.predict(diff_twice_train.index)
ses_pred = ses.predict(diff_twice_test.index)

ses_opt = AutoETS(auto=False, trend=None, seasonal=None, return_params=False, n_jobs=-1)
ses_opt.fit(diff_twice_train)
alpha = ses_opt._fitted_forecaster.smoothing_level
L0 = ses_opt._fitted_forecaster.initial_level
print(f"Smoothing parameter (alpha): {alpha}")
print(f"Initial level: {L0}")

ses_opt_fitted = ses_opt.predict(diff_twice_train.index)
ses_opt_pred = ses_opt.predict(diff_twice_test.index)

train = diff_twice_train
test = diff_twice_test
df4 = pd.concat([accuracy_df("Training", train, ses_fitted, train), \
                accuracy_df("Test", test, ses_pred, train), \
                accuracy_df("Training", train, ses_opt_fitted, train), \
                accuracy_df("Test", test, ses_opt_pred, train)])
df4['MAPE'] = df4['MAPE'].multiply(100).map('{:,.0f}'.format) ## to match book format
df5 = df4.round({'MAE': 1, 'RMSE': 1, 'MASE': 3})[['RMSE', 'MAE', 'MAPE','MASE']].reset_index()
df5.rename(columns={'index': 'Series'}, inplace=True)
df5['Model'] = ["$\\alpha=0.2$" for _ in range(2)] + ["Opt. $\\alpha$" for _ in range(2)]
df6 = df5[['Model', 'Series', 'RMSE', 'MAE', 'MAPE', 'MASE']]
print(df6)
