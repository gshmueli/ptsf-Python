""" Code to create Table 5-5 """
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
y = ridership.copy() ## shorter name
y.index = y.index.to_period('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

hwin = AutoETS(auto=False, error="mul", trend="add", seasonal="add", sp=12, 
               start_params = np.array([ 0.2, 0.2, 0.2, 1838, 0.5, 28.44, -11.22, -0.53, 
                    -124.28, 200.20, 146.81, 36.59, 76.04, 60.15, 44.18, -249.47, -206.92]), n_jobs=-1)
hwin.fit(y_train)
hwin_fitted = hwin.predict(y_train.index)
hwin_pred = hwin.predict(y_test.index)

hwin_auto = AutoETS(auto=True, sp=12)
hwin_auto.fit(y_train)
hwin_auto_fitted = hwin_auto.predict(y_train.index)
hwin_auto_pred = hwin_auto.predict(y_test.index)


df4 = pd.concat([accuracy_df("Training", y_train, hwin_fitted, y_train), \
                accuracy_df("Test", y_test, hwin_pred, y_train), \
                accuracy_df("Training", y_train, hwin_auto_fitted, y_train), \
                accuracy_df("Test", y_test, hwin_auto_pred, y_train)])
df4['MAPE'] = df4['MAPE'].multiply(100).map('{:,.2f}'.format) ## to match book format
df5 = df4.round({'MAE': 1, 'RMSE': 1, 'MASE': 3})[['RMSE', 'MAE', 'MAPE','MASE']].reset_index()
df5.rename(columns={'index': 'Series'}, inplace=True)
df5['Model'] = ["$\\alpha=0.2$" for _ in range(2)] + ["Opt. $\\alpha$" for _ in range(2)]
df6 = df5[['Model', 'Series', 'RMSE', 'MAE', 'MAPE', 'MASE']]
print(df6)
