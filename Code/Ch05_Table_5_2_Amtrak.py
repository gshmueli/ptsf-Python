""" Code to create Table 5.2 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## shorter name
y.index = y.index.to_period('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

hwin = AutoETS(auto=False, error="mul", trend="add", seasonal="add", sp=12, 
               start_params = np.array([ 0.2, 0.2, 0.2, 1838, 0.5, 28.44, -11.22, -0.53, 
                    -124.28, 200.20, 146.81, 36.59, 76.04, 60.15, 44.18, -249.47, -206.92]), n_jobs=-1)
hwin.fit(y_train)
print(hwin._fitted_forecaster.summary())
print("\n\nFinal states:\n=============")
print(hwin._fitted_forecaster.states.tail(12))
