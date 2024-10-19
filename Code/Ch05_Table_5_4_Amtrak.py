""" Code to create Table 5.4 """
import warnings
import pandas as pd
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## shorter name
y.index = y.index.to_period('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

hwin_auto = AutoETS(auto=True, sp=12)
hwin_auto.fit(y_train)
print(hwin_auto._fitted_forecaster.summary())
