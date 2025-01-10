""" Code to create Table 3-3 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsoluteError, \
     mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sktime.split import ExpandingWindowSplitter

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', index_col='Month')
ridership.index = pd.to_datetime(ridership.index, format='%Y %b')

y = ridership.copy()
y.index = y.index.to_period('M').to_timestamp('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)
forecaster = NaiveForecaster(strategy='last')

## Roll-forward calculations
mae, rmse, mape = (lambda: ([], [], []))()
for fh in np.arange(1,n_test + 1):
    cv = ExpandingWindowSplitter(initial_window=len(y_train), step_length=1, fh=[fh])
    res = evaluate(forecaster=forecaster, y=y, cv=cv, return_data=True, 
                        scoring=[MeanAbsoluteError(), mean_absolute_percentage_error])
    mae.append(res['test_MeanAbsoluteError'].mean())
    rmse.append(np.sqrt((res['test_MeanAbsoluteError']**2).mean()))
    mape.append(res['test__DynamicForecastingErrorMetric'].mean())

## Mean of roll-forward calculations
mean_values = list(map(lambda x: np.mean(np.array(x)), [mae, rmse, mape]))

## Fixed partition calculations
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=np.arange(n_test)+1)
mae_b = mean_absolute_error(y_test, y_pred)
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred))
mape_b = mean_absolute_percentage_error(y_test, y_pred)
fixed = [mae_b, rmse_b, mape_b]

a = ["Roll-forward " + str(i) + "-month-ahead" for i in range(1,n_test+1)]
df = pd.DataFrame({'Method': a, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
mean_row = pd.DataFrame([['Roll-forward overall'] + mean_values], columns=df.columns)
fixed_row = pd.DataFrame([['Fixed partitioning overall'] + fixed], columns=df.columns)
df = pd.concat([fixed_row, df, mean_row], ignore_index=True)
df['MAPE'] = df['MAPE'].multiply(100)
df[['MAE', 'RMSE', 'MAPE']] = df[['MAE', 'RMSE', 'MAPE']].round(3)
df['MAPE'] = df['MAPE'].map('{:,.3f}%'.format) ## to match book format
print(df.head(3))
print("... ... ...")
print(df.tail(2).to_string(header=False))
