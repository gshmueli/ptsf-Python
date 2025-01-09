""" Code to create Tables 3-4, 3-5, 3-6 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, MeanAbsoluteError

warnings.filterwarnings('ignore', category=FutureWarning)

def accuracy_df(method, y, y_pred):
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred)
    df = pd.DataFrame({'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, index=[method])
    return df

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', index_col='Month')
ridership.index = pd.to_datetime(ridership.index, format='%Y %b')

y = ridership.copy()
y.index = y.index.to_period('M').to_timestamp('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

## Table 3-4
y_train.index = y_train.index.to_period('M')
f_naive = NaiveForecaster(strategy='last', sp=1).fit(y_train)
f_naive_seas = NaiveForecaster(strategy='last', sp=12).fit(y_train)

pred_naive = f_naive.predict(fh=np.arange(n_test)+1).squeeze()
pred_seas = f_naive_seas.predict(fh=np.arange(n_test)+1).squeeze()

y_test.index = pred_seas.index
df = pd.DataFrame({'Actual': y_test.squeeze(), 'Naive Forecast': pred_naive, 
                   'Seasonal Naive Forecast': pred_seas}, index=y_test.index)
df.index = df.index.strftime("%b %Y")

print(df.head(3))
print("....")
print(df.tail(1).to_string(header=False))

## Table 3-5

df2 = pd.concat([accuracy_df("Naive Forecast", y_test, pred_naive), \
            accuracy_df("Seasonal Naive Forecast", y_test, pred_seas)])
df2['MAPE'] = df2['MAPE'].multiply(100)
df2[['MAE', 'RMSE', 'MAPE']] = df2[['MAE', 'RMSE', 'MAPE']].round(3)
df2['MAPE'] = df2['MAPE'].map('{:,.3f}%'.format)
print(df2)

## Table 3-6

def res_cv2df(res, method):
    df = pd.DataFrame({'MAE': res['test_MeanAbsoluteError'].mean(),
                       'RMSE': np.sqrt((res['test_MeanAbsoluteError']**2).mean()),
                       'MAPE': res['test__DynamicForecastingErrorMetric'].mean()}, 
                       index=[method])
    return df

y.index = y.index.to_period('M')
f_naive = NaiveForecaster(strategy='last', sp=1)
f_naive_seas = NaiveForecaster(strategy='last', sp=12)
cv = ExpandingWindowSplitter(initial_window=len(y_train), step_length=1, fh=[1])

res_naive = evaluate(forecaster=f_naive, y=y, cv=cv, 
                        scoring=[MeanAbsoluteError(), mean_absolute_percentage_error], 
                        return_data=True)
res_seas = evaluate(forecaster=f_naive_seas, y=y, cv=cv, 
                        scoring=[MeanAbsoluteError(), mean_absolute_percentage_error], 
                        return_data=True)

df3 = pd.concat([res_cv2df(res_naive, "Naive Forecast"), 
                 res_cv2df(res_seas, "Seasonal Naive Forecast")])
df3['MAPE'] = df3['MAPE'].multiply(100)
df3[['MAE', 'RMSE', 'MAPE']] = df3[['MAE', 'RMSE', 'MAPE']].round(3)
df3['MAPE'] = df3['MAPE'].map('{:,.3f}%'.format)
print(df3)