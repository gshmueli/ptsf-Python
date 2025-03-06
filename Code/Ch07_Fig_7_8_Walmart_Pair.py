"""Code to create Figure 7.8 and Table 7.3"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error
from PyFableARIMA import PyFableARIMA
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

style.use('ggplot')

def accuracy_df(method, y, y_pred, y_train):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred),
        'MASE': mean_absolute_scaled_error(y_true=y, y_pred=y_pred, y_train=y_train)
    }
    return pd.DataFrame(metrics, index=[method])

one_pair = pd.read_csv('ptsf-Python/Data/Walmart_One_Pair.csv', 
                       parse_dates=['Date'], index_col='Date')
one_pair.index = one_pair.index.to_period('W')
one_pair = one_pair[['Weekly_Sales','IsHoliday']].dropna()  # final rows have NA

test_size = len(one_pair.truncate(before='2012-02-06'))
train, test = temporal_train_test_split(one_pair, test_size=test_size)
y_train, X_train = train['Weekly_Sales'], train['IsHoliday'] 
y_test, X_test = test['Weekly_Sales'], test['IsHoliday']

# Automated ARIMA fitted to the training set
SAR1 = PyFableARIMA(formula='Weekly_Sales').fit(y=y_train, X=None)                 # ignore IsHoliday
AR2_IsHoliday = PyFableARIMA(formula='Weekly_Sales ~ IsHoliday').fit(y=y_train, X=X_train)      # include IsHoliday

def calc_residuals(obs, pred):
    return pd.Series(obs.values.flatten() - pred.squeeze().values, index=obs.index)

def apply_model(model, y_train, y_test, X_test, fitted, pred, resid_train, resid_test): 
    fitted.append(model.get_fitted_values())
    pred.append(model.predict(fh=test.index, X=X_test))
    resid_train.append(calc_residuals(y_train, fitted[-1]))
    resid_test.append(calc_residuals(y_test, pred[-1]))

fitted, pred, resid_train, resid_test = [], [], [], []

apply_model(SAR1, y_train, y_test, None, fitted, pred, resid_train, resid_test)
apply_model(AR2_IsHoliday, y_train, y_test, X_test, fitted, pred, resid_train, resid_test)

figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

figure.subplots_adjust(hspace=0.3)

axis[0] = plot_series(one_pair['Weekly_Sales'], fitted[0], fitted[1], pred[0], pred[1], 
                      markers=['']*5, ax=axis[0], title="Sales and Forecasts", y_label="Sales")
ptsf_theme(axis[0], colors=['black','green','red','green','red'], 
           idx=[0,1,2,3,4], lty=['-','-','-','--','--'], 
           labels=['Actual','SAR1 (fitted)','AR2_IsHoliday (fitted)',
                              'SAR1 (pred)','AR2_IsHoliday (pred)'],
           do_legend=True
           )
axis[0] = ptsf_train_test(axis[0], train.index, test.index)

axis[1] = plot_series(resid_train[0], resid_train[1], resid_test[0], resid_test[1], 
                      markers=['']*4, title="Errors", labels=None, y_label="Error", ax=axis[1])
ptsf_theme(axis[1], colors=['green','red']*2, idx=[0,1,2,3], lty=['-','-','--','--'], 
           labels = ['SAR1 (train)','AR2_IsHoliday (train)','SAR1 (test)','AR2_IsHoliday (test)'],
           do_legend=True)
plt.savefig('Ch07_Fig_7_8_Walmart_Pair.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("SAR1")
SAR1.PyFableARIMA_report()

print("AR2_IsHoliday")
AR2_IsHoliday.PyFableARIMA_report()

df4 = pd.concat([accuracy_df("SAR1 (training)", y_train, fitted[0], y_train), \
                accuracy_df("SAR1 (test)", y_test, pred[0], y_train), 
                accuracy_df("AR2_IsHoliday (training)", y_train, fitted[1], y_train), \
                accuracy_df("AR2_IsHoliday (test)", y_test, pred[1], y_train)])
df4['MAPE'] = df4['MAPE'].multiply(100).map('{:,.2f}'.format) ## to match book format
df5 = df4.round({'MAE': 1, 'RMSE': 1, 'MASE': 3})[['RMSE', 'MAE', 'MAPE','MASE']].reset_index()
df5.rename(columns={'index': 'Series'}, inplace=True)
df5['Series'] = df5['Series'].apply(lambda x: x.ljust(20)) # Left-justify the first column
print(df5.to_string(index=False))
