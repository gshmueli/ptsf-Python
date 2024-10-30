""" Code to create Figure 7.7 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

def accuracy_df(method, y, y_pred, y_train):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred) * 100,
        'MASE': mean_absolute_scaled_error(y_true=y, y_pred=y_pred, y_train=y_train)
    }
    return pd.DataFrame(metrics, index=[method])

warnings.filterwarnings('ignore', category=FutureWarning)

bike = pd.read_csv('ptsf-Python/Data/BikeSharingDaily.csv',parse_dates=True,index_col=1)
bike.index = pd.DatetimeIndex(bike.index).to_period('D')

# Create X - the exogenous variables
months = pd.get_dummies(bike.index.strftime('%b')).set_index(bike.index).drop(['Jan'], axis=1)
dow = pd.get_dummies(bike.index.strftime('%a')).set_index(bike.index).drop(['Thu'], axis=1)
working = pd.get_dummies(bike['workingday'].astype('category'), prefix='Z')
working = working.rename(columns = {'Z_0': 'Nonworking', 'Z_1': 'Working'})
weather = pd.get_dummies(bike['weathersit'].astype('category'), prefix='Z')
weather = weather.rename(columns = {'Z_1': 'Clear', 'Z_2': 'Mist', 'Z_3': 'RainSnow'})
inter_cols = [f'{col1}_{col2}' \
    for i, col1 in enumerate(working.columns) for j, col2 in enumerate(weather.columns)]
tmp_df = pd.concat([working, weather], axis=1)
inter = tmp_df.assign(**{col: tmp_df.eval(col.replace('_','*')) for col in inter_cols})
X = pd.concat([months[['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']],
    inter[['Working_Clear','Nonworking_Mist','Working_Mist','Nonworking_RainSnow','Working_RainSnow']],
    dow[['Fri', 'Sat', 'Sun','Mon','Tue','Wed']]], axis=1)

y_train, y_test = temporal_train_test_split(bike['cnt'], test_size=90)
n_train = len(y_train)

mdl = ARDL(lags=0, order=0, trend='ct', seasonal=False, auto_ardl=False)
_ = mdl.fit(y_train, X=X.iloc[:n_train,:])

fitted = mdl.predict(y_train.index, X=X.iloc[:n_train,:])
pred = mdl.predict(y_test.index, X=X.iloc[n_train:,:])

resid_train = y_train - fitted
resid_test = y_test - pred

figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))

axis[0] = plot_series(bike['cnt'], fitted, pred, markers=['']*3, ax=axis[0],
            title="Rentals and Forecasts", labels=None, y_label="Count")
ptsf_theme(axis[0], colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])
axis[0] = ptsf_train_test(axis[0], y_train.index, y_test.index)

axis[1] = plot_series(resid_train, resid_test, markers=['']*2, ax=axis[1],
            title="Errors", labels=None, y_label="Error")
ptsf_theme(axis[1], colors=['black']*2, idx=[0,1], lty=['-','--'])
plt.savefig('Ch07_Fig_7_7_Bike.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(mdl.summary())

print(pd.concat([
    accuracy_df("Train", y_train, fitted, y_train),
    accuracy_df("Test", y_test, pred, y_train)]).drop('MASE',axis=1).round(0))

""" # The following code produces the plot with the prediction intervals
import statsmodels.api as sm
import numpy as np
Z = bike['cnt'].to_frame()
Z = Z.merge(X, left_index=True, right_index=True, how='left')
Z['trend'] = np.arange(1, len(X)+1)
form = 'cnt ~ ' + ' + '.join(list(Z.columns)[1:])
Z_train = Z.iloc[:n_train,:]
Z_test = Z.iloc[n_train:,:]
bike_lm = sm.OLS.from_formula(form, data=Z_train)
bike_fit = bike_lm.fit()
bike_pred = bike_fit.get_prediction(exog=Z_test.drop(['cnt'], axis=1))
alpha=0.05
prediction_intervals = bike_pred.summary_frame(alpha=alpha) #95% confidence interval
# Extract lower and upper confidence intervals
lower = prediction_intervals['obs_ci_lower']
upper = prediction_intervals['obs_ci_upper']
pred_interval = pd.DataFrame({'lower': lower, 'upper': upper})
pred_interval.index=pred.index
pred_interval.columns = pd.MultiIndex.from_product([['cnt'],[1-alpha], ['lower', 'upper']])
fig, ax = plot_series(bike['cnt'], fitted, pred, markers=['']*3, pred_interval=pred_interval,
            title="Rentals and Forecasts", labels=None, y_label="Count")
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])
plt.show()
 """
 