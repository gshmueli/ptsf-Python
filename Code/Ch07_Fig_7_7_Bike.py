""" Code to create Figure 7.7 and Table 7.2"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error
import statsmodels.api as sm
from statsmodels.tools import add_constant
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

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

months_order = bike.index.strftime('%b').unique()
months = pd.get_dummies(pd.Categorical(bike.index.strftime('%b'), 
                        categories=months_order, ordered=True), dtype=int).drop(['Jan'], axis=1)
dow = pd.get_dummies(bike.index.strftime('%a'), dtype=int).drop(['Thu'], axis=1)
a = pd.get_dummies(bike['workingday'].astype('category'), prefix='Z', dtype=int)
a.columns = ['Nonworking' if col == 'Z_0' else 'Working' for col in a.columns]
b = pd.get_dummies(bike['weathersit'].astype('category'), prefix='Z', dtype=int)
b.columns = ['Clear' if col == 'Z_1' else 'Mist' if col == 'Z_2' else 'RainSnow' for col in b.columns]
c = pd.concat([a, b], axis=1).assign(**{f'{v}_{w}': a[v] * b[w] for v in a.columns for w in b.columns})
months, dow, c = map(lambda df: df.set_index(bike.index), [months, dow, c]) # ensure same index
X = pd.concat([months, c.iloc[:, -5:], dow], axis=1)
X['trend'] = np.arange(1, len(X) + 1)
X = add_constant(X)   # to get the intercept

train, test = temporal_train_test_split(bike['cnt'], test_size=90)

mdl = sm.OLS(train.values, X[:len(train)]).fit()

fitted = mdl.predict(X.iloc[:len(train),:])
pred = mdl.predict(X.iloc[len(train):,:])

figure, axis = plt.subplots(nrows=2, sharex=False, figsize=(12, 8))
figure.subplots_adjust(hspace=0.3)

# Extract lower and upper prediction intervals
alpha=0.05
pi = mdl.get_prediction(X.iloc[len(train):,:]).summary_frame(alpha=alpha) #95% confidence interval

pred_interval = pd.DataFrame({'lower': pi['obs_ci_lower'], 'upper': pi['obs_ci_upper']}, index=test.index)
pred_interval.columns = pd.MultiIndex.from_product([['cnt'],[1-alpha], ['lower', 'upper']])
axis[0] = plot_series(bike['cnt'], fitted, pred, markers=['']*3, ax=axis[0],
            pred_interval=pred_interval,  title="Rentals and Forecasts", labels=None, y_label="Count")
ptsf_theme(axis[0], colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])
axis[0] = ptsf_train_test(axis[0], train.index, test.index)


axis[1] = plot_series(train - fitted, test - pred, markers=['']*2, ax=axis[1],
            title="Errors", labels=None, y_label="Error")
ptsf_theme(axis[1], colors=['black']*2, idx=[0,1], lty=['-','--'])

plt.savefig('Ch07_Fig_7_7_Bike.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(mdl.summary())

print(pd.concat([
    accuracy_df("Train", train, fitted, train),
    accuracy_df("Test", test, pred, train)]).drop('MASE',axis=1).round(0))
