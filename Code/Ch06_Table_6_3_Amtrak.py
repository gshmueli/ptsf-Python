""" Code to create Table 6.3 """
import warnings
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')

# Create dummy variables for the months and drop January (Month_1)
mat = pd.get_dummies(ridership.index.month, prefix='Month', dtype=float).drop(columns=['Month_1'])
X = sm.add_constant(mat)   # add a constant column for the intercept

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

model = sm.OLS(train.values, X[:len(train)]).fit()
print(model.summary())
