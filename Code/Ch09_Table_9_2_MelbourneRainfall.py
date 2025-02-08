""" Code to create Table 9.2 """
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')

rainfall = rainfall.assign(t = np.arange(1,len(rainfall)+1,1),
    Seasonal_sine = lambda x: np.sin(2*np.pi*x['t']/365.25),
    Seasonal_cosine = lambda x: np.cos(2*np.pi*x['t']/365.25)
    )
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall = rainfall.assign(Lag1=rainfall['rainy'].shift(1))
rainfall = rainfall.iloc[1:,:] ## drop first row that has NA in lag

rain_train = rainfall.truncate(after='2009-12-31')
rain_test = rainfall.truncate(before='2010-01-01')

exog_train = rain_train[['Lag1','Seasonal_sine','Seasonal_cosine']]
exog_test = rain_test[['Lag1','Seasonal_sine','Seasonal_cosine']]

lr_model = LogisticRegression(random_state=0, fit_intercept=True, penalty=None)
lr_model.fit(exog_train, rain_train['rainy'])

print(pd.DataFrame({'Coef': ['Intercept'] + exog_train.columns.tolist(),
              'Estimate':np.append(lr_model.intercept_, lr_model.coef_[0])}))

fitted_probs = lr_model.predict_proba(exog_train)
print(pd.DataFrame(fitted_probs, columns=['Prob no rain', 'Prob rain'], index=exog_train.index))

pred_probs = lr_model.predict_proba(exog_test)
print(pd.DataFrame(pred_probs, columns=['Prob no rain', 'Prob rain'], index=exog_test.index))
