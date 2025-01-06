""" Code to create Table 9.3 """
import pandas as pd
import numpy as np
import statsmodels.api as sm

rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')

rainfall = rainfall.assign(t = np.arange(1,len(rainfall)+1,1),
    Seasonal_sine = lambda x: np.sin(2*np.pi*x['t']/365.25),
    Seasonal_cosine = lambda x: np.cos(2*np.pi*x['t']/365.25)
    )
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall = rainfall.assign(Lag1=rainfall['rainy'].shift(1)).iloc[1:,:] ## drop first row that has NA in lag

rain_train = rainfall.truncate(after='2009-12-31')
rain_test = rainfall.truncate(before='2010-01-01')

exog_train = rain_train[['Lag1','Seasonal_sine','Seasonal_cosine']]
exog_test = rain_test[['Lag1','Seasonal_sine','Seasonal_cosine']]

# Add a constant term for the intercept
exog_train = sm.add_constant(exog_train)
exog_test = sm.add_constant(exog_test)

# Fit the logistic regression model using GLM with MLE
glm_model = sm.GLM(rain_train['rainy'], exog_train, family=sm.families.Binomial())
glm_results = glm_model.fit()

# Print the summary of the model
print(glm_results.summary())
