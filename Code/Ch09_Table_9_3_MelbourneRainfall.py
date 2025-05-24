""" Code to create Table 9.3 """
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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

lr_model = LogisticRegression(random_state=0, fit_intercept=True, penalty=None)
lr_model.fit(exog_train, rain_train['rainy'])

def performance_summary(label, pred, actual):
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    print(f"{label} Confusion Matrix\n        Prediction\nActual     0     1")
    print(f"  0     {tn}   {fp}\n  1     {fn}   {tp}\n\n\nAccuracy : {(tp+tn)/len(actual):.4f}")
    print(f"Sensitivity : {(tp/(tp+fn)):.4f}\nSpecificity : {(tn/(tn+fp)):.4f}")

train_pred = lr_model.predict(exog_train)
performance_summary("Training", train_pred, rain_train['rainy'])

test_pred = lr_model.predict(exog_test)
performance_summary("Test", test_pred, rain_test['rainy'])
