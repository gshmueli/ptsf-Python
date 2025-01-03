""" Code to create Figure 9.2 """
import pandas as pd
import numpy as np

rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall['Lag1'] = rainfall['rainy'].shift(1)
rainfall = rainfall.assign(t = np.arange(1,len(rainfall)+1,1))
rainfall = rainfall.assign(
    Seasonal_sine = lambda x: np.sin(2*np.pi*x['t']/365.25),
    Seasonal_cosine = lambda x: np.cos(2*np.pi*x['t']/365.25)
    )
print(rainfall)
