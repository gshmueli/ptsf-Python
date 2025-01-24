""" Code to create Table 6.2 """
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import PolynomialTrendForecaster
from ptsf_setup import ptsf_summary

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

qm = PolynomialTrendForecaster(degree=2)  ## quadratic trend model
qm.fit(ridership_train)

ptsf_summary(qm, ridership_train)
