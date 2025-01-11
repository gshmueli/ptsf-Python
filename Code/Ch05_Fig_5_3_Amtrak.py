""" Code to create Figure 5.3 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.naive import NaiveForecaster
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## use a shorter name
y_train = y.truncate(after='2001-03-31') 
y_test = y.truncate(before='2001-04-01')

fc = NaiveForecaster(strategy='mean', sp=1, window_length=12) ## moving average forecaster
fc.fit(y_train)
fitted = fc.predict(y_train.index[12:])
pred = fc.predict(y_test.index)

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(y, fitted, pred, markers=['']*3, x_label="", y_label="Ridership", ax=ax)
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])
ptsf_train_test(ax, y_train.index, y.index)
plt.savefig('Ch05_Fig_5_3_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
