""" Code related to Footnote 6 of Chapter 6 """
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.ardl import ARDL
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.transformations.series.fourier import FourierFeatures
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

test_size = len(ridership.truncate(before='2001-04-01'))
train, test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')
figure, ax = plt.subplots(figsize=(6, 4))

t = np.arange(1, len(ridership)+1)
columns = [t**i for i in range(3)]
X = pd.DataFrame(np.vstack(columns).T, index=ridership.index)
X.columns = ['const', 't', 't**2']

transformer = FourierFeatures(sp_list=[12], fourier_terms_list=[1])
X = pd.concat([X, transformer.fit_transform(X)], axis=1)

quad = ARDL(lags=0, order=0, trend='n', seasonal=False, auto_ardl=False)
quad.fit(train, X=X.iloc[:len(train),:])

fitted = quad.predict(train.index)
pred = quad.predict(test.index, X=X.iloc[len(train):,:])

plot_series(ridership, fitted, pred, ax=ax, markers=['']*3, 
            labels=None, x_label="", y_label="Ridership")
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2],
           lty=['-','-','--'])
ax = ptsf_train_test(ax, train.index, test.index)
plt.show()

print(X.head(5))
print(quad.summary())
