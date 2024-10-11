""" Code to create Figure 5.6 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from sktime.utils import plot_series
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)
style.use('ggplot')

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
y = ridership.copy() ## shorter name
y.index = y.index.to_period('M')
n_test = 36
y_train, y_test = temporal_train_test_split(y, test_size=n_test)

hwin = AutoETS(auto=False, error="mul", trend="add", seasonal="add", sp=12, n_jobs=-1)
hwin.fit(y_train)
hwin_fitted = hwin.predict(y_train.index)
hwin_pred = hwin.predict(y_test.index)

fig, ax = plt.subplots(figsize=(5.5,3.5))
ax = plot_series(y, hwin_fitted, hwin_pred, markers=['']*3,
            x_label="", y_label="Ridership", ax=ax)

ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])

ptsf_train_test(ax, y_train.index, y_test.index)
plt.savefig('Ch05_Fig_5_6_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()
