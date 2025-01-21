""" Code to create Figure 6.4 """
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from sktime.utils import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.boxcox import LogTransformer
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test

warnings.filterwarnings('ignore', category=FutureWarning)

ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv', parse_dates=['Month'], index_col='Month')
ridership.index = ridership.index.to_period('M')
test_size = len(ridership.truncate(before='2001-04-01'))
ridership_train, ridership_test = temporal_train_test_split(ridership, test_size=test_size)

style.use('ggplot')

lm = TrendForecaster()  ## linear trend model
lm.fit(ridership_train)
lm_fitted = lm.predict(ridership_train.index)
lm_pred = lm.predict(ridership_test.index)

qm = PolynomialTrendForecaster(degree=2)  ## quadratic trend model
qm.fit(ridership_train)
qm_fitted = qm.predict(ridership_train.index)
qm_pred = qm.predict(ridership_test.index)

exp_lm = LogTransformer() * TrendForecaster()  ## exponential trend model
exp_lm.fit(ridership_train)
exp_lm_fitted = exp_lm.predict(ridership_train.index)
exp_lm_pred = exp_lm.predict(ridership_test.index)

labels=['Observed','Linear Fit','Linear Forecast','Quadratic Fit','Quadratic Forecast',
        'Exponential Fit','Exponential Forecast']
fig, ax = plt.subplots(figsize=(6,4))
ax = plot_series(ridership, lm_fitted, lm_pred, qm_fitted, qm_pred, exp_lm_fitted, exp_lm_pred,
                      markers=['']*7, labels=labels, x_label="", y_label="Ridership", ax=ax)
ptsf_theme(ax, colors=['black','green','green','blue','blue','orange','orange'], 
           idx=[0,1,2,3,4,5,6], lty=['-','-','--','-','--','-','--'], 
           labels=labels, do_legend=True)
ax = ptsf_train_test(ax, ridership_train.index, ridership_test.index, ylim=[1300, 3000])

#################################################################################################
### THIS SECTION COMPRESSES THE INFORMATION IN THE LEGEND SO THAT IT TAKES UP MINIMAL SIZE
# Create custom legend handles
observed = mlines.Line2D([], [], color='black', linestyle='-', label='Observed')
linear = [mlines.Line2D([], [], color='green', linestyle='-'), mlines.Line2D([], [], color='green', linestyle='--')]
quadratic = [mlines.Line2D([], [], color='blue', linestyle='-'), mlines.Line2D([], [], color='blue', linestyle='--')]
exponential = [mlines.Line2D([], [], color='orange', linestyle='-'), mlines.Line2D([], [], color='orange', linestyle='--')]

# Custom handler to increase the width of the legend handles and adjust label position
class HandlerTupleVertical(HandlerTuple):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        a = super().create_artists(legend, orig_handle, xdescent + 16, ydescent, width * 2, height, fontsize, trans)
        return a

# Combine handles for the legend
handles = [observed, tuple(linear), tuple(quadratic), tuple(exponential)]
labels = ['Observed', 'Linear', 'Quadratic', 'Exponential']

# Add the custom legend
ax.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTupleVertical(ndivide=None)}, bbox_to_anchor=(0.1, 0.6), loc='center left')
### END OF SECTION DEALING WITH THE LEGEND
#################################################################################################

plt.savefig('Ch06_Fig_6_4_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()


## THE FOLLOWING CODE PRODUCES THE MEAN FORECAST (RATHER THAN THE MEDIAN FORECAST)
#import numpy as np
#from sktime.forecasting.trend import PolynomialTrendForecaster
#exp_lm = PolynomialTrendForecaster(degree=1, prediction_intervals=True)
#exp_lm.fit(ridership_train.map(np.log))
#fh0 = ridership_train.index ## shorter name
#fh1 = ridership_test.index ## shorter name
#exp_lm_fitted = (exp_lm.predict(fh0) + exp_lm.predict_var(fh0).values/2).map(np.exp)
#exp_lm_pred = (exp_lm.predict(fh1) + exp_lm.predict_var(fh1).values/2).map(np.exp)
