""" Code to create Figure 8.7 """
import os
import warnings
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor

from sktime.utils import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
#from sktime.forecasting.compose import EnsembleForecaster
from sktime.forecasting.compose._reduce import RecursiveReductionForecaster
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error, \
        mean_absolute_percentage_error, mean_absolute_scaled_error
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
#from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from ptsf_setup import ptsf_theme
from ptsf_setup import ptsf_train_test


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="RecursiveReductionForecaster is experimental")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

matplotlib.use('TkAgg') 
style.use('ggplot')

# Set random seed for reproducibility
RANDOM_SEED = 48
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def accuracy_df(method, y, y_pred, y_train):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'MAPE': mean_absolute_percentage_error(y, y_pred) * 100,
        'MASE': mean_absolute_scaled_error(y_true=y, y_pred=y_pred, y_train=y_train)
    }
    return pd.DataFrame(metrics, index=[method])

#################################################################################################################

# Define the model creation function
def create_lstm_model(look_back, batch_size=1):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, look_back, 1)))
    model.add(LSTM(units=50, dropout=0.01, recurrent_dropout=0.01, stateful=True, return_sequences=True))
    model.add(LSTM(units=50, dropout=0.01, recurrent_dropout=0.01, stateful=True, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

# Custom callback to reset states
class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch=None, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

######################################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())
ridership = pd.read_csv('ptsf-Python/Data/Amtrak.csv')
ridership['Month'] = pd.to_datetime(ridership['Month'], format='%Y %b')
ridership.set_index('Month', inplace=True)
ridership.index = ridership.index.to_period('M').to_timestamp('M')  # Convert to month-end frequency

TEST_SIZE = 36
train, test = temporal_train_test_split(ridership, test_size=TEST_SIZE)

SP = 12 # SP = Seasonal Period
BATCH_SIZE = 1
regressor = KerasRegressor(build_fn=create_lstm_model, batch_size=BATCH_SIZE,
                             look_back=SP, epochs=400, verbose=2,shuffle=False,
                             callbacks=[ResetStatesCallback()])

lstm_fc = RecursiveReductionForecaster(regressor, window_length=SP)

pipe = TransformedTargetForecaster( [
    ("standardize", TabularToSeriesAdaptor(MinMaxScaler())),
    ("LSTM", lstm_fc)
    ])

pipe.fit(train)

pred = pipe.predict(ForecastingHorizon(test.index, is_relative=False))
fitted = pipe.predict(train.index[SP:])

fig, ax = plt.subplots(figsize=(6, 4))
ax = plot_series(ridership, fitted, pred, markers=['']*3,
                 colors=['black','blue','blue'],
                 y_label='Ridership',ax=ax)
ptsf_theme(ax, colors=['black','blue','blue'], idx=[0,1,2], lty=['-','-','--'])

ax = ptsf_train_test(ax, train.index, test.index)
plt.savefig('Ch08_Fig_8_7_Amtrak.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(pd.concat([accuracy_df('Train', train[SP:], fitted, train[SP:]),
    accuracy_df('Test', test, pred, train)]).round(3))
