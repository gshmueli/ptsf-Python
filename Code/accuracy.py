import pandas as pd
import numpy as np
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def accuracy(actual, predicted):
    MAE = mean_absolute_error(actual, predicted)
    ME = np.mean(actual - predicted)
    MAPE = 100 * mean_absolute_percentage_error(actual, predicted)
    RMSE = np.sqrt(mean_squared_error(actual, predicted))
    df = pd.DataFrame({'MAE': MAE, 'ME': ME, 'MAPE': MAPE, 'RMSE': RMSE}, 
                        index=['Accuracy'])
    return df
