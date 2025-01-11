""" Code to create Table 9.5 """
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Load the dataset
rainfall = pd.read_csv('ptsf-Python/Data/MelbourneRainfall.csv', parse_dates=['Date'], index_col='Date')
rainfall.index = pd.to_datetime(rainfall.index, format='%d/%m/%Y')

rainfall = rainfall.assign(t=np.arange(1, len(rainfall) + 1, 1),
                           Seasonal_sine=lambda x: np.sin(2 * np.pi * x['t'] / 365.25),
                           Seasonal_cosine=lambda x: np.cos(2 * np.pi * x['t'] / 365.25)
                           )
rainfall['rainy'] = np.where(rainfall['RainfallAmount_millimetres'] > 0, 1, 0)
rainfall = rainfall.assign(Lag1=rainfall['rainy'].shift(1)).iloc[1:, :]  # drop first row that has NA in lag

rain_train = rainfall.truncate(after='2009-12-31')
rain_test = rainfall.truncate(before='2010-01-01')

X_train = rain_train[['Lag1', 'Seasonal_sine', 'Seasonal_cosine']]
X_test = rain_test[['Lag1', 'Seasonal_sine', 'Seasonal_cosine']]
y_train = rain_train['rainy']
y_test = rain_test['rainy']

def performance_summary(label, pred, actual):
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    print(f"{label} Confusion Matrix")
    print("        Prediction")
    print("Actual     0     1")
    print(f"  0     {tn}   {fp}")
    print(f"  1     {fn}   {tp}")
    print("\n\n")
    print(f"Accuracy : {(tp+tn)/len(actual):.4f}")
    print(f"Sensitivity : {(tp/(tp+fn)):.4f}")
    print(f"Specificity : {(tn/(tn+fp)):.4f}")

# Number of MLPClassifiers
NETWORKS = 20
HIDDEN_NODES = 2

# Create an ensemble of N MLPClassifier models
estimators = [(f'mlp{i}', 
               MLPClassifier(hidden_layer_sizes=(HIDDEN_NODES,), random_state=42+i, max_iter=1000)) 
              for i in range(NETWORKS)]
ensemble = VotingClassifier(estimators=estimators, voting='soft')

ensemble.fit(X_train, y_train)
train_pred = ensemble.predict(X_train)
test_pred = ensemble.predict(X_test)

performance_summary("Training", train_pred, y_train)
performance_summary("Test", test_pred, y_test)
