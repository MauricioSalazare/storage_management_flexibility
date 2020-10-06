import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
abs_path = Path(__file__).parent
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import pickle
import seaborn as sns
import Model_XGBoost.model_utils as mu


file_name = 'ems_optimization_2.1_100.csv'

(data_train, data_test) = mu.split_data(file_name, testing_split=0.2)
mu.plot_dataset(data_train, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])
mu.plot_dataset(data_test, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])

(x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                             columns_drop=['Scenario','v_1', ' storage_Q'],
                                                             columns_predict=[' storage_P'],
                                                             testing_split=0.2)


xgb_parameter_search = mu.model_training_xgboost(x_train, y_train)
y_test_hat = xgb_parameter_search.predict(x_test)

fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.plot(y_test_hat, label='Predicted')
ax.plot(y_test.values, label='Actual')
ax.legend()

mu.plot_parameters_importances(xgb_parameter_search, variable_names=x_test.columns)


################################################### NOISE XGBOOST ######################################################
# Inject noise to see how the algorithm responds
rmse = list()
noise_values = np.linspace(0, 0.1, 100)

# for noise_level in np.logspace(-10, -1, 10, endpoint=True):
for noise_level in noise_values:
    # print(noise_level)
    x_test_noise = x_test.drop(columns='time')
    noise = (x_test_noise * noise_level) * np.random.normal(size=x_test_noise.shape)
    # x_test_noise = x_test_noise + (x_test_noise * 0.05) * np.random.normal(size=x_test_noise.shape)
    x_test_noise = x_test.add(noise, fill_value=0).loc[:, x_test.columns]

    y_test_noise_hat = xgb_parameter_search.predict(x_test_noise)

    rmse.append(np.sqrt(mean_squared_error(y_test.values.ravel(), y_test_noise_hat.ravel())))


fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.plot(noise_values, rmse, label='Predicted')


fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.plot(y_test_noise_hat, label='Predicted')
ax.plot(y_test.values, label='Actual')
ax.legend()