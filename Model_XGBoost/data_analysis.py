import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
abs_path = Path(__file__).parent
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
import pickle

data_scenarios = pd.read_csv(abs_path.parents[0] / 'Model_optimization/ems_optimization.csv')

n_scenarios = data_scenarios['Scenario'].max()
idx_scenarios = np.random.choice(range(0, n_scenarios), round(n_scenarios * .2))  # 20% Testing
idx_test = data_scenarios['Scenario'].isin(idx_scenarios)
# data_scenarios.drop(columns='Scenario', inplace=True)
data_train = data_scenarios.loc[~idx_test,:]
data_test = data_scenarios.loc[idx_test,:]

# TRAIN DATA
fig = plt.figure(figsize=(10, 3))
ax_ = fig.subplots(1, 3)
variables = ['Loading', ' SOC', ' storage_P']

for ax, variable in zip(ax_.ravel(), variables):
    for scenario in data_train['Scenario'].unique():
        ax.plot(data_train.loc[data_train['Scenario'] == scenario, [variable]].values,
                linewidth=0.3,
                color='#808080')
    ax.set_title(variable)
    ax.set_xlabel('Hour')


# TEST DATA
fig = plt.figure(figsize=(10, 3))
ax_ = fig.subplots(1, 3)
variables = ['Loading', ' SOC', ' storage_P']

for ax, variable in zip(ax_.ravel(), variables):
    for scenario in data_test['Scenario'].unique():
        ax.plot(data_test.loc[data_test['Scenario'] == scenario, [variable]].values,
                linewidth=0.3,
                color='#808080')
    ax.set_title(variable)
    ax.set_xlabel('Hour')

data_train.drop(columns='Scenario', inplace=True)


#### MODEL TRAINING
class VARIABLES:
    PREDICT = ' storage_P'

abs_path = Path(__file__).parent

param_dist = {'n_estimators': stats.randint(150, 500),
              # 'learning_rate': stats.uniform(0.01, 0.07),
              # 'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9]}
              # 'colsample_bytree': stats.uniform(0.5, 0.45),
              # 'min_child_weight': [1, 2, 3]}

xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_parameter_search = RandomizedSearchCV(xgb_regressor_model,
                                          param_distributions=param_dist,
                                          n_iter=30,
                                          scoring='neg_mean_squared_error',
                                          cv=5,
                                          refit=1)

x_train = data_train.loc[:, data_train.columns != VARIABLES.PREDICT]
y_train = data_train.loc[:, VARIABLES.PREDICT]

xgb_parameter_search.fit(x_train, y_train)

# pickle.dump(xgb_parameter_search, open(abs_path/ 'XGBoost_model.dat', 'wb'))

# xgb_parameter_search = pickle.load(open(abs_path/ 'XGBoost_model.dat', 'rb'))

xgb_parameter_search.cv_results_['mean_test_score']

data_test_ = data_test.drop(columns='Scenario')
x_test = data_test_.loc[:, data_test_.columns != VARIABLES.PREDICT]
y_test = data_test_.loc[:, VARIABLES.PREDICT]

y_test_hat = xgb_parameter_search.predict(x_test)

fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.plot(y_test_hat)
ax.plot(y_test.values)

n_features = len(xgb_parameter_search.best_estimator_.feature_importances_)
fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.bar(x_test.columns, xgb_parameter_search.best_estimator_.feature_importances_)
ax.tick_params(axis='x', rotation=90, labelsize='small')
ax.set_title('Feature importance')

