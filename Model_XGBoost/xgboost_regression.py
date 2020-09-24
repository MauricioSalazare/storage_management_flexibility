from scipy import stats
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


class VARIABLES:
    PREDICT = ' storage_P'

abs_path = Path(__file__).parent

param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]}

xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_parameter_search = RandomizedSearchCV(xgb_regressor_model,
                                          param_distributions=param_dist,
                                          n_iter=30,
                                          scoring='neg_mean_squared_error',
                                          cv=5,
                                          refit=1)

data_scenarios = pd.read_csv(abs_path.parents[0] / 'Model_optimization/ems_optimization.csv')
data_scenarios.drop(columns='Scenario', inplace=True)

n_folds = 5
k_fold = KFold(n_splits=n_folds, random_state=123, shuffle=True)

x = data_scenarios.loc[:, data_scenarios.columns != VARIABLES.PREDICT]
y = data_scenarios.loc[:, VARIABLES.PREDICT]

xgb_parameter_search.fit(x, y)

# pickle.dump(xgb_parameter_search, open(abs_path/ 'XGBoost_model.dat', 'wb'))
#
# xgb_parameter_search = pickle.load(open(abs_path/ 'XGBoost_model.dat', 'rb'))

xgb_parameter_search.cv_results_['mean_test_score']

y_hat = xgb_parameter_search.predict(x)

fig = plt.figure(figsize=(6, 6))
ax = fig.subplots(1, 1)
ax.plot(y_hat)
ax.plot(y.values)