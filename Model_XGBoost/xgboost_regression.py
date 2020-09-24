import pandas as pd
import xgboost as xgb
from pathlib import Path

class VARIABLES:
    PREDICT = ' storage_P'

abs_path = Path(__file__).parent
data_scenarios = pd.read_csv(abs_path.parents[0] / 'Model_optimization/ems_optimization.csv')

data_scenarios[VARIABLES.PREDICT]



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