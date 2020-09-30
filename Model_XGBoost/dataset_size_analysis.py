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

file_name = 'ems_optimization_1.0_1000.csv'
data_scenarios = pd.read_csv(abs_path.parents[0] / ('Model_optimization/' + file_name))
scenario_list = np.round(np.linspace(10, data_scenarios['Scenario'].max() + 1, 10))


####################### THIS TAKES TOO LONG !!!! #######################################################################
RETRAIN_ALL_MODELS = False
if RETRAIN_ALL_MODELS:
    xgb_models_trained = list()
    for scenario_number, scenario_threshold in enumerate(scenario_list):
        print(f'Scenario: {scenario_number} of {(len(scenario_list)-1)} -- Scenario threshold: {scenario_threshold}')

        idx = data_scenarios['Scenario'] <= scenario_threshold
        data_scenarios_selected = data_scenarios.loc[idx,:]

        columns_drop=['Scenario','v_1', ' storage_Q']
        columns_predict=[' storage_P']
        data_train_clean = data_scenarios_selected.drop(columns=columns_drop)
        x_train = data_train_clean.drop(columns=columns_predict)
        y_train = data_train_clean.loc[:, columns_predict]

        xgb_parameter_search = mu.model_training_xgboost(x_train, y_train)
        xgb_models_trained.append(xgb_parameter_search)

    pickle.dump((scenario_list, xgb_models_trained), open(abs_path/ 'All_XGBoost_models_dataset_size_analysis.dat', 'wb'))
    
(scenario_list, xgb_models_trained) = pickle.load(open(abs_path/ 'All_XGBoost_models_dataset_size_analysis.dat', 'rb'))

# (scenario_list, xgb_models_trained) = pickle.load(open(abs_path/ 'All_XGBoost_models_dataset_size_analysis.dat', 'rb'))
(std_upper, mean_value, std_lower) = mu.get_results_data_set_size_analysis(xgb_models_trained,
                                                                           scenario_list,
                                                                           std_=2,
                                                                           plot=True)
