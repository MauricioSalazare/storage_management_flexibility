import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy import stats
import pickle
from pathlib import Path
abs_path = Path(__file__).parent

file_name = 'ems_optimization_2.1_1000_yes.csv'
data_scenarios = pd.read_csv(abs_path.parents[0] / ('Model_optimization/' + file_name))
scenario_list = np.round(np.linspace(10, data_scenarios['Scenario'].max() + 1, 10))


RETRAIN_ALL_MODELS = True
if RETRAIN_ALL_MODELS:
    model_results = {'model_search': list(),
                     'best_model': list(),
                     'mean_norm_rmse_fold': list(),
                     'std_norm_rmse_fold': list()}
    xgb_models_trained = list()
    for scenario_number, scenario_threshold in enumerate(scenario_list):
        print(f'Scenario: {scenario_number + 1} of {(len(scenario_list))} -- Scenario threshold: {scenario_threshold}')

        idx = data_scenarios['Scenario'] <= scenario_threshold
        data_scenarios_selected = data_scenarios.loc[idx,:]

        columns_drop=['Scenario','v_1', ' storage_Q']
        columns_predict=[' storage_P']
        data_train_clean = data_scenarios_selected.drop(columns=columns_drop)

        x_train = data_train_clean.drop(columns=columns_predict)
        y_train = data_train_clean.loc[:, columns_predict]

        param_dist = {'n_estimators': stats.randint(100, 1000),
                      'learning_rate': stats.uniform(0.01, 0.1),
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 2, 3]}

        xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_regressor_search = RandomizedSearchCV(xgb_regressor_model,
                                                  param_distributions=param_dist,
                                                  n_iter=50,
                                                  scoring='neg_mean_squared_error',
                                                  cv=10,
                                                  refit=1,
                                                  n_jobs=4)
        xgb_regressor_search.fit(x_train, y_train)

        # Calculate the normalized error and standard deviation from the folds
        cv_results = pd.DataFrame(xgb_regressor_search.cv_results_)
        cv_results = cv_results.sort_values(by='rank_test_score')

        norm_factor = np.max(y_train.max()) - np.min(y_train.min())

        folds_results = cv_results.filter(regex='split', axis=1)
        norm_mse_splits = ((np.sqrt(-folds_results) / norm_factor) * 100)  # Values in percentage

        mean_norm_mse = norm_mse_splits.mean(axis=1)[0]  # Mean performance of best model
        std_norm_mse = norm_mse_splits.std(axis=1, ddof=0)[0]  # Standard deviation of performance of best model

        # Save results
        model_results['model_search'].append(xgb_regressor_search)
        model_results['best_model'].append(xgb_regressor_search.best_estimator_)

        model_results['mean_norm_rmse_fold'].append(mean_norm_mse)
        model_results['std_norm_rmse_fold'].append(std_norm_mse)

        print(f'Normalized RMSE (%): {round(mean_norm_mse, 2)}')

    pickle.dump((scenario_list, model_results), open(abs_path / 'XGBoost_scenario_analysis.dat', 'wb'))
else:
    (scenario_list, model_results) = pickle.load(open(abs_path / 'XGBoost_scenario_analysis.dat', 'rb'))

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.errorbar(scenario_list, model_results['mean_norm_rmse_fold'],
            yerr=model_results['std_norm_rmse_fold'],
            marker='o',  markersize=0, linewidth=0.3, color='k')
ax.set_ylabel('NRMSE [\%]', fontsize=7)
ax.set_xlabel('Number of scenarios', fontsize='small')
ax.set_xticks(scenario_list)
ax.tick_params(axis='both', labelsize=7)
plt.tight_layout()