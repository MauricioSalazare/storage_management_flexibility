import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import xgboost as xgb
from scipy import stats
import pickle
from sklearn.model_selection import cross_val_score
from pathlib import Path
import matplotlib
matplotlib.rc('text', usetex=True)
abs_path = Path(__file__).parent

file_name = 'ems_optimization_2.1_1000_yes.csv'
data_scenarios = pd.read_csv(abs_path.parents[0] / ('Model_optimization/' + file_name))
scenario_list = np.round(np.linspace(10, data_scenarios['Scenario'].max() + 1, 10))


RETRAIN_ALL_MODELS = False
if RETRAIN_ALL_MODELS:
    model_results = {'best_model': list(),
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

        group_kfold_inner = GroupKFold(n_splits=10)
        groups_train = data_scenarios_selected['Scenario']

        xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_regressor_search = RandomizedSearchCV(xgb_regressor_model,
                                                  param_distributions=param_dist,
                                                  n_iter=50,
                                                  scoring='neg_mean_squared_error',
                                                  cv=group_kfold_inner.split(x_train, groups=groups_train),
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
        model_results['best_model'].append(xgb_regressor_search.best_estimator_)

        model_results['mean_norm_rmse_fold'].append(mean_norm_mse)
        model_results['std_norm_rmse_fold'].append(std_norm_mse)

        print(f'Normalized RMSE (%): {round(mean_norm_mse, 2)}')

    pickle.dump((scenario_list, model_results), open(abs_path / 'XGBoost_scenario_analysis.dat', 'wb'))
else:
    (scenario_list, model_results) = pickle.load(open(abs_path / 'XGBoost_scenario_analysis.dat', 'rb'))

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.errorbar(scenario_list, model_results['mean_norm_rmse_fold'],
            yerr=model_results['std_norm_rmse_fold'],
            marker='o',  markersize=0, linewidth=0.3, color='k')
ax.set_ylabel('NRMSE [\%]', fontsize=7)
ax.set_xlabel('Number of scenarios', fontsize='small')
ax.set_xticks(scenario_list)
ax.tick_params(axis='both', labelsize=7)
ax.set_xlim((0, 800))
plt.tight_layout()

#%%
# Analyze the parameters found by the best model found by the sweep of scenarios and random hyperparameter search:
idx_ = np.argsort(model_results['mean_norm_rmse_fold'])
opt_params = model_results['best_model'][idx_[0]].get_params()
scenario_list_final = np.round(np.linspace(10, data_scenarios['Scenario'].max() + 1, 100))

if RETRAIN_ALL_MODELS:
    cross_val_results = {'norm_rmse_scores': list(),
                         'mean_norm_rmse_scores': list(),
                         'std_norm_rmse_scores': list()}

    # Re-train the model with the optimal parameters
    for scenario_number, scenario_threshold in enumerate(scenario_list_final):
        print(f'Scenario: {scenario_number + 1} of {(len(scenario_list_final))} -- Scenario threshold: {scenario_threshold}')

        idx = data_scenarios['Scenario'] <=scenario_threshold
        data_scenarios_selected = data_scenarios.loc[idx, :]

        columns_drop = ['Scenario', 'v_1', ' storage_Q']
        columns_predict = [' storage_P']
        data_train_clean = data_scenarios_selected.drop(columns=columns_drop)
        x_train = data_train_clean.drop(columns=columns_predict)
        y_train = data_train_clean.loc[:, columns_predict]

        group_kfold_inner = GroupKFold(n_splits=10)
        groups_train = data_scenarios_selected['Scenario']

        xgb_regressor_model = xgb.XGBRegressor(**opt_params)
        norm_factor = np.max(y_train.max()) - np.min(y_train.min())

        scores = cross_val_score(xgb_regressor_model,
                                 x_train, y_train,
                                 cv=group_kfold_inner.split(x_train, groups=groups_train),
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1)

        norm_rmse_scores = (np.sqrt(-scores)/norm_factor) * 100
        cross_val_results['norm_rmse_scores'].append(norm_rmse_scores)
        cross_val_results['mean_norm_rmse_scores'].append(np.std(norm_rmse_scores))
        cross_val_results['std_norm_rmse_scores'].append(np.mean(norm_rmse_scores))

    pickle.dump((scenario_list_final, cross_val_results), open(abs_path / 'XGBoost_scenario_model.dat', 'wb'))
else:
    (scenario_list_final, cross_val_results) = pickle.load(open(abs_path / 'XGBoost_scenario_model.dat', 'rb'))


mean_values = np.array(cross_val_results['mean_norm_rmse_scores'])
std_values = np.array(cross_val_results['std_norm_rmse_scores'])

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
plt.subplots_adjust(bottom=0.2, top=0.98, right=0.98, left=0.09)
ax.errorbar(np.array(scenario_list_final)[::2], mean_values[::2],
            yerr=std_values[::2],
            marker='o',
            markersize=0,
            linewidth=0.9,
            color='r',
            capsize=3,
            capthick=0.3, ecolor='k')
ax.set_ylabel('NRMSE [\%]', fontsize='small')
ax.set_xlabel('Number of scenarios', fontsize='small')
# ax.set_xticks(np.array(scenario_list_final)[::2])
ax.tick_params(axis='both', labelsize=8)
ax.axvline(x=200, color='b', linewidth=0.9, linestyle='--')
# plt.tight_layout()
