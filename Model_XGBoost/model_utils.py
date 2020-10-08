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
import pickle
import seaborn as sns
import matplotlib
matplotlib.rc('text', usetex=True)

def model_training_xgboost(x_train, y_train):
    param_dist = {'n_estimators': stats.randint(100, 1000),
                  'learning_rate': stats.uniform(0.01, 0.1),
                  'subsample': stats.uniform(0.3, 0.7),
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': stats.uniform(0.5, 0.45),
                  'min_child_weight': [1, 2, 3]}

    xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_parameter_search = RandomizedSearchCV(xgb_regressor_model,
                                              param_distributions=param_dist,
                                              n_iter=50,
                                              scoring='neg_mean_squared_error',
                                              cv=3,
                                              refit=1,
                                              n_jobs=-1)

    xgb_parameter_search.fit(x_train, y_train)

    return xgb_parameter_search


def model_training_svr(x_train, y_train, kernel):
    param_dist_svr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      # 'C': stats.expon(scale=10),
                      'gamma': stats.expon(scale=1),
                      'kernel': kernel}

    svr = SVR()
    svr_parameter_search = RandomizedSearchCV(svr,
                                              param_distributions=param_dist_svr,
                                              n_iter=50,
                                              scoring='neg_mean_squared_error',
                                              cv=3,
                                              refit=1,
                                              n_jobs=-1)

    svr_parameter_search.fit(x_train, y_train)

    return svr_parameter_search


def split_data(file_name, testing_split=0.2):
    data_scenarios = pd.read_csv(abs_path.parents[0] / ('Model_optimization/' + file_name))

    n_scenarios = data_scenarios['Scenario'].max()
    idx_scenarios = np.random.choice(range(0, n_scenarios), round(n_scenarios * testing_split))  # 20% Testing
    idx_test = data_scenarios['Scenario'].isin(idx_scenarios)
    # data_scenarios.drop(columns='Scenario', inplace=True)
    data_train = data_scenarios.loc[~idx_test,:]
    data_test = data_scenarios.loc[idx_test,:]

    return (data_train, data_test)



def plot_dataset(data_train, variables):
    fig = plt.figure(figsize=(5, 5))
    ax_ = fig.subplots(2, 2)
    plt.subplots_adjust(left=0.12, bottom=0.16, wspace=0.5, hspace=0.5)
    SCALE = 1
    # variables = ['Loading', ' SOC', ' storage_P', ' storage_Q']

    params = {'Loading': {'Scale': 1,
                          'x_label': 'Hour',
                          'y_label': 'Percentage [\%]',
                          'title': 'Transformer Loading'},
              ' SOC':{'Scale': 1,
                      'x_label': 'Hour',
                      'y_label': 'Percentage [\%]',
                      'title': 'SOC'},
              ' storage_P': {'Scale': 1000,
                             'x_label': 'Hour',
                             'y_label': 'Active Power [MW]',
                             'title': 'Batt. Operation' },
              ' storage_Q': {'Scale': 1000,
                             'x_label': 'Hour',
                             'y_label': 'Reactive Power [MVAr]',
                             'title': 'Batt. Operation'}
              }

    for ax, variable in zip(ax_.ravel(), variables):
        variable_data = list()
        for scenario in data_train['Scenario'].unique():
            data_val = data_train.loc[data_train['Scenario'] == scenario, [variable]].values / params[variable]['Scale']
            data_val = data_val.ravel()
            variable_data.append(data_val)
            ax.plot(data_train.loc[data_train['Scenario'] == scenario, [variable]].values/params[variable]['Scale'],
                    linewidth=0.3,
                    color='#808080')
            ax.set_xlabel(params[variable]['x_label'])
            ax.set_ylabel(params[variable]['y_label'])
        ax.plot(np.quantile(np.array(variable_data), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
        ax.plot(np.quantile(np.array(variable_data), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
        ax.plot(np.quantile(np.array(variable_data), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')
        ax.set_title(params[variable]['title'])
        ax.set_xlabel('Hour')
    ax.legend(loc='upper center', bbox_to_anchor=(-0.4, -0.3), fancybox=False, shadow=False, ncol=4, fontsize=7)


def split_data_for_model(file_name,  columns_drop, columns_predict, testing_split=0.2, select_input_features=False,
                         input_features_columns=None):
    #### MODEL TRAINING
    # PREPARE THE DATA
    (data_train, data_test) = split_data(file_name, testing_split=testing_split)


    data_train_clean = data_train.drop(columns=columns_drop)
    data_test_clean = data_test.drop(columns=columns_drop)

    x_train = data_train_clean.drop(columns=columns_predict)
    y_train = data_train_clean.loc[:, columns_predict]

    x_test = data_test_clean.drop(columns=columns_predict)
    y_test = data_test_clean.loc[:, columns_predict]

    if select_input_features:
        assert input_features_columns is not None, "Pass a list of strings with input features names."
        x_train = x_train.loc[:, input_features_columns]
        x_test = x_test.loc[:, input_features_columns]

    return (x_train, y_train, x_test, y_test)


def plot_parameters_importances(model, variable_names, sorted=True):

    if sorted:
        feature_importance = pd.DataFrame({'Feature':  variable_names,
                                           'Importance': model.best_estimator_.feature_importances_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        # %%
        fig = plt.figure(figsize=(6, 5))
        ax = fig.subplots(1, 1)
        plt.subplots_adjust(bottom=0.15)
        ax.bar(feature_importance.Feature, feature_importance.Importance)
        ax.tick_params(axis='x', rotation=90, labelsize='small')
        ax.set_title('Feature importance')
    else:
        # n_features = len(model.best_estimator_.feature_importances_)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.subplots(1, 1)
        ax.bar(variable_names, model.best_estimator_.feature_importances_)
        ax.tick_params(axis='x', rotation=90, labelsize='small')
        ax.set_title('Feature importance')


def get_results_data_set_size_analysis(xgb_models_trained, scenario_list, std_=2, plot=True):
    xgb_models_rmse = list()
    xgb_models_table = list()
    std_upper = list()
    std_lower = list()
    mean_value = list()

    for xgb_model in xgb_models_trained:
        table_results = pd.DataFrame(xgb_model.cv_results_)
        table_results = table_results.set_index('rank_test_score', drop=True).sort_index()
        table_results['RMSE'] = np.sqrt(-table_results['mean_test_score'])

        xgb_models_table.append(table_results)
        # xgb_models_trained.append(xgb_parameter_search)
        xgb_models_rmse.append(table_results.loc[1, ['RMSE']].values)

        std_upper.append(np.sqrt(-table_results['mean_test_score'][1])
                         + (std_ * np.sqrt(table_results['std_test_score'][1])))
        std_lower.append(np.sqrt(-table_results['mean_test_score'][1])
                         - (std_ * np.sqrt(table_results['std_test_score'][1])))
        mean_value.append(np.sqrt(-table_results['mean_test_score'][1]))


    if plot:
        fig = plt.figure(figsize=(4, 2))
        ax = fig.subplots(1, 1)
        plt.subplots_adjust(left=0.15, bottom=0.2, top=0.95)
        ax.plot(scenario_list, std_upper, color='r', linestyle=':', label='Perc. 97.5')
        ax.plot(scenario_list, mean_value, color='r', linestyle='-', label='Perc. 50')
        ax.plot(scenario_list, std_lower, color='r', linestyle='--', label='Perc. 2.5')
        # ax.set_title('RMSE vs Scenarios')
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Error [kW]')  # Percentage
        ax.legend(fontsize='small')

    return (std_upper, mean_value, std_lower)