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
from sklearn.linear_model import Lasso, LinearRegression
import pickle
import seaborn as sns
import Model_XGBoost.model_utils as mu

np.random.seed(1234)  # For reproducibility
file_name = 'ems_optimization_1.0_200.csv'

(data_train, data_test) = mu.split_data(file_name, testing_split=0)  # All data to train
(x_, y_, _, _) = mu.split_data_for_model(file_name,
                                       columns_drop=['Scenario','v_1', ' storage_Q'],
                                       columns_predict=[' storage_P'],
                                       testing_split=0.0)
x = x_.values
y = y_.values
n_folds = 5
outer_k_fold = KFold(n_splits=n_folds, random_state=123, shuffle=True)

models_results = {'fold_params': {'x_train': list(),
                                  'x_train_scaled': list(),
                                  'y_train': list(),
                                  'y_train_scaled': list(),
                                  'x_test': list(),
                                  'x_test_scaled': list(),
                                  'y_test': list(),
                                  'scaler_x': list(),
                                  'scaler_y': list(),
                                  'idx_train': list(),
                                  'idx_test': list()},
                  'xgboost': {'best_trained_model': list(),
                                  'y_hat': list(),
                                  'rmse': list(),
                                  'mse': list()},
                  'linear_model': {'best_trained_model': list(),
                                   'y_hat_scaled': list(),
                                   'y_hat': list(),
                                   'rmse': list(),
                                   'mse': list()},
                  'svr_lineal': {'best_trained_model': list(),
                                   'y_hat_scaled': list(),
                                   'y_hat': list(),
                                   'rmse': list(),
                                   'mse': list()},
                  'svr_rbf': {'best_trained_model': list(),
                                   'y_hat_scaled': list(),
                                   'y_hat': list(),
                                   'rmse': list(),
                                   'mse': list()}}


RETRAIN_ALL_MODELS = False

if RETRAIN_ALL_MODELS:
    # Outer cross-fold is guided by the Scenario number
    for ii, (scenario_train_idx, scenario_test_idx) in enumerate(outer_k_fold.split(np.arange(0, data_train.Scenario.max() + 1))):
        print("-----" * 10)
        print (f'Fold {ii} of 4')

        train_idx = data_train.Scenario.isin(scenario_train_idx).values
        test_idx = data_train.Scenario.isin(scenario_test_idx).values

        X_train, Y_train = x[train_idx, :], y[train_idx, :]
        X_test, Y_test = x[test_idx, :], y[test_idx, :]

        # Gradient boosting
        print("Fitting XGB...")
        xgb_parameter_search = mu.model_training_xgboost(X_train, Y_train)

        # -- METHODS THAT REQUIRES NORMALIZATION --------------------------
        scaler_x = StandardScaler()
        scaler_x.fit(X_train)
        scaler_y = StandardScaler()
        scaler_y.fit(Y_train)

        X_train_scaled = scaler_x.transform(X_train)
        Y_train_scaled = scaler_y.transform(Y_train)

        X_test_scaled = scaler_x.transform(X_test)

        # SVR linear kernel
        print("Fitting SVR.Lineal...")
        svr_linear_parameter_search = mu.model_training_svr(X_train_scaled, Y_train_scaled.ravel(), kernel=['linear'])

        # SVR radial kernel
        print("Fitting SVR.rbf...")
        svr_rbf_parameter_search = mu.model_training_svr(X_train_scaled, Y_train_scaled.ravel(), kernel=['rbf'])

        # Linear model
        print("Fitting Linear model...")
        lr = LinearRegression()
        lr.fit(X_train_scaled, Y_train_scaled)

        # -- SAVER RESULTS -------------------------------------
        models_results['fold_params']['x_train'].append(X_train)
        models_results['fold_params']['y_train'].append(Y_train)
        models_results['fold_params']['x_test'].append(X_test)
        models_results['fold_params']['y_test'].append(Y_test)
        models_results['fold_params']['x_train_scaled'].append(X_train_scaled)
        models_results['fold_params']['y_train_scaled'].append(Y_train_scaled)
        models_results['fold_params']['x_test_scaled'].append(X_test_scaled)
        models_results['fold_params']['scaler_x'].append(scaler_x)
        models_results['fold_params']['scaler_y'].append(scaler_y)
        models_results['fold_params']['idx_train'].append(train_idx)
        models_results['fold_params']['idx_test'].append(test_idx)


        models_results['xgboost']['best_trained_model'].append(xgb_parameter_search)
        models_results['xgboost']['y_hat'].append(xgb_parameter_search.predict(X_test))
        models_results['xgboost']['rmse'].append(np.sqrt(mean_squared_error(Y_test.ravel(),
                                                                            xgb_parameter_search.predict(X_test).ravel())))
        models_results['xgboost']['mse'].append(mean_squared_error(Y_test.ravel(),
                                                                     xgb_parameter_search.predict(X_test).ravel()))

        models_results['linear_model']['best_trained_model'].append(lr)
        models_results['linear_model']['y_hat_scaled'].append(lr.predict(X_test_scaled))
        models_results['linear_model']['y_hat'].append(scaler_y.inverse_transform(lr.predict(X_test_scaled)))
        models_results['linear_model']['rmse'].append(np.sqrt(mean_squared_error(Y_test.ravel(),
                                                                                 scaler_y.inverse_transform(lr.predict(X_test_scaled)))))
        models_results['linear_model']['mse'].append(mean_squared_error(Y_test.ravel(),
                                                                        scaler_y.inverse_transform(lr.predict(X_test_scaled))))

        models_results['svr_lineal']['best_trained_model'].append(svr_linear_parameter_search)
        models_results['svr_lineal']['y_hat_scaled'].append(svr_linear_parameter_search.predict(X_test_scaled))
        models_results['svr_lineal']['y_hat'].append(scaler_y.inverse_transform(svr_linear_parameter_search.predict(X_test_scaled)))
        models_results['svr_lineal']['rmse'].append(np.sqrt(mean_squared_error(Y_test.ravel(),
                                                                               scaler_y.inverse_transform(svr_linear_parameter_search.predict(X_test_scaled)))))
        models_results['svr_lineal']['mse'].append(mean_squared_error(Y_test.ravel(),
                                                                        scaler_y.inverse_transform(svr_linear_parameter_search.predict(X_test_scaled))))


        models_results['svr_rbf']['best_trained_model'].append(svr_rbf_parameter_search)
        models_results['svr_rbf']['y_hat_scaled'].append(svr_rbf_parameter_search.predict(X_test_scaled))
        models_results['svr_rbf']['y_hat'].append(scaler_y.inverse_transform(svr_rbf_parameter_search.predict(X_test_scaled)))
        models_results['svr_rbf']['rmse'].append(np.sqrt(mean_squared_error(Y_test.ravel(),
                                                                               scaler_y.inverse_transform(svr_rbf_parameter_search.predict(X_test_scaled)))))
        models_results['svr_rbf']['mse'].append(mean_squared_error(Y_test.ravel(),
                                                                        scaler_y.inverse_transform(svr_rbf_parameter_search.predict(X_test_scaled))))

    pickle.dump(models_results, open(abs_path/ 'All_models_model_selection.dat', 'wb'))

models_results = pickle.load(open(abs_path / 'All_models_model_selection.dat', 'rb'))

#%%
# # Plot coefficients of linear models
# fig = plt.figure(figsize=(8, 10))
# ax_ = fig.subplots(5, 1)
# plt.subplots_adjust(hspace=0)
# for ii, ax in enumerate(ax_.ravel()):
#     ax.bar(x_.columns, models_results['linear_model']['best_trained_model'][ii].coef_.ravel())
#     ax.tick_params(axis='x', rotation=90, labelsize='small')
#     ax.set_ylabel(f'Fold-{ii}')
# ax_[0].set_title('Coefficients Linear model')


#%%
# Table with the mean and standard deviation of RMSE
table_results = pd.DataFrame({'model_type': list(models_results.keys())[1:],
                              'rmse_mean': [np.mean(models_results[model_type]['rmse']) for model_type in list(models_results.keys())[1:]],
                              'std_folds': [np.std(models_results[model_type]['rmse']) for model_type in list(models_results.keys())[1:]]})
print(table_results)
table_results.round(2).to_csv(abs_path / 'algorithm_selection.csv')


#%%
# # Show the results of the algorithm in the time series
# slide = 1
# fold = 0
# fig = plt.figure(figsize=(4, 4))
# ax = fig.subplots(1, 1)
# plt.subplots_adjust(left=0.2, bottom=0.25)
# for model_type in list(models_results.keys())[1:]:
#     ax.plot(models_results[model_type]['y_hat'][fold][(48 * slide):(48 * (slide + 1))] / 1000, label=model_type)
# ax.plot(models_results['fold_params']['y_test'][fold][(48 * slide):(48 * (slide + 1))].ravel() / 1000, label='Actual')
# ax.set_ylabel('Active Power [MW]')
# ax.set_xlabel('Hour')
# ax.set_title('Algorithms response')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), fancybox=False, shadow=False, ncol=3, fontsize='small')



#%%
# Plot only one model
slide = 1
fold = 0
fig = plt.figure(figsize=(4, 2.5))
ax = fig.subplots(1, 1)
plt.subplots_adjust(left=0.15, bottom=0.15)
ax.plot(models_results['xgboost']['y_hat'][fold][(48 * slide):(48 * (slide + 1))] / 1000, label='GB-Trees')
ax.plot(models_results['fold_params']['y_test'][fold][(48 * slide):(48 * (slide + 1))].ravel() / 1000, label='Actual')
ax.set_ylabel('Active Power [MW]')
ax.set_xlabel('Hour')
# ax.set_title('Algorithms response')
ax.legend(loc='upper left', fancybox=False, shadow=False, fontsize='small')

#%% PLOT OF ALL TIME SERIES AND ZOOM IN A BOX

slide = 10 * 24  # 10 scenarios of 24 hours
from_day = 3 * 24
to_day = from_day + 1 * 24  # 1 days ahead
fold = 0
fig = plt.figure(figsize=(4, 2.5))
ax = fig.subplots(1, 1)
plt.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.95)
ax.plot(models_results['xgboost']['y_hat'][fold][0:slide] / 1000, label='GB-Trees', linewidth=0.6)
ax.plot(models_results['fold_params']['y_test'][fold][0:slide].ravel() / 1000, label='Actual', linewidth=0.6)
ax.set_ylabel('Active Power [MW]', labelpad=-4)
ax.set_xlabel('Hour')
ax.set_ylim((-2.5, 8))
ax.set_xlim((0, slide))
# ax.set_title('Algorithms response')
ax.legend(loc='upper left', fancybox=False, shadow=False, fontsize='small')

axins = ax.inset_axes([0.5, 0.55, 0.45, 0.4])
axins.plot(models_results['xgboost']['y_hat'][fold][0:slide] / 1000, label='GB-Trees', linewidth=0.6)
axins.plot(models_results['fold_params']['y_test'][fold][0:slide].ravel() / 1000, label='Actual', linewidth=0.6)
axins.set_ylim(-2, 3)
# sub region of the original image
x1, x2, y1, y2 = from_day, to_day, -2, 3
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# axins.set_xticklabels('')
# axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)

#%%
# fig = plt.figure(figsize=(4, 4))
# ax = fig.subplots(1, 1)
# ax.plot(models_results['svr_rbf']['y_hat'][1])
# ax.plot(models_results['fold_params']['y_test'][1])
# ax.set_ylabel('Active Power [MW]')
# ax.set_xlabel('Hour')
# ax.set_title('Algorithms response')





# fig = plt.figure()
# ax = fig.subplots(1, 1)
# ax.bar(x_.columns, lr.coef_.ravel())
# ax.tick_params(axis='x', rotation=90, labelsize='small')
#
#
# #%%
# feature_importance_df = pd.DataFrame({'Feature': x_.columns,
#                                       'Importance': np.abs(lr.coef_.ravel())})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# fig = plt.figure(figsize=(5.5, 4))
# ax = fig.subplots(1, 1)
# plt.subplots_adjust(bottom=0.15)
# ax.bar(feature_importance_df.Feature,
#        feature_importance_df.Importance)
# ax.tick_params(axis='x', rotation=90, labelsize='small')
# ax.set_ylabel('abs(coefficients)')
# ax.set_title('Coefficient weights - Lineal model')


