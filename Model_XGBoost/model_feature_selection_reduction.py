import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import Model_XGBoost.model_utils as mu
from sklearn.inspection import permutation_importance
from matplotlib.ticker import NullFormatter
import matplotlib
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
abs_path = Path(__file__).parent
matplotlib.rc('text', usetex=True)

np.random.seed(1234)  # For reproducibility
file_name = 'ems_optimization_2.1_200_yes.csv'

(data_train, data_test) = mu.split_data(file_name, testing_split=0)  # All data to train
(x_, y_, _, _) = mu.split_data_for_model(file_name,
                                         columns_drop=['Scenario', 'v_1', ' storage_Q'],
                                         columns_predict=[' storage_P'],
                                         testing_split=0.0)

TRAIN_ALL_MODELS = False  # If set True, the code takes around 5 hours to complete

if TRAIN_ALL_MODELS:
    models_results = {'fold_params': {'x_train': list(),
                                      'y_train': list(),
                                      'x_test': list(),
                                      'y_test': list(),
                                      'idx_train': list(),
                                      'idx_test': list()},
                      'xgboost': {'best_trained_model': list(),
                                  'cv_results': list(),
                                  'perm_importance': list(),
                                  'perm_importance_rank': list(),
                                  'y_hat': list(),
                                  'mse': list(),
                                  'neg_mse': list(),
                                  'rmse': list(),
                                  'norm_rmse': list()}}

    param_dist = {'n_estimators': stats.randint(100, 1000),
                  'learning_rate': stats.uniform(0.01, 0.1),
                  'subsample': stats.uniform(0.3, 0.7),
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': stats.uniform(0.5, 0.45),
                  'min_child_weight': [1, 2, 3]}

    n_splits = 50
    groups = data_train['Scenario']
    group_kfold = GroupKFold(n_splits=n_splits)

    rmse = list()
    for number, (train_idx, test_idx) in enumerate(group_kfold.split(data_train, groups=groups)):
        print(f'Fold {number + 1} of {n_splits}')
        X_train, Y_train = x_.iloc[train_idx, :], y_.iloc[train_idx]
        X_test, Y_test = x_.iloc[test_idx, :], y_.iloc[test_idx]
        groups_train = groups[train_idx]

        group_kfold_inner = GroupKFold(n_splits=3)

        xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_regressor_search = RandomizedSearchCV(xgb_regressor_model,
                                                  param_distributions=param_dist,
                                                  n_iter=10,
                                                  scoring='neg_mean_squared_error',
                                                  cv=group_kfold_inner.split(X_train, groups=groups_train),
                                                  refit=1,
                                                  n_jobs=-1)
        xgb_regressor_search.fit(X_train, Y_train)

        # Prediction with the best model
        y_hat = xgb_regressor_search.best_estimator_.predict(X_test)

        # Error on the prediction with the best model in the test fold
        rmse = np.sqrt(mean_squared_error(Y_test.values.ravel(), y_hat.ravel()))
        mse = mean_squared_error(Y_test.values.ravel(), y_hat.ravel())
        norm_rmse = (rmse / (Y_test.max() - Y_test.min())) * 100

        # Calculate predictor importance
        print('Calculation predictor importance')
        result_ind = permutation_importance(xgb_regressor_search.best_estimator_, X_train, Y_train, n_repeats=10,
                                            random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')

        # Save the results
        models_results['fold_params']['x_train'].append(X_train)
        models_results['fold_params']['y_train'].append(Y_train)
        models_results['fold_params']['x_test'].append(X_test)
        models_results['fold_params']['y_test'].append(Y_test)
        models_results['fold_params']['idx_train'].append(train_idx)
        models_results['fold_params']['idx_test'].append(test_idx)

        models_results['xgboost']['y_hat'].append(y_hat)
        models_results['xgboost']['best_trained_model'].append(xgb_regressor_search.best_estimator_)
        models_results['xgboost']['cv_results'].append(xgb_regressor_search.cv_results_)
        models_results['xgboost']['neg_mse'].append(-mse)
        models_results['xgboost']['mse'].append(mse)
        models_results['xgboost']['rmse'].append(rmse)
        models_results['xgboost']['norm_rmse'].append(norm_rmse)

        models_results['xgboost']['perm_importance'].append(result_ind)
        models_results['xgboost']['perm_importance_rank'].append(result_ind.importances_mean.argsort() + 1)

    pickle.dump(models_results, open(abs_path / 'XGBoost_models_feature_analysis_reduced_models.dat', 'wb'))

else:
    models_results = pickle.load(open(abs_path / 'XGBoost_models_feature_analysis_reduced_models.dat', 'rb'))

#%%
# Feature importance without permutation
feature_names = models_results['fold_params']['x_train'][0].columns

feat_importances = np.array([model.feature_importances_ for model in models_results['xgboost']['best_trained_model']])
feat_importances_frame = pd.DataFrame({'feature': feature_names,
                                       'mean_gain_importance': feat_importances.mean(axis=0),
                                       'std_gain_importance': feat_importances.std(axis=0)})
feat_importances_frame = feat_importances_frame.sort_values(by='mean_gain_importance', ascending=False)


# Feature importance with permutation
perm_feat_importance = np.array([model.importances_mean for model in models_results['xgboost']['perm_importance']])
base_line = -np.array(models_results['xgboost']['neg_mse'])[np.newaxis].T
perm_feat_importance = np.sqrt(perm_feat_importance/base_line)

perm_feat_importance_frame = pd.DataFrame({'feature': feature_names,
                                           'mean_permutation_importance': perm_feat_importance.mean(axis=0),
                                           'std_permutation_importance': perm_feat_importance.std(axis=0)})
perm_feat_importance_frame = perm_feat_importance_frame.sort_values(by='mean_permutation_importance', ascending=False)
idx_labels = (-perm_feat_importance.mean(axis=0)).argsort()

#%% PLOT OF THE RESULTS OF THE FEATURE SELECTION
n_features = feat_importances_frame.feature.shape[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.2)
ax1.bar(list(np.linspace(0.5, n_features - 0.5, n_features)),
        feat_importances_frame.mean_gain_importance.values,
        yerr=feat_importances_frame.std_gain_importance.values)
ax1.tick_params(axis='x', rotation=90, labelsize='small')
ax1.tick_params(axis='both', labelsize='small')
ax1.set_xticklabels(
                    # feat_importances_frame.feature.to_list(),
                    feat_importances_frame.feature.str.replace('_', '\_').to_list(),
                    fontsize=6)
ax1.set_ylabel('Normalized', fontsize=9)


ax1.set_xlim((0, len(feat_importances_frame.feature)))
ax1.set_xticks(list(np.linspace(0.5, n_features - 0.5, n_features)))
ax2.boxplot(perm_feat_importance[:, idx_labels],
            vert=True,
            labels=feature_names[idx_labels], showfliers=False)
ax2.tick_params(axis='x', rotation=90, labelsize='small')
ax2.tick_params(axis='both', labelsize='small')
ax2.set_xticklabels(
                    # feature_names[idx_labels],
                    feature_names[idx_labels].str.replace('_', '\_'),
                    fontsize=6)
ax2.set_ylabel('Normalized', fontsize=9)
plt.tight_layout()

#%% PLOT OF ALL TIME SERIES AND ZOOM IN A BOX
slide = 8 * 24  # 10 scenarios of 24 hours
from_day = 2 * 24
to_day = from_day + 1 * 24  # 1 days ahead
fold = 0

# Two folds are attached together to form a longer time series
x_data = np.hstack([models_results['xgboost']['y_hat'][fold][0:slide] / 1000,
                    models_results['xgboost']['y_hat'][fold+1][0:slide] / 1000])
y_data = np.hstack([models_results['fold_params']['y_test'][fold][0:slide].values.ravel() / 1000,
                    models_results['fold_params']['y_test'][fold+1][0:slide].values.ravel() / 1000])

_, ax = plt.subplots(1, 1, figsize=(4, 2.5))
plt.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.95)
ax.plot(x_data, label='GB-Trees', linewidth=0.6)
ax.plot(y_data, label='Actual', linewidth=0.6)
ax.set_ylabel('Active Power [MW]', labelpad=-4)
ax.set_ylim((-2.5, 8))
ax.set_xlim((0, slide))
# ax.set_title('Algorithms response')
ax.legend(loc='upper left', fancybox=False, shadow=False, fontsize='small')

axins = ax.inset_axes([0.5, 0.55, 0.45, 0.4])
axins.plot(x_data, label='GB-Trees', linewidth=0.6)
axins.plot(y_data, label='Actual', linewidth=0.6)
axins.set_ylim(-2, 3)
# Sub region of the original image
x1, x2, y1, y2 = from_day, to_day, -2, 3
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.xaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_major_formatter(NullFormatter())
axins.set_xticks([])
axins.set_yticks([])

ax.indicate_inset_zoom(axins)
x_ticks_pos = (np.linspace(0, slide, int(slide/24)+1)+12)[0:-1]
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels((np.arange(x_ticks_pos.shape[0]) + 1).astype('str'))
ax.set_xlabel('Scenario')

# Create the vertical lines
for x_line in np.linspace(0, slide, int(slide/24)+1):
    ax.axvline(x=x_line, linewidth=0.3, linestyle='-', color='#808080')


#%% Compute the cross validation for each group of reduced feature
if TRAIN_ALL_MODELS:
    reduced_model_results = {'model_search': list(),
                             'best_model': list(),
                             'predictors': list(),
                             'x_train': list(),
                             'y_train': list(),
                             'x_test': list(),
                             'y_test': list(),
                             'mse': list(),
                             'neg_mse': list(),
                             'rmse': list(),
                             'norm_rmse': list(),
                             'mean_norm_rmse_fold': list(),
                             'std_norm_rmse_fold': list()}

    param_dist = {'n_estimators': stats.randint(100, 1000),
                  'learning_rate': stats.uniform(0.01, 0.1),
                  'subsample': stats.uniform(0.3, 0.7),
                  'max_depth': [3, 4, 5, 6, 7, 8, 9],
                  'colsample_bytree': stats.uniform(0.5, 0.45),
                  'min_child_weight': [1, 2, 3]}

    feature_threshold = np.arange(2, 38, 2)
    models_trained = list()
    for threshold in feature_threshold:
        print(f'Threshold: {threshold}')
        input_features_columns = perm_feat_importance_frame[:threshold].feature.to_list()
        print(f'Features: {input_features_columns}')
        (x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                                     columns_drop=['Scenario', 'v_1', ' storage_Q'],
                                                                     columns_predict=[' storage_P'],
                                                                     testing_split=0.2,
                                                                     select_input_features=True,
                                                                     input_features_columns=input_features_columns)

        xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_regressor_search = RandomizedSearchCV(xgb_regressor_model,
                                                  param_distributions=param_dist,
                                                  n_iter=50,
                                                  scoring='neg_mean_squared_error',
                                                  cv=10,
                                                  refit=1,
                                                  n_jobs=-1)
        xgb_regressor_search.fit(x_train, y_train)

        y_hat = xgb_regressor_search.best_estimator_.predict(x_test)

        mse = mean_squared_error(y_test, y_hat)
        rmse = np.sqrt(mean_squared_error(y_test, y_hat))

        y_test_ = y_test.values.ravel()
        norm_rmse = (rmse / (y_test_.max() - y_test_.min())) * 100

        # Calculate the normalized error and standard deviation from the folds
        cv_results = pd.DataFrame(xgb_regressor_search.cv_results_)
        cv_results = cv_results.sort_values(by='rank_test_score')

        norm_factor = np.max([y_train.max(), y_test.max()]) - np.min([y_train.min(), y_test.min()])

        folds_results = cv_results.filter(regex='split', axis=1)
        norm_mse_splits = ((np.sqrt(-folds_results) / norm_factor) * 100)  # Values in percentage

        mean_norm_mse = norm_mse_splits.mean(axis=1)[0]  # Best model
        std_norm_mse = norm_mse_splits.std(axis=1, ddof=0)[0]  # Best model

        reduced_model_results['model_search'].append(xgb_regressor_search)
        reduced_model_results['best_model'].append(xgb_regressor_search.best_estimator_)
        reduced_model_results['predictors'].append(input_features_columns)
        reduced_model_results['x_train'].append(x_train)
        reduced_model_results['y_train'].append(y_train)
        reduced_model_results['x_test'].append(x_test)
        reduced_model_results['y_test'].append(y_test)

        reduced_model_results['mse'].append(mse)
        reduced_model_results['neg_mse'].append(-mse)
        reduced_model_results['rmse'].append(rmse)
        reduced_model_results['norm_rmse'].append(norm_rmse)
        reduced_model_results['mean_norm_rmse_fold'].append(mean_norm_mse)
        reduced_model_results['std_norm_rmse_fold'].append(std_norm_mse)

        print(f'Normalized RMSE (%): {round(norm_rmse, 2)}')
        # print(f'RMSE: {np.sqrt(-xgb_regressor_search.cv_results_["mean_test_score"].max())}')
    pickle.dump((feature_threshold, reduced_model_results), open(abs_path / 'XGBoost_reduced_models.dat', 'wb'))
else:
    (feature_threshold, reduced_model_results) = pickle.load(open(abs_path / 'XGBoost_reduced_models.dat', 'rb'))

#%% Plot error bars
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
fig.subplots_adjust(right=0.95, bottom=0.2, top=0.95)
ax.errorbar(feature_threshold, reduced_model_results['mean_norm_rmse_fold'],
            yerr=reduced_model_results['std_norm_rmse_fold'],
            marker='o',  markersize=2, linewidth=0.3, color='k')
ax.set_ylabel('Normalized RMSE [\%]', fontsize='small')
ax.set_xlabel('Number of predictors', fontsize='small')
ax.set_xticks(feature_threshold)
ax.tick_params(axis='x', labelsize='small')
# fig.tight_layout()

#%% ALL THE FIGURES IN THE SAME PLOT
n_features = feat_importances_frame.feature.shape[0]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4))
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95, hspace=0.2)
ax1.bar(list(np.linspace(0.5, n_features - 0.5, n_features)),
        feat_importances_frame.mean_gain_importance.values,
        yerr=feat_importances_frame.std_gain_importance.values,
        error_kw={'markeredgewidth': 0.3, 'elinewidth': 0.3},
        capsize=2)
ax1.tick_params(axis='x', rotation=90, labelsize='small')
ax1.tick_params(axis='both', labelsize='small')
ax1.set_xticklabels(
                    # feat_importances_frame.feature.to_list(),
                    feat_importances_frame.feature.str.replace('_', '\_').to_list(),
                    fontsize=6)
ax1.set_ylabel('Normalized', fontsize=7)


ax1.set_xlim((0, len(feat_importances_frame.feature)))
ax1.set_xticks(list(np.linspace(0.5, n_features - 0.5, n_features)))
ax2.boxplot(perm_feat_importance[:, idx_labels],
            vert=True,
            labels=feature_names[idx_labels],
            showfliers=False,
            boxprops={'linewidth': 0.3},
            medianprops={'linewidth': 0.3},
            whiskerprops={'linewidth': 0.3},
            capprops={'linewidth': 0.3})
ax2.tick_params(axis='x', rotation=90, labelsize='small')
ax2.tick_params(axis='both', labelsize='small')
ax2.set_xticklabels(
                    # feature_names[idx_labels],
                    feature_names[idx_labels].str.replace('_', '\_'),
                    fontsize=6)
ax2.set_ylabel('Normalized', fontsize=7)

ax3.errorbar(feature_threshold, reduced_model_results['mean_norm_rmse_fold'],
             yerr=reduced_model_results['std_norm_rmse_fold'],
             marker='o',  markersize=0, linewidth=0.3, color='k')
ax3.set_ylabel('NRMSE [\%]', fontsize=7)
ax3.set_xlabel('Number of predictors', fontsize='small')
ax3.set_xticks(feature_threshold)
ax3.tick_params(axis='x', labelsize='small')
plt.tight_layout()


