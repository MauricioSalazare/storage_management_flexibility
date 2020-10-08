import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
abs_path = Path(__file__).parent
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

def split_data(file_name, testing_split=0.2):
    data_scenarios = pd.read_csv(abs_path / file_name)

    n_scenarios = data_scenarios['Scenario'].max()
    idx_scenarios = np.random.choice(range(0, n_scenarios), round(n_scenarios * testing_split))  # 20% Testing
    idx_test = data_scenarios['Scenario'].isin(idx_scenarios)
    # data_scenarios.drop(columns='Scenario', inplace=True)
    data_train = data_scenarios.loc[~idx_test,:]
    data_test = data_scenarios.loc[idx_test,:]

    return (data_train, data_test)

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

#%% Load data
file_name = 'ems_optimization_1.0_100.csv'
(x_train, y_train, x_test, y_test) = split_data_for_model(file_name,
                                       columns_drop=['Scenario','v_1', ' storage_Q'],
                                       columns_predict=[' storage_P'],
                                       testing_split=0.2)
#%% Apply standarization
scaler_x = StandardScaler()
scaler_x.fit(x_train)
scaler_y = StandardScaler()
scaler_y.fit(y_train)

x_features = x_train.columns
y_features = y_train.columns

x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

x_train = pd.DataFrame(x_train, columns=x_features)
x_test = pd.DataFrame(x_test, columns=x_features)
y_train = pd.DataFrame(y_train, columns=y_features)
y_test = pd.DataFrame(y_test, columns=y_features)

#%% Feature correlation
all_data = pd.concat([x_train, y_train], axis=1)
corr = spearmanr(all_data).correlation

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.9)
image = ax.imshow(corr, cmap='RdBu')
ax.set_xticks(np.arange(all_data.columns.shape[0]))
ax.set_yticks(np.arange(all_data.columns.shape[0]))
ax.set_xticklabels(all_data.columns.to_list())
ax.set_yticklabels(all_data.columns.to_list())
ax.tick_params(axis='x', rotation=90, labelsize='small')
ax.tick_params(axis='both', labelsize='small')
ax.set_title('Feature Correlation (Spearman)', fontsize='small')
cbar = fig.colorbar(image, ax=ax, fraction=0.0468, pad=0.02)
cbar.minorticks_on()
cbar.ax.tick_params(labelsize='small')

#%% Build boosted trees with all estimators
param_dist = {'n_estimators': stats.randint(100, 1000),
              'learning_rate': stats.uniform(0.01, 0.1),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]}

xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_parameter_search = RandomizedSearchCV(xgb_regressor_model,
                                          param_distributions=param_dist,
                                          n_iter=10,
                                          scoring='neg_mean_squared_error',
                                          cv=5,
                                          refit=1,
                                          n_jobs=-1)
xgb_parameter_search.fit(x_train, y_train)
print("Accuracy on test data: {:.2f}".format(xgb_parameter_search.best_estimator_.score(x_test, y_test)))

#%% Feature importance and permutation test on the boosted trees
clf = xgb_parameter_search.best_estimator_
result = permutation_importance(clf, x_train, y_train, n_repeats=20,
                                random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
         clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(x_train.columns[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(clf.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=x_train.columns[perm_sorted_idx])
fig.tight_layout()
plt.show()


#%% HANDLING MULTICOLLINEAR FEATURES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# data_voltages = x_train.filter(regex='v_', axis=1).drop(columns='v_2')
# plt.subplots_adjust(left=0.15, right=0.95)

# data_voltages = x_train.filter(regex='v_', axis=1)
data_voltages = x_train
# data_voltages = pd.concat([x_train, y_train], axis=1)

corr = spearmanr(data_voltages).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=data_voltages.columns.to_list(), ax=ax1, leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))

im_corr = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']], cmap='RdBu')
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
ax2.set_title('Voltage Spearman Correlation', fontsize='small')
cbar = fig.colorbar(im_corr, ax=ax2, fraction=0.0468, pad=0.02)
cbar.minorticks_on()
cbar.ax.tick_params(labelsize='small')
fig.tight_layout()
plt.show()

#%%
threshold_values = np.logspace(-3,0,10)
nrmse_values= list()
selected_predictors = list()
for threshold in threshold_values:
    print(f'Threshold value: {round(threshold, 4)}')
    # Set the threshold to select the number of clusters
    cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()] # Pick the first feature of the cluster

    selected_voltages = data_voltages.columns[selected_features].to_list()
    # selected_voltages = ['time', 'v_2', ' SOC']
    selected_predictors.append(selected_voltages)

    x_train_sel = pd.concat([x_train.drop(columns=data_voltages.columns.to_list()), x_train.loc[:,selected_voltages]], axis=1)
    x_test_sel = pd.concat([x_test.drop(columns=data_voltages.columns.to_list()), x_test.loc[:,selected_voltages]], axis=1)

    #% Fit the reduced model and calculate the metrics

    # x_train_sel = x_train_sel.drop(columns=['v_2','v_3','v_4' ])
    # x_test_sel = x_test_sel.drop(columns=['v_2','v_3','v_4' ])

    # x_train_sel = x_train_sel.drop(columns=['v_2','v_3'])
    # x_test_sel = x_test_sel.drop(columns=['v_2','v_3'])

    xgb_regressor_model_sel = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_parameter_search_sel = RandomizedSearchCV(xgb_regressor_model_sel,
                                                  param_distributions=param_dist,
                                                  n_iter=10,
                                                  scoring='neg_mean_squared_error',
                                                  cv=5,
                                                  refit=1,
                                                  n_jobs=-1)
    xgb_parameter_search_sel.fit(x_train_sel, y_train)
    print("Accuracy on test data: {:.2f}".format(xgb_parameter_search_sel.best_estimator_.score(x_test_sel, y_test)))
    clf_sel = xgb_parameter_search_sel.best_estimator_

    # Do the permutation importance again
    result = permutation_importance(clf_sel, x_train_sel, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(clf_sel.feature_importances_)
    tree_indices = np.arange(0, len(clf_sel.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices, clf_sel.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticklabels(x_train_sel.columns[tree_importance_sorted_idx])
    ax1.set_yticks(tree_indices)
    ax1.set_ylim((0, len(clf_sel.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=x_train_sel.columns[perm_sorted_idx])
    fig.tight_layout()
    plt.show()

    #%% Show the time series

    y_hat_test = clf_sel.predict(x_test_sel)
    y_test_true = y_test.values.ravel()

    fig, ax = plt.subplots(1,1, figsize=(8,3))
    ax.plot(y_hat_test)
    ax.plot(y_test_true)
    nrmse = (np.sqrt(mean_squared_error(y_test_true, y_hat_test))/(y_test_true.max()-y_test_true.min())) * 100
    nrmse_values.append(nrmse)

    print(f'Normalized RMSE: {round(nrmse, 2)} %')


#%% 3D plot of the SOC, v_3 and storage P
from mpl_toolkits.mplot3d import Axes3D

X = data_voltages[' SOC'].values.ravel()
Y = data_voltages['v_3'].values.ravel()
Z = y_train.values.ravel()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('SOC')
ax.set_ylabel('V_3')
ax.set_zlabel('Storage_P')

#%%
X = data_voltages[' SOC'].values.ravel()
Y = data_voltages['time'].values.ravel()
Z = y_train.values.ravel()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('SOC')
ax.set_ylabel('Time')
ax.set_zlabel('Storage_P')


#%%
import seaborn as sns
dataset = pd.concat([data_voltages[['time','v_3',' SOC']], y_train], axis=1)
g = sns.pairplot(dataset)
g.fig.set_size_inches(8,8)


#%%
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(x_train[['time','v_3','v_2', 'v_27', ' SOC']], y_train)

y_hat_test = regr.predict(x_test[['time','v_3','v_2','v_27',  ' SOC']])

y_test_true = y_test.values.ravel()

plt.figure()
plt.plot(y_hat_test.ravel(), label='Predicted')
plt.plot(y_test.values.ravel(), label='Actual')
plt.legend()

nrmse = (np.sqrt(mean_squared_error(y_test_true, y_hat_test)) / (y_test_true.max() - y_test_true.min())) * 100
print(f'Normalized RMSE: {round(nrmse, 2)} %')
