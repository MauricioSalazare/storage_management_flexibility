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
from sklearn.linear_model import Lasso, LassoCV
import pickle
import seaborn as sns
import Model_XGBoost.model_utils as mu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

class VARIABLES:
    PREDICT = [' storage_P']

file_name = 'ems_optimization_1.0_200.csv'

# (data_train, data_test) = mu.split_data(file_name, testing_split=0.2)
# mu.plot_dataset(data_train, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])
# mu.plot_dataset(data_test, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])

(x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                             columns_drop=['Scenario','v_1', ' storage_Q'],
                                                             columns_predict=[' storage_P'],
                                                             testing_split=0.2)

scaler_x = StandardScaler()
scaler_x.fit(x_train)
scaler_y = StandardScaler()
scaler_y.fit(y_train)

X_train_scaled = scaler_x.transform(x_train)
Y_train_scaled = scaler_y.transform(y_train)

X_test_scaled = scaler_x.transform(x_test)



lasso = Lasso(max_iter = 10000, normalize = False)
coefs = []
alphas = 10**np.linspace(1,-10,100)*0.5


# alphas = 10**np.linspace(10,-2,100)*0.5
X_train, X_test , y_train, y_test = train_test_split(X_train_scaled, Y_train_scaled, test_size=0.5, random_state=1)

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)



ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


model = LassoCV(alphas = alphas, cv = 5, max_iter = 100000000, normalize = False)
model.fit(X_train, y_train.ravel())



EPSILON = 1e-4
plt.figure()
# ymin, ymax = 2300, 3800
plt.semilogx(model.alphas_ + EPSILON, model.mse_path_, ':')
plt.plot(model.alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_ + EPSILON, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

#
#
# df = pd.read_csv(abs_path / 'Hitters.csv').dropna()
# dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
# y = df.Salary
# X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
# X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
#
# lasso = Lasso(max_iter = 10000, normalize = True)
# coefs = []
#
# alphas = 10**np.linspace(10,-2,100)*0.5
# X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
#
# for a in alphas:
#     lasso.set_params(alpha=a)
#     lasso.fit(scale(X_train), y_train)
#     coefs.append(lasso.coef_)
#
#
#
# ax = plt.gca()
# ax.plot(alphas*2, coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('weights')
#
# lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
# lassocv.fit(X_train, y_train)
#
# lasso.set_params(alpha=lassocv.alpha_)
# lasso.fit(X_train, y_train)
# mean_squared_error(y_test, lasso.predict(X_test))
#
# model = LassoCV(alphas = alphas, cv = 5, max_iter = 100000, normalize = True)
# model.fit(X_train, y_train)
#
#
# EPSILON = 1e-4
# plt.figure()
# # ymin, ymax = 2300, 3800
# plt.semilogx(model.alphas_ + EPSILON, model.mse_path_, ':')
# plt.plot(model.alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k',
#          label='Average across the folds', linewidth=2)
# plt.axvline(model.alpha_ + EPSILON, linestyle='--', color='k',
#             label='alpha: CV estimate')
#
# plt.legend()
#

#
# # xgb_parameter_search = mu.model_training_xgboost(x_train, y_train)
# # pickle.dump(xgb_parameter_search, open(abs_path/ 'XGBoost_model.dat', 'wb'))
# xgb_parameter_search = pickle.load(open(abs_path/ 'XGBoost_model.dat', 'rb'))
#
# y_test_hat = xgb_parameter_search.predict(x_test)
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(y_test_hat, label='Predicted')
# ax.plot(y_test.values, label='Actual')
# ax.legend()
#
# mu.plot_parameters_importances(xgb_parameter_search, variable_names=x_test.columns)
#
# ### LINEAR MODELS
# scaler_x = StandardScaler()
# scaler_x.fit(x_train)
# scaler_y = StandardScaler()
# scaler_y.fit(y_train)
#
# x_train_scaled = scaler_x.transform(x_train)
# y_train_scaled = scaler_y.transform(y_train)
#
# x_test_scaled = scaler_x.transform(x_test)
# y_test_scaled = scaler_y.transform(y_test)
#
# # Linear and RBF support vector machine
# svr =  SVR()
# param_dist_svr = {'C': stats.expon(scale=100),
#                   'gamma': stats.expon(scale=1),
#                   'kernel': ['linear']}
#
# svr_parameter_search = RandomizedSearchCV(svr,
#                                           param_distributions=param_dist_svr,
#                                           n_iter=10,
#                                           scoring='neg_mean_squared_error',
#                                           cv=5,
#                                           refit=1)
# # svr_parameter_search.fit(x_train_scaled, y_train_scaled.ravel())
#
#
# xgb_parameter_search.best_estimator_.importance_type
#
# # svr =  SVR(C=1.0, epsilon=0.2)
# # svr.fit(x_train_scaled, y_train_scaled.ravel())
#
# lasso_model = Lasso(alpha=0.001, max_iter=1000000)
# lasso_model.fit(x_train_scaled, y_train_scaled.ravel())
# print(lasso_model.coef_)
# y_test_scaled_hat = lasso_model.predict(x_test_scaled)
# y_test_hat = scaler_y.inverse_transform(y_test_scaled_hat)
#
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(y_test_scaled_hat.ravel(), label='Predicted')
# ax.plot(y_test_scaled.ravel(), label='Actual')
# ax.legend()
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(y_test_hat.ravel(), label='Predicted')
# ax.plot(y_test.values.ravel(), label='Actual')
# ax.legend()
#
# print(f'RMSE: {np.sqrt(mean_squared_error(y_test_scaled.ravel(), y_test_scaled_hat.ravel()))}')
# print(f'RMSE: {np.sqrt(mean_squared_error(y_test.values.ravel(), y_test_hat.ravel()))}')
#
# # Inject noise to see how the algorithm responds
# rmse_lasso = list()
# noise_values = np.linspace(0, 0.1, 100)
#
# # for noise_level in np.logspace(-10, -1, 10, endpoint=True):
# for noise_level in noise_values:
#     # print(noise_level)
#     x_test_noise = x_test.drop(columns='time')
#     noise = (x_test_noise * noise_level) * np.random.normal(size=x_test_noise.shape)
#     # x_test_noise = x_test_noise + (x_test_noise * 0.05) * np.random.normal(size=x_test_noise.shape)
#     x_test_noise = x_test.add(noise, fill_value=0).loc[:, x_test.columns]  # Re-order columns
#     x_test_noise_scaled = scaler_x.transform(x_test_noise)
#     y_test_noise_scaled_hat = lasso_model.predict(x_test_noise_scaled)
#     y_test_noise_hat = scaler_y.inverse_transform(y_test_noise_scaled_hat)
#
#     rmse_lasso.append(np.sqrt(mean_squared_error(y_test.values.ravel(), y_test_noise_hat.ravel())))
#
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(noise_values, rmse_lasso, label='Predicted')
#
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(y_test_noise_hat, label='Predicted')
# ax.plot(y_test.values, label='Actual')
# ax.legend()
#
#
#
#
# ################################################### NOISE XGBOOST ######################################################
# # Inject noise to see how the algorithm responds
# rmse = list()
# noise_values = np.linspace(0, 0.1, 100)
#
# # for noise_level in np.logspace(-10, -1, 10, endpoint=True):
# for noise_level in noise_values:
#     # print(noise_level)
#     x_test_noise = x_test.drop(columns='time')
#     noise = (x_test_noise * noise_level) * np.random.normal(size=x_test_noise.shape)
#     # x_test_noise = x_test_noise + (x_test_noise * 0.05) * np.random.normal(size=x_test_noise.shape)
#     x_test_noise = x_test.add(noise, fill_value=0).loc[:, x_test.columns]
#
#     y_test_noise_hat = xgb_parameter_search.predict(x_test_noise)
#
#     rmse.append(np.sqrt(mean_squared_error(y_test.values.ravel(), y_test_noise_hat.ravel())))
#
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(noise_values, rmse, label='Predicted')
#
#
# fig = plt.figure(figsize=(6, 6))
# ax = fig.subplots(1, 1)
# ax.plot(y_test_noise_hat, label='Predicted')
# ax.plot(y_test.values, label='Actual')
# ax.legend()
#
#
#
#
# ####################################################################################################################
# ######################################   SUPPORT VECTOR MACHINES    ################################################
# ####################################################################################################################
# svr =  SVR(C=1.0, epsilon=0.2)
# svr.fit(x_train, y_train)
#
#
# # Linear and RBF support vector machine
# svr =  SVR()
# param_dist_svr = {'C': stats.expon(scale=100),
#                   'gamma': stats.expon(scale=1),
#                   'kernel': ['linear', 'rbf']}
#
# svr_parameter_search = RandomizedSearchCV(svr,
#                                           param_distributions=param_dist_svr,
#                                           n_iter=50,
#                                           scoring='neg_mean_squared_error',
#                                           cv=5,
#                                           refit=1)
# svr_parameter_search.fit(x_train, y_train)
# # pickle.dump(svr_parameter_search, open(abs_path / 'SVR_RBF_model.dat', 'wb'))
# svr_linear = pickle.load(open(abs_path/ 'SVR_linear_model.dat', 'rb'))
#
#
# # svr_linear
# #
# # fig = plt.figure(figsize=(6, 5))
# # ax = fig.subplots(1, 1)
# # plt.subplots_adjust(bottom=0.15)
# # ax.bar(x_train.columns, svr_linear.best_estimator_.coef_.ravel())
# # ax.tick_params(axis='x', rotation=90, labelsize='small')
# # ax.set_title('Feature importance')
# #
# #
# #
# #
# # y_test_hat = svr_parameter_search.predict(x_test)
# #
# # fig = plt.figure(figsize=(6, 6))
# # ax = fig.subplots(1, 1)
# # ax.plot(y_test_hat)
# # ax.plot(y_test.values)
# # ax.set_title('SVR RBF')
# #
# # corr_matrix = x_train.drop(columns='v_1').corr(method='pearson')
# # #%%
# # fig = plt.figure(figsize=(6, 6))
# # ax = fig.subplots(1, 1)
# # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
# # sns.heatmap(corr_matrix, cmap=plt.cm.get_cmap('PuOr'), ax=ax)
# # ax.tick_params(axis='y', rotation=0)
