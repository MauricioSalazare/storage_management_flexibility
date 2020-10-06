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
import matplotlib
matplotlib.rc('text', usetex=False)

class VARIABLES:
    PREDICT = [' storage_P']

file_name = 'ems_optimization_1.0_200.csv'

(data_train, data_test) = mu.split_data(file_name, testing_split=0.2)
# mu.plot_dataset(data_train, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])
# mu.plot_dataset(data_test, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])

(x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                             columns_drop=['Scenario','v_1', ' storage_Q'],
                                                             columns_predict=[' storage_P'],
                                                             testing_split=0)

####################### THIS TAKES TOO LONG !!!! #######################################################################
RETRAIN_ALL_MODELS = False
if RETRAIN_ALL_MODELS:
    # Run the fitting of the model 100 times with x-validation
    models_xgboost = list()
    for ii in range(100):
        print(f'Training model {ii} out of 99')
        models_xgboost.append(mu.model_training_xgboost(x_train, y_train))

    pickle.dump(models_xgboost, open(abs_path/ 'All_XGBoost_models_feature_analysis.dat', 'wb'))

models_xgboost = pickle.load(open(abs_path/ 'All_XGBoost_models_feature_analysis.dat', 'rb'))

# Check best models characteristics
for model in models_xgboost:
    print("*" * 7)
    print(f'Estimators: {model.best_estimator_.get_params()["n_estimators"]}')
    print(f'Max depth: {model.best_estimator_.get_params()["max_depth"]}')
    print(f'Max depth: {model.best_estimator_.get_params()["learning_rate"]}')

# Create feature importance matrix with the best estimators
feature_importance_weight = list()
feature_importance_cover = list()
feature_importance_gain = list()
for model in models_xgboost:
    model.best_estimator_.importance_type = 'weight'
    feature_importance_weight.append(model.best_estimator_.feature_importances_)
    model.best_estimator_.importance_type = 'cover'
    feature_importance_cover.append(model.best_estimator_.feature_importances_)
    model.best_estimator_.importance_type = 'gain' # Reset
    feature_importance_gain.append(model.best_estimator_.feature_importances_)

feature_importance_weight = np.array(feature_importance_weight)
feature_importance_cover = np.array(feature_importance_cover)
feature_importance_gain = np.array(feature_importance_gain)

#%%
feature_importance_df = pd.DataFrame({'Feature':  x_train.columns,
                                      'Importance': feature_importance_gain.mean(axis=0),
                                      'std_dev':  feature_importance_gain.std(axis=0)})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
fig = plt.figure(figsize=(5.5, 4))
ax = fig.subplots(1, 1)
plt.subplots_adjust(bottom=0.15)
ax.bar(feature_importance_df.Feature.str.replace('_','\_'),
       feature_importance_df.Importance,
       yerr=feature_importance_df.std_dev)
ax.tick_params(axis='x', rotation=90, labelsize='small')
ax.set_ylabel('Normalized')
ax.set_title('Feature Importance - Gain')

########################################################################################################################
#%% Filter the features, retrain and report.
####################### THIS TAKES TOO LONG !!!! #######################################################################
RETRAIN_ALL_MODELS = False
if RETRAIN_ALL_MODELS:
    feature_threshold = np.arange(2, 38, 2)
    models_trained = list()
    for threshold in feature_threshold:
        print(f'Threshold: {threshold}')
        input_features_columns = feature_importance_df[:threshold].Feature.to_list()
        (x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                                     columns_drop=['Scenario','v_1', ' storage_Q'],
                                                                     columns_predict=[' storage_P'],
                                                                     testing_split=0,
                                                                     select_input_features=True,
                                                                     input_features_columns=input_features_columns)

        xgb_parameter_search = mu.model_training_xgboost(x_train, y_train)
        models_trained.append(xgb_parameter_search)
        print(f'RMSE: {np.sqrt(-xgb_parameter_search.cv_results_["mean_test_score"].max())}')
    pickle.dump((feature_threshold, models_trained), open(abs_path/ 'All_XGBoost_models_feature_analysis_reduced_models.dat', 'wb'))

# (feature_threshold, models_trained) = pickle.load(open(abs_path/ 'All_XGBoost_models_feature_analysis_reduced_models.dat', 'rb'))

# The next 2 lines should be deleted when the model is retrained again!!!
models_trained = pickle.load(open(abs_path / 'All_XGBoost_models_feature_analysis_reduced_models.dat', 'rb'))
feature_threshold = np.arange(2, 38, 2)

best_rmse = list()
for model in models_trained:
    best_rmse.append(np.sqrt(-model.cv_results_['mean_test_score'].max()))
    print(f'RMSE: {np.sqrt(-model.cv_results_["mean_test_score"].max())}')
rmse_vector = np.array(best_rmse)

#%%
fig = plt.figure(figsize=(5, 4))
ax = fig.subplots(1, 1)
plt.subplots_adjust(left=0.15, top=0.95)
ax.plot(feature_threshold, rmse_vector)
ax.set_ylabel('RMSE [kW]')
ax.set_xlabel('Number of features')
ax.set_xticks(feature_threshold)
ax.set_yticks(list(ax.get_yticks())+[round(rmse_vector[-1])])
ax.axhline(y=round(rmse_vector[-1]), linewidth=0.7, linestyle='--', color='r')
ax.tick_params(axis='x', labelsize='small')

#%%
# The figures in in the same plot
fig = plt.figure(figsize=(4, 3))
ax = fig.subplots(2, 1)
plt.subplots_adjust(bottom=0.15, right=0.95, top=0.95, hspace=0.6)
ax[0].bar(feature_importance_df.Feature.str.replace('_','\_'),
       feature_importance_df.Importance,
       yerr=feature_importance_df.std_dev)
ax[0].tick_params(axis='x', rotation=90, labelsize='small')
ax[0].set_ylabel('Normalized', fontsize='small')
# ax[0].set_title('Feature Importance - Gain', fontsize='small')

ax[1].plot(feature_threshold, rmse_vector)
ax[1].set_ylabel('RMSE [kW]', fontsize='small')
ax[1].set_xlabel('Number of features', fontsize='small')
ax[1].set_xticks(feature_threshold)
# ax[1].set_yticks(list(ax[1].get_yticks())+[round(rmse_vector[-1])])
ax[1].axhline(y=round(rmse_vector[-1]), linewidth=0.7, linestyle='--', color='r')
ax[1].tick_params(axis='x', labelsize='small')
ax[1].set_ylim((90, 400))
