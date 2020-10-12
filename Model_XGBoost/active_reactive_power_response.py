import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import pickle
import Model_XGBoost.model_utils as mu
from matplotlib.ticker import NullFormatter
import matplotlib
abs_path = Path(__file__).parent
matplotlib.rc('text', usetex=True)
mu.set_figure_art()

opt_params = pickle.load(open(abs_path / 'XGBoost_optimal_params.dat', 'rb'))
opt_params_reactive = pickle.load(open(abs_path / 'XGBoost_optimal_params_REACTIVE.dat', 'rb'))

np.random.seed(1234)  # For reproducibility
file_name = 'ems_optimization_2.1_200_yes.csv'
(data_train, data_test) = mu.split_data(file_name, testing_split=0)  # All data to train
(x_train, y_train, x_test, y_test) = mu.split_data_for_model(file_name,
                                                             columns_drop=['Scenario', 'v_1', ' storage_Q'],
                                                             columns_predict=[' storage_P'],
                                                             testing_split=0.2)

xgb_regressor_model = xgb.XGBRegressor(**opt_params)
xgb_regressor_model.fit(x_train, y_train)



(x_train_reactive,
 y_train_reactive,
 x_test_reactive,
 y_test_reactive) = mu.split_data_for_model(file_name,
                                            columns_drop=['Scenario', 'v_1', ' storage_P'],
                                            columns_predict=[' storage_Q'],
                                            testing_split=0.2)

xgb_regressor_model_reactive = xgb.XGBRegressor(**opt_params_reactive)
xgb_regressor_model_reactive.fit(x_train_reactive, y_train_reactive)


#%% PLOT OF ALL TIME SERIES AND ZOOM IN A BOX
slide = 8 * 24  # 10 scenarios of 24 hours
from_day = 2 * 24
to_day = from_day + 1 * 24  # 1 days ahead
fold = 0

# Two folds are attached together to form a longer time series
# x_data = np.hstack([models_results['xgboost']['y_hat'][fold][0:slide] / 1000,
#                     models_results['xgboost']['y_hat'][fold+1][0:slide] / 1000])
# y_data = np.hstack([models_results['fold_params']['y_test'][fold][0:slide].values.ravel() / 1000,
#                     models_results['fold_params']['y_test'][fold+1][0:slide].values.ravel() / 1000])


x_data = xgb_regressor_model.predict(x_test) / 1000
y_data = y_test.values.ravel() / 1000

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


########################################################################################################################
#################################### REACTIVE POWER RESPONSE ###########################################################
########################################################################################################################


#%% PLOT OF ALL TIME SERIES AND ZOOM IN A BOX
slide = 8 * 24  # 10 scenarios of 24 hours
from_day = 2 * 24
to_day = from_day + 1 * 24  # 1 days ahead
fold = 0

# Two folds are attached together to form a longer time series
# x_data = np.hstack([models_results['xgboost']['y_hat'][fold][0:slide] / 1000,
#                     models_results['xgboost']['y_hat'][fold+1][0:slide] / 1000])
# y_data = np.hstack([models_results['fold_params']['y_test'][fold][0:slide].values.ravel() / 1000,
#                     models_results['fold_params']['y_test'][fold+1][0:slide].values.ravel() / 1000])


x_data_reactive = xgb_regressor_model_reactive.predict(x_test_reactive) / 1000
y_data_reactive = y_test_reactive.values.ravel() / 1000

_, ax = plt.subplots(1, 1, figsize=(4, 2.5))
plt.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.95)
ax.plot(x_data_reactive, label='GB-Trees', linewidth=0.6)
ax.plot(y_data_reactive, label='Actual', linewidth=0.6)
ax.set_ylabel('Reactive Power [MVAr]', labelpad=-4)
ax.set_ylim((-2.5, 8))
ax.set_xlim((0, slide))
# ax.set_title('Algorithms response')
ax.legend(loc='upper left', fancybox=False, shadow=False, fontsize='small')

axins = ax.inset_axes([0.5, 0.55, 0.45, 0.4])
axins.plot(x_data_reactive, label='GB-Trees', linewidth=0.6)
axins.plot(y_data_reactive, label='Actual', linewidth=0.6)
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
x_ticks_pos = (np.linspace(0, slide, int(slide / 24) + 1) + 12)[0:-1]
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels((np.arange(x_ticks_pos.shape[0]) + 1).astype('str'))
ax.set_xlabel('Scenario')

# Create the vertical lines
for x_line in np.linspace(0, slide, int(slide / 24) + 1):
    ax.axvline(x=x_line, linewidth=0.3, linestyle='-', color='#808080')


#%%
########################################################################################################################
#################################### ADD BOTH THINGS IN THE SAME PLOT ##################################################
########################################################################################################################


_, (ax, ax_) = plt.subplots(2, 1, figsize=(3.5, 2))
plt.subplots_adjust(left=0.15, bottom=0.18, right=0.97, top=0.89, hspace=0)
ax.plot(y_data, label='OEM', linewidth=0.8, color='k')
ax.plot(x_data, label='SGBT', linewidth=0.8, color='r')
ax.set_ylabel('Active\n [MW]', fontsize=7, labelpad=10)
ax.xaxis.set_major_formatter(NullFormatter())
ax.set_ylim((-3, 8))
ax.set_xlim((0, slide))
ax.tick_params(axis='both', labelsize=7)
# ax.set_title('Algorithms response')
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.35),fancybox=False, shadow=False, fontsize=7)

axins = ax.inset_axes([0.5, 0.55, 0.45, 0.4])
axins.plot(y_data, label='Actual', linewidth=0.8, color='k')
axins.plot(x_data, label='SBT', linewidth=0.8, color='r')
# axins.set_ylim(-3, 3)
# Sub region of the original image
x1, x2, y1, y2 = from_day, to_day, -2.5, 3
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.xaxis.set_major_formatter(NullFormatter())
# axins.yaxis.set_major_formatter(NullFormatter())
axins.set_xticks([])
# axins.set_yticks([])

ax.indicate_inset_zoom(axins, linewidth=0.8)

# Create the vertical lines
for x_line in np.linspace(0, slide, int(slide / 24) + 1):
    ax.axvline(x=x_line, linewidth=0.3, linestyle='-', color='#808080')


ax_.plot(y_data_reactive, label='OEM', linewidth=0.8, color='k')
ax_.plot(x_data_reactive, label='SGBT', linewidth=0.8, color='r')
ax_.set_ylabel('Reactive\n [MVAr]', fontsize=7)
# ax_.set_ylim((-2.5, 8))
ax_.set_xlim((0, slide))
ax_.tick_params(axis='both', labelsize=7)
# ax.set_title('Algorithms response')
# ax_.legend(loc='upper left', fancybox=False, shadow=False, fontsize='small')


# x_ticks_pos = (np.linspace(0, slide, int(slide / 24) + 1) + 12)[0:-1]
# ax_.set_xticks(x_ticks_pos)
# ax_.set_xticklabels((np.arange(x_ticks_pos.shape[0]) + 1).astype('str'))
# ax_.set_xlabel('Scenario')


x_ticks_pos = (np.linspace(0, slide, int(slide / 24) + 1) + 24)[0:-1]
ax_.set_xticks(x_ticks_pos)
ax_.set_xticklabels(x_ticks_pos.astype('int').astype('str'))
# ax_.set_xticklabels((np.arange(x_ticks_pos.shape[0]) + 1).astype('str'))
ax_.set_xlabel('Instance number [hours]', fontsize=7)


for x_line in np.linspace(0, slide, int(slide / 24) + 1):
    ax_.axvline(x=x_line, linewidth=0.3, linestyle='-', color='#808080')