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
matplotlib.rc('text', usetex=True)
from matplotlib.ticker import NullFormatter


file_name_with_battery = 'ems_optimization_2.1_200_yes.csv'
(data_train_with_battery, data_test_with_battery) = mu.split_data(file_name_with_battery, testing_split=0.0)

file_name_without_battery = 'ems_optimization_2.1_200_no.csv'
(data_train_without_battery, data_test_without_battery) = mu.split_data(file_name_without_battery, testing_split=0.0)

#%%
fig = plt.figure(figsize=(4, 3))
ax_ = fig.subplots(2, 2)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.95, wspace=0.4, hspace=0.15)
ax_ = ax_.ravel()

data_voltages = data_train_without_battery.filter(regex='v_', axis=1).drop(columns='v_1')
maximum_voltages = data_voltages.max(axis=1)
minimum_voltages = data_voltages.min(axis=1)

variable_data_min = list()
variable_data_max = list()
variable_data_loading = list()


for scenario in data_train_without_battery['Scenario'].unique():
    loading_values = data_train_without_battery.Loading[data_train_without_battery['Scenario'] == scenario]
    voltages_values_max = maximum_voltages[data_train_without_battery['Scenario'] == scenario]
    voltages_values_min = minimum_voltages[data_train_without_battery['Scenario'] == scenario]
    variable_data_max.append(voltages_values_max)
    variable_data_min.append(voltages_values_min)
    variable_data_loading.append(loading_values)

    ax_[0].plot(loading_values.values, linewidth=0.3, color='#808080')
    ax_[1].plot(voltages_values_max.values, linewidth=0.3, color='#9713AE')
    ax_[1].plot(voltages_values_min.values, linewidth=0.3, color='#FF8C00')

variable_data_max = np.array(variable_data_max)
variable_data_min = np.array(variable_data_min)
variable_data_loading = np.array(variable_data_loading)

ax_[0].plot(np.quantile(np.array(variable_data_loading), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
ax_[0].plot(np.quantile(np.array(variable_data_loading), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
ax_[0].plot(np.quantile(np.array(variable_data_loading), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')

ax_[1].plot(np.quantile(np.array(variable_data_max), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
ax_[1].plot(np.quantile(np.array(variable_data_max), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
ax_[1].plot(np.quantile(np.array(variable_data_max), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')

ax_[1].plot(np.quantile(np.array(variable_data_min), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
ax_[1].plot(np.quantile(np.array(variable_data_min), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
ax_[1].plot(np.quantile(np.array(variable_data_min), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')


data_voltages = data_train_with_battery.filter(regex='v_', axis=1).drop(columns='v_1')
maximum_voltages = data_voltages.max(axis=1)
minimum_voltages = data_voltages.min(axis=1)

variable_data_min = list()
variable_data_max = list()
variable_data_loading = list()

for scenario in data_train_with_battery['Scenario'].unique():
    loading_values = data_train_with_battery.Loading[data_train_with_battery['Scenario'] == scenario]
    voltages_values_max = maximum_voltages[data_train_with_battery['Scenario'] == scenario]
    voltages_values_min = minimum_voltages[data_train_with_battery['Scenario'] == scenario]
    variable_data_max.append(voltages_values_max)
    variable_data_min.append(voltages_values_min)
    variable_data_loading.append(loading_values)

    ax_[2].plot(loading_values.values, linewidth=0.3, color='#808080')
    ax_[3].plot(voltages_values_max.values, linewidth=0.3, color='#9713AE')
    ax_[3].plot(voltages_values_min.values, linewidth=0.3, color='#FF8C00')

# Trick to add the labels to the same legend
ax_[2].plot(loading_values.values, linewidth=0.3, color='#808080', label='Transformer Loading')
ax_[3].plot(voltages_values_max.values, linewidth=0.3, color='#9713AE', label='Max. Voltage')
ax_[3].plot(voltages_values_min.values, linewidth=0.3, color='#FF8C00', label='Min. Voltage')
ax_[3].plot(voltages_values_min.values, linewidth=0.3, color='#808080', label='Transformer Loading') # Trick to add label


variable_data_max = np.array(variable_data_max)
variable_data_min = np.array(variable_data_min)
variable_data_loading = np.array(variable_data_loading)

ax_[2].plot(np.quantile(np.array(variable_data_loading), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
ax_[2].plot(np.quantile(np.array(variable_data_loading), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
ax_[2].plot(np.quantile(np.array(variable_data_loading), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')

ax_[3].plot(np.quantile(np.array(variable_data_max), q=0.05, axis=0), color='r', linestyle=':', label='Perc. 5')
ax_[3].plot(np.quantile(np.array(variable_data_max), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8, label='Perc. 50')
ax_[3].plot(np.quantile(np.array(variable_data_max), q=0.95, axis=0), color='r', linestyle='--', label='Perc. 95')

ax_[3].plot(np.quantile(np.array(variable_data_min), q=0.05, axis=0), color='r', linestyle=':')
ax_[3].plot(np.quantile(np.array(variable_data_min), q=0.5, axis=0), color='k', linestyle='-', linewidth=0.8)
ax_[3].plot(np.quantile(np.array(variable_data_min), q=0.95, axis=0), color='r', linestyle='--')

# Labels and titles
ax_[0].xaxis.set_major_formatter(NullFormatter())
ax_[0].set_ylim((20, 170))
ax_[0].set_xticks([])
ax_[0].axhline(y=100, linewidth=0.7, linestyle='-', color='b')
ax_[0].set_ylabel('[\%]', fontsize='small')

ax_[1].xaxis.set_major_formatter(NullFormatter())
ax_[1].set_xticks([])
ax_[1].set_ylim((0.85, 1.15))
ax_[1].axhline(y=1.1, linewidth=0.7, linestyle='-', color='b')
ax_[1].axhline(y=0.90, linewidth=0.7, linestyle='--', color='b')
ax_[1].set_ylabel('[p.u.]', fontsize='small')

ax_[2].set_ylim((20, 170))
ax_[2].axhline(y=100, linewidth=0.7, linestyle='-', color='b')
ax_[2].set_xlabel('Hour', fontsize='small')
ax_[2].set_ylabel('[\%]', fontsize='small')

ax_[3].set_ylim((0.85, 1.15))
ax_[3].axhline(y=1.1, linewidth=0.7, linestyle='-', color='b')
ax_[3].axhline(y=0.90, linewidth=0.7, linestyle='--', color='b')
ax_[3].set_xlabel('Hour', fontsize='small')
ax_[3].set_ylabel('[p.u.]', fontsize='small')

ax_[0].tick_params(axis='both', labelsize='small')
ax_[1].tick_params(axis='both', labelsize='small')
ax_[2].tick_params(axis='both', labelsize='small')
ax_[3].tick_params(axis='both', labelsize='small')

ax_[3].legend(loc='lower center', bbox_to_anchor=(-0.2, -0.8), fancybox=False, shadow=False, ncol=3, fontsize=7)



