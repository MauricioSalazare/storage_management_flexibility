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

file_name = 'ems_optimization_1.0_200.csv'
(data_train, data_test) = mu.split_data(file_name, testing_split=0.2)
mu.plot_dataset(data_train, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])
mu.plot_dataset(data_test, variables=['Loading', ' SOC', ' storage_P', ' storage_Q'])