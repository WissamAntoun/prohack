#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
#%%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
#%%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

#%%
target = 'y'
Non_data_col = ['galaxy','y']
#%%
#Choose all predictors except target & IDcols
predictors = [x for x in train_df.columns if x not in Non_data_col]
#%%
xgb1 = XGBRegressor()
#%%
lr = [i/100.0 for i in range(1,10,2)]
l2 = [i/10.0 for i in range(1,10,2)]
lr.extend(l2)
lr.extend([1,2])

parameters = { 
                'objective':['reg:linear'],
                'learning_rate': lr, 
                'max_depth': range(10,80,10),
                'min_child_weight': range(1,7,2),
                'silent': [1],
                'subsample': [0.5,0.55,0.6],
                'colsample_bytree': [0.7,0.8,0.85],
                'n_estimators': [10,50,100,200,500],
                'objective':['reg:squarederror'],
                # 'tree_method':['gpu_hist'],
                # 'gpu_id':[0]
                'seed' : [42]
                }

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        scoring='neg_root_mean_squared_error', #this line can be commented to use XGB's default metric
                        cv = 5,
                        n_jobs = 4,
                        verbose=True)


#%%
xgb_grid.fit(train_df[predictors],train_df[target])
print(xgb_grid.best_estimator_)
print(xgb_grid.best_score_)
# %%
