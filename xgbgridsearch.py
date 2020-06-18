#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
#%%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
#%%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
labels = train_df['y']

# train_df = train_df.fillna(0)
# test_df = test_df.fillna(0)

train_df = train_df.drop('y',axis =1 )
#%%

# Non_data_col = ['galactic year','y']
# #%%
# #Choose all predictors except target & IDcols
# predictors = [x for x in train_df.columns if x not in Non_data_col]
#%%
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True)
#%%
all_data["galaxy"] = all_data["galaxy"].astype('category')
all_data["galaxy"] = all_data["galaxy"].cat.codes

#%%

all_data_without_year_name = all_data.drop(['galactic year','galaxy'],axis=1)
#%%
scaler = RobustScaler().fit(all_data_without_year_name)
all_data_without_year_name_scaled = scaler.transform(all_data_without_year_name)
#%%
year_name = all_data[['galactic year','galaxy']]
all_data_without_year_name_scaled_df = pd.DataFrame(all_data_without_year_name_scaled,columns=all_data_without_year_name.columns)
#%%
all_data_scaled = pd.concat([year_name,all_data_without_year_name_scaled_df],axis=1,sort=False)
#%%
all_data_scaled = all_data_scaled.fillna(0) 
#%%
X_train = all_data_scaled[0:len(train_df)]
X_test = all_data_scaled[len(train_df):]
#%%
xgb1 = XGBRegressor()

lr = [i/100.0 for i in range(1,10,2)]
l2 = [i/10.0 for i in range(1,10,2)]
lr.extend(l2)
lr.extend([1,2])

parameters = { 
                'objective':['reg:linear'],
                'learning_rate': lr, 
                'max_depth': range(5,16,5),
                'min_child_weight': range(1,5,1),
                'silent': [1],
                'subsample': [0.5,0.55,0.6],
                'colsample_bytree': [0.65,0.7,0.75],
                'n_estimators': [200,250,300],
                'objective':['reg:squarederror'],
                #'tree_method':['gpu_exact'],
                # 'gpu_id':[0],
                'seed' : [42]
                }

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        scoring='neg_root_mean_squared_error', #this line can be commented to use XGB's default metric
                        cv = 5,
                        n_jobs = 8,
                        verbose=True)


#%%
xgb_grid.fit(X_train,labels)
print(xgb_grid.best_estimator_)
print(xgb_grid.best_score_)
# %%
# XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints=None,
#              learning_rate=0.05, max_delta_step=0, max_depth=10,
#              min_child_weight=3, missing=nan, monotone_constraints=None,
#              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=42,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
#              subsample=0.55, tree_method=None, validate_parameters=False,
#              verbosity=None)
# -0.02789310275675531

# GBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.75, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints=None,
#              learning_rate=0.05, max_delta_step=0, max_depth=10,
#              min_child_weight=3, missing=nan, monotone_constraints=None,
#              n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=42,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
#              subsample=0.6, tree_method=None, validate_parameters=False,
#              verbosity=None)
# -0.022127200450818837

# XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints=None,
#              learning_rate=0.05, max_delta_step=0, max_depth=10,
#              min_child_weight=4, missing=nan, monotone_constraints=None,
#              n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=42,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
#              subsample=0.6, tree_method=None, validate_parameters=False,
#              verbosity=None)
# -0.021227692090664572

#%%

xgb1 = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.05, max_delta_step=0, max_depth=10,
             min_child_weight=4, monotone_constraints=None,
             n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=42,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
             subsample=0.6, tree_method=None, validate_parameters=False,
             verbosity=1)

# %%

xgb1.fit(X_train,labels)

# %%
preds= xgb1.predict(X_test)

# %%
print(preds)

# %%
X_test['y'] = preds
sub_df = X_test[['y','existence expectancy index']]
#%%
sub_df.to_csv('submission_task2.csv',index=False)# %%


# %%
