#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
RFR = RandomForestRegressor()

lr = [i/100.0 for i in range(1,10,2)]
l2 = [i/10.0 for i in range(1,10,2)]
lr.extend(l2)
lr.extend([1,2])

parameters = {
    'n_estimators' : [250,300],
    'random_state' : [0,2,42],
    'max_depth' : range(10,31,4)
}


rfr_grid = GridSearchCV(RFR,
                        parameters,
                        scoring='neg_root_mean_squared_error', #this line can be commented to use XGB's default metric
                        cv = 5,
                        n_jobs = 4,
                        verbose=True)


#%%
rfr_grid.fit(X_train,labels)
print(rfr_grid.best_estimator_)
print(rfr_grid.best_score_)

# %%
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=40, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=300, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)
# -0.02648496264778788

#Standard Scaler
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=18, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=200, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)
# -0.025359745802465854

#MinMaxScaler
# Fitting 5 folds for each of 54 candidates, totalling 270 fits
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.4min
# [Parallel(n_jobs=4)]: Done 192 tasks      | elapsed: 12.4min
# [Parallel(n_jobs=4)]: Done 270 out of 270 | elapsed: 18.3min finished
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=20, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=300, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)
# -0.026147830873975975

#RobustScaler
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=20, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=200, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)
# -0.025365427652947543

#RobustScaler
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=22, max_features='auto', max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=250, n_jobs=None, oob_score=False,
#                       random_state=0, verbose=0, warm_start=False)
# -0.02542345660373719