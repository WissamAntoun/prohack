#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
#%%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
#%%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# # %%
# print(train_df.columns)
# print(test_df.columns)

# # %%
# print(set(train_df.columns).difference((set(test_df))))

# # %%
# sns.heatmap(train_df.isnull(), cbar=False)
# plt.savefig('null_heatmap.png',dpi=600)

# # %%
# df = train_df.iloc[train_df.isnull().sum(axis=1).mul(-1).argsort()]
# sns.heatmap(df.isnull(), cbar=False)
# plt.savefig('soreted_null_heatmap.png',dpi=600)

# # %%

# sns.heatmap(test_df.isnull(), cbar=False)
# plt.savefig('test_null_heatmap.png',dpi=600)

# # %%
# df2 = test_df.iloc[test_df.isnull().sum(axis=1).mul(-1).argsort()]
# sns.heatmap(df2.isnull(), cbar=False)
# plt.savefig('soreted_test_null_heatmap.png',dpi=600)

# %%
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

        
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % mean_squared_error(dtrain[target].values, dtrain_predictions,squared=False))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    

#%%
#%%
target = 'y'
Non_data_col = ['galaxy','y']
#%%
#Choose all predictors except target & IDcols
predictors = [x for x in train_df.columns if x not in Non_data_col]

xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 objective='reg:squarederror',
 colsample_bytree=0.8,
 nthread=4,
 scale_pos_weight=1,
 seed=42)

#%%
modelfit(xgb1, train_df, predictors)

# %%
