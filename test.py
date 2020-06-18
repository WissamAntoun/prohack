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
corr_mat=train_df.corr(method='pearson')
plt.figure(figsize=(50,25))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.savefig('pearson_corr', dpi = 600)
# %%


# %%

columns_to_drop = [
    "Gender Inequality Index (GII)",
    "Private galaxy capital flows (% of GGP)",
    "Creature Immunodeficiency Disease prevalence, adult (% ages 15-49), total",
    "Adjusted net savings ",
    "Intergalactic Development Index (IDI), male, Rank",
    "Intergalactic Development Index (IDI), female, Rank",
    "Gender Development Index (GDI)",
    'Intergalactic Development Index (IDI), female',
    'Intergalactic Development Index (IDI), male',
    'Current health expenditure (% of GGP)',
    'Gross capital formation (% of GGP)', 'Population, total (millions)',
    'Population, urban (%)',
    'Mortality rate, under-five (per 1,000 live births)',
    'Mortality rate, infant (per 1,000 live births)',
    'Old age dependency ratio (old age (65 and older) per 100 creatures (ages 15-64))',
    'Population, ages 15–64 (millions)',
    'Population, ages 65 and older (millions)',
    'Life expectancy at birth, male (galactic years)',
    'Life expectancy at birth, female (galactic years)',
    'Population, under age 5 (millions)',
    'Young age (0-14) dependency ratio (per 100 creatures ages 15-64)',
    'Adolescent birth rate (births per 1,000 female creatures ages 15-19)',
    'Total unemployment rate (female to male ratio)',
    'Vulnerable employment (% of total employment)',
    'Unemployment, total (% of labour force)',
    'Employment in agriculture (% of total employment)',
    'Labour force participation rate (% ages 15 and older)',
    'Labour force participation rate (% ages 15 and older), female',
    'Employment in services (% of total employment)',
    'Labour force participation rate (% ages 15 and older), male',
    'Employment to population ratio (% ages 15 and older)',
    'Jungle area (% of total land area)',
    'Share of employment in nonagriculture, female (% of total employment in nonagriculture)',
    'Youth unemployment rate (female to male ratio)',
    'Unemployment, youth (% ages 15–24)',
    'Mortality rate, female grown up (per 1,000 people)',
    'Mortality rate, male grown up (per 1,000 people)',
    'Infants lacking immunization, red hot disease (% of one-galactic year-olds)',
    'Infants lacking immunization, Combination Vaccine (% of one-galactic year-olds)',
    'Gross galactic product (GGP) per capita',
    'Gross galactic product (GGP), total',
    'Outer Galaxies direct investment, net inflows (% of GGP)',
    'Exports and imports (% of GGP)',
    'Share of seats in senate (% held by female)',
    'Natural resource depletion',
    'Mean years of education, female (galactic years)',
    'Mean years of education, male (galactic years)',
    'Expected years of education, female (galactic years)',
    'Expected years of education, male (galactic years)',
    'Maternal mortality ratio (deaths per 100,000 live births)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Estimated gross galactic income per capita, male',
    'Estimated gross galactic income per capita, female',
    'Rural population with access to electricity (%)',
    'Domestic credit provided by financial sector (% of GGP)',
    'Population with at least some secondary education, female (% ages 25 and older)',
    'Population with at least some secondary education, male (% ages 25 and older)',
    'Gross fixed capital formation (% of GGP)',
    'Remittances, inflows (% of GGP)',
    'Population with at least some secondary education (% ages 25 and older)',
    'Intergalactic inbound tourists (thousands)',
    'Gross enrolment ratio, primary (% of primary under-age population)',
    'Respiratory disease incidence (per 100,000 people)',
    'Interstellar phone subscriptions (per 100 people)',
    'Interstellar Data Net users, total (% of population)',
]

# %%
#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

target = 'y'
Non_data_col = ['galaxy','y','galactic year']
#%%
#Choose all predictors except target & IDcols


cleaned_df = train_df.drop(columns_to_drop,axis=1)
#%%
cleaned_df = cleaned_df.dropna(axis=0)
#%%
predictors = [x for x in cleaned_df.columns if x not in Non_data_col]
# %%
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

parameters2 = { 
                'objective':['reg:linear'],
                'learning_rate': [0.1], 
                'max_depth': range(10,80,5),
                'min_child_weight': [1],
                'silent': [1],
                'subsample': [0.5,0.55,0.6],
                'colsample_bytree': [0.7,0.8,0.85],
                'n_estimators': [1000],
                'objective':['reg:squarederror'],
                # 'tree_method':['gpu_hist'],
                # 'gpu_id':[0]
                'seed' : [42]
                }

xgb_grid = GridSearchCV(xgb1,
                        parameters2,
                        scoring='neg_root_mean_squared_error', #this line can be commented to use XGB's default metric
                        cv = 5,
                        n_jobs = 4,
                        verbose=True)


#%%
xgb_grid.fit(cleaned_df[predictors],cleaned_df[target])
print(xgb_grid.best_estimator_)
print(xgb_grid.best_score_)

# %%
# didnt drop na
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=20,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=1000, n_jobs=0, num_parallel_tree=1,
#              objective='reg:squarederror', random_state=42, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, seed=42, silent=1,
#              subsample=0.55, tree_method='exact', validate_parameters=1,
#              verbosity=None)
# -0.03167322216668586