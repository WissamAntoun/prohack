# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Ömer Gözüaçık, 29/05/2020

# %%
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from scipy.optimize import minimize
from tqdm import tqdm

import xgboost as xgb
from xgboost.sklearn import XGBRegressor


import warnings 
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Importing data

# %%
pd.set_option('display.max_columns',None)

# training data
train = pd.read_csv('data/train.csv')

# test data
test = pd.read_csv('data/test.csv')
df=pd.concat([train,test], sort=False)
train.head()

# %% [markdown]
# ### Converting categorical feature to numbers
# 
# There is no specific reason. I think it is easier for the eye.

# %%
df["galaxy"] = df["galaxy"].astype('category')
df["galaxy"] = df["galaxy"].cat.codes
train = df[:3865]
test = df[3865:]
test=test.drop("y", axis = 1)
test_res= test.copy()

# %% [markdown]
# ### Checking how many galaxies are there and how many of them are distinct.
# 
# - There are **181** distinct galaxies on the training set and **172** on the test set.
# 
# - On overall they each galaxy has **20** samples on the training set and **5** on the test set.
# 
# - **Some galaxies on the training set does not exist on the test set.**
# 
# - **Galaxy 126** has only one sample. I discard it on the training phase
# 
# As far as I know, the world bank has **182** members (countries) in 2000s (IBRD). Each distinct galaxy may represent a country in real life. Every sample for a galaxy may represent the properties of the country at a time (galactic year). 

# %%
train_gal=set(train["galaxy"])
s=0
for x in train_gal:
    s=s+len(train.loc[train['galaxy'] == x])
print("Total distinct galaxies: {}".format(len(train_gal)))
print("Average samples per galaxy: {}".format(s/len(train_gal)))


# %%
test_gal=set(test["galaxy"])
s=0
for x in test_gal:
    s=s+len(test.loc[test['galaxy'] == x])
print("Total distinct galaxies: {}".format(len(test_gal)))
print("Average samples per galaxy: {}".format(s/len(test_gal)))

# %% [markdown]
# #### Number of samples and features
# Train set: 3865
# 
# Test set: 890
# 
# Features: 79

# %%
print("Train vector: " + str(train.shape))
print("Test vector: " + str(test.shape))

# %% [markdown]
# ## Methods for Cross-validating Training Data
# 
# - I trained **a model for exery distinct galaxy** in the training set (180) except the one from 126th galaxy as it has only one sample. 
# 
# - I used **features with top x correlation** with respect to y (target variable) galaxy specific. (x is found by trying different values [20,25,30,40,50,60,70])
# 
# - Missing values are filled with the galaxy specific 'mean' of the data. (Median can be used alternatively.)
# 
# - **Train and test sets are not mixed for both imputation and standardization.**
# 
# - Standard Scaler is used to standardize data.
# 
# - Gradient Boosted Regression is used as a model.

# %%
def cross_validation_loop(data,cor):
    labels= data['y']
    data=data.drop('galaxy', axis=1)    
    data=data.drop('y', axis=1)
    
    correlation=abs(data.corrwith(labels))
    columns=correlation.nlargest(cor).index
    data=data[columns]
    
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data)
    # data=imp.transform(data)

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)

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

    estimator = XGBRegressor(n_estimators=300)   
    #estimator = GradientBoostingRegressor(n_estimators=300)
    
    cv_results = cross_validate(estimator, data, labels, cv=5, scoring='neg_root_mean_squared_error')

    error=np.mean(cv_results['test_score'])
    
    return error

# %% [markdown]
# #### Code for cross-validating a model for every galaxy
# 
# I return the mean of the cross-validation scores disregarding the differences of their sample sizes.

# %%
train_gal=set(train["galaxy"])
train_gal.remove(126)
def loop_train(cor):
    errors=[]
    for gal in tqdm(train_gal):
        index = train.index[train['galaxy'] == gal]
        data = train.loc[index]
        errors.append(cross_validation_loop(data,cor))
    return np.mean(errors)

# %% [markdown]
# #### Checking which correlation threshold gives better value
# 
# The model performs best when the threshold is 20 with RMSE of 0.0063

# %%
cor=[20,25,30,40,50,60,70,80]
errors=[]
for x in cor:
    print("cor: ",x)
    errors.append(loop_train(x))


 # %%
print(errors)

# [-0.005510409192904806, -0.005474700678841418, -0.005478204236398942, -0.005493891458843025, -0.005485265856592613, -0.005493237060981963, -0.005493713846323645, -0.0055068515842603225]
# %% [markdown]
# ## Making predictions on the test data
# 
# - Similar methodology is used to fill the missing value and standardization.
# - The best covariance threshold in the cross validation, 20, is used.

# %%
def test_loop(data, test_data):
    labels= data['y']
    data=data.drop('galaxy', axis=1)    
    data=data.drop('y', axis=1)
    correlation=abs(data.corrwith(labels))
    columns=correlation.nlargest(20).index
    
    train_labels= labels
    train_data=data[columns]
    test_data= test_data[columns]
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(train_data)
    train_data=imp.transform(train_data)
    test_data=imp.transform(test_data)

    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    model = GradientBoostingRegressor(n_estimators=300)
    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    return predictions

# %% [markdown]
# #### Sorting samples with respect to their unique galaxy type. 

# %%
test=test_res
test=test.sort_values(by=['galaxy'])
test_pred = pd.DataFrame(0, index=np.arange(len(test)), columns=["predicted_y"])

# %% [markdown]
# #### Looping over all galaxy types in the test set and making predictions.

# %%
i=0
for gal in test_gal:
    count=len(test.loc[test['galaxy'] == gal])
    index = train.index[train['galaxy'] == gal]
    data = train.loc[index]
    pred=test_loop(data,test.loc[test['galaxy']==gal])
    test_pred.loc[i:i+count-1,'predicted_y'] = pred
    i=i+count 

# %% [markdown]
# #### Sorting samples with respect to the index.

# %%
test["predicted_y"]=test_pred.to_numpy()
test.sort_index(inplace=True)
predictions = test["predicted_y"]

# %% [markdown]
# ## Discussion 1
# 
# - With this approach, we are **not using 8 galaxies in the training set as they are not in the test set.** (Almost 160 samples)
# 
# - A better approach should use them as well.
# 
# - According to our theory, every galaxy represent a country and samples are its properties at a time (maybe galactic year represents time).
# 
# - Some countries may have missing values as they may have joined IBRD late. This may be organizers decision as well. Filling missing values with regression can improve performance.
# 
# - World Bank categorizes countries by both region and income: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups
# 
# 7 regions: East Asia and Pacific, Europe and Central Asia, Latin America & the Caribbean, Middle East and North Africa, North America, South Asia, Sub-Saharan Africa
# 
# 4 income groups: Low-income economies, Lower-middle-income economies, Upper-middle-income economies, High-income economies 
# 
# - Clustering galaxies may excel the performance of the model. I would try both clustering galaxies to either 4 or 7 clusters. Then try making imputation/training with respect to every cluster.
# 
# This code is a summary of what we have done. We also analyzed RMSE for cross-validation for per galaxy. 
# 
# Galaxies: {128, 2, 4, 5, 133, 11, 140, 147, 153, 154, 34, 35, 40, 43, 55, 64, 76, 78, 83, 100, 101, 102, 107, 108, 119} have RMSE over 0.008. 
# 
# The list gives them in order, 128th having 0.008559 and 119th having 0.034926. 
# 
# - Fine tuning these problematic galaxies with low cross-validation scores can excel the performance of the model
# %% [markdown]
# ## Optimization part
# 
# - Ideally giving 100 to top 500 samples with highest p^2 values should optimize the likely increase.
# - However, as the predictions can be faulty, this approach would result with lower Leaderboard Score.
# 
# E.g: If the original p^2 value is higher than the predicted p^2, it will increase the error as we are directly giving it 0.
# 
# - That's why, I believe its better to spread the risk for the samples in the bordering regions (400< [rank of p^2] <600).
# - I assign 100 energy to top 400 samples and 50 energy to the remaining top 200 samples.

# %%
index = predictions
pot_inc = -np.log(index+0.01)+3


# %%
p2= pot_inc**2


# %%
ss = pd.DataFrame({
    'Index':test.index,
    'pred': predictions,
    'opt_pred':0,
    'eei':test['existence expectancy index'], # So we can split into low and high EEI galaxies
})


# %%
ss.loc[p2.nlargest(400).index, 'opt_pred']=100
ss=ss.sort_values('pred')
ss.iloc[400:600].opt_pred = 50
ss=ss.sort_index()


# %%
increase = (ss['opt_pred']*p2)/1000


# %%
print(sum(increase), ss.loc[ss.eei < 0.7, 'opt_pred'].sum(), ss['opt_pred'].sum())


# %%
ss[['Index', 'pred', 'opt_pred']].to_csv('submission.csv', index=False)

# %% [markdown]
# ## Discussion 2
# 
# - Optimization can be done better by changing the spreading the risk part (assigning energy to the 400<p^2<600 region).
# 
# - You can give values that are decreasing starting from 400th to 600th (99, 98, 97...). 
# 
# It is less likely for the 400th sample to be out of top 500, and similarly it is less likely for the 600th sample to be in the top 500. That's why, you can give more energy to the ones in the 400-500 region and less to the 500-600.
# 
# - This approach got me and my friend to the best score of 0.04271993 which is ranked 22nd right now.
# 
# - As we are out of top 20 and reached the upload limit, we are sharing our approach publicly to help other teams that have worse results. 
# 

