#%%
import glob
import re
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm_notebook
#%%
filenames = glob.glob("clean_data/*.csv")


# %%
dict_galaxy_train = {}
dict_galaxy_test = {}
for galaxy_file in filenames:
    galaxy_df = pd.read_csv(galaxy_file).set_index('Unnamed: 0')

    train_df = galaxy_df[np.isnan(galaxy_df['y'])==False]
    test_df = galaxy_df[np.isnan(galaxy_df['y'])==True]

    name = int(re.findall(r'\d+',galaxy_file)[0])
    dict_galaxy_train[name] = train_df
    dict_galaxy_test[name] = test_df

# %%

df_galaxy_preds = pd.DataFrame([])
for key in tqdm_notebook(dict_galaxy_test.keys()):
    train_df = dict_galaxy_train[key]
    test_df = dict_galaxy_test[key]
    
    if(test_df.empty):
        continue

    RFR = RandomForestRegressor()

    parameters = {
    'n_estimators' : range(5,51,10),
    'random_state' : [0,2,42],
    'max_depth' : range(10,81,10)
    }

    rfr_grid = GridSearchCV(RFR,
                        parameters,
                        scoring='neg_root_mean_squared_error', #this line can be commented to use XGB's default metric
                        cv = 5,
                        n_jobs = 4,
                        verbose=True,
                        refit=True)
    
    rfr_grid.fit(train_df.drop('y',axis=1,inplace=False),train_df['y'])
    print(rfr_grid.best_estimator_)
    print(rfr_grid.best_score_)

    test_df['y'] = rfr_grid.best_estimator_.predict(test_df.drop('y',axis=1,inplace=False))
    df_galaxy_preds = pd.concat([df_galaxy_preds,test_df['y']],axis=0,ignore_index=False)

    



# %%
df_galaxy_preds_sorted = df_galaxy_preds.sort_index()


# %%
df_galaxy_preds_sorted.columns=['y']
#%%
test_df = pd.read_csv("data/test.csv")
# %%
df_galaxy_preds_sorted['existence expectancy index'] = list(test_df['existence expectancy index'])

# %%
eei_df= pd.DataFrame([])
for key in tqdm_notebook(dict_galaxy_test.keys()):
    test_df = dict_galaxy_test[key]    
    if(test_df.empty):
        continue    
    eei_df = pd.concat([eei_df,test_df['existence expectancy index']],axis=0,ignore_index=False)
#%%
eei_df.columns = ['existence expectancy index']
# %%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
labels = train_df['y']

train_df = train_df.drop('y',axis =1 )

#%%
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True)
#%%
all_data["galaxy"] = all_data["galaxy"].astype('category')
all_data["galaxy"] = all_data["galaxy"].cat.codes

#%%

all_data_without_year_name = all_data.drop(['galactic year','galaxy'],axis=1)
#%%
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(np.array(all_data_without_year_name['existence expectancy index']).reshape(-1,1))

# %%
tt = scaler.inverse_transform(np.array(eei_df['existence expectancy index']).reshape(-1,1))

# %%
eei_df['existence expectancy index'] = tt

# %%
eei_df_sorted = eei_df.sort_index()

# %%
df_galaxy_preds_sorted['existence expectancy index'] = eei_df_sorted['existence expectancy index']

# %%
df_galaxy_preds_sorted.to_csv('submission_task1.csv',index=False)

# %%
