#%%
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
#%%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
#%%
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
scaler = RobustScaler().fit(all_data_without_year_name)
all_data_without_year_name_scaled = scaler.transform(all_data_without_year_name)
#%%
year_name = all_data[['galactic year','galaxy']]
all_data_without_year_name_scaled_df = pd.DataFrame(all_data_without_year_name_scaled,columns=all_data_without_year_name.columns)
#%%
all_data_scaled = pd.concat([year_name,all_data_without_year_name_scaled_df],axis=1,sort=False)
# all_data_scaled['galactic year'] =all_data_scaled['galactic year'] - all_data_scaled['galactic year'][0]
#%%
#all_data_scaled = all_data_scaled.fillna(0) 
#%%
X_train = all_data_scaled[0:len(train_df)]
X_test = all_data_scaled[len(train_df):]

# %%
Non_data_col = ['galaxy','y']
predictors = [x for x in all_data_scaled.columns if x not in Non_data_col]

#%%
galaxy_dict=collections.defaultdict(dict)
for col in tqdm_notebook(all_data_scaled[predictors].columns, desc='Column Loop'):
    selectef_col = all_data_scaled[col]
    for galaxy in tqdm_notebook(all_data_scaled['galaxy'].unique(), desc='Galaxy Loop'):
        galaxy_data = all_data_scaled[all_data_scaled['galaxy']==galaxy][['galactic year',col]]
        if col == 'galactic year':
            galaxy_dict[galaxy]['index'] = galaxy_data.index
            galaxy_dict[galaxy][col]  = list(all_data_scaled[all_data_scaled['galaxy']==galaxy]['galactic year'])
            
            continue

        clean_df = galaxy_data.dropna()
        if not clean_df.empty:
            model = RandomForestRegressor(n_estimators=50)
            model.fit(np.array(clean_df['galactic year']).reshape(-1,1),clean_df[col])
            preds = model.predict(np.array(galaxy_data['galactic year']).reshape(-1,1))

            for i, row in enumerate(galaxy_data.iterrows()):
                if np.isnan(row[1][1]):
                    galaxy_data[col][row[0]] = preds[i]
            
            galaxy_dict[galaxy][col]  = list(galaxy_data[col])
        else:
            print("Empty DataFrame Found")
            print("Column: ",col)
            print("Galaxy Name: ",galaxy)

#%%
all_labels = labels.copy()
all_labels = all_labels.append(pd.Series([np.nan] * len(test_df),index=range(3865,4755,1)))
#%%
for galaxy in galaxy_dict.keys():
    df = pd.DataFrame(galaxy_dict[galaxy],index=galaxy_dict[galaxy]['index']).drop('index',axis=1)
    df['y']  = all_labels[df.index]
    df.to_csv("clean_data/{}.csv".format(galaxy))



# %%
