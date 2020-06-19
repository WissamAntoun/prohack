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
from sklearn.impute import KNNImputer
#%%
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 4
#%%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
labels = train_df['y']
eei_test = test_df["existence expectancy index"]
train_df = train_df.drop('y',axis =1 )

#%%
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True)
#%%
# all_data["galaxy"] = all_data["galaxy"].astype('category')
# all_data["galaxy"] = all_data["galaxy"].cat.codes

#%%

all_data_without_year_name = all_data.drop(['galactic year','galaxy'],axis=1)
#%%
scaler = RobustScaler().fit(all_data_without_year_name)
all_data_without_year_name_scaled = all_data_without_year_name#scaler.transform(all_data_without_year_name)
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

for galaxy in tqdm_notebook(X_test['galaxy'].unique(), desc='Galaxy Loop'):
    galaxy_data = all_data_scaled[all_data_scaled['galaxy']==galaxy]
    galaxy_labels = labels.iloc[galaxy_data.index[galaxy_data.index<len(train_df)]]
    print(len(galaxy_data.columns))
    galaxy_data.sort_values('galactic year',inplace=True)
    galaxy_data_with_labels = pd.concat([galaxy_data,galaxy_labels],axis=1,sort=False).sort_values('galactic year',inplace=False)
    #galaxy_data.dropna(how='all',axis=1,thresh=int(len(galaxy_data)*0.35),inplace=True)
    print(len(galaxy_data.columns))
    #galaxy_data.fillna()
    imputer = KNNImputer(n_neighbors=2)
    galaxy_data_imputed = pd.DataFrame(imputer.fit_transform(galaxy_data))
    break

# %%
#galaxy_data['galactic year'].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0]) rolling diff for year
#1010025

all_data_scaled_sorted = pd.concat([all_data_scaled,labels],axis=1,sort=False).sort_values(['galaxy','galactic year'])[['galaxy','galactic year','y']]

#%%
X_train_with_labels = pd.concat([X_train,labels],axis=1,sort=False).sort_values(['galaxy','galactic year'])[['galaxy','galactic year','y']]
# %%

predicted_data = pd.DataFrame([])
for galaxy in tqdm_notebook(X_test['galaxy'].unique(), desc='Galaxy Loop'):
    galaxy_data = X_train_with_labels[X_train_with_labels['galaxy']==galaxy][['galaxy','galactic year','y']]
    galaxy_data_to_predict = X_test[X_test['galaxy']==galaxy][['galaxy','galactic year']]

    model = RandomForestRegressor(n_estimators=50)
    model.fit(np.array(galaxy_data['galactic year']).reshape(-1,1),galaxy_data['y'])
    galaxy_data_to_predict['y'] = model.predict(np.array(galaxy_data_to_predict['galactic year']).reshape(-1,1))
    # trend_line = np.poly1d(np.polyfit(galaxy_data['galactic year'],galaxy_data['y'],deg=4))    
    # galaxy_data_to_predict['y'] = galaxy_data_to_predict['galactic year'].apply(lambda x: trend_line(x))
    

    eei_test = test_df[test_df['galaxy']==galaxy][['galactic year','existence expectancy index']]
    eei_test.fillna(method='ffill',axis=0,inplace=True)
    galaxy_data_to_predict['existence expectancy index'] = eei_test['existence expectancy index'].values

    predicted_data = pd.concat([predicted_data,galaxy_data_to_predict[['y','existence expectancy index']]])

    # data_to_plot = galaxy_data[['galactic year','y']]
    # data_to_plot = pd.concat([data_to_plot,galaxy_data_to_predict[['galactic year','y']]]).sort_values('galactic year')
    
    # plt.Line2D(xdata=data_to_plot['galactic year'],ydata=data_to_plot['y'])
    # plt.show()
    
    


#%%

data_to_plot.set_index('galactic year').plot()
#%%
plt.Line2D(xdata=data_to_plot['galactic year'],ydata=data_to_plot['y'])
plt.show()
# %%

test_df['y'] = predicted_data[0].values
sub_df = test_df[['y','existence expectancy index']]
#%%
predicted_data.sort_index(inplace=True)
predicted_data.to_csv('submission_task6.csv',index=False)


# %%
