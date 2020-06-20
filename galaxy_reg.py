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
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import GradientBoostingRegressor
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

#%% Creating Dev-Set
X_train_with_labels = pd.concat([X_train,labels],axis=1,sort=False).sort_values(['galaxy','galactic year'])

#%%
X_Dev_with_labels = pd.DataFrame([])
X_mini_train_with_labels = pd.DataFrame([])

train_with_test_galaxies_df = pd.DataFrame([])
len_valid_data = []

for galaxy in tqdm_notebook(X_test['galaxy'].unique(), desc='Galaxy Loop'):
    galaxy_data = X_train_with_labels[X_train_with_labels['galaxy']==galaxy]
    recent_data = galaxy_data[galaxy_data['galactic year'] >= 1010025]
    old_data = galaxy_data[galaxy_data['galactic year'] < 1010025]
    X_mini_train_with_labels = pd.concat([X_mini_train_with_labels,old_data])    
    len_valid_data.append(len(recent_data))
    if len(recent_data) > 3:
        X_Dev_with_labels = pd.concat([X_Dev_with_labels,recent_data.iloc[-2:,:]])
        X_mini_train_with_labels = pd.concat([X_mini_train_with_labels,recent_data.iloc[:-2,:]])
    else:
        X_mini_train_with_labels = pd.concat([X_mini_train_with_labels,recent_data])
    
    train_with_test_galaxies_df = pd.concat([train_with_test_galaxies_df,galaxy_data])
    


#%%

plt.hist(len_valid_data)
plt.show()

#%%

predictions_df = pd.DataFrame([])
for galaxy in tqdm_notebook(X_Dev_with_labels['galaxy'].unique(), desc='Galaxy Loop'):
    galaxy_data_train = X_mini_train_with_labels[X_mini_train_with_labels['galaxy']==galaxy]
    galaxy_data_dev = X_Dev_with_labels[X_Dev_with_labels['galaxy']==galaxy]
    
    columns_to_remove = []
    for col in galaxy_data_dev.columns:
        if galaxy_data_dev[col].isnull().values.any():
            columns_to_remove.append(col)

    galaxy_data_train.dropna(axis=1,how='all',thresh=int(len(galaxy_data)*0.4),inplace=True)
    galaxy_data_train.bfill(axis=0,inplace=True)

    columns_to_remove.extend(list(set(galaxy_data_dev.columns) - set(galaxy_data_train.columns))) 
    #corr = galaxy_data_train.corr()
    galaxy_data_dev.drop(columns_to_remove,axis=1,inplace=True,errors='ignore')
    galaxy_data_train.drop(columns_to_remove,axis=1,inplace=True,errors='ignore')
    orig_year = galaxy_data_train['galactic year'].iloc[0]
    galaxy_data_train['galactic year'] = ((galaxy_data_train['galactic year'] - orig_year +100)/1000).astype(int)
    galaxy_data_dev['galactic year'] = ((galaxy_data_dev['galactic year'] - orig_year +100)/1000).astype(int)

    galaxy_data_train_year_index = galaxy_data_train.set_index('galactic year')
    galaxy_data_train_dev_index = galaxy_data_dev.set_index('galactic year')

    data_to_fit = galaxy_data_train_year_index.drop(['galaxy'],axis=1).copy()
    for iteration in range(len(galaxy_data_train_dev_index)):
        model = VAR(endog=np.asarray(data_to_fit))
        model_fit = model.fit()
        prediction = model_fit.forecast(model_fit.y, steps=1)
        data_to_fit = data_to_fit.append(galaxy_data_train_dev_index.drop(['galaxy'],axis=1).iloc[iteration])
        data_to_fit['y'].iloc[-1]=prediction[0][-1]
    
    temp_preds = galaxy_data_dev[['y']]
    temp_preds['y'] = list(data_to_fit['y'][-len(galaxy_data_dev):].abs())
    predictions_df = pd.concat([predictions_df,temp_preds])   
    
    

#%%

mean_squared_error(X_Dev_with_labels['y'],predictions_df)

#%%
predictions_df = pd.DataFrame([])
for galaxy in tqdm_notebook(X_test['galaxy'].unique(), desc='Galaxy Loop'):
    galaxy_data = X_train_with_labels[X_train_with_labels['galaxy']==galaxy]
    galaxy_data_to_predict = X_test[X_test['galaxy']==galaxy]
    galaxy_data_to_predict['y'] = np.nan

    all_galaxy_data = pd.concat([galaxy_data,galaxy_data_to_predict]).sort_values('galactic year')

    index_of_first_na = all_galaxy_data.index.get_loc(all_galaxy_data['y'].isna().idxmax())

    data_to_fit = all_galaxy_data.drop(['galaxy'],axis=1).iloc[:index_of_first_na,:]
    for iteration in range(0,len(all_galaxy_data)-index_of_first_na):
        if np.isnan(all_galaxy_data['y'].iloc[index_of_first_na+iteration]):
            data_to_fit_cleaned = data_to_fit.dropna(axis=1,how='all',thresh=int(len(galaxy_data)*0.4))

            # data_to_fit_cleaned.fillna(0,inplace=True,axis=0)
            imputer = KNNImputer(n_neighbors=3)
            data_to_fit_cleaned = pd.DataFrame(imputer.fit_transform(data_to_fit_cleaned),columns=data_to_fit_cleaned.columns)

            orig_year = data_to_fit_cleaned['galactic year'].iloc[0]
            data_to_fit_cleaned['galactic year'] = ((data_to_fit_cleaned['galactic year'] - orig_year +100)/1000).astype(int)

            data_to_fit_cleaned_year_index = data_to_fit_cleaned.set_index('galactic year')
            model = VAR(endog=np.asarray(data_to_fit_cleaned_year_index))
            model_fit = model.fit()
            prediction = model_fit.forecast(model_fit.y, steps=1)
            prediction_series = pd.DataFrame(prediction,columns=data_to_fit_cleaned_year_index.columns)
            if np.abs(prediction_series['y'].values[0])>1 or prediction_series['y'].values[0] < 0 or prediction_series['y'].values[0] > 2 * data_to_fit_cleaned['y'].iloc[-1]:
                print(prediction_series['y'].values[0])
                trend_line = np.poly1d(np.polyfit(data_to_fit_cleaned['galactic year'],data_to_fit_cleaned['y'],deg=4))
                x = int((all_galaxy_data.iloc[index_of_first_na+iteration,:]['galactic year'] - orig_year +100)/1000)   
                prediction_series['y'].values[0] = trend_line(x) 
                print(prediction_series['y'].values[0])
                print(iteration)
                print(galaxy)
                print(data_to_fit_cleaned['y'])         
        
            data_to_fit = data_to_fit.append(all_galaxy_data.drop(['galaxy'],axis=1).iloc[index_of_first_na+iteration,:])
            data_to_fit['y'].iloc[-1]=prediction_series['y'].values[0]

            if np.isnan(all_galaxy_data.iloc[index_of_first_na+iteration,:]['existence expectancy index']):
                data_to_fit['existence expectancy index'].iloc[-1] = prediction_series['existence expectancy index'].values[0]
        else:
            data_to_fit = data_to_fit.append(all_galaxy_data.drop(['galaxy'],axis=1).iloc[index_of_first_na+iteration,:])
        
        
    
    galaxy_data_to_predict['existence expectancy index'] = data_to_fit['existence expectancy index'][galaxy_data_to_predict['existence expectancy index'].index]
    galaxy_data_to_predict['y'] = data_to_fit['y'][galaxy_data_to_predict['y'].index]
    predictions_df = pd.concat([predictions_df,galaxy_data_to_predict[['y','existence expectancy index']].abs()])  
    
        


#%%

predictions_df = predictions_df.sort_index()
predictions_df.to_csv('submission_task10.csv',index=False)












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
    galaxy_data = X_train_with_labels[X_train_with_labels['galaxy']==galaxy]
    galaxy_data_to_predict = X_test[X_test['galaxy']==galaxy][['galaxy','galactic year']]

    model = GradientBoostingRegressor(n_estimators=300)
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
predicted_data.to_csv('submission_task12.csv',index=False)


# %%
