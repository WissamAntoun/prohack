#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
#%%
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

#%%
print(train_df.columns)
# %%
target = 'y'
Non_data_col = ['galaxy','galactic year']
#%%
#Choose all predictors except target & IDcols
predictors = [x for x in train_df.columns if x not in Non_data_col]

# %%
# for col in train_df[predictors].columns:
#     print(col)
#     plt.plot(train_df[col])
#     plt.title(col)
#     plt.savefig('plots/'+col+'_plot.png',dpi=72)
#     plt.close()


# for col in train_df[predictors].columns:
#     print(col)
#     plt.hist(train_df[col])
#     plt.title(col)
#     plt.savefig('plots/' + col + '_hist.png',dpi=72)
#     plt.close()
    

# # %%
# len(train_df['galaxy'].unique())

# # %%
print(set(test_df['galaxy'].unique()).difference((set(train_df['galaxy'].unique()))))

# %%
from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()
int_names = le.fit_transform(train_df['galaxy'])
#%%
# for col in train_df[predictors].columns:
#     print(col)
#     plt.scatter(x= train_df.index,y=train_df[col],c=int_names)
#     plt.title(col)
#     plt.savefig('plots/' + col + '_scatter_groubedby_galaxy.png',dpi=72)
#     plt.close()

#%%
# i=0
# for col in train_df[predictors].columns:
#     print(col)
#     fig, ax = plt.subplots()
#     bp = train_df.groupby('galaxy')[col].plot(ax=ax,title=col)
#     fig.savefig('plots/' + col + '_line_groubedby_galaxy.png',dpi=72)
#     ax.clear()
#     plt.cla()
#     plt.close('all')


#
# labels = train_df['y']

# train_df = train_df.fillna(0)
# test_df = test_df.fillna(0)

# train_df = train_df.drop('y',axis =1 )
#%%

# Non_data_col = ['galactic year','y']
# #%%
# #Choose all predictors except target & IDcols
# predictors = [x for x in train_df.columns if x not in Non_data_col]
#%%
# test_df['y']=1
test_df['y']= np.nan
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True).reset_index()

# %%
#fig, ax = plt.subplots()
t_df = all_data[['galactic year','y','galaxy']].set_index('galactic year').sort_index()
#t_df.groupby('galaxy')['y'].plot(ax=ax)

# %%

pred_df = pd.read_csv("submission_task1.csv")

test_df['y'] = pred_df['y']
# test_df['y']=1
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True).reset_index()

# %%
fig, ax = plt.subplots()
t_df = all_data[['galactic year','y','galaxy']].set_index('galactic year').sort_index()
t_df.groupby('galaxy')['y'].plot(ax=ax)

# %%
from plotly import graph_objs as go


# %%
fig = go.Figure()
for name, group in t_df.groupby('galaxy'):
    trace = go.scatter()
    trace.name = name
    trace.x = group['y']
    fig.add_trace(trace)

# %%
import plotly.express as px

# %%
test_df['y']= np.nan
all_data = pd.concat([train_df,test_df],axis=0,ignore_index=True).sort_values('galactic year').reset_index()
t_df = all_data[['galactic year','y','galaxy']]
fig = px.scatter()(all_data,x='galactic year',y='y',color='galaxy',width=1280, height=720)
fig.show()

# %%
