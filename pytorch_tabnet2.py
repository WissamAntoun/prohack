#%%
from pytorch_tabnet import tab_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%%
df = pd.read_csv("data/train.csv")
df = df.fillna(-1)
#%%
target = 'y'
Non_data_col = ['galaxy','y','galactic year']
#Choose all predictors except target & IDcols
predictors = [x for x in df.columns if x not in Non_data_col]
#%%
df2 = df[predictors]

x_train , x_test, y_train, y_test = train_test_split(df2,df[target],test_size=0.2,random_state=42)

print(x_train)
print(y_train)

print(x_test)
print(y_test)

np_x_train = np.array(x_train)
np_y_train = np.array(y_train)
np_x_test = np.array(x_test)
np_y_test = np.array(y_test)

np_x_train = np_x_train
np_y_train = np_y_train.reshape(-1,1) 
np_x_test = np_x_test 
np_y_test = np_y_test.reshape(-1,1)

print(np_x_train.shape)
print(np_y_train.shape)
print(np_x_test.shape)
print(np_y_test.shape)


#%%
reg = tab_model.TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, epsilon=1e-15,  momentum=0.02,
                 lambda_sparse=1e-3, seed=0, lr=2e-2,verbose=1)
reg.fit(np_x_train, np_y_train , np_x_test, np_y_test)

preds = reg.predict(np_x_test)
print(mean_squared_error(np_y_test,preds,squared=False))


# %%
from sklearn.base import RegressorMixin

X_valid,y_valid = np_x_test, np_y_test
class sklearntabnetregressor(RegressorMixin):
  def __init__(self,n_d=8, n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2,  momentum=0.02,
                 lambda_sparse=1e-3, seed=2, lr=2e-2):
    self.n_d = n_d
    self.n_steps = n_steps
    self.gamma = gamma
    self.n_independent = n_independent
    self.n_shared = n_shared
    self.momentum = momentum
    self.lambda_sparse = lambda_sparse
    self.lr = lr
    self.seed = seed
    self.tabnet = tab_model.TabNetRegressor(n_d = self.n_d, n_a = self.n_d, n_steps = self.n_steps, gamma = self.gamma,
                 n_independent = self.n_independent, n_shared = self.n_shared,  momentum = self.momentum,
                 lambda_sparse = self.lambda_sparse, seed = self.seed, lr = self.lr,verbose=False)

  def fit(self, X, y):
    self.tabnet.fit(X, y, X_valid,y_valid)
    return self
  
  def predict(self, X):    
    return self.tabnet.predict(X)

  def get_params(self, deep=True):
    return {
        "n_d" : self.n_d, 
        "n_steps" : self.n_steps, 
        "gamma" : self.gamma,
        "n_independent" : self.n_independent, 
        "n_shared" : self.n_shared, 
        "momentum" : self.momentum, 
        "lambda_sparse" : self.lambda_sparse, 
        "lr" : self.lr, 
        "seed" : self.seed
    }

  def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self


# %%
from sklearn.model_selection import GridSearchCV


parameters = {
    "n_d" : range(8,65,16), 
    "n_steps" : range(3,11,2), 
    "gamma" : [i/10.0 for i in range(10,21,3)],
    "n_independent" : range(1,6,2), 
    "n_shared" : range(1,6,2), 
    "momentum" : [0.02,0.2], 
    "lambda_sparse" : [1e-3], 
    "lr" : [0.02],
    "seed" : [0]
}

parameters_test = {
    "n_d" : range(8,65,300), 
    "n_steps" : range(3,11,300), 
    "gamma" : [i/10.0 for i in range(10,21,300)],
    "n_independent" : range(1,6,200), 
    "n_shared" : range(1,6,200), 
    "momentum" : [0.02,0.2], 
    "lambda_sparse" : [1e-3], 
    "lr" : [0.02],
    "seed" : [0]
}

tabnet = sklearntabnetregressor()
tabnetgrid = GridSearchCV(tabnet,parameters,scoring='neg_root_mean_squared_error',cv=5,verbose=True,n_jobs=-1)


# %%

total=1
for k ,v in parameters.items():
  print(k,len(v))
  total*=len(v)

print(total)
#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%
tabnetgrid.fit(np_x_train, np_y_train)

print(tabnetgrid.best_estimator_)
print(tabnetgrid.best_params_)
print(tabnetgrid.best_score_)


# %%
