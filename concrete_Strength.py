# deep learning source 
# Concrete strength data (from UCI Data Repository) 
# ................................................. 
# .................................................
import pandas as pd # dataframes
import numpy as np # mathematical formulae
# import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
# set path according to your directory 
pth = "C:\\Users\\Travis\\Desktop\\Concrete_Data.csv" 
clmns = ['Mix','Slag','Ash','Water','Superplasticizer','Coarse','Fine','Age','Strength']
x = ['Mix','Slag','Ash','Water','Superplasticizer','Coarse','Fine','Age']
y = ['Strength']
df = pd.read_csv(pth, delimiter = ',', skiprows=1, names=clmns) # load dataframe
print(df.head()) 
# First column
list(df.columns.values) # get names
 
yy = df[y] # subset by y
xx = df[x] # subset by x
 
 
################ 
################
 
X_train, X_test, y_train, y_test = train_test_split(xx, yy) #use default split ratio
 
 
(X_train).head() # top records
X_train.count() # number of records
X_test.count() #
#### Playing with iloc() function
X_train.iloc[0:3] # first 4 rows
X_train.iloc[:1] # second row
X_train.iloc[:,0:2] # first two columns
 
f=xx.iloc[:,0] # first column
sec = xx.iloc[:,1] # second column
third = xx.iloc[:,2] # third column
fourth = xx.iloc[:,3] # fourth
fifth = xx.iloc[:,4] # fifth
six = xx.iloc[:,5] # sixth column
y_train.head() # view top y-values
y_train.count()
print(six)
#### ############
###### ##########
#### Preprocessing - standarize input (subtract means and divided by std)
 
 
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 
####################
######################
### Multi-perceptron network 
 
### model's container
mlp = MLPRegressor(hidden_layer_sizes=(10,10), # two hidden layers of size 10
                    activation='logistic', 
                    solver='sgd', 
                    learning_rate = 'adaptive',
                    shuffle = True,
                    max_iter=1000,
                    learning_rate_init=0.01,
                    early_stopping = True,
                    validation_fraction=0.1,
                    alpha=0.01)
# import time
# start_time = int(time.time() * 1000)
 
mlp = mlp.fit(X_train,y_train) # Train
pred = mlp.predict(X_test)  # Predict
scr = mlp.score(X_test, y_test) #R-squared
print(scr)
 
####  # # # #
#   GLM
# # # # # # #
from sklearn import linear_model
 
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
pred1 = reg.predict(X_test)
print(reg.coef_)
# bivariate plots
plt.scatter(f, yy)
plt.scatter(sec, yy)
plt.scatter(third, yy)
plt.scatter(fourth, yy)
plt.scatter(fifth, yy)
plt.scatter(six, yy)
 
print(r2_score(y_test, pred1)) # r-squared = 0.5716227972362045
 
####  # # # # # # # # 
# # # GLM + shrinkage
#### # # # # # # # #
# Ridge
reg1 = linear_model.Ridge(alpha=0.5)
reg1.fit(X_train, y_train)
reg1.coef_
reg1.intercept_
pred2 = reg1.predict(X_test)
print(r2_score(y_test ,pred2)) # r-squared = 0.571623161249849 a little smaller
 
# # # # # # # # # # # # #
# # # LASSO
# # # # # # # # # # # # # 
reg2 = linear_model.Lasso(alpha=0.1)
reg2.fit(X_train, y_train)
reg2.coef_
pred3 = reg2.predict(X_test)
print(r2_score(y_test, pred3 )) # 0.5717671364598043 a little bigger