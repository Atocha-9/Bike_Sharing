import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#%matplotlib inline
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 999

df = pd.read_csv('hour.csv')
df=df.rename(columns={'weathersit':'weather',
                      'yr':'year',
                      'mnth':'month',
                      'hr':'hour',
                      'hum':'humidity',
                      'cnt':'count'})
df = df.drop(columns=['instant','dteday','year'])

# Change int columns to category
cols = ['season', 'month','hour','holiday','weekday', 'workingday','weather']
for col in cols:
  df[col]= df[col].astype('category')
df ['count']=np.log(df['count'])
#OneHotEncoding
df_oh = df
def one_hot_encoding(data,column):
  data=pd.concat([data,pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
  data=data.drop([column],axis=1)
  return data

cols = ['season', 'month','hour','holiday','weekday', 'workingday','weather']
for col in cols:
  df_oh= one_hot_encoding(df_oh,col)

X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered' ,'count'], axis=1)
Y  = df_oh[['count']]


#Decision Trees
from sklearn.tree import DecisionTreeRegressor
bike_x_train = X.sample(frac=0.8, random_state=25)
bike_x_test = X.drop(bike_x_train.index)

# Split the targets into training/testing sets
bike_y_train = Y.sample(frac=0.8,random_state=25)
bike_y_test = Y.drop(bike_y_train.index)

print("tree regression configuration")
tree_reg

feature_name = bike_x_train.columns
print(bike_x_test.columns)
target_name = bike_y_train.columns
print(target_name)

from sklearn import tree
import graphviz

# dot is a graph description language
dot = tree.export_graphviz(tree_reg, out_file=None, 
                           feature_names=feature_name,
                           class_names=target_name,
                           filled=True, rounded=True,  
                           special_characters=True) 

# we create a graph from dot source using graphviz.Source
graph = graphviz.Source(dot) 
graph

#How good is our decision tree
from sklearn.metrics import mean_squared_error, r2_score

bike_y_pred = tree_reg.predict(bike_x_test)

# The coefficients
#print('Coefficients: \n', tree_reg.coef_) #Internal value of the parameter. In this case the "m" of the equation y = m*x + b
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(bike_y_test, bike_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(bike_y_test, bike_y_pred))


#make prediction
n=27
pred =  tree_reg.predict(bike_x_test[n:n+1])
print("Prediction \n ",pred)
print("Real value \n",bike_y_test[n:n+1])

#Return to original values of count before log transformation
prediction = np.exp(pred)
print(prediction)
Real = np.exp(bike_y_test[n:n+1])
print('Real: \n', Real)