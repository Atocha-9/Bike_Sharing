import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#%matplotlib inline
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 999

df = pd.read_csv('hour.csv')
df.head()

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

# Log transformation to get a normal distribution of data, can be used min-max transformation or standarization
df['count']=np.log(df['count'])

df_oh = df
def one_hot_encoding(data,column):
  data=pd.concat([data,pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
  data=data.drop([column],axis=1)
  return data

cols = ['season', 'month','hour','holiday','weekday', 'workingday','weather']
for col in cols:
  df_oh= one_hot_encoding(df_oh,col)

X = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered', 'count'], axis=1)
Y  = df_oh[['count']]
data = df_oh.drop(columns=['atemp', 'windspeed', 'casual', 'registered'], axis=1)


#To describe nodes
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value

#Tree class
class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    


#Train-split
data2 = data.loc[:, data.columns != 'count']
X = data2.iloc[:, :].values
Y = data.iloc[:, 2].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

#Fit
regressor = DecisionTreeRegressor(min_samples_split=10, max_depth=10)
regressor.fit(X_train,Y_train)
#regressor.print_tree()

#Test the model
Y_pred = regressor.predict(X_test) 
from sklearn.metrics import mean_squared_error , r2_score
np.sqrt(mean_squared_error(Y_test,Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_test, Y_pred))

temperatura = input("Temperature in Celcious: ")
temp=int(temperatura)/41
humidity = input("Percent Humidity: ")
humidity = int(humidity)/100
season = input("season: \n 1.winter \n 2.spring \n 3.summer \n 4.fall \n")
if season == '1':
  season_2=season_3=season_4 =0
  season_1 = 1
elif season =='2':
  season_1=season_3=season_4 =0
  season_2 =1
elif season =='3':
  season_1=season_2=season_4=0
  season_3 =1
elif season =='4':
  season_1=season_2=season_3 =0
  season_4 =1
else: 
  print("invalid value")

month = input("Month: \n 1.Jan \n 2.Feb \n 3.Mar \n 4.Apr \n 5.May \n 6.Jun \n 7.Jul \n 8.Aug \n 9.Sep \n 10.Oct \n 11.Nov \n 12.Dec\n")
month_1=month_2=month_3=month_4=month_5=month_6=month_7=month_8=month_9=month_10=month_11=month_12=0
if month == '1':
  month_1=1
elif month =='2':
  month_2=1
elif month =='3':
  month_3=1
elif month =='4':
  month_4=1
elif month =='5':
  month_5=1
elif month =='6':
  month_6=1
elif month =='7':
  month_7=1
elif month =='8':
  month_8=1
elif month =='9':
  month_9=1
elif month =='10':
  month_10=1
elif month =='11':
  month_11=1
elif month =='12':
  month_12=1
else: 
  print("invalid value")

hour = input("\n hour(0-23):")
hour_0=hour_1=hour_2=hour_3=hour_4=hour_5=hour_6=hour_7=hour_8=hour_9=hour_10=hour_11=hour_12=hour_13=hour_14=hour_15=hour_16=hour_17=hour_18=hour_19=hour_20=hour_21=hour_22=hour_23=0
if hour == '1':
  hour_1=1
elif hour == '0':
  hour_0=1
elif hour == '2':
  hour_2=1
elif hour == '3':
  hour_3=1
elif hour == '4':
  hour_4=1
elif hour == '5':
  hour_5=1
elif hour == '6':
  hour_6=1
elif hour == '7':
  hour_7=1
elif hour == '8':
  hour_8=1
elif hour == '9':
  hour_9=1
elif hour == '10':
  hour_10=1
elif hour == '11':
  hour_11=1
elif hour == '12':
  hour_12=1
elif hour == '13':
  hour_13=1
elif hour == '14':
  hour_14=1
elif hour == '15':
  hour_15=1
elif hour == '16':
  hour_16=1
elif hour == '17':
  hour_17=1
elif hour == '18':
  hour_18=1
elif hour == '19':
  hour_19=1
elif hour == '20':
  hour_20=1
elif hour == '21':
  hour_21=1
elif hour == '22':
  hour_22=1
elif hour == '23':
  hour_23=1
else: 
  print("invalid value")

holiday_1 = input("\n Holiday? \n 1.Yes \n 0.No \n :")
weekday = input("\n weekday? \n 0.Sunday \n 1.Monday \n 2.Tuesday \n 3.Wednesday \n 4.Thursday \n 5.Friday \n 6.Saturday \n :")
weekday_0=weekday_1=weekday_2=weekday_3=weekday_4=weekday_5=weekday_6=0
if weekday == '0':
  weekday_0=1
elif weekday =='1':
  weekday_1=1
elif weekday =='2':
  weekday_2=1
elif weekday =='3':
  weekday_3=1
elif weekday =='4':
  weekday_4=1
elif weekday =='5':
  weekday_5=1
elif weekday =='6':
  weekday_6=1
else: 
  print("invalid value")

workingday_1 = input("\n Workingday? \n 1.Yes \n 0.No \n :")
weather = input("\n Weather: \n 1.Clear, Few clouds, Partly cloudy, Partly cloudy \n 2.Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist \n 3.Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds \n 4.Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog  \n :")
weather_1=weather_2=weather_3=weather_4=0
if weather == '1':
  weather_1=1
elif weather =='2':
  weather_2=1
elif weather =='3':
  weather_3=1
elif weather =='4':
  weather_4=1
else: 
  print("invalid value")
O_test = [temp, humidity, season_2,	season_3,	season_4,	month_2,	month_3,	month_4,	month_5,	month_6,	month_7,	month_8,	month_9,	month_10,	month_11,	month_12,	hour_1,	hour_2,	hour_3,	hour_4,	hour_5,	hour_6,	hour_7,	hour_8,	hour_9,	hour_10,	hour_11,	hour_12,	hour_13,	hour_14,	hour_15,	hour_16,	hour_17,	hour_18,	hour_19,	hour_20,	hour_21,	hour_22,	hour_23,	holiday_1,	weekday_1,	weekday_2,	weekday_3,	weekday_4, weekday_5,	weekday_6,	workingday_1,	weather_2,	weather_3,	weather_4]

O2_test = np.array(O_test)
X_test2=X_test
X_test2[0,:] = O2_test
#X_test2[0,:]
Bike_sharing_users = regressor.predict(X_test2) 
np.exp(Bike_sharing_users[0])
