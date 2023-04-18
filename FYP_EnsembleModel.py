
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
warnings.filterwarnings("ignore")

df=pd.read_csv('FYP_Dataset_D.csv')


df=df[['Price', 'Area','Avg_Price_Area', 'Location Id',
       'No. of Bedrooms', 'New/Resale', 'Gymnasium', 'Lift Available',
       'Car Parking', 'Swimming Pool']]


z = np.abs(stats.zscore(df))
threshold = 3
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


x=df_out[[ 'Area','Avg_Price_Area', 'Location Id',
       'No. of Bedrooms', 'New/Resale', 'Gymnasium', 'Lift Available',
       'Car Parking', 'Swimming Pool']]
y=df_out[['Price']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)


#Linear Regression

lm = LinearRegression()
lm.fit(x_train,y_train)
pickle.dump(lm,open('modelLR.pkl','wb'))
modelLR=pickle.load(open('modelLR.pkl','rb'))

#KNN

from sklearn import neighbors
for K in range(20):
    K = K+1
    knn = neighbors.KNeighborsRegressor(n_neighbors = K)

    knn.fit(x_train, y_train)  
pickle.dump(knn,open('modelKNN.pkl','wb'))
modelKNN=pickle.load(open('modelKNN.pkl','rb'))

#Decision Tree

from sklearn.tree import DecisionTreeRegressor  
dt = DecisionTreeRegressor(random_state = 0)  
dt.fit(x_train, y_train)
pickle.dump(dt,open('modelDT.pkl','wb'))
modelDT=pickle.load(open('modelDT.pkl','rb'))

mean_predictions=[]
temp=0
for i in range(len(x_test)):
    temp = predictions[i]*0.1 + prediction2[i]*0.3 + prediction3[i]*0.6
    mean_predictions.append(temp)
    temp = 0


