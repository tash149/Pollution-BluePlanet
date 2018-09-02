import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import sklearn
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
#import os 
#import json





# Importing the dataset
dataset = pd.read_csv('Pollution Data.csv')

# Adding a new col for traffic
dataset['Traffic']=np.zeros(744,dtype=int)

for i in range(0,744):
    if (dataset.iloc[i,9]>200):
        dataset.iloc[i,11] = np.random.randint(low=200, high=500)
    else:
        dataset.iloc[i,11] = np.random.randint(low=20, high=100)
        
# Removing unnecessary data
X = dataset.drop(['Stn Code','State', 'City/Town/Village/Area', 'Agency'], axis=1)
X = dataset[['Sampling Date','Location of Monitoring Station','Type of Location','SO2','NO2','PM 2.5','Traffic','RSPM/PM10']]

# Forming input set and output set
y = X.iloc[:, -1].values
X = X.iloc[:, 1:7].values
y = np.reshape(y,(-1,1))
y = np.asmatrix(y)

        
# Data preprocessing

# Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:6])                                        #can be 2:5
X[:,2:6] = imputer.transform(X[:, 2:6]) 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(y[:,0])   
y[:,0] = imputer.transform(y[:,0])


df = pd.DataFrame(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)                   #Only apply when using PCA
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 13)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_'''




#Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


'''# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)'''



#Predict the Test set Results
y_pred = regressor.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()




#Building optimum model using Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((744,1)).astype(int) , values = X , axis = 1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,12,13,14]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()  #Best set of var by adjusted r sq
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,8,9,10,12,13,14]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,8,10,12,13,14]]
regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()


# Splitting the dataset into the Training set and Test set
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 42)
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, y_opt_train)

'''#Fitting Decision tree
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 42)
regressor_opt = DecisionTreeRegressor(random_state = 0)
regressor_opt.fit(X_opt_train, y_opt_train)'''




#predicting new results
y_opt_pred = regressor_opt.predict(X_opt_test)


# Saving model as pickle file
from sklearn.externals import joblib
import pickle

lin_reg = pickle.dumps(regressor)
joblib.dump(lin_reg, 'model.pkl')


#joblib.dump(regressor, 'model.pkl')
#regr=joblib.load('model.pkl')


    



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_opt_train, y = y_opt_train, cv = 10)
accuracies.mean()
accuracies.std()


