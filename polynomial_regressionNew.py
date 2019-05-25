# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import datetime as dt1
import calendar
import numpy as np
calendar.setfirstweekday(6)


# Importing the dataset
dataset = pd.read_csv('NewAgen1.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

y = np.log(y)

#df=np.array(X)
#X[:,1]=[(dt.datetime.strptime(str(d),'%m/%d/%Y').strftime('%A')) for d in X[:,1]]

#X = np.c_[X,[(dt.datetime.strptime(str(d),'%m/%d/%Y').strftime('%A')) for d in X[:,0]]]

#X = np.c_[X,[(dt.datetime.strptime(str(d),'%m/%d/%Y').strftime('%A')) for d in X]]

#X = X[:, [1]]

def get_week_of_month(xdate):
    
    #strings = time.strftime("%Y,%m,%d")
    agendate = dt1.datetime.strptime(str(xdate),"%m/%d/%Y").date()
    datearr = str(agendate).split('-')
    splitdate = [ int(x) for x in datearr ]
    
    x = np.array(calendar.monthcalendar(splitdate[0], splitdate[1]))
    week_of_month = np.where(x==splitdate[2])[0][0] + 1
    return(week_of_month)

    
X = np.c_[X,[(dt.datetime.strptime(str(d),'%m/%d/%Y').strftime('%A')) for d in X]]

X = np.c_[X,[get_week_of_month(str(d)) for d in X[:,0]]]

X = X[:,[1,2]]


#Categories data of WeekDays
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#Categories data of Weeks of Month
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 7] = labelencoder.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [7])
X = onehotencoder.fit_transform(X).toarray()


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#building optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((181,1)).astype(int), values = X, axis = 1)
#X_opt = X[:,[0,1,2,3,4,5,6,7,8,9]]
youFirts_ols = sm.OLS(endog = y, exog= X_poly).fit()
youFirts_ols.summary()



