# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NewAgen.csv')
"""X = dataset.iloc[:, 1:2].values"""
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values
y = np.log(y)





dataset1 = pd.read_csv('NewAgen.csv')
z = dataset1.iloc[:, 0:9].values

"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 4] = labelencoder.fit_transform(X[:, 4])
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()"""


#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

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

# Visualising the Linear Regression results
"""plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""

# Visualising the Polynomial Regression results
"""plt.scatter(X, y, colo////r = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
"""X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""

# Predicting a new result with Linear Regression
"""lin_reg.predict(6.5)"""

z_poly = poly_reg.fit_transform(z)
# Predicting a new result with Polynomial Regression
Predi = lin_reg_2.predict(z_poly)


#plot validation between PRED and Difference
pred = dataset.iloc[:,10].values
diff =  dataset.iloc[:,11].values
plt.scatter(pred,diff)
plt.title('Relationship between Prediction and Difference')
plt.show()



