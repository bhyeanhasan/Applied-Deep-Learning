# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Known data input to the model
X_train = np.array([280, 750, 1020, 1400, 1700, 2300, 2900]).reshape(-1, 1)
y_train = np.array([7.0, 22.4, 29.0, 40.9, 52.3, 70.5, 85.1])
X_query = np.array([500, 1500, 2000]).reshape(-1, 1)

# Fit the linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Output model parameters
print("Intercept:", lm.intercept_)
print("Coefficient:", lm.coef_)

predictions = lm.predict(X_query)
print("Predictions:", predictions)

