# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Known data input to the model
X_in = [280, 750, 1020, 1400, 1700, 2300, 2900]
Y_out = [7.0, 22.4, 29.0, 40.9, 52.3, 70.5, 85.1]
X_query = [500, 1500, 2000]

# Creating arrays
X = np.array(X_in)
Y = np.array(Y_out)
X_find = np.array(X_query)
X_max = np.max(X)
Y_max = np.max(Y)

# Normalizing with respect to the maxima
X = X/X_max
Y = Y/Y_max
X_find = X_find/X_max
m = X.size

# Initializing parameters
iterations = 1000
alpha = 0.1

# Generating input (feature) vector
X_train = np.ones([2, m])
X_train[1, :] = X 
Y_train = Y

# Initializing theta randomly
def init_theta():
    theta = np.random.rand(1, 2)
    return theta

def forward_prop(theta, X):
    h = theta.dot(X)
    return h

def error_cal(h, Y):
    cost = (1/2*m)*np.sum(np.multiply((h-Y),(h-Y)))
    return cost

def backward_prop(h, Y, X):
    del_j = h - Y
    del_J = del_j.dot(X.T)
    return del_J

def update_theta(theta, del_J, alpha):
    theta = theta - (1/m)*alpha*del_J
    return theta

def grad_descent(X, Y, alpha, iterations):
    theta = init_theta()
    print("Initial theta: ",theta)
    error = np.zeros(iterations)
    
    for i in range(iterations):
        h = forward_prop(theta, X)
        error[i] = error_cal(h, Y)
        del_J = backward_prop(h, Y, X)
        theta = update_theta(theta, del_J, alpha)
        if (i%50 == 0):
            print("Iterations: ", i)
            print("Cost: ", error_cal(h, Y))
    return theta, error


def Result(theta, X_find):
    X_result = np.ones([2, X_find.size])
    X_result[1,:] = X_find
    Y_result = theta.dot(X_result)
    return Y_result

theta, error = grad_descent(X_train, Y_train, alpha, iterations)
print("Theta after 1000 iterations: ",theta)
Y_result = Result(theta, X_find)
print("Price: ", Y_result*Y_max)


# plotting the iteration vs loss curve
plt.plot(range(1, iterations+1), error)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Iteration vs Loss curve")
plt.show()



