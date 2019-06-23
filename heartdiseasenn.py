#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 08:28:54 2019

@author: victor
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def getData(data_set_x):
    pre_data_set_y = data_set_x[:,-1]
    data_set_y = np.zeros((pre_data_set_y.shape[0], 1))
    for i in range(len(pre_data_set_y)):
        if pre_data_set_y[i] == 0:
            data_set_y[i] = np.array([0])
        else:
            data_set_y[i] = np.array([1])
    data_set_x = data_set_x[:,2:-1]
    medians = np.zeros(data_set_x[0].shape)
    
    for i in data_set_x:
        for j in range(len(i)):
            if not np.isnan(i[j]):
                medians[j] += i[j]
    medians /= data_set_x.shape[0]
    for i in range(len(data_set_x)):
        j = np.isnan(data_set_x[i])
        data_set_x[i][j] = medians[j]
        
    data_set_x = data_set_x.T
    data_set_y = data_set_y.T
    return data_set_x, data_set_y

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    
    s = 1 / (1 + np.exp(-z))
    
    
    return s



def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    
    w = np.zeros((dim, 1))
    b = 0
    
    return w, b



def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    
    A = sigmoid(np.dot(w.T, X) + b)                                    # compute activation
    cost = np.sum(Y * np.log(A) + (1-Y) * np.log(1 - A)) / -m                                 # compute cost
    
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A - Y) / m
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        
        grads, cost = propagate(w, b, X, Y)
        
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and not np.isnan(cost):
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        
        Y_prediction = np.round(A)
        
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction



def model(X_train, Y_train, X_train2, Y_train2, X_train3, Y_train3, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_train2 -- training set 2 represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train2 -- training labels 2 represented by a numpy array (vector) of shape (1, m_train)
    X_train3 -- training set 3 represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train3 -- training labels 3 represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train2, Y_train2, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train3, Y_train3, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_test = predict(w, b, X_test)

    

    # Print train/test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "acc":100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}
    
    return d

p = ["processed.hungarian.data.csv", "processed.va.data.csv", "processed.cleveland.data.csv", "processed.switzerland.data.csv"]

'''
totalAccs = []
for i in p:
    totalAcc = 0
    for j in p:
        if i != j:
            train_set_x = np.genfromtxt(i, delimiter=',')
            test_set_x = np.genfromtxt(j, delimiter=',')
            print(train_set_x.shape)
            print(test_set_x.shape)
            d = model(*getData(train_set_x), *getData(test_set_x), num_iterations = 40000, learning_rate = 0.05, print_cost = False)
            totalAcc += d["acc"]
    totalAccs.append(totalAcc)
print(totalAccs)
[171.30568300732511, 222.0034596789694, 215.11888169902105, 228.84717451336968]
'''
for j in p:
    train_set_x = np.genfromtxt("processed.switzerland.data.csv", delimiter=',')
    train_set_x2 = np.genfromtxt("processed.va.data.csv", delimiter=',')
    train_set_x3 = np.genfromtxt("processed.cleveland.data.csv", delimiter=',')
    test_set_x = np.genfromtxt(j, delimiter=',')
    print("Testing " + " ".join(j.split(".")[1:-1])+": ", end = "")
    d = model(*getData(train_set_x), *getData(train_set_x2), *getData(train_set_x3), *getData(test_set_x), num_iterations = 400000, learning_rate = 0.05, print_cost = False)
