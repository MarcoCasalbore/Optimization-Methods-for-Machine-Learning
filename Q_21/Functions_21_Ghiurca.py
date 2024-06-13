# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:26:55 2023

@author: Bianca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import time

N = 50 
seed = 87057
ro = 1e-5 
sigma = 1
k = 5 

def lettura_file(file):
    data_frame = pd.read_csv(file, header = None)
    features = []
    P = data_frame.shape[0] 
    n = len(data_frame.columns) 
    for j in range(0, n-1):
        feat = data_frame[data_frame.columns[j]].values.reshape((P,1))
        features.append(feat)
    Y = data_frame[data_frame.columns[n-1]].values
    X = np.concatenate(features, axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1836553)
    return  X_train, X_test, Y_train, Y_test

def var_initialization(seed, N):
    np.random.seed(seed)
    W = np.random.randn(N, 2)
    v = np.random.randn(1, N) 
    b = np.random.randn(N, 1)
    return W,v,b

def prediction(v,W,b,x,N,sigma):
    prod_bias = np.dot(W, x) + b
    activation = (np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1)
    predicted_y = np.dot(v, activation)
    return predicted_y

def error(v,W,b,dataset,ro,N,sigma):
    x = dataset[0] 
    y = dataset[1]
    P = x.shape[1] 
    prod_bias = np.dot(W,x) + b
    activation = (np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1) 
    y_pred= np.dot(v, activation)
    error = (1/(2*P))*np.sum((y_pred-y)**2) + 0.5*ro*np.linalg.norm(v)**2  
    return float(error)

def minimization(W,b,dataset, ro,N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[1]
    prod = np.dot(W,x)
    prod_bias = prod + b
    activation = (np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1)
    shape = (np.dot(activation, activation.T)).shape[0]
    Q_matrix = (1/P)*np.dot(activation, activation.T) + ro*np.eye(shape)
    c = (1/P)*(np.dot(y, activation.T)).T  
    x_star, residuals, rank, eigenvectors_ = np.linalg.lstsq(Q_matrix, c, rcond = None)
    return  x_star, residuals, rank, eigenvectors_ 

# parametri_multistart = {"seed": np.arange(0, 100001)}

def multistart(dataset, param):  
    x = dataset[0]
    y = dataset[1]
    minimo = float('inf') 
    kf = KFold(n_splits = k, random_state = 1836553, shuffle = True) 
    for i in range(len(param["seed"])):
        validation_error = []
        seed = param["seed"][i]  
        W,v,b = var_initialization(seed, N)
        for train_index, test_index in kf.split(x.T): 
            X_train, X_validation = x[:,train_index], x[:,test_index]
            Y_train, Y_validation = y[train_index], y[test_index] 
            data_train = (X_train, Y_train)
            data_val = (X_validation, Y_validation)
            result = minimization(W,b,data_train, ro,N, sigma) 
            v_opt = result[0]  
            error_val = error(v_opt, W, b, data_val, ro,N,sigma) 
            validation_error = np.append(validation_error, error_val)
        mean_error = np.mean(validation_error) 
        if mean_error < minimo:    
            minimo = mean_error 
            seed_opt = param["seed"][i]   
                     
    return minimo, seed_opt  

def plotting(v,W,b, N, sigma, title):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    x1 = np.linspace(-2, 2, 50) 
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2) #costruzione della griglia 
    X = np.concatenate((X1.reshape(2500,1), X2.reshape(2500,1)), axis=1)
    Y_plot = prediction(v, W, b,X.T,N,sigma).reshape(50,50)
    ax.plot_surface(X1, X2, Y_plot, rstride=1, cstride=1, cmap='plasma', edgecolor='None')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()
