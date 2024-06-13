# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:17:44 2023

@author: Bianca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from scipy.optimize import minimize
import time

N = 50
ro = 1e-5
sigma = 0.9
seed = 87226
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

def var_initialization(seed,x, N):
    np.random.seed(seed)
    v = np.random.randn(N, 1)
    indices = np.random.choice(len(x), N, replace=False) 
    selected_centres = x[indices]
    c = selected_centres.reshape((2,N))
    return v,c

def prediction(v,c,x,sigma): 
    P = x.shape[0]
    x = x.reshape(P, 2, 1)  
    activation = np.exp(-(np.sum(np.square(x-c), axis = 1))/sigma**2)
    y_pred = np.dot(activation, v)
    return y_pred

def error(v,c,dataset,ro,N,sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    v = v.reshape(N, 1)
    y_pred = prediction(v,c,x,sigma)
    error = (1/(2*P))*np.sum((y_pred-y.reshape(-1,1))**2) + 0.5*ro*np.linalg.norm(v)**2
    return float(error)

def minimization(c,dataset, ro,N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    x = x.reshape(P,2,1)
    activation = np.exp(-(np.sum(np.square(x-c), axis = 1))/sigma**2)
    shape = (np.dot(activation.T, activation)).shape[0]
    Q_matrix = (1/P)*np.dot(activation.T, activation) + ro*np.eye(shape)
    c_vector = (1/P)*(np.dot(y, activation)).T  
    x_star, residuals, rank, eigenvectors_ = np.linalg.lstsq(Q_matrix, c_vector, rcond = None)
    
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
        v,c = var_initialization(seed,x, N)
        for train_index, test_index in kf.split(x): 
            X_train, X_validation = x[train_index,:], x[test_index,:]
            Y_train, Y_validation = y[train_index], y[test_index]
            data_train = (X_train, Y_train)
            data_val = (X_validation, Y_validation)
            result = minimization(c,data_train, ro,N, sigma) 
            v_opt = result[0] 
            error_val = error(v_opt, c, data_val, ro,N,sigma)
            validation_error = np.append(validation_error, error_val) 
        mean_error = np.mean(validation_error) 
        if mean_error < minimo:    
            minimo = mean_error 
            seed_opt = param["seed"][i]   
                      
    return minimo, seed_opt 

def plotting(c,v, N, sigma, title):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d') 
    x1 = np.linspace(-2, 2, 50) 
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2) #costruzione della griglia 
    X = np.concatenate((X1.reshape(2500,1), X2.reshape(2500,1)), axis=1)
    Y_plot = prediction(v,c,X,sigma).reshape(50,50)
    ax.plot_surface(X1, X2, Y_plot, rstride=1, cstride=1, cmap='plasma', edgecolor='None')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()




