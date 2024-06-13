# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:00:19 2023

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
sigma = 1
seed = 200
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
    b = np.zeros((N, 1))
    return W, v, b

def omega_initialization(W, v, b):
    W = np.reshape(W, -1)
    v = np.reshape(v, -1)
    b = np.reshape(b, -1)
    omega = np.concatenate((W, v, b), axis = 0)
    return omega

def param(omega, N):
    n = 2
    W = np.reshape(omega[0:n*N], (N,n))
    v = np.reshape(omega[n*N:n*N+N],(1,N))
    b = np.reshape(omega[n*N+N:n*N+2*N],(N,1))
    return W, v, b

def MLP(omega,x,N,sigma):
    W, v, b = param(omega, N)
    prod_bias = np.dot(W, x) + b
    activation = (np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1)
    predicted_y = np.dot(v, activation)
    return predicted_y

def error(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[1]
    y_pred = MLP(omega, x, N, sigma)
    error = (1/(2*P))*np.sum((y_pred-y)**2) + 0.5*ro*np.linalg.norm(omega)**2
    return float(error)

def Grad(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[1]
    W, v, b = param(omega,N)
    prod = np.dot(W,x)
    prod_bias = prod + b
    activation = (np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1)
    d_activation= sigma*(1 -((np.exp(2*sigma*prod_bias)-1)/(np.exp(2*sigma*prod_bias)+1))**2)
    y_pred = MLP(omega, x, N, sigma)
    dE_v =  (1/P)*np.dot((y_pred - y), activation.T) + ro*v
    dE_W = (1/P)*np.dot((np.dot(v.T ,(y_pred-y))*d_activation), x.T) + ro*W
    dE_b_part1 = (1/P)*np.sum((np.dot(v.T ,(y_pred-y))*d_activation), axis=1)
    dE_b = dE_b_part1.reshape(N,1) + ro*b 
    gradient = omega_initialization(dE_W, dE_v, dE_b) 
    return gradient 

def minimization(omega, args):
    # args = (dataset, ro, N, sigma)
    res = minimize(error, omega, args = args, method = 'L-BFGS-B', jac=Grad, tol = 1e-7)
    return res 

def Time(omega, args):
    t_in = time.time()
    res = minimize(error, omega, args = args, method = 'L-BFGS-B', jac=Grad, tol = 1e-7)
    return time.time()-t_in    
 
def cross_validation(k, omega, dataset, ro, N, sigma): 
    x = dataset[0]
    y = dataset[1]
    kf = KFold(n_splits = k, random_state = 1836553, shuffle = True)
    validation_error = [] 
    for train_index, test_index in kf.split(x.T): 
        X_train, X_validation = x[:,train_index], x[:,test_index]
        Y_train, Y_validation = y[train_index], y[test_index]
        W, v, b = param(omega, N) 
        data_TRAIN = (X_train, Y_train)
        data_VALIDATION = (X_validation, Y_validation)
        args = (data_TRAIN, ro, N, sigma) 
        res = minimization(omega, args)
        omega_opt = res.x  
        val_error = error(omega_opt, data_VALIDATION, ro, N, sigma) 
        validation_error = np.append(validation_error, val_error)  
    mean_error_validation = np.mean(validation_error)
    return mean_error_validation, omega_opt  
 
# parametri_grid = {"neuron": [10,20,30,40,50], "ro": [1e-3, 1e-4, 1e-5], "sigma":[0.9, 1, 1.2]}  

def GRID_SEARCH(dataset, param_grid):
    minimo = 10000 
    for i in range(len(param_grid["neuron"])):
        N = param_grid["neuron"][i] 
        np.random.seed(seed) 
        W,v,b = var_initialization(seed, N)
        omega = omega_initialization(W, v, b)
        for j in range(len(param_grid["ro"])):
            ro = param_grid["ro"][j]
            for s in range(len(param_grid["sigma"])):
                sigma = param_grid["sigma"][s]
                evaluation, omega_opt = cross_validation(5, omega, dataset, ro, N, sigma)
                if evaluation < minimo:  
                    minimo = evaluation 
                    N_opt = param_grid["neuron"][i]
                    ro_opt = param_grid["ro"][j] 
                    sigma_opt = param_grid["sigma"][s]
                    
    return minimo, N_opt, ro_opt, sigma_opt 
                     
def plotting(omega, N, sigma, title):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    x1 = np.linspace(-2, 2, 50) 
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2) #costruzione della griglia 
    X = np.concatenate((X1.reshape(2500,1), X2.reshape(2500,1)), axis=1)
    Y_plot = MLP(omega, X.T, N, sigma).reshape(50,50)
    ax.plot_surface(X1, X2, Y_plot, rstride=1, cstride=1, cmap='plasma', edgecolor='None')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()
                                                      
