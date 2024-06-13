# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:33:40 2023

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
    c = np.random.randn(2, N)
    v = np.random.randn(N, 1)
    return c,v

def omega_initialization(c, v):
    c = np.reshape(c, -1)
    v = np.reshape(v, -1)
    omega = np.concatenate((c, v))
    return omega

def param(omega, N):
    c, v = omega[:2*N].reshape((2, N)), omega[2*N:].reshape((N, 1))
    return c, v 

def RBF(omega,N,x,sigma):
    c,v = param(omega,N)
    P = x.shape[0]
    # x = x.reshape(x.shape[0], x.shape[1], 1)
    x = x.reshape(P,2,1) 
    activation = np.exp(-(np.sum(np.square(x-c), axis = 1))/sigma**2)
    y_pred = np.dot(activation, v)
    return y_pred
  
def error(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    c,v = param(omega, N)
    y_pred = RBF(omega,N,x,sigma)
    error = (0.5/P)*np.sum((y_pred - y.reshape(-1,1))**2)+0.5*ro*np.linalg.norm(omega)**2
    return float(error)
  
def Grad(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    c,v = param(omega, N) 
    x = x.reshape(P,2,1)
    activation = np.exp(-(np.sum(np.square(x-c), axis=1))/sigma**2)
    y_pred = RBF(omega, N, x, sigma)
    dE_v = (1/P)*np.dot(activation.T, (y_pred - y.reshape(-1,1))) + ro*v 
    dE_c = np.sum((1/P)*(np.dot((y_pred - y.reshape(-1,1)), v.T)[:,np.newaxis,:])*(2/(sigma**2)*(activation[:,np.newaxis,:]*(x - c))), axis=0) + ro*c
    gradient = omega_initialization(dE_c, dE_v)
    return gradient 
 
def minimization(omega, args):
    #args = (dataset, ro, N, sigma) 
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
    for train_index, test_index in kf.split(x): 
        X_train, X_validation = x[train_index,:], x[test_index,:]
        Y_train, Y_validation = y[train_index], y[test_index]
        c,v = param(omega, N) 
        data_TRAIN = (X_train, Y_train)
        data_VALIDATION = (X_validation, Y_validation)
        args = (data_TRAIN, ro, N, sigma)
        res = minimization(omega, args)
        omega_opt = res.x 
        val_error = error(omega_opt, data_VALIDATION, ro, N, sigma)  
        validation_error = np.append(validation_error, val_error)
    mean_error_validation = np.mean(validation_error)
    return mean_error_validation, omega_opt 
   
parametri_grid = {"neuron": [10,20,30,40,50], "ro": [1e-3, 1e-4, 1e-5], "sigma":[0.9, 1, 1.2]} 

def GRID_SEARCH(dataset, param_grid): 
    minimo = 10000
    for i in range(len(param_grid["neuron"])):
        N = param_grid["neuron"][i]
        np.random.seed(seed)
        c,v = var_initialization(seed, N)
        omega = omega_initialization(c,v)
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
    Y_plot = RBF(omega, N, X, sigma).reshape(50,50)
    ax.plot_surface(X1, X2, Y_plot, rstride=1, cstride=1, cmap='plasma', edgecolor='None')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()


