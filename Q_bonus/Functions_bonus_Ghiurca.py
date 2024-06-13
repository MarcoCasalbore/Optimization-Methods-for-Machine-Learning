# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:17:51 2023

@author: Bianca
"""

import pandas as pd
import numpy as np 
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
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
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1836553)
    return  X, Y 

def var_initialization(seed,N):
    np.random.seed(seed)
    v = np.random.randn(N, 1)
    c = np.random.randn(2,N)
    return v,c

def omega_initialization(c, v):
    c = np.reshape(c, -1)
    v = np.reshape(v, -1)
    omega = np.concatenate((c, v))
    return omega

def param(omega, N):
    c, v = omega[:2*N].reshape((2, N)), omega[2*N:].reshape((N, 1))
    return c, v 

def prediction(v,c,x,sigma): 
    P = x.shape[0]
    x = x.reshape(P, 2, 1)  
    activation = np.exp(-(np.sum(np.square(x-c), axis = 1))/sigma**2)
    y_pred = np.dot(activation, v)
    return y_pred

def V_minimization(c,dataset, ro,N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    x = x.reshape(P,2,1)
    activation = np.exp(-(np.sum(np.square(x-c), axis = 1))/sigma**2)
    shape = (np.dot(activation.T, activation)).shape[0]
    Q_matrix = (1/P)*np.dot(activation.T, activation) + ro*np.eye(shape)
    c_vector = (1/P)*(np.dot(y, activation)).T  
    x_star, residuals, rank, eigenvectors_ = np.linalg.lstsq(Q_matrix, c_vector, rcond = None)
    
    return  x_star

def ERROR(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    c,v = param(omega, N)
    y_pred = prediction(v,c,x,sigma) 
    error = (0.5/P)*np.sum((y_pred - y.reshape(-1,1))**2)+0.5*ro*np.linalg.norm(omega)**2
    return float(error)

def Grad(omega, dataset, ro, N, sigma):
    x = dataset[0]
    y = dataset[1]
    P = x.shape[0]
    c,v = param(omega, N) 
    x = x.reshape(P,2,1)
    activation = np.exp(-(np.sum(np.square(x-c), axis=1))/sigma**2)
    y_pred = prediction(v,c,x,sigma)
    dE_v = (1/P)*np.dot(activation.T, (y_pred - y.reshape(-1,1))) + ro*v 
    dE_c = np.sum((1/P)*(np.dot((y_pred - y.reshape(-1,1)), v.T)[:,np.newaxis,:])*(2/(sigma**2)*(activation[:,np.newaxis,:]*(x - c))), axis=0) + ro*c
    gradient = omega_initialization(dE_c, dE_v)
    return gradient 
 
def minimization(omega, args):
    #args = (dataset, ro, N, sigma) 
    res = minimize(ERROR, omega, args = args, method = 'L-BFGS-B', jac=Grad, tol = 1e-7)
    c_optimized = res.x[:2*N].reshape((2, N))
    return res 

def final_minimization(dataset, ro,N, sigma, max_iterations=100, tol=1e-6, patience=10):
    v,c = var_initialization(seed, N)
    count = 0
    best_error = float('inf')
    number_fun_evaluations = 0
    number_grad_evaluations = 0
    t_in = time.time()

    for it in range(max_iterations):  
        v_opt = V_minimization(c,dataset, ro,N, sigma) 
        omega = omega_initialization(c, v_opt)
        args = (dataset, ro, N, sigma) 
        result = minimization(omega,args)
        c_opt = result.x[:2*N].reshape((2, N)) 
        number_fun_evaluations += result.nfev 
        number_grad_evaluations += result.njev
        # y_pred = prediction(c_opt,v_opt,x,sigma)
        omega_opt = omega_initialization(c_opt, v_opt) 
        error = ERROR(omega_opt, dataset, ro, N, sigma) 
        if error < best_error:
            best_error = error
            count = 0
        else:
           count += 1
           
        if count >= patience:
            break
    
        if error < tol:
            break  
        c = c_opt
        v = v_opt 
    t_fin = time.time() - t_in
        
    print("Number of function evaluations: ",  number_fun_evaluations)
    print("Number of gradient evaluations: ", number_grad_evaluations) 
    print("Time used to optimize:", t_fin) 
    
    return  c_opt , v_opt

      
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

