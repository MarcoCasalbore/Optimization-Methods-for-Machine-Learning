# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:14:08 2023

@author: Bianca
"""

from Functions_11_Ghiurca import *
 
file = 'Dataset.csv' 

(X_train, X_test, Y_train, Y_test) = lettura_file(file) 
dataset_train = (X_train.T, Y_train)     
(W,v,b) = var_initialization(seed, N) 
omega_in = omega_initialization(W, v, b) 
initial_error = error(omega_in, dataset_train, 0, N, sigma) 
initial_error_objfun = error(omega_in, dataset_train, ro, N, sigma) 

# print("Initial error without R term", initial_error) 
# print("Starting value O.F.", initial_error_objfun)
# print('Starting value for the norm grad=',np.linalg.norm(Grad(omega_in, dataset_train, ro, N, sigma)))
print("Number of neurons:", N)
print("Ro:", ro) 
print("Sigma:", sigma) 
print("Solver: L-BFGS-B") 
   
args = (dataset_train, ro, N, sigma)
result = minimization(omega_in, args) 
# print("Minimization result:", result) 
omega_opt = result.x  
# print(omega_opt) 
print("Number of function evaluations:", result.nfev)
print("Number of gradient evaluations:", result.njev)
print("Number of iterations:", result.nit)
# print("Final value O.F.", error(omega_opt, dataset_train, ro, N, sigma))
# print('Final value for the norm grad=',np.linalg.norm(Grad(omega_opt, dataset_train, ro, N, sigma)))
 
t = Time(omega_in, args)
print("Time used to optimize:", t) 
 
dataset_test = (X_test.T, Y_test) 
error_training = error(omega_opt, dataset_train, 0, N, sigma)
error_test = error(omega_opt, dataset_test, 0, N, sigma)
print("Training error", error_training)  
print("Test error", error_test)   
 
plotting(omega_opt, N, sigma, title='ES. 1.1 - MLP')  

