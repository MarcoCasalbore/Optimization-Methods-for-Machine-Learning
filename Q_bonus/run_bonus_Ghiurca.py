# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:01:34 2023

@author: Bianca
"""

from Functions_bonus_Ghiurca import *

file = 'Dataset.csv'
dataset = lettura_file(file)
v,c = var_initialization(seed, N)
omega = omega_initialization(c, v) 
initial_error = ERROR(omega,dataset,0,N,sigma)
initial_error_objfun = ERROR(omega,dataset,ro,N,sigma) 

# print("Initial error without R term", initial_error) 
# print("Starting value O.F.", initial_error_objfun)


print("Number of neurons:", N)
print("Ro:", ro)
print("Sigma:", sigma)
print("Solvers: L-BFGS-B and LSTSQ")  

c_opt, v_opt = final_minimization(dataset, ro,N, sigma, max_iterations=100, tol=1e-6, patience=10)
omega_optimized = omega_initialization(c_opt, v_opt)

error_training = ERROR(omega_optimized, dataset, ro, N, sigma)
print("Training error", error_training) 

plotting(c_opt,v_opt,N,sigma, title = 'BONUS')  

# print(multistart(dataset, parametri_multistart)) 