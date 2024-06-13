# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:16:46 2023

@author: Bianca
"""


from Functions_21_Ghiurca import *

file = 'Dataset.csv'
(X_train, X_test, Y_train, Y_test) = lettura_file(file)
dataset_train = (X_train.T, Y_train)
(W,v,b) = var_initialization(seed, N)  
# initial_error = error(v,W,b, dataset_train, 0,N,sigma)
# initial_error_objfun = error(v,W,b, dataset_train, ro, N,sigma)

# print("Initial error without R term", initial_error) 
# print("Starting value O.F.", initial_error_objfun)
print("Number of neurons:", N)
print("Ro:", ro) 
print("Sigma:", sigma)
print("Solver: LSTSQ")   

print("Number of gradient evaluations:", 1)
print("Number of function evaluations:", 1)

t1 = time.time()
result = minimization(W,b, dataset_train,ro,N,sigma) 
t2 = time.time() 
t_tot = t1-t2
cifre_significative = 5
tempo = f"{t_tot:.{cifre_significative}f}"   
print("Time used to optimize:", tempo) 

v_optimized = result[0] 
# print("Final value O.F.", V_error(v_optimized,W,b, dataset_train, ro, N,sigma))
dataset_test = (X_test.T, Y_test)
error_training = error(v_optimized, W, b, dataset_train, 0,N,sigma)
error_test = error(v_optimized, W, b, dataset_test, 0,N,sigma)
print("Training error", error_training)
print("Test error", error_test)  

plotting(v_optimized,W,b,N, sigma, title = 'ES. 2.1 - MLP, extreme learning')


 