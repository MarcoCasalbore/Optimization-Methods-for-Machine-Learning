# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:58:44 2023

@author: Bianca
"""


from Functions_22_Ghiurca import *

file = 'Dataset.csv'
(X_train, X_test, Y_train, Y_test) = lettura_file(file)
dataset_train = (X_train, Y_train)
v,c = var_initialization(seed,X_train,N) 
initial_error = error(v,c,dataset_train,0,N,sigma)
# initial_error_objfun = error(v,c,dataset_train,ro,N,sigma) 

# print("Initial error without R term", initial_error) 
# print("Starting value O.F.", initial_error_objfun)
print("Number of neurons:", N)
print("Ro:", ro)
print("Sigma:", sigma)
print("Solver: LSTSQ")  

print("Number of gradient evaluations:", 1)
print("Number of function evaluations:", 1)

t1 = time.time() 
result = minimization(c,dataset_train, ro,N, sigma)
t2 = time.time()
t_tot= t1-t2
cifre_significative = 5
tempo = f"{t_tot:.{cifre_significative}f}"
print("Time used to optimize:", tempo)  

v_optimized = result[0] 
dataset_test = (X_test, Y_test)
error_training = error(v_optimized,c, dataset_train, 0,N,sigma)
error_test = error(v_optimized, c, dataset_test, 0,N,sigma)
print("Training error", error_training)
print("Test error", error_test)  

plotting(c,v_optimized,N,sigma, title ='ES. 2.2 - RBF,with unsupervised selection of the centers')



