#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:57:51 2019

@author: Kartik
"""


import numpy as np #arrays
from numpy import linalg as LA #Linear Algebra
import matplotlib.pyplot as plt #plotting
import sympy #symbolic computing package
from sympy.utilities.lambdify import lambdify #convert sympy objects to python interpretable

##################################################
# CREATING FUNCTIONS USING SYMPY
##################################################

#Create functions using SymPy
v = sympy.Matrix(sympy.symbols('x[0] x[1]'))                    #import SymPy objects

#create a function as a SymPy expression
f_sympy1 = v[0]**2 - 2.0 * v[0] * v[1] + 4 * v[1]**2            #first function
print('This is what the function looks like: ', f_sympy1)

f_sympy2 = 0.5*v[0]**2 + 2.5*v[1]**2                            #second function

f_sympy3 = 4*v[0]**2 + 2*v[1]**2 + 4*v[0]*v[1] - 3*v[0]         #third function

##################################################
# CONVERTING SYMPY EXPRESSIONS INTO REGULAR EXPRESSIONS
##################################################

#Extract Main function
def f_x(f_expression, values):
    '''Takes in SymPy function expression along with values of dim 1x2 and return output of the function'''
    f = lambdify((v[0],v[1]), f_expression)                     #convert to function using lambdify
    return f(values[0],values[1])                               #Evaluate the function at the given value

#Extract gradients
def df_x(f_expression, values):
    '''Takes in SymPy function expression along with values of dim 1x2 and returns gradients of the original function'''
    df1_sympy = np.array([sympy.diff(f_expression, i) for i in v])       #first order derivatives
    dfx_0 = lambdify((v[0],v[1]), df1_sympy[0])                          #derivative w.r.t x_0
    dfx_1 = lambdify((v[0],v[1]), df1_sympy[1])                          #derivative w.r.t x_1
    evx_0 = dfx_0(values[0], values[1])                                  #evaluating the gradient at given values          
    evx_1 = dfx_1(values[0], values[1])
    return(np.array([evx_0,evx_1]))
    
#Extract Hessian
def hessian(f_expression):
    '''Takes in a SymPy expression and returns a Hessian'''
    df1_sympy = np.array([sympy.diff(f_expression, i) for i in v])              #first order derivatives
    hessian = np.array([sympy.diff(df1_sympy, i) for i in v]).astype(np.float)  #hessian
    return(hessian)

##################################################
# FUNCTIONS TO VISUALIZE
##################################################

#Function to create a 3-D plot of the loss surface
def loss_surface(sympy_function): 
    '''Plots the loss surface for the given function'''
    #x = sympy.symbols('x')
    return(sympy.plotting.plot3d(sympy_function, adaptive=False, nb_of_points=400))
    
#Function to create a countour plot 
def contour(sympy_function):
    '''Takes in SymPy expression and plots the contour'''
    x = np.linspace(-3, 3, 100)                         #x-axis
    y = np.linspace(-3, 3, 100)                         #y-axis
    x, y = np.meshgrid(x, y)                            #creating a grid using x & y
    func = f_x(sympy_function, np.array([x,y]))
    plt.axis("equal")
    return plt.contour(x, y, func)

#Function to plot contour along with the travel path of the algorithm
def contour_travel(x_array, sympy_function):
    '''Takes in an array of output points and the corresponding SymPy expression to return travel contour plot '''
    x = np.linspace(-2, 2, 100)                         #x-axis
    y = np.linspace(-2, 2, 100)                         #y-axis
    x, y = np.meshgrid(x, y)                            #creating a grid using x & y
    func = f_x(sympy_function, np.array([x,y]))
    plt.axis("equal")
    plt.contour(x, y, func)
    plot = plt.plot(x_array[:,0],x_array[:,1],'x-')
    return (plot)
    
##################################################
#ALGORITHMS
##################################################

####Newton Method
def Newton(sympy_function, max_iter, start, step_size = 1, epsilon = 10**-2):
    i = 0
    x_values = np.zeros((max_iter+1,2))
    x_values[0] = start
    norm_values = []
    while i < max_iter:
        norm = LA.norm(df_x(sympy_function, x_values[i]))
        if norm < epsilon:
            break
        else:
            grad = df_x(sympy_function, x_values[i])
            hessian_inv = LA.inv(hessian(sympy_function))
            p = -np.dot(grad, hessian_inv)
            x_values[i+1] = x_values[i] + step_size*p
            norm_values.append(norm)
        i+=1
    print('No. of steps Newton takes to converge: ', len(norm_values))
    return(x_values, norm_values)
    
####Steepest Descent Method
def SDM(sympy_function, max_iter, start, step_size, epsilon = 10**-2):
    i = 0
    x_values = np.zeros((max_iter+1,2))
    x_values[0] = start
    norm_values = []
    while i < max_iter:
        norm = LA.norm(df_x(sympy_function, x_values[i]))
        if norm < epsilon:
            break
        else:
            p = -df_x(sympy_function, x_values[i])                        #updated direction to move in 
            x_values[i+1] = x_values[i] + step_size*p                     #new x-value
            norm_values.append(norm)   
        i+=1
    print('No. of steps SDM takes to converge: ', len(norm_values))
    return(x_values, norm_values)


#### Conjugate Gradient Method
def CGM(sympy_function, max_iter, start, step_size, epsilon = 10**-2):
    i = 0
    x_values = np.zeros((max_iter+1,2))
    x_values[0] = start
    grad_fx = np.zeros((max_iter+1,2))
    p = np.zeros((max_iter+1,2))
    norm_values = []
    while i < max_iter:
        grad_fx[i] = df_x(sympy_function, x_values[i])
        norm = LA.norm(df_x(sympy_function, x_values[i]))
        if norm < epsilon:
            break
        else:
            if i == 0:
                beta = 0
                p[i] = - np.dot(step_size,df_x(sympy_function, x_values[i]))
            else:
                beta = np.dot(grad_fx[i],grad_fx[i]) / np.dot(grad_fx[i-1],grad_fx[i-1])
                p[i] =  -df_x(sympy_function, x_values[i]) + beta * p[i-1]
        x_values[i+1] = x_values[i] + step_size*p[i]
        norm_values.append(norm)
        i += 1
    print('No. of steps CDM takes to converge: ', len(norm_values))
    return(x_values, norm_values)

##################################################
#IMPLEMENTATION
##################################################
  
#Case 1
loss_surface(f_sympy1)
contour(f_sympy1)

#Newton
newton_values1, newton_norm1 = Newton(f_sympy1, 9, [-3.0,2.0])
contour_travel(newton_values1, f_sympy1)

#SDM
sdm_values1, sdm_norm1 = SDM(f_sympy1, 40, [-3.0,2.0], 0.15)
contour_travel(sdm_values1, f_sympy1)

#CGM
cgm_values1, cgm_norm1 = CGM(f_sympy1, 50, [-3.0,2.0], 0.15)   
contour_travel(cgm_values1, f_sympy1)


#Case2
loss_surface(f_sympy2)
contour(f_sympy2)

#Newton
newton_values2, newton_norm2 = Newton(f_sympy2, 9, [-3.0,2.0])
contour_travel(newton_values2, f_sympy2)

#SDM
sdm_values2, sdm_norm2 = SDM(f_sympy2, 50, [-3.0,2.0], 0.15)
contour_travel(sdm_values2, f_sympy2)

#CGM
cgm_values2, cgm_norm2 = CGM(f_sympy2, 50, [-3.0,2.0], 0.15)   
contour_travel(cgm_values2, f_sympy2)

#Case3
loss_surface(f_sympy3)
contour(f_sympy3)

#Newton
newton_values3, newton_norm3 = Newton(f_sympy3, 9, [-3.0,2.0])
contour_travel(newton_values3, f_sympy3)

#SDM
sdm_values3, sdm_norm3 = SDM(f_sympy3, 50, [-3.0,2.0], 0.15)
contour_travel(sdm_values3, f_sympy3)

#CGM
cgm_values3, cgm_norm3 = CGM(f_sympy3, 50, [-3.0,2.0], 0.15)   
contour_travel(cgm_values3, f_sympy3)



