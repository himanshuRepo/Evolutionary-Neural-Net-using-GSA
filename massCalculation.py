# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

Purpose: Defining the massCalculation Function
            for calculating the mass

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy 

def massCalculation(fit,PopSize,M):
    Fmax = max(fit)
    Fmin = min(fit)
    Fsum = sum(fit)        
    Fmean = Fsum/len(fit)
        
    if Fmax == Fmin:
        M = numpy.ones(PopSize)
    else:
        best = Fmin
        worst = Fmax
        
        for p in range(0,PopSize):
           M[p] = (fit[p]-worst)/(best-worst)
            
    Msum=sum(M)
    for q in range(0,PopSize):
        M[q] = M[q]/Msum
            
    return M
