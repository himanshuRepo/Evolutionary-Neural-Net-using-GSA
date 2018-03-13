# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

Purpose: Defining the move Function
            for calculating the updated position

Code compatible:
 -- Python: 2.* or 3.*
"""

import random

def move(PopSize,dim,pos,vel,acc):
    for i in range(0,PopSize):
        for j in range (0,dim):
            r1=random.random()
            vel[i,j]=r1*vel[i,j]+acc[i,j]
            pos[i,j]=pos[i,j]+vel[i,j]
    
    return pos, vel
