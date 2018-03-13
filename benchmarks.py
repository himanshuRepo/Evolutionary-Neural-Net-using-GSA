# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.
 
 -- Purpose: Define the objective function for training Neural Net
              with each individual of the GSA population using accuracy
              as the performance cost function

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def perf(X,Y,ai=None):
	y_pred=numpy.zeros((numpy.shape(X)[0]))
  	for i in range(numpy.shape(X)[0]):
  		x_pass=list(X[i,:])
  		if ai.forward(x_pass) > 0.5:
  			y_pred[i]=1
  		else:
  			y_pred[i]=0
  	
  	s=(accuracy_score(Y, y_pred))
  	return s