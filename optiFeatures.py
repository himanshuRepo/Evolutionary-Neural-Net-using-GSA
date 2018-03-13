# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Measuring the performance parameters of optimal trained Neural Net after all the 
              iterations of GSA with different accuracy measures (TP, FP, TN, FN)
              
Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def optiFeats(X,Y,ai=None):
  y_pred=numpy.zeros((numpy.shape(X)[0]))
  for i in range(numpy.shape(X)[0]):
    x_pass=list(X[i,:])
    if ai.forward(x_pass) > 0.5:
      y_pred[i]=1
    else:
      y_pred[i]=0
  s=accuracy_score(Y, y_pred)
  ConfMatrix=confusion_matrix(Y, y_pred)
  time.sleep(5)
  ConfMatrix1D=ConfMatrix.flatten()
  #print(ConfMatrix1D)
  printAcc=[]
  printAcc.append(accuracy_score(Y, y_pred,normalize=True)) 

  classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
  # print(classification_results)
  return classification_results