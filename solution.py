# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Defining the solution class
 
Code compatible:
 -- Python: 2.* or 3.*
"""

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.gBest=[]
        self.bestNet=None
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        self.trainAcc=None
        self.trainTP=None
        self.trainFN=None
        self.trainFP=None
        self.trainTN=None
        self.testAcc=None
        self.testTP=None
        self.testFN=None
        self.testFP=None
        self.testTN=None