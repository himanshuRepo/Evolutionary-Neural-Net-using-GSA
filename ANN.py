# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Define the Neural Network class along with all relevant functions

Code compatible:
 -- Python: 2.* or 3.*
"""

import numpy
import math
import random

######################NEURAL NET CLASS######################
class NeuralNet():
    #Constructor for the class which initializes the parameters of the NeuralNet
    def __init__(self, inpnodes, lb, ub, base, weights1=None, weights2=None):
        #The number of input, hidden and output nodes is set
        self.inputnodes = inpnodes
        self.hiddennodes = 2
        self.outputnodes = 1
        netwSize=self.inputnodes*self.hiddennodes+self.hiddennodes*self.outputnodes
        # self.base = [-0.252165132632152, -0.125465654654654165, -0.62625521521526,-0.365545284852,-0.4054554656265656,1.982126516512]
        for i in range (netwSize):
            base.append(random.uniform(ub,lb))
        #If weights are provided in the constructor call, then set them as it is
        if weights1 and weights2:
	    #Weights1 is a set of 2X2-tuples which represents the weights from the input layer to the hidden layer
            self.weights1 = weights1
	    #Weights2 is represents 2 weights which correspond to the edges from the 2 hidden nodes to the single output node
            self.weights2 = weights2
        else:
            #set the weights randomly around the base value
            self.random(base)

    #creating the weights of the NeuralNet if no parameters are given while forming the NeuralNet
    def random(self,base):
        weightslist = []

        for weight in base:
            #Perturbing the weight by an amount proportional to it and also taken from a random triangular distribution
            # weightslist.append(weight + random.triangular(weight - .3, weight + .3))
            weightslist.append(weight)
            #Adding the weights to the NeuralNet by using the listtonet function
        self.listtonet(weightslist)

    #converts a list of weights to the weight of a NeuralNet
    def listtonet(self, weightslist):
        weights1 = []
        weights2 = []
        count = 0
    #Initially parsing through the list for first filling the values of weights1
        for i in xrange(self.inputnodes):
            n = [] 
            for j in xrange(self.hiddennodes):
                n.append(weightslist[count])
                count += 1
            weights1.append(n)
    #Now parsing through the list for filling the values of weights2
        for i in xrange(self.hiddennodes):
            n = [] 
            for j in xrange(self.outputnodes):
                n.append(weightslist[count])
                count += 1
            weights2.append(n)
    #setting these weights to the NeuralNet
        self.weights1 = weights1
        self.weights2 = weights2

    #Getting output value for the NeuralNet from the input
    def forward(self, inputvalues):
        #First finding out the value of the hidden layer nodes using the weights from the input layer
        hiddenlayeroutput = self.sigmoid(numpy.dot(inputvalues, self.weights1))
        #Then finding the value of the output nodes using the weight and values of the hidden layer 
        output = self.sigmoid(numpy.dot(hiddenlayeroutput, self.weights2))
        return output
	


    #Converting the NeuralNet weights into a list
    def nettolist(self):
        weightslist = []
        for i in self.weights1:
            weightslist.extend(i)
        for i in self.weights2:
            weightslist.extend(i)
        return weightslist



    #Sigmoid funtion returns the value of the sigmoid function the input provided
    def sigmoid(self, n):
        return 1/(1 + numpy.exp(-n))
