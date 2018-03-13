# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Initialization of  Neural Nets 

Code compatible:
 -- Python: 2.* or 3.*
"""

import ANN
import numpy as np
import math
import random

def oldnetlist(netsize,inpnodes,lb,ub):
    netlist=[]

    for i in range (0,netsize):
    	base=[]
	#Creating a default NeuralNet with weights randomly distributed around the base value (triangular distribution)
        netelement = ANN.NeuralNet(inpnodes,lb,ub,base)
	#Adding this NeuralNet to the netlist
        netlist.append(netelement)
    return netlist