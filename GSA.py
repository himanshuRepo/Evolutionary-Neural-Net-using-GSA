# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            with minimizing of the Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""

import random
import Initialize as initi
import numpy
import math
from solution import solution
import time
import massCalculation
import gConstant
import gField
import move
import ANN

        
# def GSA(objf,lb,ub,dim,PopSize,iters,df):
def GSA(objf,lb,ub,PopSize,iters,X,Y):
    # GSA parameters
    ElitistCheck =1
    Rpower = 1 
     
    s=solution()
    inputLen=numpy.shape(X)[1]
    nets=initi.oldnetlist(PopSize,inputLen,lb,ub)
    dim=len(numpy.array(nets[0].nettolist()))
        
    """ Initializations """
    pos=numpy.zeros((PopSize,dim))
    vel=numpy.zeros((PopSize,dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest=numpy.zeros(dim)
    gBestScore=float("inf")
    
    for i in range(PopSize):
        pos[i,:]=numpy.array(nets[i].nettolist())

    convergence_curve=numpy.zeros(iters)
    
    print("GSA is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(0,iters):
        """Neural N/w updated"""
        for i in range(PopSize):
            nets[i].listtonet(pos[i,:])

        for i in range(0,PopSize):
            l1 = [None] * dim
            l1=numpy.clip(pos[i,:], lb, ub)
            nets[i].listtonet(l1)
            pos[i,:]=numpy.array(nets[i].nettolist())

            #Calculate objective function for each particle
            fitness=[]
            fitness=-objf(X,Y,nets[i])
            fit[i]=fitness
    
                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=l1
                bestNet=nets[i]

        
        
        """ Calculating Mass """
        M = massCalculation.massCalculation(fit,PopSize,M)

        """ Calculating Gravitational Constant """        
        G = gConstant.gConstant(l,iters)        
        
        """ Calculating Gfield """        
        acc = gField.gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower)
        
        """ Calculating Position """        
        pos, vel = move.move(PopSize,dim,pos,vel,acc)


        
        convergence_curve[l]=gBestScore
      
        if (l%1==0):
                # Just for display, added the negative sign.
               print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(-(gBestScore))]);
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.gBest=gBest
    s.bestNet=bestNet
    s.Algorithm="GSA"
    s.objectivefunc=objf.__name__

    return s