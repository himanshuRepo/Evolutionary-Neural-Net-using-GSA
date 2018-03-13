# -*- coding: utf-8 -*-
"""
Python code of Evolving Neural Networks with one input layer (number of units = number of input features), one hidden layer 
(number of  units = 2) and one output layer (number of  units = 1) using Gravitational Search Algorithm (GSA)

Coded by: Himanshu Mittal (emailid: himanshu.mittal224@gmail.com)
The code template used is similar given at link: https://github.com/himanshuRepo/GSA_PythonCode and https://github.com/7ossam81/EvoloPy.

 -- Purpose: Main File::
                Calling the Gravitational Search Algorithm(GSA) Algorithm 
                for training a Neural Net
Code compatible:
 -- Python: 2.* or 3.*
"""
import GSA as gsa
import benchmarks
import csv
import numpy
import time
import pandas as pd
import optiFeatures
from sklearn.model_selection import train_test_split


def selector(algo,func_details,popSize,Iter,dataset):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    df=pd.read_csv(dataset)
    X=df.iloc[:,1:-1]
    X=X.values
    y=df.iloc[:, -1]
    y_act=y.astype('category').cat.codes
    y_act=y_act.values
    X_train, X_test, y_train, y_test = train_test_split(X, y_act,test_size=0.30)
    

    if(algo==0):
        x=gsa.GSA(getattr(benchmarks, function_name),lb,ub,popSize,Iter,X_train,y_train)

        # Evaluate MLP classification model based on the training set
        trainClassification_results=optiFeatures.optiFeats(X_train,y_train,x.bestNet)
        x.trainAcc=trainClassification_results[0]
        x.trainTP=trainClassification_results[1]
        x.trainFN=trainClassification_results[2]
        x.trainFP=trainClassification_results[3]
        x.trainTN=trainClassification_results[4]

        # Evaluate MLP classification model based on the testing set
        testClassification_results=optiFeatures.optiFeats(X_test,y_test,x.bestNet)  
        x.testAcc=testClassification_results[0]
        x.testTP=testClassification_results[1]
        x.testFN=testClassification_results[2]
        x.testFP=testClassification_results[3]
        x.testTN=testClassification_results[4] 
    return x
    
    
# Select optimizers
GSA= True # Code by Himanshu Mittal




Algorithm=[GSA]
datasets=["Iristr"]
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
Runs=1

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 3
iterations= 10

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
atLeastOneIteration=False


# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,iterations):
	CnvgHeader.append("Iter"+str(l+1))


for j in range (0, len(datasets)):        # specfiy the number of the datasets
    # df=pd.read_csv("Iris.csv")
    dataset=datasets[j]+".csv"

    for i in range (0, len(Algorithm)):
        if(Algorithm[i]==True): # start experiment if an Algorithm and an objective function is selected
            for k in range (0,Runs):
                
                func_details=["perf",-100,100]
                x=selector(i,func_details,PopSize,iterations,dataset)
                if(Export==True):
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (atLeastOneIteration==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime","trainAcc", "trainTP","trainFN","trainFP","trainTN", "testAcc", "testTP","testFN","testFP","testTN"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.Algorithm,datasets[j],x.objectivefunc,k+1,x.startTime,x.endTime,x.executionTime,x.trainAcc, x.trainTP,x.trainFN,x.trainFP,x.trainTN, x.testAcc, x.testTP,x.testFN,x.testFP,x.testTN],x.convergence])
                        writer.writerow(a)
                    out.close()
                atLeastOneIteration=True # at least one experiment
                
if (atLeastOneIteration==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 