# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:23:49 2018

@author: puning
"""

import numpy as np
import AKNN
import matplotlib.pyplot as plt
from sklearn import neighbors
from scipy.stats import norm
import pandas as pd
class Generate:
    def __init__(self, N, distribution):
        if distribution=='Gaussian1':
            self.y=np.random.choice([-1,1],N)
            self.x=np.random.normal(0,1,N)*(self.y+3)/2
        if distribution=='Gaussian2':
            self.y=np.random.choice([-1,1],N)
            self.x=np.random.normal(0,1,N)+1*self.y  
        if distribution=='Gaussian3':
            self.x=np.random.normal(0,1,N)
            self.y=[0]*N
            for i in range(N):
                self.y[i]=np.random.choice([-1,1],p=[0.5-0.5*np.cos(5*self.x[i]),0.5+0.5*np.cos(5*self.x[i])])
            self.y=np.asarray(self.y)
        if distribution=='Laplace':
            self.x=np.random.laplace(0,1,N)
            self.y=[0]*N
            for i in range(N):
                self.y[i]=np.random.choice([-1,1],p=[0.5-0.5*np.cos(5*self.x[i]),0.5+0.5*np.cos(5*self.x[i])])
            self.y=np.asarray(self.y) 
        if distribution=='t2':
            self.x=np.random.standard_t(2,N)
            self.y=[0]*N
            for i in range(N):
                self.y[i]=np.random.choice([-1,1],p=[0.5-0.5*np.cos(5*self.x[i]),0.5+0.5*np.cos(5*self.x[i])])
            self.y=np.asarray(self.y)             
def evaluate(model,dtrain,dtest):
    model.fit(dtrain.x,dtrain.y)
    ypredict=model.predict(dtest.x)
    err=np.sum(ypredict!=dtest.y)
    return err
def Rbayes(distribution):
    if distribution=='Gaussian3':
        N=10000000
        x=np.random.normal(0,1,N)
        eta=np.cos(5*x)
        return (1-np.mean(abs(eta)))/2
    if distribution=='Laplace':
        N=10000000
        x=np.random.laplace(0,1,N)
        eta=np.cos(5*x)
        return (1-np.mean(abs(eta)))/2  
    if distribution=='t2':
        R=[0]*100
        for i in range(100):
            N=10000000
            x=np.random.standard_t(2,N)
            eta=np.cos(5*x)
            R[i]=(1-np.mean(abs(eta)))/2 
        return np.mean(R)
def compare(M,Ntrain,Ntest,dist,k):
    if dist=='Cauchy':
        model1=AKNN.Brute(q=0.8,K=0.6,A=1)
    else:
        model1=AKNN.Approx(q=0.8,K=0.6,A=1,m=20)
    model3=neighbors.KNeighborsClassifier(n_neighbors=k,weights='uniform')
    err1=[0]*M
    err2=[0]*M
    err3=[0]*M
    for i in range(M):
        dtrain=Generate(Ntrain,dist)
        dtest=Generate(Ntest,dist)
        err1[i]=evaluate(model1,dtrain,dtest)
        dtrain.x=dtrain.x.reshape(-1,1)
        dtest.x=dtest.x.reshape(-1,1)
        err3[i]=evaluate(model3,dtrain,dtest)  
    R1=np.mean(err1)/Ntest
    R3=np.mean(err3)/Ntest
    print('Classification Error Rate:')
    print(R1)
    print(R3)    
    if dist=='Gaussian3':
        Rbayes=0.1817
    if dist=='Gaussian2':
        Rbayes=norm.cdf(-1)
    if dist=='Laplace':
        Rbayes=0.1797
    if dist=='Cauchy' or dist=='t2':
        Rbayes=0.18168
    print('Estimated Excess Risk:')
    print(R1-Rbayes)
    print(R3-Rbayes) 
    return [R1,R3,R1-Rbayes,R3-Rbayes]
def bestk(M,Ntrain,Ntest,dist,kcandidates):
    #Select best $k$ for standard nearest neighbor classifier.
    L=len(kcandidates)
    R=[[0 for i in range(M)] for j in range(L)]
    for i in range(M):
        dtrain=Generate(Ntrain,dist)
        dtest=Generate(Ntest,dist)
        dtrain.x=dtrain.x.reshape(-1,1)
        dtest.x=dtest.x.reshape(-1,1)
        for j in range(L):
            model=neighbors.KNeighborsClassifier(n_neighbors=kcandidates[j],weights='uniform')
            R[j][i]=evaluate(model,dtrain,dtest)
    Risk=[[np.mean(R[j])/Ntest] for j in range(L)]
    print('For the following k:')
    print(kcandidates)
    print('The estimated risk is:')
    print(Risk)
    return kcandidates[np.argmin(Risk)]
def bestK(M,Ntrain,Ntest,dist,Kcandidates):
    L=len(Kcandidates)
    R=[[0 for i in range(M)] for j in range(L)]
    for i in range(M):
        print(i)
        dtrain=Generate(Ntrain,dist)
        dtest=Generate(Ntest,dist)
        for j in range(L):
            model=AKNN.Approx(q=0.8,K=Kcandidates[j],A=1,m=20)
            R[j][i]=evaluate(model,dtrain,dtest)
    Risk=[[np.mean(R[j])/Ntest] for j in range(L)]
    print('For the following K:')
    print(Kcandidates)
    print('The estimated risk is:')
    print(Risk)  
    return Kcandidates[np.argmin(Risk)]
def compute(distribution):
    Narray=[50,100,500,1000,5000,10000]
    N_test=1000
    Result=[]
    for N in Narray:
        print('Training sample size:',N)
        if distribution=='Gaussian3':
            R=compare(M=1000,Ntrain=N,Ntest=N_test,dist='Gaussian3',k=np.round(0.9474*np.power(N,4/9)).astype(int))
        if distribution=='Laplace':
            R=compare(M=1000,Ntrain=N,Ntest=N_test,dist='Laplace',k=np.round(0.6316*np.power(N,4/9)).astype(int))
        if distribution=='Cauchy':
            R=compare(M=1000,Ntrain=N,Ntest=N_test,dist='Cauchy',k=np.round(1.4776*np.power(N,4/13)).astype(int))
        if distribution=='t2':
            R=compare(M=1000,Ntrain=N,Ntest=N_test,dist='t2',k=np.round(2.7848*np.power(N,4/11)).astype(int))
        Result=Result+[[N]+R]
    Output=pd.DataFrame(Result)
    filename=distribution+'.csv'
    Output.to_csv(filename,header=['Ntrain','AdaErr','OrgErr','AdaExc','OrgExc'])
#k_list=[1,2,4,5,10,15,20]
#K_list=[0.2,0.4,0.6,0.8,1]
#print(Rbayes('t2'))
#kopt=bestk(M=300,Ntrain=500,Ntest=800,dist='t2',kcandidates=k_list)
#Kopt=bestK(M=300,Ntrain=500,Ntest=800,dist='t2',Kcandidates=K_list)
compute('t2')



















