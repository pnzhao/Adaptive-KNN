# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:31:02 2018

@author: puning
"""
import numpy as np
class Brute: #Brute Calculation.
    def __init__(self,q,K,A):
        self.q=q
        self.K=K
        self.A=A
    def fit(self,xtrain,ytrain):
        self.xtrain=xtrain
        self.ytrain=ytrain
    def predict(self,xtest):
        Ntest=len(xtest)
        xtrain=self.xtrain
        ytrain=self.ytrain
        q=self.q
        K=self.K
        A=self.A
        Ntrain=len(xtrain)
        ypredict=[0]*Ntest
        for i in range(Ntest):
            n=0
            d=[np.abs(xtrain[j]-xtest[i]) for j in range(Ntrain)]
            for j in range(Ntrain):
                if d[j]<A:
                    n=n+1
            k=int(np.floor(K*np.power(n,q)))+1
            dsort=np.argsort(d)
            s=0
            for j in range(k):
                s=s+ytrain[dsort[j]]
            if s>0:
                ypredict[i]=1
            else:
                ypredict[i]=-1
        return np.asarray(ypredict).T
class Approx: #Fast approximate implementation (1 dimensional)
    def __init__(self,q,K,A,m):
        self.q=q
        self.K=K
        self.A=A
        self.m=m
    def fit(self,xtrain,ytrain):
        A=self.A
        K=self.K
        q=self.q
        m=self.m
        xmin=np.min(xtrain)-2.2*A
        xmax=np.max(xtrain)+2.2*A #Leave enough margins.
        grid_length=A/(2*m+1)
        ngrid=int(np.ceil((xmax-xmin)/grid_length))
        count=[0]*ngrid
        value=[0]*ngrid #sum of the value of all labels in this grid.
        cum_count=[0]*ngrid
        cum_value=[0]*ngrid
        N=len(xtrain)
        num=np.floor((xtrain-xmin)/grid_length).astype(int)
        for i in range(N):
            count[num[i]]=count[num[i]]+1
            value[num[i]]=value[num[i]]+ytrain[i]
        for i in range(1,ngrid):
            cum_count[i]=cum_count[i-1]+count[i]
            cum_value[i]=cum_value[i-1]+value[i]
        nlist=[0]*ngrid
        for i in range(m+1,ngrid-m):
            nlist[i]=cum_count[i+m]-cum_count[i-m-1] #Count the approximate numbers.
        klist=np.floor(K*np.power(nlist,q)).astype(int)+1 #Calculate k.
        predlist=[0]*ngrid
        for i in range(m+1,ngrid-m):
            if klist[i]<nlist[i]: #Use binary search. 
                L=0
                R=m
                while L<R-1:
                    mid=np.floor((L+R)/2).astype(int)
                    current_count=cum_count[i+mid]-cum_count[i-mid-1]
                    if current_count<klist[i]:
                        L=mid
                    elif current_count>klist[i]:
                        R=mid
                    else:
                        L=mid
                        R=mid
                value_under=cum_value[i+L]-cum_value[i-L-1]
                value_over=cum_value[i+R]-cum_value[i-R-1]
                if value_under+value_over>0:
                    predlist[i]=1
                else:
                    predlist[i]=-1
            elif klist[i]>nlist[i]:
                L=m
                R=ngrid-m
                while L<R-1:
                    mid=np.floor((L+R)/2).astype(int)
                    current_count=cum_count[min(i+mid,ngrid-1)]-cum_count[max(i-mid-1,0)]
                    if current_count<klist[i]:
                        L=mid
                    elif current_count>klist[i]:
                        R=mid
                    else:
                        L=mid
                        R=mid
                value_under=cum_value[min(i+L,ngrid-1)]-cum_value[max(i-L-1,0)]
                value_over=cum_value[min(i+R,ngrid-1)]-cum_value[max(i-R-1,0)]
                if value_under+value_over>0:
                    predlist[i]=1
                else:
                    predlist[i]=-1
            else:
                value=cum_value[i+m]-cum_value[i-m-1]
                if value>0:
                    predlist[i]=1
                else:
                    predlist[i]=-1
        for i in range(m+1):
            predlist[i]=predlist[m+1]
        for i in range(ngrid-m,ngrid):
            predlist[i]=predlist[ngrid-m-1]
        self.predlist=predlist
        self.grid_length=grid_length
        self.ngrid=ngrid
        self.xmin=xmin
        self.xmax=xmax
    def predict(self,xtest):
        predlist=self.predlist
        xmin=self.xmin
        xmax=self.xmax
        ngrid=self.ngrid
        grid_length=self.grid_length
        Ntest=len(xtest)
        ypredict=[0]*Ntest
        for i in range(Ntest):
            if xtest[i]<=xmin:
                ypredict[i]=predlist[0]
            elif xtest[i]>=xmax:
                ypredict[i]=predlist[ngrid-1]
            else:
                index=np.floor((xtest[i]-xmin)/grid_length).astype(int)
                ypredict[i]=predlist[index]
        return np.asarray(ypredict).T

        
        
        
        
        
        
        
        
        
        
        
        