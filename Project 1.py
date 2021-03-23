#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import matplotlib.pyplot as plt
import random as rd
from datetime import datetime
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


#### Initialisation
T_W = np.array([1/6,1/3,1/2,2/3,5/6])
x_1 = np.array([1,0,0,0,0])
x_2 = np.array([0,1,0,0,0])
x_3 = np.array([0,0,1,0,0])
x_4 = np.array([0,0,0,1,0])
x_5 = np.array([0,0,0,0,1])
x = [x_1,x_2,x_3,x_4,x_5]

#### Build Simulations
rd.seed(datetime.now())
num = 100
trial = 10
sequences = []
sequence = []
for i in range(num):
    sequence=[]
    for j in range(trial):
        seq = []
        s = 3
        while 0<s<6:
            seq.append(s)
            if rd.random() < 0.5:
                s = s-1
            else:
                s = s+1            
        if seq[-1] ==5:
            seq.append(6)
        else:
            seq.append(0)
        sequence.append(seq)
    sequences.append(sequence)


# In[60]:


#### Functions to solve each figure

def F3(alpha,l,sequences,T_W,convergence):
    res = []
    for sequence in sequences:
        # initialise w
        w = np.array([0.5,0.5,0.5,0.5,0.5])
        epsilon = 1
        # loop through until convergence
        while epsilon > convergence:
            delta_w = np.zeros(5)
            for seq in sequence:
                m = len(seq)-1
                r = int(seq[-1] == 6)
                for t in range(1,m+1):
                    sumGrad = np.zeros(5)
                    for k in range(1,t+1):
                        grad = np.power(l,t-k) * x[seq[k-1]-1]
                        sumGrad += grad
                    if t == m:
                        delta_wt = alpha * (r-w[seq[t-1]-1]) * sumGrad   
                    else:
                        delta_wt = alpha * (w[seq[t]-1]-w[seq[t-1]-1]) * sumGrad
                    delta_w += delta_wt
            w = w+delta_w 
            epsilon = np.amax(np.absolute(delta_w ))
        # Calculate RMSE for each training set
        res.append(np.sqrt(np.sum((T_W-w)**2)/5))
    return np.average(res)


def F4(alpha,l,sequences,T_W):
    res = []
    for sequence in sequences:
        # initialise w
        w = np.array([0.5,0.5,0.5,0.5,0.5]) 
        for seq in sequence:
            m = len(seq)-1
            z = int(seq[-1] == 6)
            # sum delta_wt from initial state to termination state
            delta_wt_sum = np.zeros(5)
            for t in range(1,m+1):
                sumGrad = np.zeros(5)
                for k in range(1,t+1):
                    grad = np.power(l,t-k) * x[seq[k-1]-1]
                    sumGrad += grad 
                if t == m:
                    delta_wt = alpha * (z-w[seq[t-1]-1]) * sumGrad   
                else:
                    delta_wt = alpha * (w[seq[t]-1]-w[seq[t-1]-1]) * sumGrad
                delta_wt_sum += delta_wt
            # Update w vector after each trial
            w = w + delta_wt_sum

        # Calculate RMSE for each training set
        res.append(np.sqrt(np.sum(np.power(T_W-w,2))/5))
    return np.average(res)


# In[61]:


#### Figure 3
alpha = 0.01
lamdas = np.array([0,0.1,0.3,0.5,0.7,0.9,1])

rmse_lamdas = []
for l in lamdas:
    print(l)
    rmse_lamdas.append(F3(alpha,l,sequences,T_W,0.0001))

plt.figure(figsize=(10,8))
plt.plot(lamdas,rmse_lamdas,'-o')
plt.ylabel('ERROR',size=24)
plt.xlabel('λ',size=24)
plt.annotate('Widrow-Hoff',(0.75, rmse_lamdas[-1]),fontsize=20)
plt.show()


# In[65]:



#### Figure 4
lamdas = np.arange(0.0, 1.1, 0.1)
alphas = np.arange(0.0, 0.65, 0.05)
errors = np.zeros([len(lamdas),len(alphas)])

for l in range(len(lamdas)):
    for a in range(len(alphas)):          
        errors[l][a] = F4(alphas[a],lamdas[l],sequences,T_W)

plt.figure(figsize=(10,8))
plt.plot(alphas,errors[0],'-o',label='λ = {}'.format(lamdas[0]))
plt.plot(alphas,errors[3],'-o',label='λ = {}'.format(lamdas[3]))
plt.plot(alphas,errors[8],'-o',label='λ = {}'.format(lamdas[8]))
plt.plot(alphas,errors[10],'-o',label='λ = {} (Widrow-Hoff)'.format(lamdas[10]))
plt.ylim(0.0,0.75)
plt.xlim(-0.05,0.65)
plt.ylabel('ERROR',size=20)
plt.xlabel('α',size=20)
plt.legend()
plt.show()


# In[64]:


#### Figure 5

best_error = np.min(errors,axis=1)
plt.figure(figsize=(10,8))
plt.plot(lamdas,best_error,'-o')
plt.ylabel('ERROR USING BEST α',size=24)
plt.xlabel('λ',size=24)
plt.annotate('Widrow-Hoff',(0.75,best_error[-1]),fontsize=20)
plt.show()


# In[ ]:




