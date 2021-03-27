import numpy as np
import sys
import math
import random
learning_factor=0.3
entry=[[1,0,1],[1,1,0],[0,0,1],[0,1,1]]
theoretical_output=[[0,0,1],[1,1,1],[0,1,0]]
def train(r,c,x,t):
    #Transform X and T to numpy arrays
    t = np.asarray(t)
    x = np.asarray(x)
    #Generate random weights matrix
    w=np.random.rand(r,c)
    #Add Epsilon to make T different from O
    o=t+sys.float_info.epsilon
    epoch=0
    while (np.array_equal(t,o)==False):
        a = np.matmul(np.transpose(w), x)
        ro = random.uniform(0, 1)
        probability=1/(1+np.exp(-a))
        o=np.where(probability>ro,1,0)
        print("--- Epoch N°", epoch + 1, "---", "\n", o)
        w=w+learning_factor*(np.matmul(x,(np.transpose(t-o))))
        epoch+=1

    print("--- Weights Matrix ---","\n",w)



train(4,3,entry,theoretical_output)



