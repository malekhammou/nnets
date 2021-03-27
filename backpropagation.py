import numpy as np
from numpy import linalg as LA
import sys
learning_factor=0.3
entry=[1,2,3]
theoretical_output=[0.1,0.3,0.7]
boltzmann=lambda x: 1/(1+np.exp(-x))
normalize=np.vectorize(boltzmann)


def train(wr,wc,zr,zc,x,t):
    # Transform X and T to numpy arrays
    t = np.asarray(t)
    x = np.asarray(x)

    # Generate random weights matrix W and Z

    W = np.random.rand(wr,wc)
    Z=np.random.rand(zr,zc)

    # Add Epsilon to make T different from O
    o = t + sys.float_info.epsilon
    while(LA.norm(t-o)>sys.float_info.epsilon):
        b=np.matmul(x,np.transpose(W))
        h = normalize(b)
        a = np.matmul(h,np.transpose(Z))
        o = normalize(a)
        error = t-o
        output_error=np.matmul(o,1-o,error)
        reshaped_output_error=output_error.reshape(3,1)
        Z=Z+learning_factor*reshaped_output_error* (np.transpose(h))
        hidden_error = np.matmul(h, 1 - h, (np.transpose(Z) * output_error))
        W = W + learning_factor * hidden_error* (np.transpose(x))
        print(o)
train(2,3,3,2,entry,theoretical_output)
