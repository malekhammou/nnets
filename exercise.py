import numpy as np
import sys
learning_factor=0.3
#PROCESSING
a=[[1,0,0],[0,1,0],[0,0,1]]
c=[[0,1],[1,0]]
entries=[a,a,c]
matrix=[]
for i in range(len(a)):
    for j in range(len(a)):
        for k in range(len(c)):
            column_vector=a[i]+a[j]+c[k]
            matrix.append(column_vector)

X=np.asarray(matrix)
T=np.asarray(matrix)
for elem in (T):
    if(elem[0]==1 and elem[3]==1 and elem[6]==1):
        pass
    else:
     elem[6] = 0 if elem[6]==1 else 1
     elem[7] = 0 if elem[7] == 1 else 1
X=X.transpose()
T=T.transpose()
#TRAINING_FUNCTION
def train(r,c,x,t):
    #Generate random weights matrix
    w=np.random.rand(r,c)
    #Add Epsilon to make T different from O
    o=t+sys.float_info.epsilon
    epoch=0
    while (np.array_equal(t,o)==False):
        a=np.matmul(np.transpose(w),x)
        o=np.where(a>0,1,0)
        print("--- Epoch N°", epoch + 1, "---", "\n", o)
        w=w+learning_factor*(np.matmul(x,(np.transpose(t-o))))
        epoch+=1

    print("--- Weights Matrix ---","\n",np.asarray(w))



train(8,8,X,T)

