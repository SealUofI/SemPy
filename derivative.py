import numpy as np

def lagrange(x):
    assert x.ndim==1

    n=x.size

    a=np.ones((n,),dtype=np.float64)

    for i in range(n):
        for j in range(i):
            a[i]=a[i]*(x[i]-x[j])
        for j in range(i+1,n):
            a[i]=a[i]*(x[i]-x[j])
    a=1.0/a
    D=np.zeros((n,n))

    for i in range(n):
        D[i,:]=a[i]*(x[i]-x)
        D[i,i]=1
    for j in range(n):
        D[:,j]=D[:,j]/a[j]
    D=1.0/D

    for i in range(n):
        D[i,i]=0
        D[i,i]=-sum(D[i,:])
    return D
