import numpy as np

def legendp(x,N, normalized=False):
    #
    #     Compute the Legendre polynomials up to degree N evaluated at points x
    #
    m=len(x)
    l=np.ones((m,N+1),order='F')
    l[:,1]=x

    for k in range(2,N+1):
        l[:,k] = ((2*k-1)*x*l[:,k-1]-(k-1)*l[:,k-2] )/k

    if normalized:
        for k in range(0,N+1):
            alpha = np.sqrt(2/(2*k+1))
            l[:,k] = l[:,k] / alpha;

    return l
