import numpy as np
from sempy.quadrature import gauss_lobatto

def reference_mass_matrix_1D(p):
    z,w=gauss_lobatto(p)
    return np.diag(w)

def reference_mass_matrix_3D(p):
    z,w=gauss_lobatto(p)

    n=p+1
    B=np.zeros((n*n*n,),dtype=np.float64)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                B[k*n*n+j*n+i]=w[i]*w[j]*w[k]

    return B
