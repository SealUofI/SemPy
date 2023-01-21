from sempy.kron import KroneckerProductOperator#, KroneckerProductSumOperator
import numpy as np
from scipy.sparse import csr_matrix

A = np.random.rand(2,2)
B = np.random.rand(3,3)
C = np.random.rand(4,4)

D = np.random.rand(2,2)
E = np.random.rand(3,3)
F = np.random.rand(4,4)



# Test assembly
M_kp = KroneckerProductOperator(A,B,C,order='F')
N_kp = KroneckerProductOperator(D,E,F,order='F')
M = np.kron(np.kron(A,B),C)
N = np.kron(D,np.kron(E,F))

v = np.random.rand(M.shape[0])
w = np.random.rand(N.shape[0])

#assert np.allclose(M, M_kp.to_sparse().A)
#assert np.allclose(M.diagonal(), M_kp.diagonal())
assert np.allclose(M.shape, M_kp.shape)
assert M.dtype == M_kp.dtype
assert np.allclose(M@v, M_kp@v)
assert np.allclose(M.T@v, M_kp.T@v)

#print(M.T@M)
assert np.allclose((M + N)@v, (M_kp + N_kp)@v)
