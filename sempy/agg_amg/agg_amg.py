from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
from sempy.agg_amg.project import project

import numpy as np
np.set_printoptions(threshold=np.inf)

pts = np.loadtxt("pts.dat")
tri = np.loadtxt("tri.dat", dtype=np.int32) - 1

AL, BL, Q, R, xb, yb, p, t = stiffness_mat(pts, tri)

# print(np.linalg.norm(AL.todense()))
# print(np.linalg.norm(BL.todense()))
# print(np.linalg.norm(Q.todense()))
# print(np.linalg.norm(R.todense()))
# exit()

# print(AL)

Ab = Q.T.dot(AL.dot(Q))
A = R.dot(Ab.dot(R.T))
Bb = Q.T.dot(BL.dot(Q))
B = R.dot(Bb.dot(R.T))
x = R.dot(xb)
y = R.dot(yb)

# print(np.linalg.norm(Ab.todense()))
# print(np.linalg.norm(A.todense()))
# print(np.linalg.norm(Bb.todense()))
# print(np.linalg.norm(B.todense()))
# exit()

J0 = get_J0(x, y)
n, ncuts = J0.shape
print("J0: {} x {}".format(n, ncuts))

rr = x**2 + y**2  # np.multiply(x, x)+np.multiply(y, y)
#Arr = A.dot(rr)
# print(np.linalg.norm(np.cumsum(Arr)))
# exit()
uex = 1.0 - rr
f = 4*B.dot(np.ones((n, 1)))
print("norm rr: {}, Ar: {} uex: {}, f:{} {}".format(
    np.linalg.norm(rr), np.linalg.norm(A.dot(rr)), np.linalg.norm(uex), np.linalg.norm(f), f.shape))

tol = 1e-8
prec = 1
u, res, n_iter = project(f, A, J0, tol, prec, verbose=1)
print("n_iter: {} norm x: {}".format(n_iter, np.linalg.norm(u)))
