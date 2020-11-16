from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
import numpy as np

pts = np.loadtxt("pts.dat")
tri = np.loadtxt("tri.dat", dtype=np.int32) - 1

AL, BL, Q, R, xb, yb, p, t = stiffness_mat(pts, tri)

Ab = Q.T.dot(AL.dot(Q))
A = R.dot(Ab.dot(R.T))
Bb = Q.T.dot(BL.dot(Q))
B = R.dot(Bb.dot(R.T))
x = R.dot(xb)
y = R.dot(yb)

J0 = get_J0(x, y)
n, ncuts = J0.shape
print("J0: {} x {}".format(n, ncuts))

rr = np.multiply(x, x)+np.multiply(y, y)
uex = 1.0 - rr
f = 4*B.dot(np.ones(n,))
print("norm rr: {}, uex: {}, f:{}".format(
    np.linalg.norm(rr), np.linalg.norm(uex), np.linalg.norm(f)))
