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

#x = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
#y = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

J0 = get_J0(x, y)
print(J0.T)
