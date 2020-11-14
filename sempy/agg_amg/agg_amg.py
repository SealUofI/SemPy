from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
import numpy as np

pts = np.loadtxt("pts.dat")
tri = np.loadtxt("tri.dat",dtype=np.int32) - 1
   

AL, BL, Q, R, xb, yb, p, t =  stiffness_mat(pts,tri)
J0 = get_J0(x, y)
