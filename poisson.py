import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,reference_2d,box_ab
from sempy.mesh import load_mesh

from sempy.mass import reference_mass_matrix_1d
from sempy.stiffness import geometric_factors,geometric_factors_2d
from sempy.stiffness import gradient,gradient_2d,gradient_transpose,\
    gradient_transpose_2d

from sempy.iterative import cg,pcg

from mayavi import mlab
import matplotlib.pyplot as plt

mesh=load_mesh("box004.msh")
print("num elements: {}".format(mesh.get_num_elements()))
print(mesh.elem_to_elem_map)
print(mesh.elem_to_face_map)

example_2d = False
plot_on    = True

N  = 15
n  = N+1

if example_2d:
    X,Y=box_ab(0.,1.,N)
    G,J,B=geometric_factors_2d(X,Y,n)
else:
    X,Y,Z=box_ab(0.,1.,N)
    G,J,B=geometric_factors(X,Y,Z,n)

def mask_2d(W):
    W=W.reshape((n,n))
    W[0  ,:]=0
    W[n-1,:]=0
    W[:,0  ]=0
    W[:,n-1]=0
    return W.reshape((n*n),)

def mask(W):
    W=W.reshape((n,n,n))
    W[0  ,:,:]=0
    W[n-1,:,:]=0
    W[:,0  ,:]=0
    W[:,n-1,:]=0
    W[:,:,0  ]=0
    W[:,:,n-1]=0
    return W.reshape((n*n*n),)

def Ax_2d(x):
    Ux,Uy=gradient_2d(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy

    W=gradient_transpose_2d(Wx,Wy,n)
    return mask_2d(W)

def Ax(x):
    Ux,Uy,Uz=gradient(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=gradient_transpose(Wx,Wy,Wz,n)
    return mask(W)

Minv=1.0/(B*J)
def precon_mass(r):
    return Minv*r

if example_2d:
    x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)
    x_analytic=mask(x_analytic.reshape((n*n,)))
    b=2*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)
    b=mask(b.reshape(n*n,)*B*J)
else:
    x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    x_analytic=mask(x_analytic.reshape((n*n*n,)))
    b=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    b=mask(b.reshape(n*n*n,)*B*J)

tol=1.e-8
maxit=1000
verbose=0

#if example_2d:
#    x_cg  ,niter_cg  =cg (Ax_2d            ,b,tol,maxit,verbose)
#    x_mass,niter_mass=pcg(Ax_2d,precon_mass,b,tol,maxit,verbose)
#else:
#    x_cg  ,niter_cg  =cg (Ax            ,b,tol,maxit,verbose)
#    x_mass,niter_mass=pcg(Ax,precon_mass,b,tol,maxit,verbose)
#
#print("     niter/error")
#print("cg  : {}/{}".format(niter_cg  ,\
#  np.abs(np.max(x_cg  -x_analytic))))
#print("mass: {}/{}".format(niter_mass,\
#  np.abs(np.max(x_mass-x_analytic))))
#
#if plot_on:
#    if example_2d:
#      print("N/A")
#    else:
#        mlab.figure()
#        mlab.points3d(X,Y,Z,(x_cg-x_analytic).reshape((n,n,n)),\
#          scale_mode="none",scale_factor=0.1)
#        mlab.axes()
#        mlab.show()
