import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,reference_2d,box_ab

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.iterative import cg,pcg

from sempy.derivative import reference_derivative_matrix

from sempy.mass import reference_mass_matrix_1d

from mayavi import mlab
import matplotlib.pyplot as plt

import time

example_2d=0
plot_on=0

elapsed_cg =[]
elapsed_fdm=[]
niters_cg =[]
niters_fdm=[]
orders =[]

for N in range(2,10):
    n=N+1
    
    if example_2d:
        X,Y=reference_2d(N)
        G,J,B=geometric_factors_2d(X,Y,n)
    else:
        X,Y,Z=trapezoid(N)
        G,J,B=geometric_factors(X,Y,Z,n)
    
    def fdm_d_inv_2d(p):
        n=p+1
    
        Bh=reference_mass_matrix_1d(p);
        Dh=reference_derivative_matrix(p)
        Ah=Dh.T@Bh@Dh; Ah=0.5*(Ah+Ah.T)
    
        I=np.identity(n,dtype=np.float64)
        Ry=I[1:,:]
    
        Ax=Rx@Ah@Rx.T; nx=Ax.shape[0]
        Ay=Ry@Ah@Ry.T; ny=Ay.shape[0]
        Bx=Rx@Bh@Rx.T
        By=Ry@Bh@Ry.T
    
        Lx,Sx=sla.eigh(Ax,Bx); Lx=np.diag(Lx); Ix=np.identity(nx,dtype=np.float64)
        Ly,Sy=sla.eigh(Ay,By); Ly=np.diag(Ly); Iy=np.identity(ny,dtype=np.float64)
    
        Lx=Lx.real
        Ly=Ly.real
    
        D=np.kron(Iy,Lx)+np.kron(Ly,Ix)
        dinv=1.0/np.diag(D)
    
        return Rx,Ry,Sx.real,Sy.real,dinv
    
    def fdm_d_inv(p):
        n=p+1
    
        Bh=reference_mass_matrix_1d(p);
        Dh=reference_derivative_matrix(p)
        Ah=Dh.T@Bh@Dh; Ah=0.5*(Ah+Ah.T)
    
        I=np.identity(n,dtype=np.float64)
        Rx=Rz=I
        Ry=I[1:,:]
    
        Ax=Rx@Ah@Rx.T; nx=Ax.shape[0]
        Ay=Ry@Ah@Ry.T; ny=Ay.shape[0]
        Az=Rz@Ah@Rz.T; nz=Az.shape[0]
        Bx=Rx@Bh@Rx.T
        By=Ry@Bh@Ry.T
        Bz=Rz@Bh@Rz.T
    
        Lx,Sx=sla.eigh(Ax,Bx); Lx=np.diag(Lx); Ix=np.identity(nx,dtype=np.float64)
        Ly,Sy=sla.eigh(Ay,By); Ly=np.diag(Ly); Iy=np.identity(ny,dtype=np.float64)
        Lz,Sz=sla.eigh(Az,Bz); Lz=np.diag(Lz); Iz=np.identity(nz,dtype=np.float64)
    
        Lx=Lx.real
        Ly=Ly.real
        Lz=Lz.real
    
        D=np.kron(Iz,np.kron(Iy,Lx))+np.kron(Iz,np.kron(Ly,Ix))+\
            np.kron(Lz,np.kron(Iy,Ix))
        dinv=1.0/np.diag(D)
    
        return Rx,Ry,Rz,Sx.real,Sy.real,Sz.real,dinv
    
    def mask(W):
        return np.dot(R.T,np.dot(R,W))
    
    def Ax_2d(x):
        Ux,Uy=gradient_2d(x,n)
    
        Wx=G[0,0,:]*Ux+G[0,1,:]*Uy
        Wy=G[1,0,:]*Ux+G[1,1,:]*Uy
    
        W=gradient_transpose_2d(Wx,Wy,n)
        return mask(W)
    
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
    
    def fast_kron_2d(Sy,Sx,U):
        nx,mx=Sx.shape
        ny,my=Sy.shape
    
        U=U.reshape((my,mx))
        U=np.dot(U,Sx.T)
    
        V=U.reshape((my,nx))
        U=np.dot(Sy,V)
    
        return U.reshape((nx*ny,))
    
    def fast_kron(Sz,Sy,Sx,U):
        nx,mx=Sx.shape
        ny,my=Sy.shape
        nz,mz=Sz.shape
    
        U=U.reshape((my*mz,mx))
        U=np.dot(U,Sx.T)
    
        U=U.reshape((mz,my,nx))
        V=np.zeros ((mz,ny,nx))
        for i in range(mz):
            V[i,:,:]=np.dot(Sy,U[i,:,:])
    
        V=V.reshape((mz,nx*ny))
        U=np.dot(Sz,V)
    
        return U.reshape((nx*ny*nz,))
    
    if example_2d:
        Rx,Ry,Sx,Sy,dinv=fdm_d_inv_2d(N)
        R=np.kron(Ry,Rx)
    else:
        Rx,Ry,Rz,Sx,Sy,Sz,dinv=fdm_d_inv(N)
        R=np.kron(Rz,np.kron(Ry,Rx))
    
    def precon_fdm_2d(r):
        r=np.dot(R,r)
        b=fast_kron_2d(Sy.T,Sx.T,r)
        b=dinv*b
        return np.dot(R.T,fast_kron_2d(Sy,Sx,b))
    
    def precon_fdm(r):
        r=np.dot(R,r)
        b=fast_kron(Sz.T,Sy.T,Sx.T,r)
        b=dinv*b
        return np.dot(R.T,fast_kron(Sz,Sy,Sx,b))
    
    def precon_jacobi_2d(r):
        r=np.dot(R,r)
        b=fast_kron_2d(Sy.T,Sx.T,r)
        b=dinv*b
        return np.dot(R.T,fast_kron_2d(Sy,Sx,b))
    
    def precon_jacobi(r):
        r=np.dot(R,r)
        b=fast_kron(Sz.T,Sy.T,Sx.T,r)
        b=dinv*b
        return np.dot(R.T,fast_kron(Sz,Sy,Sx,b))
    
    if example_2d:
        b=np.exp(10*Y)*np.sin(10*X)
        b=mask(b.reshape((n*n,))*B*J)
    else:
        b=np.exp(10*Y*Z)*np.sin(10*X)
        b=mask(b.reshape((n*n*n,))*B*J)
    
    tol=1.e-8
    maxit=1000
    verbose=0
    
    t=time.process_time() 

    if example_2d:
        x_cg    ,niter_cg    =cg (Ax_2d                 ,b,tol,maxit,verbose)
        x_mass  ,niter_mass  =pcg(Ax_2d,precon_mass     ,b,tol,maxit,verbose)
        x_jacobi,niter_jacobi=pcg(Ax_2d,precon_jacobi_2d,b,tol,maxit,verbose)
        x_fdm   ,niter_fdm   =pcg(Ax_2d,precon_fdm_2d   ,b,tol,maxit,verbose)
    else:
        x_cg    ,niter_cg    =cg (Ax              ,b,tol,maxit,verbose)
        tt=time.process_time()-t
        elapsed_cg.append(tt)
    #    x_mass  ,niter_mass  =pcg(Ax,precon_mass  ,b,tol,maxit,verbose)
    #    x_jacobi,niter_jacobi=pcg(Ax,precon_jacobi,b,tol,maxit,verbose)
        t=time.process_time() 
        x_fdm   ,niter_fdm   =pcg(Ax,precon_fdm   ,b,tol,maxit,verbose)
        tt=time.process_time()-t
        elapsed_fdm.append(tt)

    niters_fdm.append(niter_fdm)
    niters_cg.append(niter_cg)
    orders.append(N)


plt.figure()
plt.plot(orders,elapsed_cg,'-o')
plt.title("Order vs Elapsed time for CG",fontsize=20)
plt.xlim(1,N+1)
plt.xlabel("N - order",fontsize=16)
plt.ylabel("time (s)" ,fontsize=16)
plt.savefig('elapsed_cg.pdf',bbox_inches='tight')

plt.figure()
plt.plot(orders,niters_cg,'-o')
plt.title("Order vs number of iterations for CG",fontsize=20)
plt.xlim(2,N+1)
plt.ylabel("# iterations",fontsize=16)
plt.xlabel("N - order"   ,fontsize=16)
plt.savefig('niter_cg.pdf',bbox_inches='tight')

plt.figure()
plt.semilogy(orders,elapsed_fdm,'b-o',label='pcg(fdm)')
plt.semilogy(orders,elapsed_cg,'g-o',label='cg')
plt.title("Order vs Elapsed time",fontsize=20)
plt.xlim(1,N+1)
plt.xlabel("N - order",fontsize=16)
plt.ylabel("time (s)" ,fontsize=16)
plt.legend(loc=0)
plt.savefig('elapsed_fdm_cg.pdf',bbox_inches='tight')

plt.figure()
plt.semilogy(orders,niters_fdm,'b-o',label='pcg(fdm)')
plt.semilogy(orders,niters_cg ,'g-o',label='cg')
plt.title("Order vs number of iterations for FDM",fontsize=20)
plt.xlim(2,N+1)
plt.ylabel("# iterations",fontsize=16)
plt.xlabel("N - order"   ,fontsize=16)
plt.legend(loc=0)
plt.savefig('niter_fdm_cg.pdf',bbox_inches='tight')

"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from plot_stuff import *
# SEM stuff
#from make_q          import make_Q
#from make_r          import make_R
from setup_geometry  import setup_geometry
from make_coef       import make_coef
from semhat          import semhat
from abar            import abar
from zwgll           import zwgll
from interp_mat      import *
from convl           import *
"""

#Ra = 120000;
Pr = 0.8;
N  = 30; Np1 = N + 1
E  = 1
nL = Np1*Np1*E

periodic=False
Q    =  sp.eye(Np1*Np1)#Assume one element for now #make_Q(E, N, periodic=periodic)
#R    =  np.eye(Np1*Np1)#make_R(Q,E,N, periodic=periodic)
#R[0,0] = 0; R[-1,-1] = 0
Mask = np.ones((Np1, Np1)); Mask[0,:]= 0; Mask[-1,:] = 0; Mask[:,-1] = 0; Mask[:,0] = 0;
uMask = Mask.copy()
uMask = uMask.flatten(order='F')
vMask = Mask.copy()
vMask = vMask.flatten(order='F')
#IMask = Mask.copy(); #Add top condition for lid driven cavity
print(Mask)

X, Y = setup_geometry(E, N)

G,J,ML,RX       = make_coef(X,Y)
Ah,Bh,Ch,Dh,z,w = semhat(N)


# Form A-bar
n, nb = Mask.shape#R.shape
nl = Np1*Np1*E

Ab = Ah
'''
Ab_dense = np.zeros((nb,nb))
for j in range(nb):
    x = np.zeros((nb,))
    x[j] = 1
    Ab_dense[:,j] = abar(x,Dh,G,Q).round(18)
Ab = sp.csr_matrix(Ab_dense)
'''

A = sp.kron(Ab,Ab)#Ab #A  = R*Ab*R.T
Bb = ML.reshape((nl,),  order="F")
#Bb = np.diag(Bb)
#Bb = sp.diags(Bb, 0)
#Bb = Q.T*Bb*Q
Ma = Bb#Ma = R*Bb*R.T

# compute dt according to CFL
dxmin  = np.pi*(z[Np1-1]-z[N-1])/(2*E); # Get min dx for CFL constraint
Tmax   = 150;
CFL    = 0.3;
dt     = CFL*dxmin;
nSteps = int(np.ceil(Tmax/dt))
dt     = Tmax/nSteps

u   = 0*ML;
v   = 0*ML;

IC_type = 'KovaFlow'

if IC_type == 'Eddy':
    plotf = 100
    # https://relate.cs.illinois.edu/course/tam570-f19/f/notes/walsh_1992.pdf
    Re = 1000 #1e30 #np.sqrt(Ra);
    Pe = Re*Pr;

    X *= np.pi
    Y *= np.pi

    #v = d/dx psi
    v = np.cos(3*X)*np.cos(4*Y)+np.sin(5*Y)
    #u = -d/dy psi
    u = (3./4.)*np.sin(3*X)*np.sin(4*Y)+np.cos(5*X)

    X /= np.pi
    Y /= np.pi
elif IC_type == 'KovaFlow':
    plotf = 10
    Re = 40 #1e30 #np.sqrt(Ra);
    Pe = Re*Pr;
    # https://doc.nektar.info/userguide/4.3.4/user-guidese45.html
    lmbd = 0.5*Re-np.sqrt(0.25*Re*Re+4*np.pi*np.pi)
    v = 1.-np.exp(lmbd*Y)*np.cos(2*np.pi*X)
    v = np.maximum(np.zeros_like(v), v)
    u = (lmbd/2./np.pi)*np.exp(lmbd*Y)*np.sin(2*np.pi*X)
    u = np.maximum(np.zeros_like(u), u)
else:
    y0 = 0; x0 = 0.5
    rr = np.sqrt(np.power(X-x0, 2) + np.power(Y-y0,2))
    u = np.maximum(np.zeros_like(rr), 1-rr*3)+v

ps = PlotStuff(u, X, Y, IC_type)
utmp = (Q*Q.T*u.reshape((Np1*Np1*E,))).reshape((Np1, Np1, E), order="F")
vtmp = (Q*Q.T*v.reshape((Np1*Np1*E,))).reshape((Np1, Np1, E), order="F")
#ps.plot(utmp, X, Y, 0)
ps.plot(utmp, vtmp, X, Y, 0)

u2  = u.copy(); u3  = u.copy();
v2  = v.copy(); v3  = v.copy();
fy3 = v.copy(); fy2 = v.copy(); fy1 = v.copy();
fx3 = u.copy(); fx2 = u.copy(); fx1 = u.copy();

for step in range(nSteps):

    if step == 0:
        b0 = 1.0
        b  = np.array([-1, 0, 0])
        a  = np.array([ 1, 0, 0])
    elif step == 1:
        b0 = 1.5
        b  = np.array([-4, 1, 0])/2
        a  = np.array([ 2,-1, 0])
    elif step == 2:
        b0 = 11./6.
        b  = np.array([-18, 9, -2])/6
        a  = np.array([  3,-3,  1])

    if step<=3:
        #A = np.eye(Np1*Np1) # Overwrite for now to get past
        H     = (Ma+ A*dt/(b0*Re))
        LUH   = splu(H)
        b0i=1./b0
    vc = X
    uc = -Y

    #   Nonlinear step - unassembled, not multiplied by mass matrix
    fx1 = -convl(u,RX,Dh,uc,vc); # du = Cu  (without mass matrix)
    fy1 = -convl(v,RX,Dh,uc,vc); # dv = Cv


    rx  = a[0]*fx1+a[1]*fx2+a[2]*fx3; # kth-order extrapolation
    ry  = a[0]*fy1+a[1]*fy2+a[2]*fy3;
    fx3 = fx2; fx2 = fx1
    fy3 = fy2; fy2 = fy1

    # Add BDF terms and save old values
    rx  = dt*rx - (b[0]*u+b[1]*u2+b[2]*u3)
    u3  = u2
    u2  = u
    ry  = dt*ry - (b[0]*v+b[1]*v2+b[2]*v3)
    v3  = v2
    v2  = v

    # Tentative fields
    ut  = b0i*rx;
    vt = b0i*ry;

    uL = ut
    vL = vt

 # divergence free pressure update
 # check the HW writeup for details
 #   uL,vL, pr = pressure_project(uL, vL, );
    Qfudge     = b0*Bh # Q is not the same as gather/scatter
    Qfudge_inv = np.linalg.inv(Qfudge)
    E_inv      = np.linalg.inv(Dh*Qfudge_inv*Dh.T)
    def Hfun(vec):
        tmp = (vec).reshape((nL,), order="F")
        #tmp = R*(Q.T*tmp)
        vec = H.dot(tmp)
       #vec = Q*(R.T*vec)
        vec = vec.reshape((Np1,Np1,), order="F");
        return vec

    u  = u.reshape((u.shape[0], u.shape[1],))
    dp = -(1./dt)*E_inv*Dh*u
    u  = u+dt*Hfun(Qfudge*Dh.T*dp)
    uL  = u.reshape((u.shape[0], u.shape[1],1))

    v  = v.reshape((v.shape[0], v.shape[1],))
    dp = -(1./dt)*E_inv*Dh*v
    v  = v+dt*Hfun(Qfudge*Dh.T*dp)
    vL  = v.reshape((v.shape[0], v.shape[1],1))

    # Set RHS.
    ML = ML.reshape((Np1,Np1), order="F")
    tmp = np.array((ML*uL).reshape((nL,), order="F")).flatten()
    u = uMask*tmp#Pointwise multiplication
    tmp = np.array((ML*vL).reshape((nL,), order="F")).flatten()
    #v   = R*(Q.T*tmp)
    v = vMask*tmp
    # Viscous update.
    u   = LUH.solve(u)
    #u   = Q*(R.T*u)
    v   = LUH.solve(v)
    #v   = Q*(R.T*v)
    #Convert to local form.
    u   = u.reshape((Np1,Np1,E), order="F");
    v   = v.reshape((Np1,Np1,E), order="F")


    #if step == 100:
    #    exit()

    if step == 0:
        print('step\tvin\tvmax\tumin\tumax')
    if step % plotf == 0:
        utmp = (Q*Q.T*u.reshape((Np1*Np1*E,))).reshape((Np1, Np1, E), order="F")
        vtmp = (Q*Q.T*v.reshape((Np1*Np1*E,))).reshape((Np1, Np1, E), order="F")
        #ps.plot(vtmp, X, Y, step+1)
        ps.plot(utmp, vtmp, X, Y, step+1)
        print("%d\t%f\t%f\t%f\t%f" %
                (step, min(v.flatten()), max(v.flatten()),
                    min(u.flatten()), max(u.flatten())))



#if plot_on:
#    if example_2d:
#      print("N/A")
#    else:
#        mlab.figure()
#        mlab.points3d(X,Y,Z,(x_cg-x_fdm).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
#        mlab.axes()
#        mlab.show()
