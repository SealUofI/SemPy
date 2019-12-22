import numpy as np

from sempy.mesh import load_mesh

from sempy.quadrature import gauss_lobatto

from sempy.interpolation import lagrange

from sempy.derivative import lagrange_derivative_matrix,\
    reference_derivative_matrix

from sempy.helmholtz import helmholtz

from mayavi import mlab

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def get_ic(mesh,IC_type):
    X=mesh.get_x().copy()
    Y=mesh.get_y().copy()

    if IC_type == 'Eddy':
        # Walsh paper on RELATE
        plotf = 100
        Re = 1000
        Pe = Re*Pr;
    
        X *= np.pi
        Y *= np.pi
    
        #v = d/dx psi
        v = np.cos(3*X)*np.cos(4*Y)+np.sin(5*Y)
        #u = -d/dy psi
        u = (3./4.)*np.sin(3*X)*np.sin(4*Y)+np.cos(5*X)
    
        X /= np.pi
        Y /= np.pi

        return u,v,plotf

    elif IC_type == 'KovaFlow':
        plotf = 10
        Re = 40
        Pe = Re*Pr;
        # https://doc.nektar.info/userguide/4.3.4/user-guidese45.html
        lmbd = 0.5*Re-np.sqrt(0.25*Re*Re+4*np.pi*np.pi)
        v = 1.-np.exp(lmbd*Y)*np.cos(2*np.pi*X)
        v = np.maximum(np.zeros_like(v), v)
        u = (lmbd/2./np.pi)*np.exp(lmbd*Y)*np.sin(2*np.pi*X)
        u = np.maximum(np.zeros_like(u), u)

        return u,v,plotf
    else:
        y0 = 0; x0 = 0.5
        rr = np.sqrt(np.power(X-x0, 2) + np.power(Y-y0,2))
        u = np.maximum(np.zeros_like(rr), 1-rr*3)+v
        raise Exception("Unknown IC")

def convl(TT,uu,vv,mesh):
    E=mesh.get_num_elems()
    ndim=mesh.get_ndim()
    Nq=mesh.get_1d_dofs()
    Np=mesh.get_local_dofs()

    RX=mesh.get_derv()
    Dh=reference_derivative_matrix(Nq-1)

    if mesh.get_ndim()==3:
        T=TT.reshape((E,Nq,Nq,Nq))
        dT=np.zeros_like(TT).reshape((E,Nq*Nq*Nq))
        u=uu.reshape((E,Nq*Nq*Nq))
        v=vv.reshape((E,Nq*Nq*Nq))
        raise Exception("Not implemented yet.")
    else:
        T=TT.reshape((E,Nq,Nq))
        dT=np.zeros_like(TT).reshape((E,Nq*Nq))
        u=uu.reshape((E,Nq*Nq))
        v=vv.reshape((E,Nq*Nq))

    for e in range(E):
        Ts = (T[e,:,:]@Dh.T).reshape((Nq*Nq,))
        Tr = (Dh@T[e,:,:]  ).reshape((Nq*Nq,))

        Tx = Tr*RX[e,0,0,:] + Ts*RX[e,0,1,:]
        Ty = Tr*RX[e,1,0,:] + Ts*RX[e,1,1,:]

        dT[e,:] = u[e,:]*Tx + v[e,:]*Ty

    return dT.reshape((E*Nq*Nq))

def semhat(N):
    z,w = gauss_lobatto(N)
    Bh  = np.diag(w)
    Dh  = lagrange_derivative_matrix(z)

    Ah = Dh.T@Bh@Dh
    Ch = Bh@Dh

    return Ah,Bh,Ch,Dh,z,w

class PlotStuff:
    def __init__(self,E,N1,X,Y,Title):
        self.Title = Title
        self.E =E
        self.N1=N1

        self.Nf= 3*(N1-1)
        zf,wf  = gauss_lobatto(self.Nf)
        z ,w   = gauss_lobatto(N1-1)

        self.Jh    = lagrange(zf,z);

        Nf1    = self.Nf+1;
        self.Xf= np.zeros((self.E,Nf1,Nf1));
        self.Yf= self.Xf.copy()

        xx=X.reshape((E,N1,N1))
        yy=Y.reshape((E,N1,N1))

        for e in range(self.E):
            self.Xf[e,:,:] = self.Jh@xx[e,:,:]@self.Jh.T
            self.Yf[e,:,:] = self.Jh@yy[e,:,:]@self.Jh.T

        self.Uf=np.zeros((self.E,Nf1,Nf1))
        self.Vf=np.zeros((self.E,Nf1,Nf1))

    def plot(self,uu,vv,step):
        U=uu.reshape((self.E,self.N1,self.N1))
        V=vv.reshape((self.E,self.N1,self.N1))

        for e in range(self.E):
            self.Uf[e,:,:] = self.Jh@U[e,:,:]@self.Jh.T
            self.Vf[e,:,:] = self.Jh@V[e,:,:]@self.Jh.T

        for e in  range(self.E):
            fig, ax = plt.subplots(ncols=2,figsize=(10,4))

            ax = plt.subplot(1,2,1)
            ax.contour(self.Xf[e,:,:],self.Yf[e,:,:],self.Vf[e,:,:])
            ax.set_title("v")
            ax = plt.subplot(1,2,2)
            ax.set_title("u")
            ax.contour(self.Xf[e,:,:],self.Yf[e,:,:],self.Uf[e,:,:])

        plt.show(block=False)
        #plt.savefig('%s-%d.pdf' % (self.Title, step))
        plt.pause(0.2)
        plt.close()

N=20
n=N+1

mesh=load_mesh("quad001.msh")
mesh.find_physical_coordinates(N)
mesh.establish_global_numbering()
mesh.calc_geometric_factors()
mesh.setup_mask()
masked_ids=mesh.get_mask_ids()
global_to_local,global_start=mesh.get_global_to_local_map()

nelem=mesh.get_num_elems()

X =mesh.get_x()
Y =mesh.get_y()
Z =mesh.get_z()
G =mesh.get_geom()
J =mesh.get_jaco()
B =mesh.get_mass()
RX=mesh.get_derv()

Ah,Bh,Ch,Dh,z,w = semhat(N)

Pr = 0.8;
Np1= N + 1
E  = 1
nL = Np1*Np1*E

Q = np.identity(Np1)

# TODO: R implemented as masks, make this a function
Mask=np.zeros((Np1,Np1));
Mask[0,:]=1;

# compute dt according to CFL
dxmin = np.pi*(z[Np1-1]-z[N-1])/(2*E); # Get min dx for CFL constraint
Tmax  = 150;
CFL   = 0.3;
dt    = CFL*dxmin;
nSteps= int(np.ceil(Tmax/dt))
dt    = Tmax/nSteps

u= np.zeros_like(X);
v= np.zeros_like(X);

IC_type = 'KovaFlow'
u,v,plotf=get_ic(mesh,IC_type)

ps=PlotStuff(1,Np1,X,Y,IC_type)
ps.plot(u,v,0)

u2 =u.copy(); u3 =u.copy();
v2 =v.copy(); v3 =v.copy();
fy3=v.copy(); fy2=v.copy(); fy1 = v.copy();
fx3=u.copy(); fx2=u.copy(); fx1 = u.copy();

for step in range(nSteps):
    print("Step {}: ".format(step))

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
        b0i=1.0/b0

    vc= X
    uc=-Y

    #   Nonlinear step - unassembled, not multiplied by mass matrix
    fx1 = -convl(u,uc,vc,mesh); # du = Cu  (without mass matrix)
    fy1 = -convl(v,uc,vc,mesh); # dv = Cv

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
    ut = b0i*rx;
    vt = b0i*ry;

    uL = ut
    vL = vt

"""
 # divergence free pressure update
 # check the HW writeup for details
 # uL,vL, pr = pressure_project(uL, vL, );

    Qfudge     = b0*Bh # Q is not the same as gather/scatter
    Qfudge_inv = np.linalg.inv(Qfudge)
    E_inv      = np.linalg.inv(Dh*Qfudge_inv*Dh.T)

    def Hfun(vec):
        # helmhotlz_ax
        tmp = (vec).reshape((nL,), order="F")
        vec = H.dot(tmp)
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
    B = B.reshape((Np1,Np1), order="F")
    tmp = np.array((B*uL).reshape((nL,), order="F")).flatten()
    u = Mask*tmp#Pointwise multiplication
    tmp = np.array((B*vL).reshape((nL,), order="F")).flatten()
    #v   = R*(Q.T*tmp)
    v = Mask*tmp
    # Viscous update.
    u = LUH.solve(u)
    #u   = Q*(R.T*u)
    v = LUH.solve(v)
    #v   = Q*(R.T*v)
    #Convert to local form.
    u = u.reshape((Np1,Np1,E), order="F");
    v = v.reshape((Np1,Np1,E), order="F")


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

"""
