import numpy as np

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

def elliptic_ax(mesh,p):
    nelem=mesh.get_num_elems()
    Np=mesh.Np
    Nq=mesh.Nq

    p=p.reshape((nelem,Np))
    for e in range(nelem):
        g=mesh.geom[e,:]
        px,py,pz=gradient(p[e,:],Nq)

        pu=g[0,0,:]*px+g[0,1,:]*py+g[0,2,:]*pz
        pv=g[1,0,:]*px+g[1,1,:]*py+g[1,2,:]*pz
        pw=g[2,0,:]*px+g[2,1,:]*py+g[2,2,:]*pz

        p[e,:]=gradient_transpose(pu,pv,pw,Nq)

    mesh.apply_mask(p)
    return p.reshape((nelem*Np,))

def elliptic_cg(mesh,b,tol=1e-12,maxit=100,verbose=0):
    rmult=mesh.get_rmult()
    norm_b=np.dot(np.multiply(rmult,b),b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(np.multiply(rmult,r),r)
    if verbose:
        print('Initial rnorm={}'.format(rdotr))

    x=0*b
    niter=0
    if rdotr<1.e-20:
        return x,niter

    return

    p=r
    while niter<maxit and rdotr>TOL:
        # TODO: do this
        Ap=elliptic_ax(mesh,p)
        mesh.dssum(Ap)

        pAp=np.dot(np.multiply(rmult,p),Ap)

        alpha=rdotr/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(np.multiply(rmult,r),r)

        beta=rdotr/rdotr0

        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}"\
                .format(niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        niter=niter+1

    return x,niter
