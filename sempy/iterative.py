import numpy as np

def cg(A,b,tol=1e-8):
    assert b.ndim==1

    n=b.size
    x=np.zeros((n,),dtype=np.float64)

    r=b-A(x)
    p=r
    r_init=np.dot(r,r)
    tol2=tol*tol

    niter=1
    r_old=r_init

    for i in range(n):
        Ap=A(p)
        alpha=r_old/np.dot(p,Ap)

        x=x+alpha*p
        r=r-alpha*Ap

        r_new=np.dot(r,r)
        if r_new*r_new < tol2*r_init*r_init:
            return x,niter

        p=r+(r_new/r_old)*p
        r_old=r_new

    return x,niter
