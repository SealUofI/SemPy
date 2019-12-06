import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

import loopy_kernels as lpk

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
#ctx = cl.Context(devices=my_gpu_devices)
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)


def cg(A,b,tol=1e-12,maxit=100,verbose=0):

    assert b.ndim==1

    n=b.size

    ip = lpk.gen_inner_prod_knl(n)
    x=np.zeros((n,),dtype=np.float64)

    
    
    evt, (norm_b_loopy,) = ip(queue,x=b,y=b.copy())
    #print(norm_b)

    norm_b=np.dot(b,b)
    print(norm_b)

    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(r,r)
    niter=0

    if verbose:
        print('Initial rnorm={}'.format(rdotr))
    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<maxit and rdotr>TOL:
        Ap=A@p
        pAp=np.dot(p,Ap)

        alpha=rdotr/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(r,r)
        beta=rdotr/rdotr0
        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}".format( \
                niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        niter=niter+1

    return x,niter
   
##Test
A = np.random.rand(10,10)
x = np.random.rand(10)
cg(A,x)
