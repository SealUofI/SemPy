import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

# Add to path so can import from above directory
import sys
sys.path.append('../')
from sempy_types import SEMPY_SCALAR

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False


import loopy_kernels as lpk

platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
#ctx = cl.Context(devices=my_gpu_devices)
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)


def cg(A,b,tol=1e-12,maxit=100,verbose=0):

    assert b.ndim==1

    m,n = A.shape

    Ax = lpk.gen_Ax_knl()
    norm = lpk.gen_norm_knl()
    cgi = lpk.gen_CG_iteration()
    Ax = lp.set_options(Ax, "write_code")
    #print(lp.generate_code_v2(Ax).device_code())

    x=np.zeros((n,),dtype=SEMPY_SCALAR)
    
    evt, (norm_b_lp,) = norm(queue,x=b)
    print(norm_b_lp)

    norm_b=np.dot(b,b)
    print(norm_b)

    TOL=max(tol*tol*norm_b,tol*tol)

    r=b

    evt, (rdotr_lp,) = norm(queue, x=r)
    print(rdotr_lp)

    rdotr=np.dot(r,r)
    print(rdotr)
    niter=0

    if verbose:
        print('Initial rnorm={}'.format(rdotr))
    if rdotr<1.e-20:
        return x,niter

    p=r

    x_lp=x.copy()
    r_lp=r.copy()
    p_lp = r_lp
    #rdotr_lp = rdotr
    #p_lp=p.copy()

    while niter<maxit and rdotr>TOL and rdotr_lp > TOL:
        niter+=1

        # We need to define this for multiple elements
        evt, (Ap_lp,) = Ax(queue, A=A, x=p_lp)

        evt, (p_lp,r_lp,rdotr_lp,x_lp) = cgi(queue, Ap=Ap_lp, p=p_lp, r=r_lp, rdotr_prev=rdotr_lp, x=x_lp)
        print("CL: {}".format(rdotr_lp))

        # Numpy version for comparison
        """
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
        print("Numpy: {}".format(rdotr))
        """

 
    return x,niter
   
##Test
A = np.float32(np.random.rand(10,10))
A += A.T
A += np.diag(np.sum(A, axis=0))
x = np.float32(np.random.rand(10))
cg(A,x)
