# Add to path so can import from above directory
import sys
from warnings import catch_warnings, filterwarnings

import loopy as lp
import loopy.options
import loopy_kernels as lpk
import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.kernel.data import AddressSpace
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from sempy_types import SEMPY_SCALAR

sys.path.append("../")

# setup
# -----
lp.set_caching_enabled(False)
filterwarnings("error", category=lp.LoopyWarning)
loopy.options.ALLOW_TERMINAL_COLORS = False


platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
# ctx = cl.Context(devices=my_gpu_devices)
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)


def cg(A, b, tol=1e-12, maxit=100, verbose=0):

    assert b.ndim == 1

    m, n = A.shape

    Ax = lpk.gen_Ax_knl()
    norm = lpk.gen_norm_knl()
    ip = lpk.gen_inner_product_knl()
    cgi = lpk.gen_CG_iteration()
    vupdt = lpk.gen_inplace_xpay_knl()
    axpy = lpk.gen_inplace_axpy_knl()
    # Ax = lp.set_options(Ax, "write_code")
    # print(lp.generate_code_v2(Ax).device_code())

    x = np.zeros((n,), dtype=SEMPY_SCALAR)

    evt, (norm_b_lp,) = norm(queue, x=b)
    print(norm_b_lp)

    norm_b = np.dot(b, b)
    print(norm_b)

    TOL = max(tol * tol * norm_b, tol * tol)

    r = b

    rdotr = np.dot(r, r)
    print(rdotr)
    niter = 0

    if verbose:
        print("Initial rnorm={}".format(rdotr))
    if rdotr < 1.0e-20:
        return x, niter

    p = r

    x_lp = cl.array.to_device(queue, x)  # x.copy()
    r_lp = cl.array.to_device(queue, r)  # r.copy()
    p_lp = r_lp.copy()
    A_lp = cl.array.to_device(queue, A)
    # A_lp = A.copy()
    evt, (rdotr_lp,) = norm(queue, x=r_lp)
    rdotr_lp = rdotr_lp.get()

    # rdotr_lp = rdotr
    # p_lp=p.copy()

    while niter < maxit and rdotr > TOL and rdotr_lp > TOL:
        niter += 1
        """
        <> a = rdotr_prev / sum(j, p[j]*Ap[j]) {id=a}
        x[l] = x[l] + a*p[l] {id=x, dep=a}
        r[l] = r[l] - a*Ap[l] {id=r, dep=a}
        rdotr = sum(k, r[k]*r[k]) {id=rdotr, dep=r}
        p_out[i] = r[i] + (rdotr/rdotr_prev) * p[i] {id=p, dep=rdotr}
        """

        # We need to define this for multiple elements
        # p_lp=p; rdotr_lp = rdotr #Delete this line when finished

        evt, (Ap_lp,) = Ax(queue, A=A_lp, x=p_lp)
        # evt, (p_lp,r_lp,rdotr_lp,x_lp) = cgi(queue, Ap=Ap_lp, p=p_lp, r=r_lp, rdotr_prev=rdotr_lp, x=x_lp)

        # """
        evt, (pAp_lp,) = ip(queue, x=p_lp, y=Ap_lp)
        # Host operation because rdotr is on the host anyway with gslib
        a_lp = rdotr_lp / pAp_lp.get()
        evt, (x_lp,) = vupdt(queue, a=a_lp, x=x_lp, y=p_lp)
        evt, (r_lp,) = vupdt(queue, a=-a_lp, x=r_lp, y=Ap_lp)
        rdotr_prev_lp = rdotr_lp  # Host operation
        evt, (rdotr_lp,) = norm(queue, x=r_lp)
        rdotr_lp = rdotr_lp.get()
        evt, (p_lp,) = axpy(queue, a=(rdotr_lp / rdotr_prev_lp), x=p_lp, y=r_lp)
        # """

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

    # print(np.linalg.norm(x - x_lp))
    return x, niter


# Test
A = np.float32(np.random.rand(10, 10))
A += A.T
A += np.diag(np.sum(A, axis=0))
x = np.float32(np.random.rand(10))
x, niter = cg(A, x)
print(niter)
