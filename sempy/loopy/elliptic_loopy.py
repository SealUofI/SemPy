# Add to path so can import from above directory
import sys
from warnings import filterwarnings

import loopy as lp
import loopy.options
import loopy_kernels as lpk
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

sys.path.append("../")

# setup
# -----
lp.set_caching_enabled(False)
filterwarnings("error", category=lp.LoopyWarning)
loopy.options.ALLOW_TERMINAL_COLORS = False


"""
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
#ctx = cl.Context(devices=my_gpu_devices)
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)
"""

mxm = lpk.gen_mxm_knl()
tp = lpk.gen_tensor_product_2dx3d_knl()
tvs = lpk.gen_triple_vector_sum_knl()
g_app = lpk.gen_apply_geometric_factors_knl()

# Input should be a pyopencl array or a numpy array


def gradient(queue, U, D):
    n = D.shape[0]
    nn = n * n

    evt, (Ur,) = mxm(queue, A=U.reshape(nn, n), X=D.transpose())
    evt, (Us,) = tp(queue, A2d=D, X3d=U.reshape(n, n, n))
    evt, (Ut,) = mxm(queue, A=D, X=U.reshape(n, nn))

    return cl.array.concatenate([Ur.ravel(), Us.ravel(), Ut.ravel()], axis=0)


def gradient_tranpose(queue, W, D):
    n = D.shape[0]
    nn = n * n

    evt, (Ur,) = mxm(queue, A=W[0, :].reshape(nn, n), X=D)
    evt, (Us,) = tp(queue, A2d=D.transpose(), X3d=W[1, :].reshape(n, n, n))
    evt, (Ut,) = mxm(queue, A=D.transpose(), X=W[2, :].reshape(n, nn))
    evt, (result,) = tvs(queue, Ur.ravel(), Us.ravel(), Ut.ravel())

    return result


def elliptic_ax(queue, mesh, p, D):
    nelem = mesh.get_num_elems()
    Np = mesh.Np

    p_ = p.reshape((nelem, Np))
    ap = cl.array.empty_like(p_)
    for e in range(nelem):
        evt, (px,) = gradient(queue, p_[e, :], D)
        evt, (apx,) = g_app(queue, px=px)
        # This will probably break
        evt, (ap[e, :],) = gradient_transpose(queue, apx, D)

    ap = ap.reshape(nelem * Np)
    return mesh.apply_mask(ap)
