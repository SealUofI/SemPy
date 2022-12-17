from warnings import filterwarnings

import numpy as np

from sempy.derivative import reference_derivative_matrix
from sempy.gradient import gradient, gradient_transpose


def elliptic_ax(mesh, p):
    nelem = mesh.get_num_elems()
    Np = mesh.Np
    Nq = mesh.Nq

    g = mesh.get_geom()

    p_ = p.reshape((nelem, Np))
    ap = np.zeros_like(p_)
    for e in range(nelem):
        px, py, pz = gradient(p_[e, :], Nq)

        apx = g[e, 0, 0, :] * px + g[e, 0, 1, :] * py + g[e, 0, 2, :] * pz
        apy = g[e, 1, 0, :] * px + g[e, 1, 1, :] * py + g[e, 1, 2, :] * pz
        apz = g[e, 2, 0, :] * px + g[e, 2, 1, :] * py + g[e, 2, 2, :] * pz

        ap[e, :] = gradient_transpose(apx, apy, apz, Nq)

    return ap.reshape((nelem * Np,))


def elliptic_cg(mesh, b, tol=1e-12, maxit=100, verbose=0):
    rmult = mesh.get_rmult()

    norm_b = np.dot(np.multiply(rmult, b), b)
    TOL = max(tol * tol * norm_b, tol * tol)

    r = b
    rdotr = np.dot(np.multiply(rmult, r), r)
    if verbose:
        print("Initial rnorm={}".format(rdotr))

    x = 0 * b
    niter = 0
    if rdotr < 1.0e-20:
        return x, niter

    p = r
    while niter < maxit and rdotr > TOL:
        Ap = elliptic_ax(mesh, p)
        mesh.apply_mask(Ap)

        pAp = np.dot(Ap, p)
        alpha = rdotr / pAp

        Ap = mesh.dssum(Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rdotr0 = rdotr
        rdotr = np.dot(np.multiply(rmult, r), r)
        beta = rdotr / rdotr0

        if verbose:
            print(
                "niter={} r0={} r1={} alpha={} beta={} pap={}".format(
                    niter, rdotr0, rdotr, alpha, beta, pAp
                )
            )

        p = r + beta * p
        niter = niter + 1

    return x, niter


def elliptic_cg_loopy(mesh, b, tol=1e-12, maxit=100, verbose=0):
    import loopy as lp
    import loopy.options
    import pyopencl as cl
    import pyopencl.array
    import pyopencl.clrandom
    from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

    import sempy.loopy.loopy_kernels as lpk

    lp.set_caching_enabled(False)
    filterwarnings("error", category=lp.LoopyWarning)
    loopy.options.ALLOW_TERMINAL_COLORS = False

    # Setup loopy
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    knl_wnorm = lpk.gen_weighted_norm_knl()
    knl_inner = lpk.gen_inner_product_knl()
    knl_xpay = lpk.gen_inplace_xpay_knl()
    knl_axpy = lpk.gen_inplace_axpy_knl()
    knl_mask = lpk.gen_zero_boundary_knl()
    knl_dssum = lpk.gen_gather_scatter_knl()

    nelem = mesh.get_num_elems()
    ndofs_1d = mesh.Nq
    knl_ax_lp = lpk.gen_elliptic_Ax_knl(nelem, ndofs_1d)

    # Get mesh data
    d_masked_ids = cl.array.to_device(queue, mesh.get_mask_ids())
    global_to_local, global_start = mesh.get_global_to_local_map()
    max_iter = np.max(global_start[1:] - global_start[:-1])

    d_global_to_local = cl.array.to_device(queue, global_to_local)
    d_global_start = cl.array.to_device(queue, global_start)

    d_D = cl.array.to_device(queue, reference_derivative_matrix(ndofs_1d - 1))
    d_G = cl.array.to_device(queue, mesh.get_geom())

    d_b = cl.array.to_device(queue, b)
    d_rmult = cl.array.to_device(queue, mesh.get_rmult())

    event, (d_norm_b,) = knl_wnorm(queue, w=d_rmult, x=d_b)
    TOL = max(tol * tol * d_norm_b.get(), tol * tol)

    d_r = cl.array.to_device(queue, b)
    event, (d_rdotr,) = knl_wnorm(queue, w=d_rmult, x=d_r)
    rdotr = d_rdotr.get()
    if verbose:
        print("Initial rnorm={}".format(rdotr))

    d_x = cl.array.to_device(queue, np.zeros_like(b))
    niter = 0
    if rdotr < 1.0e-20:
        return d_x.get(), niter

    d_p = cl.array.to_device(queue, b)
    while niter < maxit and rdotr > TOL:
        event, (d_Ap,) = knl_ax_lp(queue, D=d_D, U=d_p, g=d_G)
        event, (d_Ap,) = knl_mask(
            queue, boundary_indices=d_masked_ids, dofs=d_Ap
        )
        event, (d_pAp,) = knl_inner(queue, x=d_Ap, y=d_p)

        pAp = d_pAp.get()
        alpha = rdotr / pAp

        event, (Ap,) = knl_dssum(
            queue,
            max_iter=max_iter,
            gather_ids=d_global_to_local,
            gather_start=d_global_start,
            q=d_Ap,
        )

        event, (d_x,) = knl_xpay(queue, x=d_x, a=alpha, y=d_p)
        event, (d_r,) = knl_xpay(queue, x=d_r, a=-alpha, y=d_Ap)

        rdotr0 = rdotr
        event, (d_rdotr,) = knl_wnorm(queue, w=d_rmult, x=d_r)
        rdotr = d_rdotr.get()

        beta = rdotr / rdotr0

        if verbose:
            print(
                "niter={} r0={} r1={} alpha={} beta={} pap={}".format(
                    niter, rdotr0, rdotr, alpha, beta, pAp
                )
            )

        event, (d_p,) = knl_axpy(queue, x=d_p, a=beta, y=d_r)

        niter = niter + 1

    return d_x.get(), niter
