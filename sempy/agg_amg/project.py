import numpy as np


def precond(r, A, J0, prec):
    if prec == 0:
        Di = 1.0/np.diagonal(A)
        z = np.dor(Di, r)
    elif prec == 1:
        z = vcycle(r, A, 0, J0)
    elif prec == 2:
        z = kcycle(r, A, 0, J0)

    return z


def project(r, A, J0, tol, prec):
    n_iter, max_iter = 0, 1000

    if tol < 0:
        max_iter = abs(tol)
        tol = 1e-3

    n, _ = r.shape

    x = np.zeros_like(r)

    z = precond(r, A, J0, prec)
    rz1 = np.dot(r, z)
    p = z

    P = np.zeros(n, 5)
    W = P

    res = []

    for k in range(max_iter):
        w = np.dot(A, p)
        pAp = np.dot(p, w)
        alpha = rz1/pAp

        if prec > 0:
            scale = 1./np.sqrt(pAp)
            W[:, k] = scale*w
            P[:, k] = scale*p

        x = x+alpha*p
        r = r-alpha*w

        ek = np.norm(r)  # 2-norm
        res.append(ek)

        if ek < tol:
            break

        zo = z
        z = precond(r, A, J0, prec)
        dz = z-zo

        rz0 = rz1
        rz1 = np.dot(r, z)
        rz2 = np.dot(r, dz)
        beta = rz2/rz0

        p = z+beta*p

        if prec > 0:
            a = np.dot(W[:, 0:k], p)
            p = p-np.dot(P[:, 0:k])*a
    return x, res, k
