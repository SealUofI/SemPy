import numpy as np

from sempy.agg_amg.vcycle import vcycle


def precond(r, A, J0, prec):
    n = A.shape[0]
    if prec == 0:
        Di = 1.0 / A.diagonal()
        Di = Di.reshape((n,))
        z = np.multiply(Di, r)
    elif prec == 1:
        z = vcycle(r, A, 0, J0)
    elif prec == 2:
        # z = kcycle(r, A, 0, J0)
        print("K-cycle is not implemented yet.")
        exit()
    else:
        z = r.copy()

    return z.reshape((n,))


def project(r, A, J0, tol, prec, verbose=0):
    max_iter = 100
    if tol < 0:
        max_iter = abs(tol)
        tol = 1e-3

    n = r.shape[0]

    z = precond(r, A, J0, prec)
    rz1 = np.dot(r, z)
    if verbose > 0:
        print(
            "z: {} r: {} rz1: {}".format(
                np.linalg.norm(z), np.linalg.norm(r), rz1
            )
        )

    x = np.zeros_like(r)
    p = z.copy()

    P = np.zeros((n, max_iter))
    W = np.zeros_like(P)

    res = []
    res.append(np.linalg.norm(r))

    if verbose > 0:
        print("A.shape: {} p.shape: {}".format(A.shape, p.shape))

    for k in range(max_iter):
        w = A.dot(p)
        pAp = np.dot(p, w)
        alpha = rz1 / pAp

        if prec > 0:
            scale = 1.0 / np.sqrt(pAp)
            W[:, k] = scale * w.reshape((n,))
            P[:, k] = scale * p.reshape((n,))

        x = x + alpha * p
        r = r - alpha * w

        ek = np.linalg.norm(r)
        res.append(ek)
        if verbose > 0:
            print("iter: {} ek: {} alpha: {}".format(k, ek, alpha))

        if ek < tol:
            break

        zo = z.copy()
        z = precond(r, A, J0, prec)
        dz = z - zo

        rz0 = rz1
        rz1 = np.dot(r, z)
        rz2 = np.dot(r, dz)
        beta = rz2 / rz0
        if verbose > 0:
            print("iter: {} beta: {}".format(k, beta))

        p = z + beta * p
        if verbose > 0:
            print("iter: {} norm: {}".format(k, np.linalg.norm(p)))

        if prec > 0:
            a = np.dot(W[:, 0 : k + 1].T, p)
            p = p - np.dot(P[:, 0 : k + 1], a)
        if verbose > 0:
            print("iter: {} norm: {}".format(k, np.linalg.norm(p)))
    return x, res, k + 1
