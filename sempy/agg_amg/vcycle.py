import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def sigma_cheb(k, n, lmin, lmax):
    k = ((k-1) % n)+1
    theta = np.pi*(k-0.5)/n
    lamk = lmin+0.5*(lmax-lmin)*(np.cos(theta)+1)
    sigma = 1.0/lamk
    return sigma


def vcycle(rhs, A, level, J0, verbose=1):
    n = rhs.shape[0]
    rhs = rhs.reshape((n, 1))
    if verbose > 0:
        print("level: {} rhs: {}".format(level, rhs.shape))

    if n == 2:
        u = sla.spsolve(A, rhs)
    else:
        nsmooth = 1
        sigma = sigma_cheb(1, nsmooth+1, 1.0, 2.0)
        D = 1.0/A.diagonal()
        D = D.reshape((n, 1))

        u = sigma*np.multiply(D, rhs)
        if verbose > 0:
            print("level: {} u: {}".format(level, u.shape))

        r = rhs-A.dot(u)

        for smooth in range(1, nsmooth+1):
            sigma = sigma_cheb(smooth+1, nsmooth+1, 1.0, 2.0)
            s = sigma*np.multiply(D, r)
            u = u+s
            r = r-A.dot(s)

        if level == 0:
            J = J0.copy()
        else:
            n_half = n/2
            I = sp.eye(n_half)
            e2 = np.ones((2, 1))
            J = sp.kron(I, e2)

        if verbose > 0:
            print("level: {} J: {}".format(level, J.shape))

        r = (J.T).dot(r)
        Ar = (J.T).dot(A.dot(J))

        if verbose > 0:
            print("level: {} Ar: {}, r: {}".format(level, Ar.shape, r.shape))

        e = vcycle(r, Ar, level+1, J)

        over = 1.4
        u = u+over*J.dot(e)

        r = rhs-A.dot(u)
        for smooth in range(1):
            sigma = 0.66
            s = sigma*np.multiply(D, r)
            u = u+s
            r = r-A.dot(s)

    return u
