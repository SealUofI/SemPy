import numpy as np
import scipy.sparse as sp


def sigma_cheb(k, n, lmin, lmax):
    k = ((k-1) % n)+1
    theta = np.pi*(k-0.5)/n
    lamk = lmin+0.5*(lmax-lmin)*(cos(theta)+1)
    sigma = 1.0/lamk
    return sigma


def vcycle(rhs, A, level, J0):
    n, _ = rhs.shape

    if n == 1:
        u = 0.0
        if abs(A) > sqrt(np.finfo(float).eps):
            u = np.linalg.solve(A, rhs)
    else:
        nsmooth = 1
        sigma = sigma_cheb(1, nsmooth+1, 1.0, 2.0)
        D = 1.0/np.diag(A)

        u = sigma*np.multiply(D, r)
        r = rhs-np.dot(A, u)

        for smooth in range(1, nsmooth+1):
            sigma = sigma_chb(smooth+1, nsmooth+1, 1.0, 2.0)
            s = sigma*np.multiply(D, r)
            u = u+s
            r = r-np.dot(A, s)

        if level == 0:
            J = J0
        else:
            n_half = n/2
            I = sp.eye(n_half)
            e2 = np.ones((2, 1))
            J = sp.kron(I, e2)

        r = np.dot(J.T, r)
        Ar = np.dot(J.T, np.dot(A, J))

        over = 1.4
        e = vcycle(r, Ar, level+1, J)
        u = u+over*np.dot(J, e)

        r = rhs-np.dot(A, u)
        for smooth in range(1):
            sigma = 0.66
            s = sigma*np.multiply(D, r)
            u = u+s
            r = r-np.dot(A, s)

    return u
