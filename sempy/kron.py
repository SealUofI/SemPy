import numpy as np


def kron_2d(Sy, Sx, U, order=None):
    nx, mx = Sx.shape
    ny, my = Sy.shape

    U = U.reshape((my, mx), order=order)
    if all([isinstance(X, np.ndarray) for X in [Sy, Sx]]):
        U = np.einsum('ai,ij,bj->ab', Sy, U, Sx, order=order, optimize=True)
    else:
        U = Sy @ U @ Sx.T

    return U.reshape((nx * ny,), order=order)


def kron(Sz, Sy, Sx, U, order=None):
    nx, mx = Sx.shape
    ny, my = Sy.shape
    nz, mz = Sz.shape

    if all([isinstance(X, np.ndarray) for X in [Sz, Sy, Sx]]):
        U = U.reshape((mz, my, mx), order=order)
        U = np.einsum('ai,bj,ijk,ck->abc', Sz, Sy, U,
                      Sx, order=order, optimize=True)
    else:
        U = U.reshape((my * mz, mx), order=order)
        U = U @ Sx.T

        U = U.reshape((mz, my, nx), order=order)

        if isinstance(Sy, np.ndarray):
            V = np.einsum('mj,ijk->imk', Sy, U, order=order, optimize=True)
        else:
            V = np.empty((mz, ny, nx), order=order)
            for i in range(mz):
                V[i, :, :] = Sy @ U[i, :, :]

        V = V.reshape((mz, nx * ny), order=order)
        U = Sz @ V

    return U.reshape((nx * ny * nz,), order=order)
