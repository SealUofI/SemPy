import numpy as np
from sempy.quadrature import gauss_lobatto
from sempy.interpolation import lagrange


def reference(M):
    x = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    z = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    return box(x, y, z, M)


def reference_2d(M):
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    return box_2d(x, y, M)


def box_01(M):
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return box(x, y, z, M)


def box_01_2d(M):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    return box(x, y, z, M)


def box_ab(a, b, M):
    x = np.array([a, b, a, b, a, b, a, b])
    y = np.array([a, a, b, b, a, a, b, b])
    z = np.array([a, a, a, a, b, b, b, b])
    return box(x, y, z, M)


def box_ab_2d(a, b, M):
    x = np.array([a, b, a, b])
    y = np.array([a, a, b, b])
    return box_2d(x, y, M)


def box(x, y, z, M):
    N = 1
    n = N + 1
    nn = n * n
    m = M + 1
    mm = m * m

    zn, wn = gauss_lobatto(N)
    zm, wm = gauss_lobatto(M)
    J = lagrange(zm, zn)

    x = x.reshape((nn, n))
    y = y.reshape((nn, n))
    z = z.reshape((nn, n))
    Jx = np.dot(x, J.T)
    Jy = np.dot(y, J.T)
    Jz = np.dot(z, J.T)

    x = Jx.reshape((n, n, m))
    y = Jy.reshape((n, n, m))
    z = Jz.reshape((n, n, m))

    Jx = np.zeros((n, m, m))
    Jy = np.zeros((n, m, m))
    Jz = np.zeros((n, m, m))
    for i in range(n):
        Jx[i, :, :] = np.dot(J, x[i, :, :])
        Jy[i, :, :] = np.dot(J, y[i, :, :])
        Jz[i, :, :] = np.dot(J, z[i, :, :])
    x = Jx.reshape((n, mm))
    y = Jy.reshape((n, mm))
    z = Jz.reshape((n, mm))

    Jx = np.dot(J, x)
    Jy = np.dot(J, y)
    Jz = np.dot(J, z)

    x = Jx.reshape((m, m, m))
    y = Jy.reshape((m, m, m))
    z = Jz.reshape((m, m, m))

    return x, y, z


def box_2d(x, y, M):
    N = 1
    n = N + 1
    m = M + 1

    zn, wn = gauss_lobatto(N)
    zm, wm = gauss_lobatto(M)
    J = lagrange(zm, zn)

    x = x.reshape((n, n))
    y = y.reshape((n, n))
    Jx = np.dot(x, J.T)
    Jy = np.dot(y, J.T)

    x = Jx.reshape((n, m))
    y = Jy.reshape((n, m))

    Jx = np.dot(J, x)
    Jy = np.dot(J, y)

    x = Jx.reshape((m, m))
    y = Jy.reshape((m, m))

    return x, y
