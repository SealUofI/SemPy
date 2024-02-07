import numpy as np
from pytools import memoize

def lagrange(x_out, x_in, dtype=np.float64):
    assert x_out.ndim == x_in.ndim == 1

    n_out = x_out.size
    n_in = x_in.size

    a = np.ones((n_in,), dtype=np.longdouble)
    for i in range(n_in):
        for j in range(i):
            a[i] = a[i] * (x_in[i] - x_in[j])
        for j in range(i + 1, n_in):
            a[i] = a[i] * (x_in[i] - x_in[j])
    a = 1.0 / a

    J = np.zeros((n_out, n_in), dtype=dtype)
    s = np.ones((n_in,), dtype=np.longdouble)
    t = np.ones((n_in,), dtype=np.longdouble)

    for i in range(n_out):
        x = x_out[i]
        for j in range(1, n_in):
            s[j] = s[j - 1] * (x - x_in[j - 1])
            t[n_in - j - 1] = t[n_in - j] * (x - x_in[n_in - j])
        J[i, :] = a * s * t

    return J
