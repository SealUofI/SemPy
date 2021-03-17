import numpy as np


def rcb_cut(x, y):
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    x_len = x_max - x_min
    y_len = y_max - y_min

    if x_len > y_len:
        return rcb_cut_graph(x)
    else:
        return rcb_cut_graph(y)


def rcb_cut_graph(x):
    indices = np.argsort(x)

    n = x.shape[0]
    n_high = int(np.ceil(n / 2.0))
    n_low = n - n_high

    return indices[0:n_low], indices[n_low:n]
