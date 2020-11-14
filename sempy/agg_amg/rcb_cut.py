def rcb_cut(x, y):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    x_len = x_max-x_min
    y_len = y_max-y_min

    if x_len > y_len:
        return rcb_cut_graph(x)
    else:
        return rcb_cut_graph(y)


def rcb_cut_graph(x):
    indices = np.argsort(x)

    n = x.shape
    n_high = np.ceil(n/2.0)
    n_low = n-n_high

    return indices[0:n_low], indices[n_low:n]
