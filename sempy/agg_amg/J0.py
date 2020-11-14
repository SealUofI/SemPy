# indices = np.arange(1, n+1) ## 1..n
# rcb_order = cut(x, y, indices)
def cut(x, y, indices):
    if n < 4:
        return indices
    else:
        low, high = rcb_cut(x, y)
        x_low, y_low, indices_low = x[low], y[low], indices[low]
        x_high, y_high, indices_high = x[high], y[high], indices[high]
        ind_low = cut(x_low, y_low, indices_low)
        ind_high = cut(x_high, y_high, indices_high)
        return np.concatenate(ind_low, ind_high)


def get_J0(x, y):
    n, _ = x.shape
    log_n = np.ceil(np.log2(n))

    ind_list = np.arange(0, n)
    ind_list_temp = np.zeros_like(ind_list)

    n_cuts, ja = 1, []
    ja[0], ja[1] = 0, n
    for level in range(log_n-1):
        it, jt = 0, []
        jt.append(ja[it])
        for i in range(n_cuts):
            j0, j1 = ja[i], ja[i+1]
            ll = ind_list[j0:j1]

            xl, yl = x[ll], y[ll]
            low, high = rcb_cut(xl, yl)
            low_n, _ = low.shape

            jt.append(j0+low_n)
            jt.append(j1)
            it = it + 2
            ind_list_tmp[j0:j0+low_n] = ll[low]
            ind_list_tmp[j0+low_n:j1] = ll[high]
        ja = jt
        ind_list = ind_list_tmp
        n_cuts = n_cuts*2

    J0 = np.zeros((n, n_cuts))
    for j in range(n_cuts):
        i0, i1 = ja[j], ja[j+1]
        for i in range(i0, i1):
            J0[ind_list[i], j] = 1
    return J0
