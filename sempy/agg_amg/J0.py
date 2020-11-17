import numpy as np
import scipy.sparse as sp
from sempy.agg_amg.rcb_cut import rcb_cut


def get_J0(x, y):
    n = x.shape[0]
    log_n = int(np.ceil(np.log2(n)))

    ind_lst = np.arange(0, n)
    ind_lst_tmp = np.zeros_like(ind_lst)

    n_cuts, ja = 1, []
    ja.append(0)
    ja.append(n)
    for level in range(log_n-1):
        it, jt = 0, []
        jt.append(ja[it])
        for i in range(n_cuts):
            j0, j1 = ja[i], ja[i+1]
            ll = ind_lst[j0:j1]

            xl, yl = x[ll], y[ll]
            low, high = rcb_cut(xl, yl)
            low_n = low.shape[0]

            jt.append(j0+low_n)
            jt.append(j1)
            it = it + 2
            ind_lst_tmp[j0:j0+low_n] = ll[low]
            ind_lst_tmp[j0+low_n:j1] = ll[high]
        ja = jt
        ind_lst = ind_lst_tmp
        n_cuts = n_cuts*2

    cols = []
    J0 = np.zeros((n, n_cuts))
    for j in range(n_cuts):
        i0, i1 = ja[j], ja[j+1]
        for i in range(i0, i1):
            cols.append(j)
            J0[ind_lst[i], j] = 1

    n_vals = ja[n_cuts]
    vals = np.ones((n_vals,))

    J0 = sp.coo_matrix((vals, (ind_lst, cols)), shape=(n, n_cuts)).tocsr()

    return J0
