# Tools
import numpy as np


def granule_performance_index(gran_original, gran_updated, xk, g, wk_i, wk, l0_i, l0, l1_i, l1, f_i, f):

    p = l1.shape[1]

    aux_l0 = np.zeros((len(wk), p))
    aux_l1 = np.zeros((len(wk), p))

    sum_w = np.sum(wk, axis = 0)
    sum_w2 = (sum_w) ** 2

    sum_l0 = np.zeros((1, p))
    sum_l1 = np.zeros((1, p))

    for i in range(len(wk)):
        for j in range(p):
            aux_l0[i, j] = (wk[i, 0] * l0[i, j])*1.0 / (f[i, 0])
            aux_l1[i, j] = (wk[i, 0] * l1[i, j])*1.0 / (f[i, 0])

    for j in range(p):
        sum_l0[0, j] = np.sum(aux_l0[:, j], axis = 0)
        sum_l1[0, j] = np.sum(aux_l1[:, j], axis = 0)

    dg_da = (np.dot(wk_i , sum_l1)*1.0 / (sum_w2)) - (np.dot(wk_i , l1_i)*1.0 / np.dot(f_i , sum_w))
    dg_dm = (np.dot(wk_i , l0_i)*1.0 / np.dot(f_i , sum_w)) - (np.dot(wk_i , sum_l0)*1.0 / (sum_w2))
    dg_db = (np.dot(wk_i , l1_i)*1.0 / np.dot(f_i , sum_w)) - (np.dot(wk_i , sum_l1)*1.0 / (sum_w2))

    delta_a = gran_updated.a - gran_original.a
    delta_m = gran_updated.m - gran_original.m
    delta_b = gran_updated.b - gran_original.b

    gran_updated.gsum = gran_updated.gsum + (g - np.dot(dg_da, (delta_a.T)) + (np.dot(dg_dm , (delta_m.T))) + (np.dot(dg_db, (delta_b.T))))

    mahal_xk = np.dot((xk - gran_updated.m) , np.dot(gran_updated.C , (xk - gran_updated.m).T)) # mahalanobis distance

    Q_it = np.dot(mahal_xk , gran_updated.gsum)
    gran_updated.Q_it = np.vstack((gran_updated.Q_it,Q_it))
    gran_updated.Q = np.cumsum(gran_updated.Q_it)

    if ((gran_updated.Q[-1]) >  (gran_original.Q[-1])):
        updated_gran = gran_updated
        updated_gran.is_improved = 1

    else:
        updated_gran = gran_original
        updated_gran.is_improved = 0

    return updated_gran
