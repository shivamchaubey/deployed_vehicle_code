# Tools
import numpy as np


# Ellipsoidal membership degree computation for ONLY ONE Graule/Cluster

# x: state + input
# m: center Graule/cluster (Hyper Ellipsoide Center)
# a & b: (optimal) upper and lower bounds Granule/cluster

# NOTE: For more information on this process look at the article
def mdg (x, m, a, b):

    upsilon = np.zeros(x.shape)
    lambda0 = np.zeros(x.shape)
    lambda1 = np.zeros(x.shape)

    diff = b - a
    dist = x - m

    for j in range(len(x)):
        
        if diff[j] == 0:
            continue

        upsilon[j] = pow(dist[j] / diff[j], 2)
        lambda0[j] = 2.0 * dist[j] / pow(diff[j], 2)
        lambda1[j] = 2.0 * pow(dist[j], 2) / pow(diff[j], 3)
        
    theta = 2.0 * np.sqrt(np.sum(upsilon))
    f = theta / 2.0
    w_xk = np.exp(-theta) # not normalized weight

    return w_xk, lambda0, lambda1, f