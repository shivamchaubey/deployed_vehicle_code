# Tools
import numpy as np
import numpy.linalg as la
from scipy.stats.distributions import chi2
from copy import deepcopy

# Imports
from . import granule_performance_index


# NOTE: A Granule represent one cluster which contains a set of LPV Models (A, B).
# Granules can evolve, new one can be created and old ones can be modified
class Granule (object):

    # Initialization
    def __init__ (self, p, xk_samples = None):

        self.p = p # Number of states + inputs

        # Threshold Mahalanobis Distance xk (admitance xk sample)
        self.thr_mahal_xk = chi2.ppf(0.999, self.p)

        self.sp = 20 # Stability Period Condition

        self.sensitivity = 0

        self.A = np.array([]) # LPV Matrice
        self.B = np.array([]) # LPV Matrice

        # Fill granule structure with default values for each attribute
        self.counter = 0 # number of xk_samples (lenght)
        self.alpha = 0
        self.beta = 0

        # Covariance Matrix ~ Default Value on Initialization
        # NOTE: It is not recommended to use this value. It is better to calculate the covariance
        #       during the offline learning phase, using the default value is not accurate, at all.
        self.C = np.eye(self.p)

        # Constant
        self.L = np.zeros(self.p) # Left Boundary
        self.R = np.zeros(self.p) # Right Boundary
        self.m = np.zeros(self.p) # Medium
        self.a = np.zeros(self.p) # Left Optimal Boundary
        self.b = np.zeros(self.p) # Right Optimal Boundary
        self.Q = 0 # Improvement Indicator

        self.Q_it = 0 # Improvement Indicator - Iterative

        self.k = np.zeros(self.p)
        self.w = np.zeros(self.p)

        self.gsum = 0

        # NOTE: If the Granule initialization contains a list of xk_samples
        # then initialize the propieties with the samples iteratively
        if xk_samples is not None:

            # Calculate Initial Covariance Granule
            self.C = la.pinv(np.cov(xk_samples, rowvar=False))

            if self.p != xk_samples.shape[1]:
                raise ValueError ("p (state + input) has a diferent size than xk_samples")

            self.initialize_with_samples(xk_samples)


    # Initialize With Samples of xk
    # NOTE: xk_samples: states + inputs samples from k instants that conform a Granule
    def initialize_with_samples (self, xk_samples):

        # NOTE: The granule is generated iteratively, thus here the medium is created with only 2 xk points
        # Latter the rest of xk_samples will be added iteratively
        self.m = np.mean(xk_samples[0:2, :], axis = 0)
        self.counter = 2 # number of xk_samples (lenght)

        # Find Bounds For Each State/Row
        # NOTE: The small values prevent L and R from being equal
        self.L = np.min(xk_samples, axis=0) - 0.001
        self.R = np.max(xk_samples, axis=0) + 0.001

        # Optimal Bounds (eq. 6 & 7)
        for j in range(self.p):
            self.a[j] = np.maximum(self.m[j] - 4 * self.L[j], 0)        + (self.m[j] - np.maximum(self.m[j] - 4 * self.L[j], 0))        * self.sensitivity
            self.b[j] = np.minimum(self.m[j] + 4 * self.R[j], np.inf)   - (np.minimum(self.m[j] + 4 * self.R[j], np.inf) - self.m[j])   * self.sensitivity

        self.alpha = 2
        self.beta = 2

        # Improvement Indicator
        self.gsum = 1 # Only one sample, thus gsum = 1
        mahal_xk = np.dot((xk_samples[0, :] - self.m) , np.dot(self.C , (xk_samples[0, :] - self.m).T))
        Q_xk0 = mahal_xk * self.gsum

        self.gsum += 1 # Second sample added to the same granule, thus g=1 and gsum = 2
        mahal_xk = np.dot((xk_samples[1, :] - self.m) , np.dot(self.C , (xk_samples[1, :] - self.m).T))
        Q_xk1 = mahal_xk * self.gsum

        self.Q_it = np.vstack([Q_xk0, Q_xk1])
        self.Q = np.cumsum(self.Q_it, axis = 0)

        # Empty A and B, thus it can be detected as a new granule by EEFIG
        self.A = np.array([]) # Used for LPV_MODEL
        self.B = np.array([]) # Used for LPV_MODEL

        # Add the rest of the xk_samples iteratively
        for i in range(2, xk_samples.shape[0]):
            xk = xk_samples[i, :].reshape(1, self.p)
            self.initialize_with_samples_evolution(xk)


    # Initial Granule Evolution
    def initialize_with_samples_evolution (self, xk):

        g = np.array([1])
        mahal_xk = np.dot((xk - self.m) , np.dot(self.C , (xk - self.m).T)) # mahalanobis distance
        self.aux_function (xk, g, 1)

        self.gsum += g
        Q_xk = np.dot(mahal_xk , self.gsum)

        self.Q_it = np.vstack([self.Q_it , Q_xk])
        self.Q = np.cumsum(self.Q_it, axis = 0)


    # Identify Anomaly
    def check_if_admitted (self, xk):

        # Check if xk is part of the Granule
        mahal_xk = np.dot((xk - self.m) , np.dot(self.C , (xk - self.m).T))

        if mahal_xk > self.thr_mahal_xk:
            return 0
        return 1


    # Granule Evolution
    def evolution (self, xk, g, wk_i, wk, l0_i, l0, l1_i, l1, f_i, f, force_update):

        # NOTE: This copy is used to compare in "granule_performance_index" is the granule distribution has improved
        gran_updated = deepcopy(self)

        # Update Covariance / Average / Boundaries
        admitted = gran_updated.aux_function(xk, g, force_update)
        if not admitted:
            return

        # Check is the granular distribution has improved
        gran_new = granule_performance_index(self, gran_updated, xk, g, wk_i, wk, l0_i, l0, l1_i, l1, f_i, f)
        self.copy(gran_new)


    # NOTE: Used by "evolution" & "initialize_with_samples_evolution"
    def aux_function (self, xk, g, force_update):

        # Check if xk is part of the Granule
        admitted = self.check_if_admitted(xk)
        if not admitted:
            return admitted

        # If admitted update xk counter
        self.counter += 1

        # Update Center Grandule
        self.m += np.dot((g / self.gsum) , (xk - self.m))

        # Update: Beta, Alpha + Calculate Gama
        beta_updated = self.beta + pow(g, 2)
        alpha_updated = self.alpha + g

        gama = self.alpha * (pow(alpha_updated, 2) - beta_updated)
        gama = gama / (np.dot(alpha_updated , (pow(self.alpha, 2) - beta_updated)))
        gama = np.squeeze(gama)

        delta = alpha_updated * (pow(self.alpha, 2) - self.beta)
        delta = delta / (np.dot((self.alpha * g) , (g + alpha_updated - 2)))

        # Update the Covariance Matrix
        mahal_xk = np.dot((xk - self.m) , np.dot(self.C , (xk - self.m).T))

        self.C = self.C - (np.dot(self.C , np.dot((xk - self.m).T , np.dot((xk - self.m) , self.C)))) / (mahal_xk + delta)
        self.C = gama * self.C

        self.beta = beta_updated
        self.alpha = alpha_updated

        kk1 = np.zeros(self.p)
        ww1 = np.zeros(self.p)

        a_new = np.zeros(self.p)
        b_new = np.zeros(self.p)
        L_new = np.zeros(self.p)
        R_new = np.zeros(self.p)
        m_new = np.zeros(self.p)

        for j in range(self.p):

            m_new[j] = (self.counter - 1.0) / self.counter * self.m[j] + 1.0 / self.counter * xk[0, j]

            if (xk[0, j] < m_new[j]):
                kk1[j] = self.k[j] + 1
                ww1[j] = self.w[j]

                L_new[j] = (self.L[j] * (kk1[j] - 1.0) + m_new[j] - xk[0, j]) / kk1[j]
                R_new[j] = self.R[j]

            else:
                kk1[j] = self.k[j]
                ww1[j] = self.w[j] + 1

                R_new[j] = (self.R[j] * (ww1[j] - 1.0) - m_new[j] + xk[0, j]) / ww1[j]
                L_new[j] = self.L[j]

            a_new[j] = np.maximum(m_new[j] - 4 * L_new[j], 0) + (m_new[j] - np.maximum(m_new[j] - 4 * L_new[j], 0)) * self.sensitivity
            b_new[j] = np.minimum(m_new[j] + 4 * R_new[j], np.inf) - (np.minimum(m_new[j] + 4 * R_new[j], np.inf) - m_new[j]) * self.sensitivity

        self.L = L_new
        self.R = R_new
        self.a = a_new
        self.b = b_new
        self.m = m_new

        self.k = kk1
        self.w = ww1

        return admitted


    # Granular Step
    def gran_step (self, xk, g, wk_i, wk, lambda0_i, lambda0, lambda1_i, lambda1, f_i, f):

        # NOTE:"force update" forces the creation of a new granule. If this boolean value was not there then the granule would 
        #       be continuously moving and thus all xk values would be admitted to it, generating only one big granule.

        # NOTE: If the counter is bigger than the stability period (p), then change "force_update" from mode 0 to 1
        if (self.counter < self.sp):
            self.evolution(xk, g, wk_i, wk, lambda0_i, lambda0, lambda1_i, lambda1, f_i, f, 0)
        # else:
        #     self.evolution(xk, g, wk_i, wk, lambda0_i, lambda0, lambda1_i, lambda1, f_i, f, 1)


    # Copy Granule Propierties
    def copy (self, gran_new):

        # only copy instance attributes from parents
        # and make a deepcopy to avoid unwanted side-effects

        for k, v in gran_new.__dict__.items():
            self.__dict__[k] = deepcopy(v)