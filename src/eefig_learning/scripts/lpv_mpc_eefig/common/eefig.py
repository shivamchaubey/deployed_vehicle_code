# Tools
import numpy as np
from numpy import linalg as la
import pickle

# Imports
from . import Granule
from . import mdg


class EEFIG (object):

    def __init__ (self, settings):

        self.EEFIG = [] # list of granules that compose the cluster
        self.nEEFIG = 0 # number of granules

        self.nx = settings["nx"] # number of inputs
        self.nu = settings["nu"] # number of states
        self.p = self.nu + self.nx # number of inputs + state

        # NOTE: An anomaly is considered as an xk value which is not fed to a granule
        self.continuous_anomalies = []      # all xk values which are anomalies are added here.
                                            # If one xk value is not an anomaly, continuous_anomalies is emptied
        self.len_continuous_anomalies = 0   # length
        self.thr_continuous_anomalies = settings["thr_continuous_anomalies"] # this parameter controls the threshold of CA to create a new granule

        # NOTE: These values are updated in each iteration to check if a granule if c-separable
        # NOTE: tracker_m and tracker_c represent an ellipsoid which is moving in the p-dimensions space.
        #       Generally, this ellipsoid shares the same space as a Granule, which means that it is not c-separable.
        #       However, when the trackers move into a space without a Granule, thus c-separable,
        #       then a new Granule can be generated if we have enough anomalies.
        #       A forgetting factor helps the ellipsoid move only taking into account the last xk samples

        self.tracker_C = np.eye(self.p) * 1e6           # inverse covariance
        self.tracker_m = np.zeros(self.p)               # average

        # - propieties to calculate tracker_C & tracker_m
        self.lmbd = settings["lmbd"]                # lambda: (0.9-1) decaying factor
        self.effective_N = settings["effective_N"]  # 1 / (1 - lambda)

        self.separation = settings["separation"]    # "separation" in [0,inf) specifies the minimum required separation level
                                                    # between two information granules (eq. 19)

        self.ff     = settings["ff"]                # forgeting factor used to calculate P and K
        self.P      = 1e5 * np.eye(self.p)          # estimate of the inverted regularised data autocorrelation matrix
        self.K      = 0                             # RLS gain vector

        self.last_samples = np.array([])            # Set of last nphi samples
        self.nphi = settings["nphi"]                # Size of last samples, thus oldest xk value saved


    # Modify EEFIG
    def set_granules (self, EEFIG):
        self.EEFIG = EEFIG
        self.nEEFIG = len(EEFIG)
    

    # Identidy Anomalies
    def identify_anomalies (self, xk):

        # NOTE: Check for every Granule if xk is or not admitted
        #       Break if admitted
        for i in range(self.nEEFIG):
            if self.EEFIG[i].check_if_admitted(xk):
                return 0 # not an anomaly
        return 1 # is an anomaly


    # Anomaly Detection and Data Labeling
    def update_anomalies (self, xk):

        # NOTE: This anomaly has already been taken into account, skip point
        if xk.tolist() in self.continuous_anomalies:
            return

        # NOTE: Check if the xk value is admitted in any of the Granules of EEFIG
        if self.identify_anomalies(xk): # is anomaly
            self.len_continuous_anomalies += 1
            self.continuous_anomalies.append(xk.tolist())

        else: # not anomaly
            self.len_continuous_anomalies = 0
            self.continuous_anomalies = []


    # Check New Granule Conditions (c-separable & length "continuous_anomalies")
    def check_new_granule_creation (self, xk, multiplierN = -1):

        # Tracker Covariance & Tracker Average -> C-Separation
        # NOTE: optional parameter effective_N / multiplierN
        if multiplierN != -1:
            self.tracker_C = self.update_tracker_C(xk, multiplierN)
        else:
            self.tracker_C = self.update_tracker_C(xk)

        self.tracker_m = self.update_tracker_m(xk)

        c_separable = self.check_c_separable()

        # return true if c-seprable & continuous anomalies conditions
        new_granule_creation = c_separable and self.len_continuous_anomalies > self.thr_continuous_anomalies

        return new_granule_creation


    # Create New Granule
    def create_new_granule (self):
        new_granule = Granule(self.p, np.array(self.continuous_anomalies))
        self.EEFIG.append(new_granule)

        self.continuous_anomalies = [] # empty
        self.len_continuous_anomalies = 0

        self.nEEFIG += 1 # Increment Counter Granules


    # # Update Tracker C (inverse covariance)
    # def update_tracker_C (self, xk, multiplier_N = 1e6):

    #     multiplier_N = min(multiplier_N, self.effective_N) # default use effective_N

    #     aux = (xk - self.tracker_m) @ self.tracker_C @ (xk - self.tracker_m).T
    #     aux += (multiplier_N - 1) / self.lmbd

    #     tracker_C = self.tracker_C - self.tracker_C * ((xk - self.tracker_m).T @ (xk - self.tracker_m)) * self.tracker_C / aux # eq. 28 - (1)

    #     mplier = multiplier_N / ((multiplier_N - 1) * self.lmbd)
    #     tracker_C *= mplier

    #     return tracker_C


    # # Update Tracker M (average)
    # def update_tracker_m (self, xk):

    #     tracker_m = self.lmbd * self.tracker_m + (1 - self.lmbd) * xk # eq. 27 - (1)
    #     return tracker_m


    # Update Tracker C (inverse covariance)
    def update_tracker_C (self, xk, multiplier_N = 1e6):
        return la.pinv(np.cov(self.last_samples, rowvar=False))


    # Update Tracker M (average)
    def update_tracker_m (self, xk):
        return np.mean(self.last_samples, axis=0)


    # Check If a Granule is Separable
    def check_if_granule_is_c_separable (self, gran):

        T1, _ = la.eig(la.pinv(gran.C))
        T2, _ = la.eig(la.pinv(self.tracker_C))

        # Biggest Eigen Value
        T1 = T1[np.where(T1 == np.max(T1))][0]
        T2 = T2[np.where(T2 == np.max(T2))][0]

        # Verify if this is the covariance REVIEW!
        temp = self.separation * np.sqrt(self.p * np.maximum(T1, T2)) # eq.19 - (2)

        if la.norm(gran.m - self.tracker_m) > temp: # eq.19 - (2)
            return True
        return False


    # Check For Any C-Separable Granule
    def check_c_separable (self):

        # Security Step
        if np.sum(np.isnan(self.tracker_C)) or np.sum(np.isinf(self.tracker_C)):
            return True

        # Check For c-separable Granule
        # NOTE: If the Granule is c-separeble stop search
        for i in range(self.nEEFIG):
            if not self.check_if_granule_is_c_separable(self.EEFIG[i]):
                return False
        return True


    # Update a Granule's A and B (LPV Model) using Recursive Least Square (RLS)
    def update_using_RLS (self, idx, xk, psik):

        theta = np.hstack([self.EEFIG[idx].A, self.EEFIG[idx].B]).T
        res = xk[:self.nx] - np.dot(psik[-1, :], theta)

        # REVIEW: Make a better implementation of this
        # NOTE: Changes can be to big sometimes, limiting their size stabalizes the
        #       matrices helping maintain coherance in the system.
        if la.norm(res) > 0.002:
            return self.EEFIG[idx].A, self.EEFIG[idx].B

        for j in range(self.nx):
            theta[:,j] = theta[:,j] + np.squeeze(np.dot(self.K, res[j]))

        # Update A & B with theta
        self.EEFIG[idx].A = theta.T[:, :self.nx]
        self.EEFIG[idx].B = theta.T[:, self.nx:]

        # Optional Return
        return self.EEFIG[idx].A, self.EEFIG[idx].B


    # Create a Granule's A and B (LPV Model) using Window Least Square (WLS)
    def create_using_WLS (self, idx, xr, psik):

        theta = np.zeros([self.p, self.nx]) # shell for creating theta in the for loop

        pseudo_inv = la.pinv(psik)
        for j in range(self.nx):
            theta[:,j] = np.dot(pseudo_inv , xr[:, j]) # eq. 24 - (2)

        # Update A & B with theta
        self.EEFIG[idx].A = theta.T[:, :self.nx]
        self.EEFIG[idx].B = theta.T[:, self.nx:]

        # Optional Return
        return self.EEFIG[idx].A, self.EEFIG[idx].B


    # Update K (Used to obtain LPV Model)
    def update_K (self, xk):
        return np.dot(self.P , xk.T) * pow(self.ff + np.dot(xk , np.dot(self.P , xk.T)), -1) # eq. 21 - (2)


    # Update P (Used to obtain LPV Model)
    def update_P (self, xk):
        return np.dot((np.eye(self.p) - (self.K * xk)) , self.P) / self.ff # eq. 22 - (2)
        # return (np.eye(self.p) - (self.K * xk)) @ self.P / self.ff # eq. 22 - (2)


    # Update Granule Distribution (m, a, b, L ...) For All Granules
    def data_evaluation (self, xk):

        xk = xk.reshape([1, self.p])    # This reshape is necesary to work with m, a and b in mdg.
                                        # Work to remove this step

        # Save Values
        wk = np.zeros([self.nEEFIG, 1]) # Weights NOT normalized
        lambda0 = np.zeros([self.nEEFIG, self.p])
        lambda1 = np.zeros([self.nEEFIG, self.p])
        f = np.zeros([self.nEEFIG, 1])

        ## ROUND 1 MDG
        # Obtain the Membership Degree of the State Relative to each Granule/Cluster
        for i in range(self.nEEFIG):
            wk[i, 0], lambda0[i, :], lambda1[i, :], f[i, 0] = mdg(xk, self.EEFIG[i].m, self.EEFIG[i].a, self.EEFIG[i].b)

        g = wk / sum(wk) # normalize the membership degree weights

        ## GRANULAR STEP
        # Update the Granular Distribution Based on New Data
        for i in range(self.nEEFIG):

            # Check that the normalized membership weight is not null (Security Step)
            if g[i] < 1e-6 or np.isnan(g[i]):
                continue

            self.EEFIG[i].gran_step(xk, g[i], wk[i, 0], wk, lambda0[i, :], lambda0, lambda1[i, :], lambda1, f[i, 0], f)

        ## ROUND 2 MDG
        # NOTE: The MDG of the new representation after updating the granules (GRANULAR STEP)
        # Obtain the Membership Degree of the State Relative to each Granule/Cluster
        for i in range(self.nEEFIG):
            wk[i,0], lambda0[i, :], lambda1[i, :], f[i, 0] = mdg(xk, self.EEFIG[i].m, self.EEFIG[i].a, self.EEFIG[i].b)

        g = wk / sum(wk) # normalize the membership degree weights
        gran_idx =  np.where(g == np.amax(g))[0][0] # biggest weight

        return gran_idx


    # Save Last Smaples (maximum size nphi)
    # IMPORTANT NOTE:   This method return a boolean value which indicates if "last_samples" contains "nphi" values.
    #                   This boolean value allows eefig online to transition to learning phase (granule creation/update).
    def update_last_samples (self, xk):

        if not np.any(self.last_samples):
            self.last_samples = xk

            return False

        elif len(self.last_samples) < self.nphi:
            self.last_samples = np.vstack([self.last_samples, xk])

            return False

        else:
            self.last_samples = np.vstack([self.last_samples[1:,:], xk])

            return True


    # Save EEFIG Configuration
    def save (self, file_path):

        file = open(file_path, "wb")
        file.write(pickle.dumps(self.__dict__))
        file.close()
        
    # Load EEFIG Configuration
    def load (self, file_path):

        file = open(file_path, "rb")
        self.__dict__ = pickle.loads(file.read())
        file.close()


# (1) PAPER: (Granules) Uncertain Data Modeling Based on Evolving Ellipsoidal Fuzzy Information Granules
# (2) PAPER: EEFIG_MPC+estimator
