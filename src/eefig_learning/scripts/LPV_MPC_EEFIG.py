# Tools
import numpy as np
import warnings

# Imports
from lpv_mpc_eefig.common import EEFIG
from lpv_mpc_eefig.common import mdg


class LPV_MPC_EEFIG (EEFIG):

    def __init__ (self, settings, configuration_file = None):
        # super().__init__(settings) #python3
        super(LPV_MPC_EEFIG, self).__init__(settings) #python2.7

        self.ready = False # Ready to start predicting A & B (LPV Matrices)
        
        # If an EEFIG configuration file is provided use it
        if configuration_file is not None:
            self.load(configuration_file)
    

    # Update EEFIG
    def update (self, xk):

        # Detect & Update Counter
        self.update_anomalies(xk)

        self.K = self.update_K(xk)
        self.P = self.update_P(xk)

        if not self.update_last_samples(xk):
            return

        # If New Granule Create
        if self.check_new_granule_creation(xk):
            self.create_new_granule()

        # NOTE: If we do not have a Granule yet just skip the rest
        #       of the process
        if self.nEEFIG < 1:
            return
        self.ready = True

        # Update Granule Distribution
        gran_idx = self.data_evaluation(xk)

        # Update/Create A and B in Granules
        psik = self.last_samples[:-1, :]

        # WLS
        # NOTE: Used to create new Granules A's & B's
        if self.EEFIG[gran_idx].A.size == 0:
            xr = self.last_samples[1:, 0:self.nx] # xr contains the states x of the buffer (eq. 24)
            self.create_using_WLS (gran_idx, xr, psik)

        # RLS
        # NOTE: Used to update existing A's & B's in Granules
        else:
            self.update_using_RLS (gran_idx, xk, psik)

        # WARNING:
        for i in range(self.nEEFIG):
            if self.EEFIG[i].A.size == 0:

                warnings.warn("LPC_MPC_EEFIG.py: One granule had a zero size A matrix. We used WLS to solve this issue.")

                xr = self.last_samples[1:, 0:self.nx] # xr contains the states x of the buffer (eq. 24)
                self.create_using_WLS (i, xr, psik)


    # Obtain the Linear Paramenter Variant Matrixes & Update EEFIG
    def get_LPV_matrices (self, xk):

        # NOTE: Obtain for xk all the normalized weights for each granule
        wk = np.zeros([self.nEEFIG, 1])

        for i in range(self.nEEFIG):
            wk[i, 0], _, _, _ = mdg(xk, self.EEFIG[i].m, self.EEFIG[i].a, self.EEFIG[i].b)
            
        gsum = sum(wk)
        g    = wk / gsum

        # LPV Model A & B
        # NOTE: We make a weighted medium to extract the A and B using all the granules
        A = np.zeros([self.nx, self.nx])
        B = np.zeros([self.nx, self.nu])

        for i in range(self.nEEFIG):
            A += g[i] * self.EEFIG[i].A
            B += g[i] * self.EEFIG[i].B

        return A, B