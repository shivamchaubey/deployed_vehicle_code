# Tools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Imports
from . import Granule

from mpl_toolkits.mplot3d import Axes3D

# Find the cluster for a dataset provided
def find_clusters (data, n_clusters, silhouette=False, debug=False):

    # Fuse multiple datasets if provided
    data_idx   = []
    old_idx    = 0
    for i in range(len(data)):
        new_idx = old_idx + len(data[i])
        data_idx.append([old_idx, new_idx])
        old_idx = new_idx
    
    data = np.vstack(data)

    # Normalize data
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)

    data_normalized = data - min_
    data_normalized /= (max_ - min_)

    # Find Best Number of Clusters
    if silhouette:
        n_clusters = compute_silhouette(data_normalized, debug=debug)

    # Create Cluster with Best Fit
    print("Looking for clusters...")
    clusters = AgglomerativeClustering(n_clusters=n_clusters, ).fit(data_normalized)
    print("Found clusters!\n")

    # Plot Cluster
    if debug:

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax = fig.gca()

        # ax = fig.add_subplot(projection='3d')

        cmap = get_cmap(max(clusters.labels_) + 1) # color
        cmap = [cmap(i) for i in clusters.labels_.astype(int)] # color
        
        ax.scatter(data_normalized[:, 0], data_normalized[:, 3], data_normalized[:, 4], color= cmap, s=20, alpha=0.5)
        plt.title("3D Representation of Best Cluster Selected")
        plt.show()

    # Get labels based on origin dataset
    labels = []
    for i in range(len(data_idx)):
        labels.append(clusters.labels_[data_idx[i][0]: data_idx[i][1]])

    return labels, n_clusters


# Create the granular structure from the cluster labels provided
def train (lpv_mpc_eefig, data, labels, expand_packs=True, remove=True):

    # Verify that both data and labels list are the same size
    if len(labels) != len(data):
        print("ERROR: In function 'train' the data and labels list have different sizes!")
        return


    ###########
    # TRAINER #
    ###########

    print("Creating Training Packs from Datasets...")
    trainers = []
    for i in range(len(data)):
        t = Trainer(labels[i], data[i])
        trainers.append(t)

    if expand_packs:
        print("Expanding Training Packs...")
        for t in trainers:
            t.expand_packages(lpv_mpc_eefig.nphi)

    print("Adding Training Packs...")
    general_trainer = trainers[0]
    for i in range(1, len(trainers)):
        general_trainer.add_trainers(trainers[i])
    
    print("Success!\n")


    #################
    # INIT GRANULES #
    #################

    print("Granules WLS initialization...")

    # Generate Granule With Cluster Points
    granules = []
    WLS_packages = general_trainer.get_WLS_packages()

    # Create the Granule
    for key in range(max(WLS_packages.keys()) + 1): # Process the dictionary in numerical order.
                                                    # This is important to update the Granules using RLS latter.

        cluster_points = WLS_packages[key][0]
        granule = Granule(lpv_mpc_eefig.p, cluster_points)
        granules.append(granule)

    # Change EEFIG List For New List (of Granules)
    lpv_mpc_eefig.set_granules(granules)

    # Obtain the A & B Matrices
    for idx in range(lpv_mpc_eefig.nEEFIG):

        cluster_points = WLS_packages[idx][0]

        psik = cluster_points[:-1, :] # only old data
        xr = cluster_points[1:, 0:lpv_mpc_eefig.nx] # xr contains the states x of the buffer (eq. 24)
        
        lpv_mpc_eefig.create_using_WLS(idx, xr, psik)

    print("Success!\n")

    if remove:
        lpv_mpc_eefig = remove_failed_granules(lpv_mpc_eefig)


    ################
    # RLS GRANULES #
    ################

    print("Granules RLS update...")

    RLS_packages = general_trainer.get_RLS_packages()

    for idx_granule in RLS_packages:
        for pack in RLS_packages[idx_granule]:
            lpv_mpc_eefig.update_using_RLS(idx_granule, pack[-1],  pack[:-1])

    print("Success!\n")


    ##########
    # OTHERS #
    ##########

    if remove:
        lpv_mpc_eefig = remove_failed_granules(lpv_mpc_eefig)

    return lpv_mpc_eefig


# Remove Failed Granules
def remove_failed_granules (lpv_mpc_eefig):
    
    granules = []

    # Remove Granule With No A and B Matrices:
    for i in range(lpv_mpc_eefig.nEEFIG):
        if lpv_mpc_eefig.EEFIG[i].A.size == 0:
            print("WARNING: Granule/CLuster number {} has been removed because matrices A & B are empty.".format(i))
        else:
            granules.append(lpv_mpc_eefig.EEFIG[i])
    
    # Change EEFIG List For New List
    lpv_mpc_eefig.set_granules(granules)

    return lpv_mpc_eefig


# Find Best Cluster
def compute_silhouette (data_normalized, debug=True, n_clusters_range=(3, 40)):

    min_clusters, max_clusters = n_clusters_range

    # Find the Best Cluster
    all_scores = []
    best_score = 0
    best_n_clusters = min_clusters
    for n_clusters in range(min_clusters, max_clusters):

        print("compute silhouette -> n_cluster = {}".format(n_clusters))

        clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(data_normalized)
        
        # Silhouette Score
        score = silhouette_score(data_normalized, clusters.labels_)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
        
        all_scores.append(score)

    # Plot Scores
    if debug:
        print("Number of clusters with the best score: {}".format(best_n_clusters))
        print("(best score): {:.4f}".format(best_score))

        plt.plot(all_scores)
        plt.title("Silhouette Score for Best Cluster Size Fit")
        plt.show()

    return best_n_clusters


# Divide in Training Pack for WLS (Window Least Sqare) LPV-Matrix

# NOTE: The objective of to divide the data in packs of continious data points which have the same labels.
#       Each one of these packs can be used to obtin the LPV matrix using WLS. For multiple packs in the same cluster (label)
#       we will average the resulting matrices.

class Trainer (object):

    def __init__ (self, _labels, _data):

        self._labels = _labels
        self._data = _data

        self.n = len(self._labels)

        self.compute_packages()


    """
    Find each package
    """
    def compute_packages (self):

        # These 3 arrays should be treated as a structure class
        self.label  = np.array([self._labels[0]])   # label id
        self.length = np.array([1])                 # has length 1
        self.start  = np.array([0])                 # starts in index 0

        for i in range(1, self.n):
            
            # new label
            if self._labels[i-1] != self._labels[i]:
                self.label  = np.hstack([self.label, np.array([self._labels[i]])])  # label id
                self.length = np.hstack([self.length, np.array([1])])               # has length 1
                self.start  = np.hstack([self.start, np.array([i])])                # starts in index "i"

            # same label
            else:
                self.length[-1] += 1    # increment length
        

    """
    Given a minimum number of points per package we expand the package to achieve this threshold
    """
    def expand_packages (self, threshold):

        under_performing_packages = np.where(self.length < threshold)[0]

        # expand packages
        for idx_package in under_performing_packages:

            label = self.label[idx_package]
            flag_start = True

            while self.length[idx_package] - threshold < 0:

                # NOTE: this value switches from 0 to 1 every iteration.
                #       flag to add point at beginning or end
                if flag_start:
                    flag_start = False
                else:
                    flag_start = True

                start   = self.start[idx_package]
                length  = self.length[idx_package]

                # add point at the beginning
                if flag_start and start > 0:

                    self.start[idx_package]  -= 1
                    self.length[idx_package] += 1

                # add point at the end
                elif not flag_start and start + (length-1) < self.n:

                    self.length[idx_package] += 1

                else:
                    print("error")
                

    def add_trainers (self, addvalue):

        self._labels = np.hstack([self._labels, addvalue._labels])
        self._data   = np.vstack([self._data,   addvalue._data])

        self.label  = np.hstack([self.label,    addvalue.label])
        self.length = np.hstack([self.length,   addvalue.length])
        self.start  = np.hstack([self.start,    addvalue.start + self.n]) # careful
        
        self.n += addvalue.n


    """
    Returns a dictionary of packages were keys are labels
    """
    def get_RLS_packages (self):

        packages = {}

        for idx_package in range(len(self.label)):

            label   = self.label[idx_package]
            start   = self.start[idx_package]
            length  = self.length[idx_package]

            if label not in packages:
                packages[label] = [self._data[start:start+length]]
            else:
                packages[label].append(self._data[start:start+length])

        for idx in packages:
            packages[idx].pop(0)    # Remove the first package because it is contained in WLS_packages, 
                                    # thus it has already been used to initialize a Granule.
        return packages
    

    """
    Returns the first set of time continuous xk values to initialize all labeled Granules.
    Returns a dictionary of packages were keys are labels
    """
    def get_WLS_packages (self):

        packages = {}

        for idx_package in range(len(self.label)):

            label   = self.label[idx_package]
            start   = self.start[idx_package]
            length  = self.length[idx_package]

            if label not in packages:
                packages[label] = [self._data[start:start+length]]
            else:
                continue

        return packages


# AUXILIARY PLOT FUNCTION
# Get Different Colors For Each Pack
def get_cmap (n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)