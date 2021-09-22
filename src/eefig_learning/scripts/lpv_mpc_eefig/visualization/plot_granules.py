# Tools - Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# Tools - Numpy
import numpy as np
import numpy.linalg as la


# Plot Animated EEFIG Granules 
class PlotGranules (object):

    def __init__(self):

        self.active = [0, 0, 0]

        self.xk = []
        self.xk_lines = {}

        self.granules = []
        self.ellipses_gran = {}

        self.tracker = []
        self.ellipses_tracker = {}

        self.granule_colors = []


    def animate(self, i):
        
        ret = [] # Return Figures

        # Modify xk
        if self.active[0]:
            for key in self.xk_lines.keys():
                dim0, dim1 = key

                self.xk_lines[key].set_xdata(self.xk[:i, dim0])
                self.xk_lines[key].set_ydata(self.xk[:i, dim1])

            ret += list(self.xk_lines.values())

        # Modify Ellipses Granule
        if self.active[1]:
            ret_ellipse_gran = []
            for key in self.ellipses_gran.keys():
                dim0, dim1 = key
                
                for j in range(len(self.granules[i])):
                    self.update_ellipse(self.ellipses_gran[key][j], self.granules[i][j][0], self.granules[i][j][1], dim0, dim1)
                    self.ellipses_gran[key][j].set_visible(True)

                    ret_ellipse_gran.append(self.ellipses_gran[key][j])

            ret += ret_ellipse_gran

        # Modify Ellipses Tracker
        if self.active[2]:
            for key in self.ellipses_tracker.keys():
                dim0, dim1 = key
                self.update_ellipse(self.ellipses_tracker[key], self.tracker[i][0], self.tracker[i][1], dim0, dim1, debug=False)    

            ret += list(self.ellipses_tracker.values())

        return ret


    def init(self):

        # Set Granule Ellipses to Invisible
        if self.active[1]:
            ret_ellipse = []
            for key in self.ellipses_gran.keys():
                for i in range(len(self.ellipses_gran[key])):
                    self.ellipses_gran[key][i].set_visible(False)
                    ret_ellipse.append(self.ellipses_gran[key][i])
                
            return ret_ellipse
        return []


    def plot (self, eefig):

        if self.active[0]:
            self.xk = np.array(self.xk) # Once all data saved change [np.array, np.array, np.array, ...] -> np.array
            frames = len(self.xk)
        elif self.active[1]:
            frames = len(self.granules)
        elif self.active[2]:
            frames = len(self.tracker)

        fig = self.generate_static_plot(eefig)
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init, frames=frames, interval=0.01, blit=True)
        return anim

    
    # NOTE: https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    # NOTE: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    def update_ellipse (self, ellipse, m, C, dim0, dim1, debug=False, nstd=2):

        try:
            C = la.inv(C) # The covariance matrix of the granule is the inverse
        except:
            return

        if debug:
            print(ellipse)

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        C = np.array([[C[dim0,dim0], C[dim0,dim1]], [C[dim1,dim0], C[dim1,dim1]]])
        vals, vecs = eigsorted(C)
                
        # Center
        center = np.array([m[dim0], m[dim1]])
        ellipse.set_center(center)

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)

        ellipse.set_height(height)
        ellipse.set_width(width)

        # Angle
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        ellipse.set_angle(theta)


    # Transform a Granule into an Ellipse
    def get_ellipse (self):

        ellipse = Ellipse(xy = (0, 0), width = 0, height = 0, angle = 0)
        return ellipse


    # This Function Generates a Representation of P Spaces With Granules & xk 
    def generate_static_plot (self, eefig):

        cols = eefig.p
        rows = eefig.p

        fig = plt.figure(figsize = (rows * 3, cols * 3))

        for i in range(1, cols * rows + 1):

            dim0 = (i - 1) // rows 
            dim1 = (i - 1) % cols

            # Skip Unnecesary Figures:
            if dim0 > dim1 - 1:
                continue

            ax = fig.add_subplot(rows - 1, cols, i) # select the plot

            # OPTIONAL - Deactivate Axis Numbers
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Set Limits
            if self.active[0]:
                porc = 0.8

                min_ = np.min(self.xk[:, dim0])
                max_ = np.max(self.xk[:, dim0])
                ax.set_xlim(min_ - abs(min_) * porc, max_ + abs(max_) * porc)

                min_ = np.min(self.xk[:, dim1])
                max_ = np.max(self.xk[:, dim1])
                ax.set_ylim(min_ - abs(min_) * porc, max_ + abs(max_) * porc)
            
            ax.set_aspect('equal')

            key = (dim0, dim1)

            # Add xk
            if self.active[0]:
                line, = ax.plot([], [], 'xg', markersize=0.4)
                self.xk_lines[key] = line

            # Add Ellipse
            if self.active[1]:

                self.get_granule_colors()

                ellipses_gran = []
                for j in range(len(self.granules[-1])):
                    ellipse = self.get_ellipse()

                    ellipse.set_edgecolor(self.granule_colors[j])
                    ellipse.set_fill(False)

                    ellipses_gran.append(ellipse)
                    ax.add_patch(ellipse)

                self.ellipses_gran[key] = ellipses_gran

            # Add Tracker
            if self.active[2]:
                ellipse = self.get_ellipse()
                ax.add_patch(ellipse)
                self.ellipses_tracker[key] = ellipse            

        return fig


    # SAVE VALUES
    def save_xk (self, xk):
        self.active[0] = 1
        self.xk.append(xk)

    def save_granules (self, eefig):
        self.active[1] = 1
        self.granules.append([[gran.m, gran.C] for gran in eefig.EEFIG])

    def save_tracker (self, eefig):
        self.active[2] = 1
        self.tracker.append([eefig.tracker_m, eefig.tracker_C])


    # Get Different Colors For Each Ellipse
    def get_cmap (self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
    
    def get_granule_colors (self):
        ngran = len(self.granules[-1])
        cmap = self.get_cmap(ngran + 1)
        for i in range(ngran):
            self.granule_colors.append(cmap(i))

    
    def show (self):
        plt.show()