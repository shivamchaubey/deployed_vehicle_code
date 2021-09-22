# Tools
import os, sys
import numpy as np
from math import sin, cos, tan, atan, asin, acos, pi
import matplotlib.pyplot as plt
from copy import copy

abspath = os.path.dirname(os.path.abspath(__file__))
abspath = os.path.split(abspath)[0] # taking the head of the path (Floder "EEFIG_Learing")

# NOTE: Making python files in forlder scripts available to this file
sys.path.insert(1, os.path.join(abspath, "scripts"))

# Imports
from lpv_mpc_eefig.visualization import PlotGranules
from LPV_MPC_EEFIG import LPV_MPC_EEFIG

# Others
from utils import angle_wrap, load_data, normalize_brake_BCNeMotorsport, normalize_torque_BCNeMotorsport

# Settings
np.set_printoptions(precision=3, suppress=True)


"""
GROUND TRUTH LPV MODEL EEFIG
This function returns and A and B matrix that is used as a reference to check if EEFIG is working properly
"""
# LPV MATRIX: Vx,Vy,Omega - X, Y, Yaw
def LPV_model (vx, vy, omega, yaw, delta):

    # Parameters
    m       = 2.424
    rho     = 1.225
    lr      = 0.1203
    lf      = 0.1377
    Cm0     = 10.1305
    Cm1     = 1.05294
    C0      = 3.68918
    C1      = 0.0306803
    Cd_A    = -0.657645
    Caf     = 1.3958
    Car     = 1.6775
    Iz      = 0.02

    F_flat  = 0
    Fry     = 0
    Frx     = 0
    
    A31     = 0
    A11     = 0
    
    eps     = 0
    if abs(vx)> 0:
        F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)))
        Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) 
        A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2)
        A31 = -Fry*lr/((vx+eps)*Iz)
        
    A12 = omega
    A21 = -omega
    A22 = 0
    
    if abs(vy) > 0.01:
        A22 = Fry/(m*(vy+eps))

    A41 = cos(yaw)
    A42 = -sin(yaw)
    A51 = sin(yaw)
    A52 = cos(yaw)


    B12 = 0
    B32 = 0
    B22 = 0
    
    if abs(delta) > 0.202: # NOTE: You finnd this value by trying, no math behind (no even sure it is safe to do this mehhh) REVIEW
        B12 = -F_flat * sin(delta) / (m * (delta + eps))
        B22 = F_flat  * cos(delta) / (m * (delta + eps))    
        B32 = F_flat  * cos(delta) * lf / (Iz * (delta + eps))

    B11 = (1/m)*(Cm0 - Cm1*vx)
    
    # Columns: vx, vy, omega, X, Y, yaw
    A = np.array(  [[A11, A12, 0,  0,   0,  0],
                    [A21, A22, 0,  0,   0,  0],
                    [A31,  0 , 0,  0,   0,  0],
                    [A41, A42, 0,  0,   0,  0], 
                    [A51, A52, 0,  0,   0,  0],
                    [ 0 ,  0 , 1,  0,   0,  0]])
    
    # Columns: accel, delta
    B = np.array(  [[B11, B12],
                    [ 0,  B22],
                    [ 0,  B32],
                    [ 0 ,  0 ],
                    [ 0 ,  0 ],
                    [ 0 ,  0 ]])
    
    return A, B


"""
This class uses the A and B matrices, plus the control inputs u, to track the state of the vehicle.
Once finished you can plot the predicted trajectory/states of the vehicle.
"""
class State (object):

    def __init__ (self, n, dt, init_state = None):

        if init_state is None:
            self.state = np.zeros(n)
        else:
            self.state = copy(init_state)
            print("Initial State is: {}".format(self.state))
        
        self.dt = dt
        self.hist = self.state

    # Using A & B and the control inputs u, update the state
    def update (self, A, B, u):

        # angle wrap
        if wrap_flag:
            self.state[5] = angle_wrap(self.state[5])

        dstate = np.dot(A,  self.state) + np.dot(B , u)
        self.state += dstate * self.dt
        self.save_history()

    # Save Each State
    def save_history (self):
        self.hist = np.vstack([self.hist, self.state])

    # Plot (matplotlib)
    def plotXY (self, ax, label = ""):
        ax.plot(self.hist[:, 3], self.hist[:, 4], label=label) # x, y

    def plotVelocity (self, label = ""):
        plt.plot(self.hist[:, 0:2], label=label)

    def plotOrientation (self, label = ""):
        plt.plot(self.hist[:, 5], label=label)


# Main
def main():

    path_csv_file = os.path.join(abspath, "data", "csv", "2021-09-22-08-41-01.csv")
    topics = ["__time","/est_state_info/vx","/est_state_info/vy","/est_state_info/yaw_rate","/est_state_info/X","/est_state_info/Y","/est_state_info/yaw","/control/accel/data","/control/steering/data"]

    desired_dt = 1./20 # the controller frequency
    data, dt, time = load_data(path_csv_file, topics, desired_dt, mode="interp")
    print("dt: ", dt)

    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=topics[i+1])
    plt.legend(loc="upper left") # activate legend
    plt.title("Data Selected")
    plt.show()

    ###############
    # SETUP EEFIG #
    ###############

    settings = {}

    settings["nx"] = 3
    settings["nu"] = 2
    settings["lmbd"] = 0.9          # USELESS?
    settings["effective_N"] = 200   # USELESS?
    settings["thr_continuous_anomalies"] = 5
    settings["separation"] = 2
    settings["ff"] = 0.975
    settings["nphi"] = 20

    # With Online Learning ACTIVE   ~ EEFIG 
    lpv_mpc_eefig1 = LPV_MPC_EEFIG(settings)
    lpv_mpc_eefig1.load(os.path.join(abspath, "saves", "rc_car_model1"))

    # ONLY with Offline Learning    ~ EEFIG 
    lpv_mpc_eefig2 = LPV_MPC_EEFIG(settings)
    lpv_mpc_eefig2.load(os.path.join(abspath, "saves", "rc_car_model1"))

    # PLOT
    plot = PlotGranules()

    state_eefig1        = State(lpv_mpc_eefig1.nx + lpv_mpc_eefig1.nu, dt, init_state=data[0, 0:6])
    state_eefig2        = State(lpv_mpc_eefig2.nx + lpv_mpc_eefig2.nu, dt, init_state=data[0, 0:6])
    state_test_control  = State(6, dt, init_state=data[0, 0:6])

    # NOTE: Since we are testing offline learning we need to set this value to True exceptionally
    lpv_mpc_eefig2.ready = True

    ###############
    # EXEC  EEFIG #
    ###############

    i_range = len(data)
    for i in range(i_range):

        print("it: {}".format(i))

        #####################################################
        #                                                   #
        # vx:       linear velocity in X (robot frame)      #
        # vy:       linear velocity in Y (robot frame)      #
        # omega:    angular velocity in Z (robot frame)     #
        #                                                   #
        # X:        position (world frame)                  #
        # Y:        position (world frame)                  #
        # yaw:      angular orientation in Z (world frame)  #
        #                                                   #
        # accel:    acceleration                            #
        # delta:    steering angle                          #
        #                                                   #
        #####################################################

        vx, vy, omega, X, Y, yaw, accel, delta = data[i]

        # Open Loop LPV
        A_lpv, B_lpv = LPV_model(vx, vy, omega, yaw, delta)

        # print("test_control - LPV Shivam")
        # print(A_lpv)
        # print(B_lpv)
        # print("--------------------------------")

        # EEFIG        
        xk1 = np.array([state_eefig1.state[0], state_eefig1.state[1], state_eefig1.state[2], accel, delta])
        xk2 = np.array([state_eefig2.state[0], state_eefig2.state[1], state_eefig2.state[2], accel, delta])

        # --------------
        # Online Learning
        lpv_mpc_eefig1.update(xk1)
        A1, B1 = lpv_mpc_eefig1.get_LPV_matrices(xk1)
        
        # Offline
        A2, B2 = lpv_mpc_eefig2.get_LPV_matrices(xk2)
        # --------------
        
        # Expand A & B
        # LPV MATRIX: Vx,Vy,Omega - X, Y, Yaw

        # --------------
        # Online Learning
        A_lpv_eefig1 = copy(A_lpv)
        B_lpv_eefig1 = copy(B_lpv)

        A_lpv_eefig1[0:3, 0:3] = (A1 - np.eye(lpv_mpc_eefig1.nx)) / dt
        B_lpv_eefig1[0:3, 0:2] = B1 / dt

        # Offline
        A_lpv_eefig2 = copy(A_lpv)
        B_lpv_eefig2 = copy(B_lpv)

        A_lpv_eefig2[0:3, 0:3] = (A2 - np.eye(lpv_mpc_eefig2.nx)) / dt
        B_lpv_eefig2[0:3, 0:2] = B2 / dt
        # --------------

        # print("EEFIG")
        # print(A_lpv_eefig1)
        # print(B_lpv_eefig1)
        # print("--------------------------------")
        # print("difference")
        # print(A_lpv_eefig1 - A_lpv)
        # print(B_lpv_eefig1 - B_lpv)

        # Find Next State
        u = np.array([accel, delta])

        state_eefig1.update(A_lpv_eefig1, B_lpv_eefig1, u)
        state_eefig2.update(A_lpv_eefig2, B_lpv_eefig2, u)
        state_test_control.update(A_lpv, B_lpv, u)

        # print("*******************************")
        # print(lpv_mpc_eefig1.nEEFIG)
        # for j in range(lpv_mpc_eefig1.nEEFIG):
            # print(j)
            # print((lpv_mpc_eefig1.EEFIG[j].A - np.eye(lpv_mpc_eefig1.nx)) / dt)
            # print(lpv_mpc_eefig1.EEFIG[j].B / dt)
            # print("+++++++++++++++++++++++++++++++")
        
        # Plot Granules
        plot.save_xk(xk1)
        plot.save_granules(lpv_mpc_eefig1)
        plot.save_tracker(lpv_mpc_eefig1)

        # print("################################")

    anim = plot.plot(lpv_mpc_eefig1)
    plot.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    state_eefig1.plotXY(ax, "eefig")
    state_eefig2.plotXY(ax, "eefig off")

    #state_test_control.plotXY(ax, "test_control")

    ax.plot(data[:, 3], data[:, 4], label="real")

    plt.legend(loc='upper left')
    plt.show()

    # -------------------

    state_eefig1.plotVelocity("eefig")
    state_eefig2.plotVelocity("eefig off")

    #state_test_control.plotVelocity("test_control")
    plt.plot(data[:, 0:2], label="real")
    plt.legend(loc='upper left')
    plt.show()

    # -------------------

    state_eefig1.plotOrientation("eefig")
    state_eefig2.plotOrientation("eefig off")

    #state_test_control.plotOrientation("test_control")
    plt.plot(data[:, 5], label="real")
    plt.legend(loc='upper left')
    plt.show()

    return 1


if __name__ == "__main__":

    global wrap_flag
    wrap_flag = False

    main()