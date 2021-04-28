#!/usr/bin/env python

from scipy import linalg, sparse
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from utilities import Curvature
import datetime
from numpy import linalg as la
import pdb
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
import rospy
from math import cos, sin, atan, pi
import osqp

solvers.options['show_progress'] = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class PathFollowingLPV_MPC:
    """Create the Path Following LMPC controller with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, Q, R, dR, N, vt, dt, map, planning_mode, Solver, steeringDelay, velocityDelay):

        # Vehicle parameters:
        # self.lf             = rospy.get_param("lf")
        # self.lr             = rospy.get_param("lr")
        # self.m              = rospy.get_param("m")
        # self.I              = rospy.get_param("Iz")
        # self.Cf             = rospy.get_param("Cf")
        # self.Cr             = rospy.get_param("Cr")
        # self.mu             = rospy.get_param("mu")

        self.slow_mode_th   = rospy.get_param("slow_mode_th")

        # Vehicle variables limits:
        self.max_vel        = rospy.get_param("max_vel")
        self.min_vel        = rospy.get_param("min_vel")
        self.accel_max      = rospy.get_param("dutycycle_max")
        self.accel_min      = rospy.get_param("dutycycle_min")
        self.steer_max      = rospy.get_param("steer_max")
        self.steer_min      = rospy.get_param("steer_min")

 
        self.A    = []
        self.B    = []
        self.C    = []
        self.N    = N
        self.n    = Q.shape[0]
        self.d    = R.shape[0]
        self.vt   = vt
        self.Q    = Q
        self.R    = R
        self.dR   = dR              # Slew rate
        self.LinPoints = np.zeros((self.N+2,self.n))
        self.dt = dt                # Sample time 33 ms

        self.map = map
        self.halfWidth = map.halfWidth

        self.first_it = 1

        self.steeringDelay = steeringDelay
        self.velocityDelay = velocityDelay

        self.OldSteering = [0.0]*int(1 + steeringDelay)

        self.OldAccelera = [0.0]*int(1)

        self.OldPredicted = [0.0]*int(1 + steeringDelay + N)

        self.Solver = Solver

        self.F, self.b = _buildMatIneqConst(self)

        self.G = []
        self.E = []
        self.L = []
        self.Eu =[]

        self.feasible = 1


# Controller.solve(LPV_States_Prediction[0,:], LPV_States_Prediction, Controller.uPred, vel_ref, curv_ref, A_L, B_L, C_L, first_it)


    def solve(self, x0, Last_xPredicted, uPred, vel_ref, curv_ref, A_L, B_L ,C_L, first_it):
        """Computes control action
        Arguments:
            x0: current state position
            EA: Last_xPredicted: it is just used for the warm up
            EA: uPred: set of last predicted control inputs used for updating matrix A LPV
            EA: A_L, B_L ,C_L: Set of LPV matrices
        """
        startTimer = datetime.datetime.now()


        #Euge: No tengo claro por que hacia esto... repasar en algun momento
        if first_it < 5:
            self.A, self.B, self.C  = _EstimateABC(self, Last_xPredicted, uPred, curv_ref)
        else:
            self.A = A_L
            self.B = B_L
            self.C = C_L

        self.G, self.E, self.L, self.Eu  = _buildMatEqConst(self) # It's been introduced the C matrix (L in the output)

        self.M, self.q          = _buildMatCost(self, uPred[0,:], vel_ref)

        endTimer                = datetime.datetime.now()
        deltaTimer              = endTimer - startTimer
        self.linearizationTime  = deltaTimer

        M = self.M   #P
        q = self.q   #q
        G = self.G   #A,  A * x == b
        E = self.E   #b, A * x == b
        L = self.L
        Eu= self.Eu
        F = self.F   #G, G * x <= h
        b = self.b   #h, G * x <= h
        n = self.n
        N = self.N
        d = self.d

        uOld  = [self.OldSteering[0], self.OldAccelera[0]]

        startTimer = datetime.datetime.now()


        # osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
            Solve a Quadratic Program defined as:
            minimize
                (1/2) * x.T * P * x + q.T * x
            subject to
                G * x <= h
                A * x == b
        """

        res_cons, self.feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F),
         b, sparse.csr_matrix(G), np.add( np.dot(E,x0),L[:,0],np.dot(Eu,uOld) ) )

        if self.feasible == 0:
            print("QUIT...")

        Solution = res_cons.x

        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n * (N + 1))]), (N + 1, n))))
        self.uPred = np.squeeze(np.transpose(np.reshape((Solution[n * (N + 1) + np.arange(d * N)]), (N, d))))

        self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T



    def MPC_solve(self, A_vec, B_vec, u, x0, vel_ref):
    
        ## x0:: should be the inital state at i=0 horizon
        ## xr:: = should be the reference, set Q = 0 weight for states which is not
        ## required to be minimized in objective function such as v_y, s, omega
        ## we want the vehicle to follow some velocity v_x so the weight will be set for it and lateral error
        ## and deviation will also be given some weight as we want the vehicle to follow the track
        

        uOld  = [self.OldSteering[0], self.OldAccelera[0]]

        [N,nx,nu] = B_vec.shape 
        
        N = self.N
    #     ## Discretizing using Euler###
    #     A_d =  sparse.eye(nx)+ sparse.csc_matrix(A_sys*dt)
    #     B_d = sparse.csc_matrix(B_sys*dt)  
        
        ## States and control input limits ##
        
        # Objective function

        
        # #maximum values
        # v_max               = rospy.get_param("max_vel")
        # v_min               = rospy.get_param("min_vel")
        # ac_max              = rospy.get_param("dutycycle_max")
        # ac_min              = rospy.get_param("dutycycle_min")
        # str_max             = rospy.get_param("steer_max")
        # str_min             = rospy.get_param("steer_min")
        # ey_max              = rospy.get_param("lat_e_max")
        # etheta_max          = rospy.get_param("orient_e_max")

        v_max               = 5.
        v_min               = -5.
        ac_max              = 1.
        ac_min              = -1.
        str_max             = 0.25
        str_min             = -0.25
        ey_max              = 0.4
        etheta_max          = 0.4   

        dstr_max            = str_max*0.1
        dstr_min            = str_min*0.1
        dac_max             = ac_max*0.1
        dac_min             = ac_min*0.1

        vx_scale            = 1/((v_max-v_min)**2)
        acc_scale           = 1/((ac_max-ac_min)**2)
        str_scale           = 1/((str_max-str_min)**2)
        ey_scale            = 1/((ey_max+ey_max)**2)
        etheta_scale        = 1/((etheta_max+etheta_max)**2)
        dstr_scale          = 1/((dstr_max-dstr_min)**2)
        dacc_scale          = 1/((dac_max-dac_min)**2)



        ## Steering, Dutycyle
        umin = np.array([str_min, ac_min]) 
        umax = np.array([str_max, ac_max])

        # umin = np.array([-0.25, -1.]) 
        # umax = np.array([0.25, 1.])
                        
        ## vx,vy,omega,theta deviation 15deg , distance,lateral error from the track:: no constraint on s
        # xmin = np.array([-v_min, -10000, -30, -1000, -10000, -ey_max])
        # xmax = np.array([v_max, 10000, 30, 1000, 10000, ey_max])
        

        # xmin = np.array([v_min, -3., -3., -etheta_max, -10000., -ey_max])
        # xmax = np.array([v_max, 3., 3., etheta_max, 10000., ey_max])

        xmin = np.array([v_min, -3., -3., -100, -10000., -ey_max])
        xmax = np.array([v_max, 3., 3., 100, 10000., ey_max])

        Q  = 0.9 * np.array([0.6*vx_scale, 0.0, 0.00, 0.2*etheta_scale, 0.0, 0.6*ey_scale])
        R  = 0.05 * np.array([0.0005*str_scale,0.1*acc_scale])     # delta, a




        # Q = [5., 0., 0., 15., 0., 5.]
        # R = [1., 5.]
        Q = sparse.diags(Q)
        QN = Q
        R = sparse.diags(R)

        # Initial and reference states
    #     x0 = np.zeros(12)
        xr = np.array([vel_ref,0.,0.,0.,0.,0.])

        # Prediction horizon
    #     N = 10

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                               sparse.kron(sparse.eye(N), R)], format='csc')


        print "shape qn",QN.shape,'xr',xr.shape,"-QN.dot(xr)",-QN.dot(xr)
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(N*nu)])

        # - linear dynamics
        A_tr = sparse.lil_matrix((nx*len(A_vec)+nx, nx*len(A_vec)+nx))
        for i in range(1,len(A_vec)+1):
    #         print i, nx*i,nx*i+nx,(i-1)*nx,(i-1)*nx+ nx, A_vec[i-1]
            A_tr[nx*i:nx*i+nx,(i-1)*nx:(i-1)*nx+ nx] = A_vec[i-1]

        B_tr = sparse.lil_matrix((nx*len(B_vec)+nx, nu*len(B_vec)))
        for i in range(1,len(B_vec)+1):
    #         print i, nx*i,nx*i+nx,(i-1)*nx,(i-1)*nx+ nx, A_vec[i-1]
            B_tr[nx*i:nx*i+nx,(i-1)*nu:(i-1)*nu+ nu] = B_vec[i-1]
        
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + A_tr
        Bu = B_tr
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        
        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True, polish=True)
        
        
        startTimer = datetime.datetime.now()

        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            print ('OSQP did not solve the problem!')
        

        Solution = res.x

        print "Solution shape", Solution.shape
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        
        print "control to be applied",Solution[-N*nu:-(N-1)*nu]


        self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(nx * (N + 1))]), (N + 1, nx))))
        self.uPred = np.squeeze(np.transpose(np.reshape((Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu))))

        # self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T



        # self.xPred = Solution[:(N+1)*nx]
        # self.xPred = self.xPred.reshape(nx,N+1) 
        print "xppredshape", self.xPred.shape
        # self.uPred = Solution[(N+1)*nx:]
        # self.uPred = self.uPred.reshape(nu,N)
        print "uppredshape", self.uPred.shape
        # self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n * (N + 1))]), (N + 1, n))))
        # self.uPred = np.squeeze(np.transpose(np.reshape((Solution[n * (N + 1) + np.arange(d * N)]), (N, d))))

        # self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        # self.xPred = self.xPred.T
        # self.uPred = self.uPred.T

        print "\n self.uPred", self.uPred

        print "\n self.xPred", self.xPred

        


    def LPVPrediction(self, x, u, vel_ref, curv_ref, planning_mode):
        #############################################
        ## States:
        ##   long velocity    [vx]
        ##   lateral velocity [vy]
        ##   angular velocity [wz]
        ##   theta error      [epsi]
        ##   distance traveled[s]
        ##   lateral error    [ey]
        ##
        ## Control actions:
        ##   Steering angle   [delta]
        ##   Acceleration     [a]
        ##
        ## Scheduling variables:
        ##   vx, vy, epsi, ey, cur
        #############################################

        # lf  = self.lf
        # lr  = self.lr
        # m   = self.m
        # I   = self.I
        # Cf  = self.Cf
        # Cr  = self.Cr
        # mu  = self.mu


        m = 2.424;
        rho = 1.225;
        lr = 0.1203;
        lf = 0.1377;
        Cm0 = 10.1305;
        Cm1 = 1.05294;
        C0 = 3.68918;
        C1 = 0.0306803;
        Cd_A = -0.657645;
        Caf = 1.3958;
        Car = 1.6775;
        Iz = 0.02;

        STATES_vec = np.zeros((self.N, 6))

        Atv = []
        Btv = []
        Ctv = []


        for i in range(0, self.N):

            if i==0:
                states  = np.reshape(x, (6,1))

            # vx      = float(states[0])
            vy      = float(states[1])
            omega   = float(states[2])
            epsi    = float(states[3])
            s       = float(states[4])
            ey      = float(states[5])

            if s < 0:
                s = 0

            if planning_mode == 2:

                PointAndTangent = self.map.PointAndTangent
                cur     = Curvature(s, PointAndTangent) # From map
                # print "PointAndTangent",PointAndTangent,"s", s 

            else:
                cur     = float(curv_ref[i,0]) # From planner

            vx      = float(vel_ref[i,0])

            print "vx",vx
            print "delta", u[i,0] 

            # small fix for none value
            if u[i,0] is None:
                u[i,0] = 0
            if u[i,1] is None:
                u[i,1] = 0

            delta = float(u[i,0])

            F_flat = 0;
            Fry = 0;
            Frx = 0;
            
            A31 = 0;
            A11 = 0;


            eps = 0.000001
            # if abs(vx)> 0.0:

            F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));        
            Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
            A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
            A31 = -Fry*lr/((vx+eps)*Iz);
                
            A12 = omega;
            A21 = -omega;
            A22 = 0;
            
            # if abs(vy) > 0.0:
            A22 = Fry/(m*(vy+eps));

            B11 = 0;
            B31 = 0;
            B21 = 0;
            
            # if abs(delta) > 0:
            B11 = -F_flat*sin(delta)/(m*(delta+eps));
            B21 = F_flat*cos(delta)/(m*(delta+eps));    
            B31 = F_flat*cos(delta)*lf/(Iz*(delta+eps));


            B12 = (1/m)*(Cm0 - Cm1*vx);

            A51 = (1/(1-ey*cur)) * ( -cos(epsi) * cur )
            A52 = (1/(1-ey*cur)) * ( +sin(epsi)* cur )
            A61 = cos(epsi) / (1-ey*cur)
            A62 = sin(epsi) / (1-ey*cur) #should be mulitplied by -1
            A7  = sin(epsi) 
            A8  = cos(epsi)


            Ai = np.array([ [A11    ,  A12 ,   0. ,  0., 0., 0.],  # [vx]
                            [A21    ,  A22 ,   0  ,  0., 0., 0.],  # [vy]
                            [A31    ,   0 ,    0  ,  0., 0., 0.],  # [wz]
                            [A51    ,  A52 ,   1. ,  0., 0., 0.],  # [epsi]
                            [A61    ,  A62 ,   0. ,  0., 0., 0.],  # [s]
                            [A7     ,   A8 ,   0. ,  0., 0., 0.]]) # [ey]

            Bi  = np.array([[ B11, B12 ], #[delta, a]
                            [ B21, 0 ],
                            [ B31, 0 ],
                            [ 0,   0 ],
                            [ 0,   0 ],
                            [ 0,   0 ]])

            Ci  = np.array([[ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ],
                            [ 0 ]])


            Ai = np.eye(len(Ai)) + self.dt * Ai
            Bi = self.dt * Bi
            Ci = self.dt * Ci

            print "np.transpose(np.reshape(u[i,:],(1,2)))",np.transpose(np.reshape(u[i,:],(1,2)))
            states_new = np.dot(Ai, states) + np.dot(Bi, np.transpose(np.reshape(u[i,:],(1,2))))

            STATES_vec[i] = np.reshape(states_new, (6,))

            states = states_new

            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)


        return STATES_vec, np.array(Atv), np.array(Btv), np.array(Ctv)





# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================




def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    # EA: P represents the quadratic weight composed by N times Q and R matrices.
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """

    def is_pos_semidef(x):
        return np.all(np.linalg.eigvals(x) >= 0)


    def is_symmetric(a, tol=1e-8):
        return np.allclose(a, a.T, atol=tol)


    if not (is_symmetric(P.todense()) and is_pos_semidef(P.todense())):
        raise ValueError("Matrix is not symmetric positive definite")


    osqp = OSQP()
    if G is not None:
        l = -inf * ones(len(h))
        if A is not None:
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l, b])
            qp_u = hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=True, polish=True)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()

    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible



def _buildMatIneqConst(Controller):
    N = Controller.N
    n = Controller.n
    max_vel = Controller.max_vel
    min_vel = Controller.min_vel
    steer_max = Controller.steer_max
    accel_max = Controller.accel_max

    Fx = np.array([[-1., 0., 0., 0., 0., 0.],
                   [+1., 0., 0., 0., 0., 0.]])
    bx = np.array([[min_vel],
                   [max_vel]]) # vx min

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[steer_max],     # Max right Steering
                   [steer_max],     # Max left Steering
                   [accel_max],     # Max Acceleration
                   [accel_max]])    # Max DesAcceleration


    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    F = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    F_return = F

    return F_return, b



def _buildMatCost(Controller, uOld, vel_ref):
    # EA: This represents to be: [(r-x)^T * Q * (r-x)] up to N+1
    # and [u^T * R * u] up to N

    Q  = Controller.Q
    n  = Q.shape[0]
    R  = Controller.R
    dR = Controller.dR

    # P  = Controller.Q
    N  = Controller.N

    uOld  = [Controller.OldSteering[0], Controller.OldAccelera[0]]

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    #c = [R] * (N)
    c = [R + 2 * np.diag(dR)] * (N) # Need to add dR for the derivative input cost

    Mu = linalg.block_diag(*c)

    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)

    # This is without slack lane:
    M0 = linalg.block_diag(Mx, Q, Mu)

    xtrack = np.array([vel_ref[0], 0, 0, 0, 0, 0])
    for i in range(1, N):
        xtrack = np.append(xtrack, [vel_ref[i], 0, 0, 0, 0, 0])

    xtrack = np.append(xtrack, [vel_ref[-1], 0, 0, 0, 0, 0])
    #print xtrack
    #xtrack = np.append(xtrack, [vel_ref[N], 0, 0, 0, 0, 0])

    q = - 2 * np.dot(np.append(xtrack, np.zeros(R.shape[0] * N)), M0)


    # Derivative Input
    q[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    M_return = M

    return M_return, q



def _buildMatEqConst(Controller):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicted input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    # G * z = L + E * x(t) + Eu * OldInputs

    A = Controller.A
    B = Controller.B
    C = Controller.C
    N = Controller.N
    n = Controller.n
    d = Controller.d

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    #E = np.zeros((n * (N + 1), n))
    E = np.zeros((n * (N + 1) + Controller.steeringDelay, n)) #new
    E[np.arange(n)] = np.eye(n)

    Eu = np.zeros((n * (N + 1) + Controller.steeringDelay, d)) #new

    # L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
    # L = np.zeros((n * (N + 1), 1))
    # L[-1] = 1 # Summmation of lamba must add up to 1

    L = np.zeros((n * (N + 1) + Controller.steeringDelay, 1))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))

    # Delay implementation:
    if Controller.steeringDelay > 0:
        xZerosMat = np.zeros((Controller.steeringDelay, n *(N+1)))
        uZerosMat = np.zeros((Controller.steeringDelay, d * N))
        for i in range(0, Controller.steeringDelay):
            ind2Steer = i * d
            L[n * (N + 1) + i, :] = Controller.OldSteering[i+1]
            uZerosMat[i, ind2Steer] = 1.0

        Gdelay = np.hstack((xZerosMat, uZerosMat))
        G = np.vstack((G, Gdelay))

    return G, E, L, Eu





#############################################
## States:
##   long velocity    [vx]
##   lateral velocity [vy]
##   angular velocity [wz]
##   theta error      [epsi]
##   distance traveled[s]
##   lateral error    [ey]
##
## Control actions:
##   Steering angle   [delta]
##   Acceleration     [a]
##
## Scheduling variables:
##   vx
##   vy
##   epsi
##   ey
##   cur
#############################################

def _EstimateABC(Controller, Last_xPredicted, uPredicted, curv_ref):

    N   = Controller.N
    dt  = Controller.dt
    # lf  = Controller.lf
    # lr  = Controller.lr
    # m   = Controller.m
    # I   = Controller.I
    # Cf  = Controller.Cf
    # Cr  = Controller.Cr
    # mu  = Controller.mu


    m = 2.424;
    rho = 1.225;
    lr = 0.1203;
    lf = 0.1377;
    Cm0 = 10.1305;
    Cm1 = 1.05294;
    C0 = 3.68918;
    C1 = 0.0306803;
    Cd_A = -0.657645;
    Caf = 1.3958;
    Car = 1.6775;
    Iz = 0.02;

    Atv = []
    Btv = []
    Ctv = []

    for i in range(0, N):

        PointAndTangent = Controller.map.PointAndTangent

        vy      = Last_xPredicted[i,1]
        omega   = Last_xPredicted[i,2]
        epsi    = Last_xPredicted[i,3]
        s       = Last_xPredicted[i,4]
        if s < 0:
            s = 0
        ey      = Last_xPredicted[i,5]
        cur     = Curvature(s, PointAndTangent)
        vx      = Last_xPredicted[i,0]
        delta   = uPredicted[i,0]             #EA: set of predicted steering angles
        # dutycycle = uPredicted[i,1]

        # if vx < Controller.slow_mode_th:

        F_flat = 0;
        Fry = 0;
        Frx = 0;
        
        A31 = 0;
        A11 = 0;


        eps = 0.000001
        # if abs(vx)> 0.0:

        F_flat = 2*Caf*(delta- atan((vy+lf*omega)/(vx+eps)));        
        Fry = -2*Car*atan((vy - lr*omega)/(vx+eps)) ;
        A11 = -(1/m)*(C0 + C1/(vx+eps) + Cd_A*rho*vx/2);
        A31 = -Fry*lr/((vx+eps)*Iz);
            
        A12 = omega;
        A21 = -omega;
        A22 = 0;
        
        # if abs(vy) > 0.0:
        A22 = Fry/(m*(vy+eps));

        B11 = 0;
        B31 = 0;
        B21 = 0;
        
        # if abs(delta) > 0:
        B11 = -F_flat*sin(delta)/(m*(delta+eps));
        B21 = F_flat*cos(delta)/(m*(delta+eps));    
        B31 = F_flat*cos(delta)*lf/(Iz*(delta+eps));


        B12 = (1/m)*(Cm0 - Cm1*vx);

        A51 = (1/(1-ey*cur)) * ( -cos(epsi) * cur )
        A52 = (1/(1-ey*cur)) * ( +sin(epsi)* cur )
        A61 = cos(epsi) / (1-ey*cur)
        A62 = sin(epsi) / (1-ey*cur) #should be mulitplied by -1
        A7  = sin(epsi) 
        A8  = cos(epsi)


        Ai = np.array([ [A11    ,  A12 ,   0. ,  0., 0., 0.],  # [vx]
                        [A21    ,  A22 ,   0  ,  0., 0., 0.],  # [vy]
                        [A31    ,   0 ,    0  ,  0., 0., 0.],  # [wz]
                        [A51    ,  A52 ,   1. ,  0., 0., 0.],  # [epsi]
                        [A61    ,  A62 ,   0. ,  0., 0., 0.],  # [s]
                        [A7     ,   A8 ,   0. ,  0., 0., 0.]]) # [ey]

        Bi  = np.array([[ B11, B12 ], #[delta, a]
                        [ B21, 0 ],
                        [ B31, 0 ],
                        [ 0,   0 ],
                        [ 0,   0 ],
                        [ 0,   0 ]])

        Ci  = np.array([[ 0 ],
                        [ 0 ],
                        [ 0 ],
                        [ 0 ],
                        [ 0 ],
                        [ 0 ]])




        Ai = np.eye(len(Ai)) + dt * Ai
        Bi = dt * Bi
        Ci = dt * Ci

        #############################################
        Atv.append(Ai)
        Btv.append(Bi)
        Ctv.append(Ci)

    return np.array(Atv), np.array(Btv), np.array(Ctv)

