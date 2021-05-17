#!/usr/bin/env python

from scipy import linalg, sparse
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
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


'''
Important issue OSQP: https://groups.google.com/g/osqp/c/ZFvblAQdUxQ
'''


class PathFollowingLPV_MPC:
    """Create the Path Following LMPC controller with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, N, vx_ref, dt, map):


        # Vehicle parameters:
        self.m     = rospy.get_param("m")
        self.rho   = rospy.get_param("rho")
        self.lr    = rospy.get_param("lr")
        self.lf    = rospy.get_param("lf")
        self.Cm0   = rospy.get_param("Cm0")
        self.Cm1   = rospy.get_param("Cm1")
        self.C0    = rospy.get_param("C0")
        self.C1    = rospy.get_param("C1")
        self.Cd_A  = rospy.get_param("Cd_A")
        self.Caf   = rospy.get_param("Caf")
        self.Car   = rospy.get_param("Car")
        self.Iz    = rospy.get_param("Iz")

    

        self.slow_mode_th   = rospy.get_param("slow_mode_th")


        #maximum values
        self.vx_max              = rospy.get_param("max_vel")
        self.vx_min              = rospy.get_param("min_vel")
        self.duty_max            = rospy.get_param("dutycycle_max")
        self.duty_min            = rospy.get_param("dutycycle_min")
        self.str_max             = rospy.get_param("steer_max")
        self.str_min             = rospy.get_param("steer_min")
        self.ey_max              = rospy.get_param("lat_e_max")
        self.etheta_max          = rospy.get_param("orient_e_max")

        self.dstr_max            = self.str_max*0.5
        self.dstr_min            = self.str_min*0.5
        self.dduty_max           = self.duty_max*0.5
        self.dduty_min           = self.duty_min*0.5

        vx_scale            = 1/((self.vx_max-self.vx_min)**2)
        duty_scale          = 1/((self.duty_max-self.duty_min)**2)
        str_scale           = 1/((self.str_max-self.str_min)**2)
        ey_scale            = 1/((self.ey_max+self.ey_max)**2)
        etheta_scale        = 1/((self.etheta_max+self.etheta_max)**2)
        dstr_scale          = 1/((self.dstr_max-self.dstr_min)**2)
        dduty_scale         = 1/((self.dduty_max-self.dduty_min)**2)

        self.Q  = 0.7 * np.array([0.4*vx_scale, 0.0, 0.00, 0.2*etheta_scale, 0.0, 0.8*ey_scale])
        self.R  = 0.1 * np.array([0.05*str_scale, 0.05*duty_scale])     # delta, a
        self.dR = 0.2 * np.array([0.2*dstr_scale,0.2*dduty_scale])  # Input rate cost u
        self.Qe = np.array([1, 0, 0, 1, 0, 1])*(10.0e8)
        
        # Create an OSQP object
        self.prob = osqp.OSQP()
        self.N    = N
        self.nx    = self.Q.shape[0]
        self.nu    = self.R.shape[0]
        self.vx_ref   = vx_ref

        self.LinPoints = np.zeros((self.N+2,self.nx))
        self.xPred = []
        self.uPred = []
        self.dt = dt                # Sample time 33 ms

        self.slew_rate_on = True
        self.soft_constraints_on = True
        self.uminus1 = np.array([0,0]).T

        self.map = map
        self.halfWidth = map.halfWidth

        self.first_it = 1

        self.feasible = 1


    def simple_MPC_setup(self, A_vec, B_vec, u, x0, vel_ref):
    
        ## x0:: should be the inital state at i=0 horizon
        ## xr:: = should be the reference, set Q = 0 weight for states which is not
        ## required to be minimized in objective function such as v_y, s, omega
        ## we want the vehicle to follow some velocity v_x so the weight will be set for it and lateral error
        ## and deviation will also be given some weight as we want the vehicle to follow the track
    

        [N,nx,nu] = B_vec.shape 
        
        N = self.N
    #     ## Discretizing using Euler###
    #     A_d =  sparse.eye(nx)+ sparse.csc_matrix(A_sys*dt)
    #     B_d = sparse.csc_matrix(B_sys*dt)  
        
        ## States and control input limits ##
        


        # umin = np.array([-0.25, -1.]) 
        # umax = np.array([0.25, 1.])
                        
        ## vx,vy,omega,theta deviation 15deg , distance,lateral error from the track:: no constraint on s
        # xmin = np.array([-v_min, -10000, -30, -1000, -10000, -ey_max])
        # xmax = np.array([v_max, 10000, 30, 1000, 10000, ey_max])
        

        # xmin = np.array([v_min, -3., -3., -etheta_max, -10000., -ey_max])
        # xmax = np.array([v_max, 3., 3., etheta_max, 10000., ey_max])

        xmin = np.array([self.vx_min, -10., -100., -100, -10000., -self.ey_max])
        xmax = np.array([self.vx_max, 10., 100., 100, 10000., self.ey_max])




        # Q = [5., 0., 0., 15., 0., 5.]
        # R = [1., 5.]
        Q = sparse.diags(self.Q)
        QN = Q
        R = sparse.diags(self.R)

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
        self.A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])
        
        # Create an OSQP object

        # Setup workspace
        self.prob.setup(P, q, self.A, self.l, self.u, warm_start=True, polish=True)
        
        

    def simple_MPC_update(self, A_vec, B_vec, u, x0, vel_ref):

        [N,nx,nu] = self.N, self.nx, self.nu 
        
        # - LPV dynamics
        A_tr = sparse.lil_matrix((nx*len(A_vec)+nx, nx*len(A_vec)+nx))
        for i in range(1,len(A_vec)+1):
            A_tr[nx*i:nx*i+nx,(i-1)*nx:(i-1)*nx+ nx] = A_vec[i-1]

        B_tr = sparse.lil_matrix((nx*len(B_vec)+nx, nu*len(B_vec)))
        for i in range(1,len(B_vec)+1):
            B_tr[nx*i:nx*i+nx,(i-1)*nu:(i-1)*nu+ nu] = B_vec[i-1]
        
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + A_tr
        Bu = B_tr
        
        Aeq = sparse.hstack([Ax, Bu])

        self.A[:Aeq.shape[0], :Aeq.shape[1]] = Aeq
        self.l[:nx] = -x0
        self.u[:nx] = -x0


        self.prob.update( Ax = self.A.data, l= self.l, u= self.u)

    def simple_MPC_solve(self):

        [N,nx,nu] = self.N, self.nx, self.nu 

        startTimer = datetime.datetime.now()

        # Solve
        res = self.prob.solve()

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

        print "xppredshape", self.xPred.shape

        print "uppredshape", self.uPred.shape


        print "\n self.uPred", self.uPred

        print "\n self.xPred", self.xPred



    def MPC_integral_solve2(self, A_vec, B_vec, u, x0, vel_ref):

        [N,nx,nu] = self.N, self.nx, self.nu 

        xr = np.array([vel_ref,0.,0.,0.,0.,0.])
        u_ref = np.array([0.0,0.0])

        umin = np.array([self.str_min, self.duty_min]) 
        umax = np.array([self.str_max, self.duty_max])

        dumin = np.array([self.dstr_min, self.dduty_min]) 
        dumax = np.array([self.dstr_max, self.dduty_max])

        xmin = np.array([self.vx_min, -10., -100., -100, -10000., -self.ey_max])
        xmax = np.array([self.vx_max, 10., 100., 100, 10000., self.ey_max])

    
        '''Objective function formulation for integral action as well as slack variables'''
        #################### P formulation ################################
        
        Q  = sparse.diags(self.Q)
        QN = Q
        R  = sparse.diags(self.R)
        dR = sparse.diags(self.dR)
        
        Qeps  = sparse.diags(self.Qe)
        

        PQx = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN], format='csc')
        PQu = sparse.kron(sparse.eye(N), R)
        idu = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
        PQdu = sparse.kron(idu, dR)
        PQeps = sparse.kron(sparse.eye(N+1), Qeps)
        
            
        #################### q formulation ################################
        

        qQx  = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr)])
        qQu  = np.kron(np.ones(N), -R.dot(u_ref))
        qQdu = np.hstack([-dR.dot(self.uminus1), np.zeros((N - 1) * nu)])
        qQeps = np.zeros((N+1)*nx)
        
        print "qQx.shape",qQx.shape, "qQu.shape", qQu.shape, 'qQdu.shape', qQdu.shape, 'qQeps.shape', qQeps.shape
        

        if self.soft_constraints_on and self.slew_rate_on:
            self.P = sparse.block_diag([PQx, PQu + PQdu, PQeps], format='csc')
            self.q = np.hstack([qQx, qQu + qQdu, qQeps])

        elif self.slew_rate_on:
            self.P = sparse.block_diag([PQx, PQu + PQdu], format='csc')
            self.q = np.hstack([qQx, qQu + qQdu])

        elif self.soft_constraints_on:
            self.P = sparse.block_diag([PQx, PQu, PQeps], format='csc')
            self.q = np.hstack([qQx, qQu, qQeps])

        else:
            self.P = sparse.block_diag([PQx, PQu], format='csc')
            self.q = np.hstack([qQx, qQu])
            
        
        '''Equality constraints'''
        
        # - LPV dynamics
        A_tr = sparse.lil_matrix((nx*len(A_vec)+nx, nx*len(A_vec)+nx))
        for i in range(1,len(A_vec)+1):
            A_tr[nx*i:nx*i+nx,(i-1)*nx:(i-1)*nx+ nx] = A_vec[i-1]

        B_tr = sparse.lil_matrix((nx*len(B_vec)+nx, nu*len(B_vec)))
        for i in range(1,len(B_vec)+1):
            B_tr[nx*i:nx*i+nx,(i-1)*nu:(i-1)*nu+ nu] = B_vec[i-1]
        
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + A_tr
        Bu = B_tr
        
        n_eps = (N + 1) * nx
        self.Aeq = sparse.hstack([Ax, Bu])
        
        if self.soft_constraints_on:
            self.Aeq = sparse.hstack([self.Aeq, sparse.csc_matrix((self.Aeq.shape[0], n_eps))])
        


        self.leq = np.hstack([-x0, np.zeros(N*nx)])
        self.ueq = self.leq

                                                      
        '''Inequality constraints'''
                             
        Aineq_x = sparse.hstack([sparse.eye((N + 1) * nx), sparse.csc_matrix(((N+1)*nx, N*nu))])
        if self.soft_constraints_on:
            Aineq_x = sparse.hstack([Aineq_x, sparse.eye(n_eps)]) # For soft constraints slack variables
        lineq_x = np.kron(np.ones(N + 1), xmin) # lower bound of inequalities
        uineq_x = np.kron(np.ones(N + 1), xmax) # upper bound of inequalities

        Aineq_u = sparse.hstack([sparse.csc_matrix((N*nu, (N+1)*nx)), sparse.eye(N * nu)])
        if self.soft_constraints_on:
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], n_eps))]) # For soft constraints slack variables
        lineq_u = np.kron(np.ones(N), umin)     # lower bound of inequalities
        uineq_u = np.kron(np.ones(N), umax)     # upper bound of inequalities


        # - bounds on \Delta u
        if self.slew_rate_on == True:
            Aineq_du = sparse.vstack([sparse.hstack([np.zeros((nu, (N + 1) * nx)), sparse.eye(nu), np.zeros((nu, (N - 1) * nu))]),  # for u0 - u-1
                                      sparse.hstack([np.zeros((N * nu, (N+1) * nx)), -sparse.eye(N * nu) + sparse.eye(N * nu, k=1)])  # for uk - uk-1, k=1...Np
                                      ]
                                     )
            if self.soft_constraints_on:
                Aineq_du = sparse.hstack([Aineq_du, sparse.csc_matrix((Aineq_du.shape[0], n_eps))])

            uineq_du = np.kron(np.ones(N+1), dumax) #np.ones((Nc+1) * nu)*Dumax
            uineq_du[0:nu] += self.uminus1[0:nu]

            lineq_du = np.kron(np.ones(N+1), dumin) #np.ones((Nc+1) * nu)*Dumin
            lineq_du[0:nu] += self.uminus1[0:nu] # works for nonscalar u?

        if self.slew_rate_on: 

            self.A = sparse.vstack([self.Aeq, Aineq_x, Aineq_u, Aineq_du]).tocsc()
            self.l = np.hstack([self.leq, lineq_x, lineq_u, lineq_du])
            self.u = np.hstack([self.ueq, uineq_x, uineq_u, uineq_du])

        else:

            self.A = sparse.vstack([self.Aeq, Aineq_x, Aineq_u]).tocsc()
            self.l = np.hstack([self.leq, lineq_x, lineq_u])
            self.u = np.hstack([self.ueq, uineq_x, uineq_u])


        # print "self.P.shape", self.P.shape, 'self.q.shape', self.q.shape, 'self.l.shape', self.l.shape, 'self.u.shape' ,self.u.shape
        
        # Setup workspace
        prob = osqp.OSQP()
        
        #self.
        prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, polish=True)
        

        res = prob.solve()
        # Check solver status
        if res.info.status != 'solved':
            print ('OSQP did not solve the problem!')
        

        Solution = res.x

        print "controller to be applied", Solution[(N+1)*nx:(N+1)*nx + nu]
        print "Solution shape", Solution.shape
        

        self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(nx * (N + 1))]), (N + 1, nx)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu)))).T

        print 'Solution', Solution





    def MPC_integral_setup(self, A_vec, B_vec, u, x0, vel_ref):

        [N,nx,nu] = self.N, self.nx, self.nu 

        xr = np.array([vel_ref,0.,0.,0.,0.,0.])
        u_ref = np.array([0.0,0.0])

        umin = np.array([self.str_min, self.duty_min]) 
        umax = np.array([self.str_max, self.duty_max])

        dumin = np.array([self.dstr_min, self.dduty_min]) 
        dumax = np.array([self.dstr_max, self.dduty_max])

        xmin = np.array([self.vx_min, -10., -100., -100, -10000., -self.ey_max])
        xmax = np.array([self.vx_max, 10., 100., 100, 10000., self.ey_max])

    
        '''Objective function formulation for integral action as well as slack variables'''
        #################### P formulation ################################
        
        Q  = sparse.diags(self.Q)
        QN = Q
        R  = sparse.diags(self.R)
        dR = sparse.diags(self.dR)
        
        Qeps  = sparse.diags(self.Qe)
        

        PQx = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN], format='csc')
        PQu = sparse.kron(sparse.eye(N), R)
        idu = (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
        PQdu = sparse.kron(idu, dR)
        PQeps = sparse.kron(sparse.eye(N+1), Qeps)
        
            
        #################### q formulation ################################
        

        qQx  = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr)])
        qQu  = np.kron(np.ones(N), -R.dot(u_ref))
        qQdu = np.hstack([-dR.dot(self.uminus1), np.zeros((N - 1) * nu)])
        qQeps = np.zeros((N+1)*nx)
        
        print "qQx.shape",qQx.shape, "qQu.shape", qQu.shape, 'qQdu.shape', qQdu.shape, 'qQeps.shape', qQeps.shape
        

        if self.soft_constraints_on and self.slew_rate_on:
            self.P = sparse.block_diag([PQx, PQu + PQdu, PQeps], format='csc')
            self.q = np.hstack([qQx, qQu + qQdu, qQeps])

        elif self.slew_rate_on:
            self.P = sparse.block_diag([PQx, PQu + PQdu], format='csc')
            self.q = np.hstack([qQx, qQu + qQdu])

        elif self.soft_constraints_on:
            self.P = sparse.block_diag([PQx, PQu, PQeps], format='csc')
            self.q = np.hstack([qQx, qQu, qQeps])

        else:
            self.P = sparse.block_diag([PQx, PQu], format='csc')
            self.q = np.hstack([qQx, qQu])
            
        
        '''Equality constraints'''
        
        # - LPV dynamics
        A_tr = sparse.lil_matrix((nx*len(A_vec)+nx, nx*len(A_vec)+nx))
        for i in range(1,len(A_vec)+1):
            A_tr[nx*i:nx*i+nx,(i-1)*nx:(i-1)*nx+ nx] = A_vec[i-1]

        B_tr = sparse.lil_matrix((nx*len(B_vec)+nx, nu*len(B_vec)))
        for i in range(1,len(B_vec)+1):
            B_tr[nx*i:nx*i+nx,(i-1)*nu:(i-1)*nu+ nu] = B_vec[i-1]
        
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + A_tr
        Bu = B_tr
        
        n_eps = (N + 1) * nx
        self.Aeq = sparse.hstack([Ax, Bu]).tocsc()
        


        # Aeqt = self.Aeq.transpose(copy=True) # Have to copy or it transposes in place

        # Aeqt.sort_indices()

        # # note that col,row are transposed

        # (self.col_indices, self.row_indices) = Aeqt.nonzero()

        # print 'row_indices, col_indices',self.row_indices, self.col_indices
        # print(self.Aeq[self.row_indices, self.col_indices].A1)

        if self.soft_constraints_on:
            self.Aeq = sparse.hstack([self.Aeq, sparse.csc_matrix((self.Aeq.shape[0], n_eps))])
        
        # print ('self.Aeq',self.Aeq.toarray()[:2*nx,:])
        # print ('self.Aeq',self.Aeq.data)

        self.leq = np.hstack([-x0, np.zeros(N*nx)])
        self.ueq = self.leq

                                                      
        '''Inequality constraints'''
                             
        self.Aineq_x = sparse.hstack([sparse.eye((N + 1) * nx), sparse.csc_matrix(((N+1)*nx, N*nu))])
        if self.soft_constraints_on:
            self.Aineq_x = sparse.hstack([self.Aineq_x, sparse.eye(n_eps)]) # For soft constraints slack variables
        self.lineq_x = np.kron(np.ones(N + 1), xmin) # lower bound of inequalities
        self.uineq_x = np.kron(np.ones(N + 1), xmax) # upper bound of inequalities

        self.Aineq_u = sparse.hstack([sparse.csc_matrix((N*nu, (N+1)*nx)), sparse.eye(N * nu)])
        if self.soft_constraints_on:
            self.Aineq_u = sparse.hstack([self.Aineq_u, sparse.csc_matrix((self.Aineq_u.shape[0], n_eps))]) # For soft constraints slack variables
        self.lineq_u = np.kron(np.ones(N), umin)     # lower bound of inequalities
        self.uineq_u = np.kron(np.ones(N), umax)     # upper bound of inequalities


        # - bounds on \Delta u
        if self.slew_rate_on == True:
            self.Aineq_du = sparse.vstack([sparse.hstack([np.zeros((nu, (N + 1) * nx)), sparse.eye(nu), np.zeros((nu, (N - 1) * nu))]),  # for u0 - u-1
                                      sparse.hstack([np.zeros((N * nu, (N+1) * nx)), -sparse.eye(N * nu) + sparse.eye(N * nu, k=1)])  # for uk - uk-1, k=1...Np
                                      ]
                                     )
            if self.soft_constraints_on:
                self.Aineq_du = sparse.hstack([self.Aineq_du, sparse.csc_matrix((self.Aineq_du.shape[0], n_eps))])

            self.uineq_du = np.kron(np.ones(N+1), dumax) #np.ones((Nc+1) * nu)*Dumax
            self.uineq_du[0:nu] += self.uminus1[0:nu]

            self.lineq_du = np.kron(np.ones(N+1), dumin) #np.ones((Nc+1) * nu)*Dumin
            self.lineq_du[0:nu] += self.uminus1[0:nu] # works for nonscalar u?



        if self.slew_rate_on: 

            self.A = sparse.vstack([self.Aeq, self.Aineq_x, self.Aineq_u, self.Aineq_du]).tocsc()
            self.l = np.hstack([self.leq, self.lineq_x, self.lineq_u, self.lineq_du])
            self.u = np.hstack([self.ueq, self.uineq_x, self.uineq_u, self.uineq_du])

        else:

            self.A = sparse.vstack([self.Aeq, self.Aineq_x, self.Aineq_u]).tocsc()
            self.l = np.hstack([self.leq, self.lineq_x, self.lineq_u])
            self.u = np.hstack([self.ueq, self.uineq_x, self.uineq_u])


        At = self.A.transpose(copy=True) # Have to copy or it transposes in place

        At.sort_indices()

        # note that col,row are transposed

        (self.col_indices, self.row_indices) = At.nonzero()
        # print "self.P.shape", self.P.shape, 'self.q.shape', self.q.shape, 'self.l.shape', self.l.shape, 'self.u.shape' ,self.u.shape
        
        # Setup workspace
        # self.prob = osqp.OSQP()
        
        print ('self.A',self.A)
        print ('self.A',self.A.data)

        print ('slef Aeq', self.Aeq)
        #self.
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, polish=True)
        

     
    def MPC_integral_update(self, A_vec, B_vec, u, x0, vel_ref):
        [N,nx,nu] = self.N, self.nx, self.nu 


        umin = np.array([self.str_min, self.duty_min]) 
        umax = np.array([self.str_max, self.duty_max])

        dumin = np.array([self.dstr_min, self.dduty_min]) 
        dumax = np.array([self.dstr_max, self.dduty_max])

        xmin = np.array([self.vx_min, -3., -3., -100, -10000., -self.ey_max])
        xmax = np.array([self.vx_max, 3., 3., 100, 10000., self.ey_max])

        '''Equality constraints'''
        
        # - LPV dynamics
        A_tr = sparse.lil_matrix((nx*len(A_vec)+nx, nx*len(A_vec)+nx))
        for i in range(1,len(A_vec)+1):
            A_tr[nx*i:nx*i+nx,(i-1)*nx:(i-1)*nx+ nx] = A_vec[i-1]

        B_tr = sparse.lil_matrix((nx*len(B_vec)+nx, nu*len(B_vec)))
        for i in range(1,len(B_vec)+1):
            B_tr[nx*i:nx*i+nx,(i-1)*nu:(i-1)*nu+ nu] = B_vec[i-1]
        
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + A_tr
        Bu = B_tr
        

        self.Aeq = sparse.hstack([Ax, Bu]).tocsc()
        print ('self.Aeq',self.Aeq)
        print ('self.Aeq',self.Aeq.data)

        if self.soft_constraints_on:
            self.Aeq = sparse.hstack([self.Aeq, sparse.csc_matrix((self.Aeq.shape[0], (N + 1) * nx))])
        
        self.A[:self.Aeq.shape[0], :self.Aeq.shape[1]] = self.Aeq


        self.l[:nx] = -x0
        self.u[:nx] = -x0


        if self.slew_rate_on == True:

            self.l[(N+1)*nx + (N+1)*nx + (N)*nu:(N+1)*nx + (N+1)*nx + (N)*nu + nu] = dumin + self.uminus1[0:nu]  # update constraint on \Delta u0: Dumin <= u0 - u_{-1}
            self.u[(N+1)*nx + (N+1)*nx + (N)*nu:(N+1)*nx + (N+1)*nx + (N)*nu + nu] = dumax + self.uminus1[0:nu]  # update constraint on \Delta u0: u0 - u_{-1} <= Dumax


        # self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, polish=True)
        # print ('self.A',self.A)
        # print ('self.A',self.A.data)

        indices = np.array(zip(self.row_indices, self.col_indices))
        
        Ax_value = self.A[self.row_indices, self.col_indices].A1


        self.prob.update( Ax = Ax_value , l= self.l, u= self.u)

        # print 'len(indices)', len(indices)
        # print 'self.Aeq[self.row_indices, self.col_indices].A1',self.Aeq[self.row_indices, self.col_indices].A1.shape
        # self.prob.update( Ax = self.Aeq[self.row_indices, self.col_indices].A1, Ax_idx = indices , l= self.l, u= self.u)

    def MPC_integral_solve(self):
        [N,nx,nu] = self.N, self.nx, self.nu 

        res = self.prob.solve()
        # Check solver status
        if res.info.status != 'solved':
            print ('OSQP did not solve the problem!')
        

        Solution = res.x

        print "controller to be applied", Solution[(N+1)*nx:(N+1)*nx + nu]
        print "Solution shape", Solution.shape
        

        self.xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(nx * (N + 1))]), (N + 1, nx)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((Solution[nx * (N + 1) + np.arange(nu * N)]), (N, nu)))).T

        print 'Solution', Solution


    def output(self, return_x_seq=False, return_u_seq=False, return_eps_seq=False, return_status=False, return_obj_val=False):
        """ Return the MPC controller output uMPC, i.e., the first element of the optimal input sequence and assign is to self.uminus1_rh.
        Parameters
        ----------
        return_x_seq : bool
                       If True, the method also returns the optimal sequence of states in the info dictionary
        return_u_seq : bool
                       If True, the method also returns the optimal sequence of inputs in the info dictionary
        return_eps_seq : bool
                       If True, the method also returns the optimal sequence of epsilon in the info dictionary
        return_status : bool
                       If True, the method also returns the optimizer status in the info dictionary
        return_obj_val : bool
                       If True, the method also returns the objective function value in the info dictionary
        Returns
        -------
        array_like (nu,)
            The first element of the optimal input sequence uMPC to be applied to the system.
        dict
            A dictionary with additional infos. It is returned only if one of the input flags return_* is set to True
        """
        Nc = self.Nc
        Np = self.Np
        nx = self.nx
        nu = self.nu

        # Extract first control input to the plant
        if self.res.info.status == 'solved':
            uMPC = self.res.x[(Np+1)*nx:(Np+1)*nx + nu]
        else:
            uMPC = self.u_failure

        # Return additional info?
        info = {}
        if return_x_seq:
            seq_X = self.res.x[0:(Np+1)*nx]
            seq_X = seq_X.reshape(-1,nx)
            info['x_seq'] = seq_X

        if return_u_seq:
            seq_U = self.res.x[(Np+1)*nx:(Np+1)*nx + Nc*nu]
            seq_U = seq_U.reshape(-1,nu)
            info['u_seq'] = seq_U

        if return_eps_seq:
            seq_eps = self.res.x[(Np+1)*nx + Nc*nu : (Np+1)*nx + Nc*nu + (Np+1)*nx ]
            seq_eps = seq_eps.reshape(-1,nx)
            info['eps_seq'] = seq_eps

        if return_status:
            info['status'] = self.res.info.status

        if return_obj_val:
            obj_val = self.res.info.obj_val + self.J_CNST # constant of the objective value
            info['obj_val'] = obj_val

        self.uminus1_rh = uMPC

        if len(info) == 0:
            return uMPC

        else:
            return uMPC, info




    def LPVPrediction(self, x, u, vel_ref):
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

        planning_mode = 2 

        STATES_vec = np.zeros((self.N, 6))

        Atv = []
        Btv = []
        Ctv = []


        for i in range(0, self.N):

            if i==0:
                states  = np.reshape(x, (6,1))

            vx      = float(states[0])
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

            # print "vx",vx
            # print "delta", u[i,0] 

            # # small fix for none value
            # if u[i,0] is None:
            #     u[i,0] = 0
            # if u[i,1] is None:
            #     u[i,1] = 0

            delta = float(u[i,0])
            dutycycle = float(u[i,1])

            # if abs(dutycycle) <= 0.15:
            #     u[i,1] = 0.  
            #     vx = 0.0           
            #     vy = 0.0

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

            states_new = np.dot(Ai, states) + np.dot(Bi, np.transpose(np.reshape(u[i,:],(1,2))))


            STATES_vec[i] = np.reshape(states_new, (6,))

            states = states_new

            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)


        return STATES_vec, np.array(Atv), np.array(Btv), np.array(Ctv)


    def LPVPrediction_setup(self, x, u, vel_ref):
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


        STATES_vec = np.zeros((self.N, 6))

        Atv = []
        Btv = []
        Ctv = []

        u = np.array([1, 1]).T
        states = np.array([1,1,1,1,1,1]).T
        for i in range(0, self.N):


            Ai = np.array([ [1.0    ,  1.0 ,   0. ,  0., 0., 0.],  # [vx]
                            [1.0    ,  1.0 ,   0  ,  0., 0., 0.],  # [vy]
                            [1.0    ,   0 ,    0  ,  0., 0., 0.],  # [wz]
                            [1.0    ,  1.0 ,   1. ,  0., 0., 0.],  # [epsi]
                            [1.0    ,  1.0 ,   0. ,  0., 0., 0.],  # [s]
                            [1.0     ,   1.0 ,   0. ,  0., 0., 0.]]) # [ey]

            Bi  = np.array([[ 1.0, 1.0 ], #[delta, a]
                            [ 1.0, 0 ],
                            [ 1.0, 0 ],
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

            states_new = np.dot(Ai, states) + np.dot(Bi,u)


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
def Curvature(s, PointAndTangent):
    """curvature and desired velocity computation
    s: curvilinear abscissa at which the curvature has to be evaluated
    PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
    """
    TrackLength = PointAndTangent[-1,3]+PointAndTangent[-1,4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength
    #     print(s)
    #     print("\n")
    # print(PointAndTangent)
    # print("\n")

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
    # print("\n")
    # print(index)
    # print(np.where(np.squeeze(index))[0])
    # print("\n")
    i = int(np.where(np.squeeze(index))[0]) #EA: this works
    #i = np.where(np.squeeze(index))[0]     #EA: this does not work

    curvature = PointAndTangent[i, 5]

    return curvature



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

