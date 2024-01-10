import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import math

class MPCWholeBody:
    def __init__(self,
        robot, 
        obstacle_list,
        N = 10, 
        Q = 5*np.diag([5, 5, 0, 0, 0, 1, 1, 1, 1]), # x y psi dx dy dpsi q1 q2 q3
        P = 5*np.diag([5, 5, 0, 0, 0, 1, 1, 1, 1]), 
        R = np.diag([0.1, 0.1, 0.0, 0.0, 0.0]),  # dV, dw, dq1, dq2, dq3
        S = np.diag([1e5]), # slack variable s, cost += S*s**2
        W = np.diag([0, 0, 1e-1, 1e-1, 1e-1]), # ddV, ddw, ddq1, ddq2, ddq3
        ulim=np.array([[-2, -ca.pi, -1, -1, -1],[2, ca.pi, 1, 1, 1]]), # dV, dw, dq1, dq2, dq3
        xlim=np.array([
            [-100, -100, -ca.inf, -2, -2, -ca.pi, -ca.pi/2, -ca.pi, 0],
            [100, 100, ca.inf, 2, 2, ca.pi, ca.pi/2, 0, ca.pi] 
        ]), # x, y, psi, dx, dy, dpsi, q1, q2, q3
        dulim=np.array([[-ca.inf, -ca.inf, -0.5, -0.5, -0.5], [ca.inf, ca.inf, 0.5, 0.5, 0.5]])
        ):

        self.N = N
        self.Q_value = Q
        self.R_value = R
        self.P_value = P
        self.S_value = S
        self.W_value = W
        self.dt = robot.dt
        self.dulim = dulim
        self.ulim = ulim
        self.xlim = xlim
        self.f_dynamics = robot.f_kinematics # member function
        self.robot_model = robot
        self.base_radius = robot.base.base_radius() # member variable
        self.obstacle_list = obstacle_list
        self.reset()


    def obsAvoid(self, obstacle_list, x):
        g = []
        threshold = 0.0
        for obs in obstacle_list:
            g.append((obs.radius + self.base_radius) - ca.sqrt((x[0]-obs.x)**2 + (x[1]-obs.y)**2) + threshold) # should be <= 0
        return g # all elements should be <= 0

    def angleDiff(self, a, b):
        """
        input angle from any range, output a-b converted to [-pi, pi)
        in this program a and b are equivalent because we use the squared error
        """
        a = ca.fmod((a + ca.pi), (2*ca.pi)) - ca.pi # convert to [-pi.pi)
        b = ca.fmod((b + ca.pi), (2*ca.pi)) - ca.pi
        
        angle_diff = ca.if_else(
            a * b >= 0, 
            a - b, 
            ca.if_else(
                a > b, 
                ca.if_else(
                    a - b <= ca.pi, 
                    a - b, 
                    a - b - 2 * ca.pi
                ), 
                ca.if_else(
                    a - b > -ca.pi, 
                    a - b, 
                    a - b + 2 * ca.pi
                )
            )
        )

        return angle_diff

    def setWeight(self, Q=None, R=None, P=None, S=None, W=None):
        if Q is not None: 
            self.Q_value = Q
            
        if R is not None: 
            self.R_value = R
            
        if P is not None: 
            self.P_value = P
            
        if S is not None: 
            self.S_value = S

        if W is not None:
            self.W_value = W
            
        self.opti.set_value(self.Q, self.Q_value)
        self.opti.set_value(self.R, self.R_value)
        self.opti.set_value(self.P, self.P_value)
        self.opti.set_value(self.S, self.S_value)
        self.opti.set_value(self.W, self.W_value)


    def reset(self):
        # Define optimization variables
        self.opti = ca.Opti()

        '''
        reason for the implementation that X is Nx2 not 2xN: 
        given p is a vector of same dim of x
        if we use Nx2, then matrix mult = X @ p; and elementwise mult = X * p, 
        of we use 2xN, then matrix mult = (X.T @ p).T; and elementwise mult = (X.T * p).T
        '''
        self.X = self.opti.variable(self.N+1, 9)    # states = x y psi dx dy dpsi of base, q1 q2 q3 of manipulator
        self.U = self.opti.variable(self.N, 5)      # inputs = dV, dw, dq1, dq2, dq3
        self.s = self.opti.variable(self.N+1, 1)    # slack variable = largest violation of obstacle constraints

        self.U_last = self.opti.parameter(self.N, 5)
        self.X_init = self.opti.parameter(1, 9)
        
        self.X_ref = self.opti.parameter(self.N+1, 9) # x y psi dx dy dpsi of base, q1 q2 q3 of manipulator
        self.U_ref = self.opti.parameter(self.N, 5)

        self.x_guess = None
        self.u_latest = None

        self.Q = self.opti.parameter(9,9)
        self.R = self.opti.parameter(5,5)
        self.P = self.opti.parameter(9,9)
        self.S = self.opti.parameter(1,1)
        self.W = self.opti.parameter(5,5)

        self.setWeight()

        cost = 0
        # Define constraints and cost, casadi requires x[k, :] instead of x[k] (which will be shape(1,1)) when calling row vector
        for k in range(self.N):
            
            # kinematic model
            self.opti.subject_to(self.X[k+1, :] == self.f_dynamics(self.X[k, :], self.U[k, :]))
            # state error
            # if pose ref:
                # pose_endpoint, pos_joint_2, pos_joint_3 = self.robot_model.forward_tranformation(self.X[k, :])
                # state_error = pose_endpoint - self.X_ref[k, :]
            # else:
            state_error = ca.horzcat(
                self.X[k, :2] - self.X_ref[k, :2],
                self.angleDiff(self.X[k, 2], self.X_ref[k, 2]),
                self.X[k, 3:] - self.X_ref[k, 3:]
            )

            # control error
            control_error = self.U[k, :] - self.U_ref[k, :]
            # control change of manipulator
            control_change = self.U[k, :] - self.U_last[k, :]
            # add to cost
            cost += ca.mtimes([state_error, self.Q, state_error.T])
            cost += ca.mtimes([control_error, self.R, control_error.T])
            cost += ca.mtimes([control_change, self.W, control_change.T])
            # boundaries
            self.opti.subject_to(self.opti.bounded(self.ulim[0].reshape(1,-1), self.U[k, :], self.ulim[1].reshape(1,-1))) # control input
            self.opti.subject_to(self.opti.bounded(self.xlim[0].reshape(1,-1), self.X[k, :], self.xlim[1].reshape(1,-1))) # state
            self.opti.subject_to(self.opti.bounded(self.dulim[0].reshape(1,-1), control_change, self.dulim[1].reshape(1,-1))) # control input change

            # obstacles on ground
            for g in self.obsAvoid(self.obstacle_list, self.X[k,:]):
                self.opti.subject_to(g <= self.s[k])
            cost += ca.mtimes([self.s[k], self.S, self.s[k]])

            # onstacles 3D 
            # TODO
        
        # terminal state error and boundaries
        # if pose ref:
            # self.terminal_pose_endpoint, terminal_pos_joint_2, terminal_pos_joint_3 = self.robot_model.forward_tranformation(self.X[self.N, :])
            # terminal_state_error =  self.terminal_pose_endpoint - self.X_ref[self.N, :]
        # else:
        terminal_state_error =  ca.horzcat(
            self.X[self.N, :2] - self.X_ref[self.N, :2],
            self.angleDiff(self.X[self.N, 2], self.X_ref[self.N, 2]),
            self.X[self.N, 3:] - self.X_ref[self.N, 3:]
        )

        cost += ca.mtimes([terminal_state_error, self.P, terminal_state_error.T])

        self.opti.subject_to(self.X[0, :] == self.X_init) # Initial state as constraints
        self.opti.subject_to(self.opti.bounded(self.xlim[0].reshape(1,-1), self.X[self.N, :], self.xlim[1].reshape(1,-1))) # state
        
        # terminal state obstacle on ground
        for g in self.obsAvoid(self.obstacle_list, self.X[self.N,:]):
                self.opti.subject_to(g <= self.s[self.N])
        cost += ca.mtimes([self.s[self.N], self.S, self.s[self.N]])

        # terminal state onstacles 3D 
        # TODO
        
        self.opti.minimize(cost)

        # Set solver options
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}
        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_init, traj_ref, u_ref):

        # debug, disable the infeasible joint angle sensor feedback, TODO: slack variable
        x_init[6:] = np.maximum(np.minimum(x_init[6:], self.xlim[1, 6:]), self.xlim[0, 6:]).squeeze()
        x_init = np.maximum(np.minimum(x_init, self.xlim[1]), self.xlim[0]).squeeze()
        assert x_init[7] <= 0 and x_init[8] >= 0

        # Set initial guess for the optimization problem
        if self.x_guess is None:
            self.x_guess = np.ones((self.N+1, 9)) * x_init

        if self.u_latest is None:
            self.u_latest = np.zeros((self.N, 5))
        
        self.opti.set_initial(self.X, self.x_guess)
        self.opti.set_initial(self.U, self.u_latest)
        self.opti.set_initial(self.s, np.zeros((self.N+1, 1)))

        self.opti.set_value(self.X_ref, traj_ref)
        self.opti.set_value(self.U_ref, u_ref)

        # set the U_last for next solve() call
        self.opti.set_value(self.U_last, self.u_latest)

        self.opti.set_value(self.X_init, x_init)

        try:
            sol = self.opti.solve()
            # s = self.opti.debug.value(self.s) # for debug
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("here should be a debug breakpoint")
            x = self.opti.debug.value(self.X)
            for x_k in x:
                print(self.obsAvoid(self.obstacle_list, x_k))
            print("x:", self.opti.debug.value(self.X))
            print("y:", self.opti.debug.value(self.U))
            print("s:", self.opti.debug.value(self.s))
        
        ## obtain the initial guess of solutions of the next optimization problem
        self.x_guess = sol.value(self.X)
        self.u_latest = sol.value(self.U) 
        return self.u_latest[0, :]
        
