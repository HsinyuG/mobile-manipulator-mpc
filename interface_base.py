import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import copy

import simulation.albert_robot as sim

class Interface:
    def __init__(self, dt, t_total, x_start, pose_target, controller, physical_sim=False):
        '''
        TODO: 
        1. assign values
        2. compute global reference trajectory 
        '''
        self.dt = dt
        self.desired_t_total = t_total
        self.pose_target = pose_target
        self.x_start = x_start
        self.controller = controller
        self.physical_sim = physical_sim

        self.x_log = []
        self.u_log = []

        # counters and states
        self.sim_dt = 0.01
        self.timer_counter = 0
        self.mpc_step_counter = 0
        self.is_active = False

        if self.physical_sim: 
            init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            idx_3dof = np.array([4,6,8])
            idx_base = np.array([0,1,2])
            init_state[idx_base] = self.x_start[:len(idx_base)]
            self.env, self.ob = sim.setup_environment(render=True, reconfigure_camera=True, obstacles=True, mode='vel',\
                initial_state=init_state, dt=self.sim_dt) 
                # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger] TODO: check this

            self.vel_command_sim = np.zeros(2) # V and omega, integrated by dV and dw command and used by simulation

        self.globalPlan2D()

    def run(self, ):
        '''
        TODO: 
        while(not end)
            1 call observationCallback to get current state (or its estimation)
            2 check if we finish the task or need to change the task
            3 calculate local reference for each optimization problem
            4 solve the mpc problem
            5 apply commands to simulation, get raw observation
            (6 calculate expected robot state if simulation part is not yet implemented)
        '''
        self.current_state = self.x_start if self.physical_sim is False else None # processed in observationCallback() in loop
        self.task_flag = 'move'
        self.is_active = True
        self.mpc_step_counter = 0
        while(self.is_active):
            self.pseudoTimer()

    
    def pseudoTimer(self):
        """ 
        This function acts as a timer, synchronized with the simulation time.
        No need to acquire sim time because gym won't update unless step() is called
        """
        if int(self.dt / self.sim_dt) <= 1:
            self.timerCallback()
            return
        if self.timer_counter == 0:
            self.timerCallback()
        else: self.actuateBase()
        self.timer_counter += 1
        if self.timer_counter == int(self.dt / self.sim_dt):
            self.timer_counter = 0

        
    def timerCallback(self):
        self.mpc_step_counter += 1
        print(self.mpc_step_counter, end=': ')

        # step 1
        if self.physical_sim is True: 
            self.observationCallback()
        print(self.current_state)
        self.x_log.append(copy.deepcopy(self.current_state))

        # step 2
        self.checkFinish2D()
        if self.task_flag == 'finish': 
            self.is_active = False # can be calling another global planner
            return

        # step 3
        # self.calcLocalRefTraj([0]) # 1d
        # self.calcLocalRefTraj([0,1,2]) # manipulator
        
        # self.calcLocalRefPose() # manipulator
        if self.task_flag == 'approach':
            self.calcLocalRefPose()
        else: 
            self.calcLocalRefTraj([0,1,2]) # base, with orientation error when finding corresponding point on global ref traj

        # step 4
        self.command = self.controller.solve(self.current_state, self.local_traj_ref, self.local_u_ref)
        self.u_log.append(copy.deepcopy(np.asarray(self.command)))

        # step 5
        if self.physical_sim is True: 
            self.actuateBase()

        # step 6
        if self.physical_sim is False:
            self.current_state = np.asarray(self.controller.f_dynamics(self.current_state, self.command)).squeeze()


    def globalPlan1D(self):
        '''
        TODO: 
        compute global reference trajectory 
        '''
        traj_length = int(self.desired_t_total/self.dt)

        # only give the positions, velocity references are not used thus will not be panalized
        self.traj_ref = np.array([
            np.linspace(self.x_start[0], self.pose_target[0], int(traj_length + 1)),
            np.zeros(int(traj_length + 1))
        ]).T

        self.u_ref = np.zeros(traj_length, 1)

    
    def globalPlan2D(self):
        '''
        TODO: 
        compute global reference trajectory 
        '''
        traj_length = int(self.desired_t_total/self.dt)

        # only give the positions, linear velocity references are not given thus will not be panalized
        self.traj_ref = np.array([
            np.linspace(self.x_start[0], self.pose_target[0], int(traj_length + 1)),
            np.linspace(self.x_start[1], self.pose_target[1], int(traj_length + 1)),
            np.linspace(self.x_start[2], self.pose_target[2], int(traj_length + 1)),
            # np.zeros(int(traj_length + 1)),
            np.zeros(int(traj_length + 1)),
            np.zeros(int(traj_length + 1)),
            np.zeros(int(traj_length + 1))
        ]).T

        self.u_ref = np.zeros((traj_length,2))


    def globalPlanManipulator(self):
        '''
        x_start: joint angles
        pose_target: endpoint pose
        '''
        traj_length = int(self.desired_t_total/self.dt)
        ''' pose_start, _, _ = np.asarray(self.controller.robot_model.forward_tranformation(self.x_start)).squeeze()
        self.traj_ref = np.array([
            np.linspace(pose_start[0], self.pose_target[0], int(traj_length + 1)),
            np.linspace(pose_start[1], self.pose_target[1], int(traj_length + 1)),
            np.linspace(pose_start[2], self.pose_target[2], int(traj_length + 1)),
        ]).T

        not working because linear reference trajectory in cartesian space cause infeasibility'''

        x_target = self.controller.robot_model.inverse_transformation(self.x_start, self.pose_target)
        self.traj_ref = np.array([
            np.linspace(self.x_start[0], x_target[0], int(traj_length + 1)),
            np.linspace(self.x_start[1], x_target[1], int(traj_length + 1)),
            np.linspace(self.x_start[2], x_target[2], int(traj_length + 1)),
        ]).T

        self.u_ref = np.zeros((traj_length, 3))


    def checkFinish1D(self):
        '''
        check if we reach the goal
        '''
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= 0.5) and \
            (abs(self.current_state[1] - self.u_ref[-1, 0]) <= 1e-2):
            self.task_flag = 'finish'


    def checkFinish2D(self):
        '''
        check if we reach the goal
        '''
        
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= 2) and \
            (abs(self.current_state[1] - self.traj_ref[-1, 1]) <= 2) and \
            self.task_flag == 'move': 
            # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
            # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
            self.task_flag = 'approach'
            self.controller.setWeight(
                P = np.diag([5, 5, 5, 0, 0, 1.]), # 5e2 1
                Q = np.diag([5, 5, 5, 0, 0, 1.]),
                # R = np.diag([0, 0])
            )
            N = self.controller.N
            self.controller.opti.subject_to(self.controller.X[N, :] == self.controller.X_ref[N, :])

        threshold = 0.01
        if (abs(self.current_state[0] - self.traj_ref[-1, 0]) <= threshold) and \
            (abs(self.current_state[1] - self.traj_ref[-1, 1]) <= threshold) and \
            (abs(self.current_state[2] - self.traj_ref[-1, 2]) <= 10*ca.pi/180): 
            # (abs(self.command[0,0] - self.u_ref[-1, 0]) <= 1e-2) and \
            # (abs(self.command[0,1] - self.u_ref[-1, 1]) <= 1e-2)
            self.task_flag = 'finish'


    def checkFinishManipulator(self):
        '''
        check if manipulator reach the target pose
        '''
        if (ca.norm_2(self.current_joints_pose[:3] - self.pose_target) <= 0.02):
            self.task_flag = 'finish'

    def calcLocalRefTraj(self, distance_index):
        '''
        x_ref: global trajectory given in cartesian space
        distance_index: list, the indices of distance variable in the state vector
        TODO: 
        compute global reference trajectory 
        if the goal lies within the horizon, repeat the last reference point
        '''
        distance_index = np.asarray(distance_index)
        min_distance = 1e5
        min_idx = -1e5
        for i, x_ref in enumerate(self.traj_ref):
            # if state_space == 'c':
            distance = np.linalg.norm(self.current_state[distance_index] - x_ref[distance_index])
            # elif state_space == 'j':
            #     current_pose, _, _ = np.asarray(self.controller.robot_model.forward_tranformation(self.current_state)).squeeze()
            #     distance = np.linalg.norm(current_pose - x_ref[distance_index])
            # else: raise ValueError("state_space is c or j!")

            if distance < min_distance:
                min_distance = distance
                min_idx = i

        terminal_index = min_idx + self.controller.N + 1   # reference (N+1) states and N inputs    
        if terminal_index <= self.traj_ref.shape[0]:
            self.local_traj_ref = self.traj_ref[min_idx : terminal_index]
            self.local_u_ref = self.u_ref[min_idx : terminal_index-1]
        else:
            # print(self.local_traj_ref.shape)
            # print(self.local_u_ref.shape)

            last_traj_ref = self.traj_ref[-1]
            last_u_ref = self.u_ref[-1]
            repeat_times = terminal_index - self.traj_ref.shape[0]
            self.local_traj_ref = np.vstack([self.traj_ref[min_idx :], np.tile(last_traj_ref, (repeat_times, 1))])
            self.local_u_ref = np.vstack([self.u_ref[min_idx :], np.tile(last_u_ref, (repeat_times, 1))])

            # print(self.local_traj_ref.shape)
            # print(self.local_u_ref.shape)
            # print(self.local_traj_ref)
        
        assert self.local_traj_ref.shape[0] == self.controller.N + 1
        assert self.local_u_ref.shape[0] == self.controller.N

    def calcLocalRefPose(self):
        # self.local_traj_ref = np.tile(self.pose_target, (self.controller.N + 1, 1))
        self.local_traj_ref = np.tile(self.traj_ref[-1], (self.controller.N + 1, 1))
        self.local_u_ref = np.tile(self.u_ref[-1], (self.controller.N, 1))

    def observationCallback(self):
        '''
        TODO: get the state feedback in simulation, 
        e.g. 
        self.observation = some function from simulation
        self.current_state = somehow(self.observation)
        '''
        # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger]
        idx_3dof = np.array([4,6,8])
        idx_base = np.array([0,1,2])
        if self.current_state is None:
            self.current_state = np.hstack([
                self.ob[0]['robot_0']['joint_state']['position'][idx_base],
                self.ob[0]['robot_0']['joint_state']['velocity'][idx_base]
            ])
        else: self.current_state = np.hstack([
                self.ob['robot_0']['joint_state']['position'][idx_base],
                self.ob['robot_0']['joint_state']['velocity'][idx_base]
            ])
        # self.current_state = copy.deepcopy(self.current_state)
        # self.current_state[2] -= ca.pi
    def actuateBase(self):
        '''
        TODO: use the return of mpc.solve() to set commands in simulation
        e.g. 
        some function from simulation(self.command)
        '''
        if not self.physical_sim: return
        action = np.zeros(self.env.n())
        # shape=(11,), [v, w, joint1~7, left finger, right finger]
        idx_3dof = np.array([3,5,7])
        idx_base = np.array([0,1])
        self.vel_command_sim += self.sim_dt * self.command
        action[idx_base] = self.vel_command_sim
        self.ob = sim.run_step(self.env, action)

    def actuate3DoFManipulator(self):
        if not self.physical_sim: return
        action = np.zeros(self.env.n())
        # shape=(11,), [v, w, joint1~7, left finger, right finger]
        idx_3dof = np.array([3,5,7])
        action[idx_3dof] = self.command
        self.ob = sim.run_step(self.env, action)

    def plot1D(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        plt.subplot(411)
        plt.plot(t, self.x_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('p')
        plt.grid()

        plt.subplot(412)
        plt.plot(t, self.x_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('v')
        plt.grid()

        plt.subplot(413)
        plt.plot(t[:-1], self.u_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('a')
        plt.grid()

        plt.subplot(414)
        plt.plot(t[:self.traj_ref.shape[0]], self.traj_ref[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('p ref')
        plt.grid()        

        plt.show()

    def plotManipulator(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        self.manipulator_pose_log = np.asanyarray(self.manipulator_pose_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        # q1 q2 q3
        plt.figure(1)
        plt.subplot(121)
        print(self.x_log[1])
        plt.plot(t, self.x_log[:, 0], label='q1')
        plt.plot(t, self.x_log[:, 1], label='q2')
        plt.plot(t, self.x_log[:, 2], label='q3')
        plt.xlabel('Time Step')
        plt.ylabel('joint angles')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(t[:-1], self.u_log[:, 0], label='dq1')
        plt.plot(t[:-1], self.u_log[:, 1], label='dq2')
        plt.plot(t[:-1], self.u_log[:, 2], label='dq3')
        plt.xlabel('Time Step')
        plt.ylabel('joint velocities')
        plt.legend()
        plt.grid()

        plt.show(block=False)

        # x y z
        plt.figure(2)   
        plt.subplot(131)
        plt.plot(t, self.manipulator_pose_log[:, 0], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 1], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 2], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('end point')
        plt.legend()
        plt.grid()

        plt.subplot(132)
        plt.plot(t, self.manipulator_pose_log[:, 3], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 4], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 5], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('joint 2')
        plt.legend()
        plt.grid()       

        plt.subplot(133)
        plt.plot(t, self.manipulator_pose_log[:, 6], label='x')
        plt.plot(t, self.manipulator_pose_log[:, 7], label='y')
        plt.plot(t, self.manipulator_pose_log[:, 8], label='z')
        plt.xlabel('Time Step')
        plt.ylabel('joint 3')
        plt.legend()
        plt.grid()   

        plt.show()

    def plot2D(self):
        self.x_log = np.asarray(self.x_log)
        self.u_log = np.asarray(self.u_log)
        # Plot the results
        t = np.arange(len(self.x_log))

        # x y psi
        plt.figure()
        plt.subplot(311)
        plt.plot(t, self.x_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('x')
        plt.grid()

        plt.subplot(312)
        plt.plot(t, self.x_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('y')
        plt.grid()

        plt.subplot(313)
        plt.plot(t, self.x_log[:, 2])
        plt.xlabel('Time Step')
        plt.ylabel('psi')
        plt.grid()

        # dx, dy, dpsi
        plt.figure()
        plt.subplot(311)
        plt.plot(t, self.x_log[:, 3])
        plt.xlabel('Time Step')
        plt.ylabel('dx')
        plt.grid()

        plt.subplot(312)
        plt.plot(t, self.x_log[:, 4])
        plt.xlabel('Time Step')
        plt.ylabel('dy')
        plt.grid()

        plt.subplot(313)
        plt.plot(t, self.x_log[:, 5])
        plt.xlabel('Time Step')
        plt.ylabel('dpsi')
        plt.grid()
        
        # dV, dw
        plt.figure()
        plt.subplot(211)
        plt.plot(t[:-1], self.u_log[:, 0])
        plt.xlabel('Time Step')
        plt.ylabel('v_dot')
        plt.grid()

        plt.subplot(212)
        plt.plot(t[:-1], self.u_log[:, 1])
        plt.xlabel('Time Step')
        plt.ylabel('w_dot')
        plt.grid()
        
        # obs
        plt.figure()
        plt.plot(self.x_log[:, 0], self.x_log[:, 1],label = 'actual postion')
        plt.plot(self.traj_ref[:, 0], self.traj_ref[:, 1],label = 'reference position')
        for obs in self.controller.obstacle_list:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='green', fill=False)
            plt.gca().add_artist(circle)
            plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid()
        plt.show()