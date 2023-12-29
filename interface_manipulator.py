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

        self.manipulator_pose_log = [] # [endpoint x y z, joint 2 x y z, joint 3 x y z], shape=(9,)
        self.x_log = [] # q1 q2 q3 for 3Dof manipulator
        self.u_log = []

        # counters and states
        self.sim_dt = 0.01
        self.timer_counter = 0
        self.mpc_step_counter = 0
        self.is_active = False

        if self.physical_sim: 
            init_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            idx_3dof = np.array([4,6,8])
            init_state[idx_3dof] = self.x_start
            self.env, self.ob = sim.setup_environment(render=True, reconfigure_camera=False, obstacles=True, mode='vel',\
                initial_state=init_state, dt=self.sim_dt) 
                # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger] TODO: check this

        # self.globalPlan1D() # no need, just use single point as reference?
        self.globalPlanManipulator()

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
        self.task_flag = 'in progress'
        self.is_active = True
        self.mpc_step_counter = 0
        while(self.is_active):
            self.pseudoTimer()

    
    def pseudoTimer(self):
        """ 
        This function acts as a timer, synchronized with the simulation time.
        No need to acquire sim time because gym won't update unless step() is called
        """
        if self.timer_counter == 0:
            self.timerCallback()
        else: self.actuate3DoFManipulator()
        self.timer_counter += 1
        if self.timer_counter == int(self.dt / self.sim_dt) - 1:
            self.timer_counter = 0

        
    def timerCallback(self):
        self.mpc_step_counter += 1
        print(self.mpc_step_counter, end=': ')

        # step 1
        if self.physical_sim is True: 
            self.observationCallback()
        print(self.current_state)
        self.x_log.append(copy.deepcopy(self.current_state))
        self.current_joints_pose = np.asarray(self.controller.robot_model.forward_tranformation(self.current_state)).flatten()
        self.manipulator_pose_log.append(copy.deepcopy(self.current_joints_pose))

        # step 2
        # self.checkFinish1D()
        self.checkFinishManipulator()
        if self.task_flag != 'in progress': 
            self.is_active = False # can be calling another global planner
            return

        # step 3
        # self.calcLocalRefTraj([0])
        self.calcLocalRefTraj([0,1,2])
        # self.calcLocalRefPose()

        # step 4
        self.command = self.controller.solve(self.current_state, self.local_traj_ref, self.local_u_ref)
        self.u_log.append(copy.deepcopy(np.asarray(self.command)))

        # step 5
        if self.physical_sim is True: 
            # self.actuate()
            self.actuate3DoFManipulator()

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
        self.local_traj_ref = np.tile(self.pose_target, (self.controller.N + 1, 1))
        self.local_u_ref = np.zeros((self.controller.N, 3))

    def observationCallback(self):
        '''
        TODO: get the state feedback in simulation, 
        e.g. 
        self.observation = some function from simulation
        self.current_state = somehow(self.observation)
        '''
        # shape=(12,), [x, y, yaw, joint1~7, left finger, right finger]
        idx_3dof = np.array([4,6,8])
        if self.current_state is None:
            self.current_state = self.ob[0]['robot_0']['joint_state']['position'][idx_3dof]
        else: self.current_state = self.ob['robot_0']['joint_state']['position'][idx_3dof]

    def autuate(self):
        '''
        TODO: use the return of mpc.solve() to set commands in simulation
        e.g. 
        some function from simulation(self.command)
        '''
        pass

    def actuate3DoFManipulator(self):
        action = np.zeros(self.env.n()) # shape=(11,), [v, w, joint1~7, left finger, right finger]
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