import numpy as np 
from hybrid_planners.Utils.TD3 import TD3
from hybrid_planners.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from hybrid_planners.Utils.utils import init_file_struct, calculate_speed

from matplotlib import pyplot as plt

from hybrid_planners.Planners.PurePursuit import PurePursuit

class E2eArchitecture:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = conf.max_v
        self.max_steer = conf.max_steer
        self.vehicle_speed = conf.vehicle_speed

        self.action_space = 1
        self.racing = run.racing

        self.n_scans = conf.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.n_beams))
        self.state_space *= self.n_scans

        self.angles = np.linspace(np.pi/2 * 0.7, -np.pi/2 * 0.7, self.n_beams)
        self.sines = np.sin(self.angles)
        self.cosines = np.cos(self.angles)

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scan']) 

        # plt.figure(1)
        # plt.clf()
        # plt.plot(scan*self.sines, scan*self.cosines, 'b.')
        # plt.plot(0, 0, 'ro')
        # plt.xlim(-2, 2)
        # plt.ylim(-1, 10)
        # plt.pause(0.000001)
        scaled_scan = scan/self.range_finder_scale
        scan = np.clip(scaled_scan, 0, 1)


        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        # fig, ax = plt.subplots(1, self.n_scans, num=1)
        # for i in range(self.n_scans):
        #     ax[i].cla()
        #     ax[i].plot(self.scan_buffer[i]*self.sines, self.scan_buffer[i]*self.cosines)
        #     ax[i].set_title(f"Scan {i}")
        #     ax[i].set_xlim(-0.5, 0.5)
        #     ax[i].set_ylim(0, 1.1)
        # plt.pause(0.000001)
        # plt.show()


        nn_obs = np.reshape(self.scan_buffer, (self.n_beams * self.n_scans))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        if self.racing:
            speed = calculate_speed(steering_angle)
        else:
            speed = self.vehicle_speed

        action = np.array([steering_angle, speed])

        return action


class SerialArchitecture:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams +1 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = conf.max_v
        self.max_steer = conf.max_steer
        self.vehicle_speed = conf.vehicle_speed

        self.action_space = 1
        self.racing = run.racing

        self.pp_planner = PurePursuit(conf, run)

        self.n_scans = conf.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.state_space))
        self.state_space *= self.n_scans

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """

        scan = np.array(obs['scan']) 
        scaled_scan = scan/self.range_finder_scale

        scan = np.clip(scaled_scan, 0, 1)
        pp_steering = self.pp_planner.plan(obs)[0]
        pp_steering = pp_steering / 0.4 # scale to [-1, 1] - note this is different to the rest.

        current_obs = np.concatenate((scan, np.array([pp_steering])))

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = current_obs 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = current_obs

        nn_obs = np.reshape(self.scan_buffer, (self.state_space))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        if np.isnan(steering_angle):
            print(f"Steering Nannnnn")
        if self.racing:
            speed = calculate_speed(steering_angle)
        else:
            speed = self.vehicle_speed

        action = np.array([steering_angle, speed])

        return action

class ModHistory:
    def __init__(self):
        self.pp_history = []
        self.nn_history = []
        self.act_history = []

        self.lap_n = 0

    def add(self, pp, nn, act):
        self.pp_history.append(pp)
        self.nn_history.append(nn)
        self.act_history.append(act)

    def save(self, path):
        pp = np.array(self.pp_history)
        nn = np.array(self.nn_history)
        act = np.array(self.act_history)

        save_arr = np.concatenate((pp[:, None], nn[:, None], act[:, None]), axis=1)
        np.save(path + f"ModHistory_Lap_{self.lap_n}", save_arr)

        self.lap_n += 1
        self.pp_history = []
        self.nn_history = []
        self.act_history = []


class ModArchitecture:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams +1 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = conf.max_v
        self.max_steer = conf.max_steer
        self.vehicle_speed = conf.vehicle_speed

        self.action_space = 1
        self.racing = run.racing

        self.pp_planner = PurePursuit(conf, run)
        self.pp_steering = 0

        self.n_scans = conf.n_scans
        self.scan_buffer = np.zeros((self.n_scans, self.state_space))
        self.state_space *= self.n_scans

        # self.history = ModHistory()

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """

        scan = np.array(obs['scan']) 
        scaled_scan = scan/self.range_finder_scale

        scan = np.clip(scaled_scan, 0, 1)
        self.pp_steering = self.pp_planner.plan(obs)[0]

        current_obs = np.concatenate((scan, np.array([self.pp_steering])))

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = current_obs 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = current_obs

        nn_obs = np.reshape(self.scan_buffer, (self.state_space))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer + self.pp_steering # no clipping.
        if np.isnan(steering_angle):
            print(f"Steering Nannnnn")
        if self.racing:
            speed = calculate_speed(steering_angle)
        else:
            speed = self.vehicle_speed

        action = np.array([steering_angle, speed])
        # self.history.add(self.pp_steering, nn_action[0], steering_angle)

        return action

