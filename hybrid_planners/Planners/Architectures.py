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
        if run.racing: self.action_space += 2

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

        return scan

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        if self.action_space == 2:
            speed = (nn_action[1] + 1) * (self.max_v  / 2 - 0.5) + 1
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
        if run.racing: self.action_space += 2

        self.pp_planner = PurePursuit(conf, run)

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
        pp_steering = self.pp_planner.plan(obs)

        nn_obs = np.concatenate((scan, np.array([pp_steering[0]])))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        if np.isnan(steering_angle):
            print(f"Steering Nannnnn")
        if self.action_space == 2:
            speed = (nn_action[1] + 1) * (self.max_v  / 2 - 0.5) + 1
        else:
            speed = self.vehicle_speed

        action = np.array([steering_angle, speed])

        return action


class ModArchitecture:
    def __init__(self, run, conf):
        self.state_space = conf.n_beams +1 
        self.range_finder_scale = conf.range_finder_scale
        self.n_beams = conf.n_beams
        self.max_v = conf.max_v
        self.max_steer = conf.max_steer
        self.vehicle_speed = conf.vehicle_speed

        self.action_space = 1
        if run.racing: self.action_space += 2

        self.pp_planner = PurePursuit(conf, run)
        self.pp_steering = 0

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

        nn_obs = np.concatenate((scan, np.array([self.pp_steering])))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer + self.pp_steering
        if np.isnan(steering_angle):
            print(f"Steering Nannnnn")
        if self.action_space == 2:
            speed = (nn_action[1] + 1) * (self.max_v  / 2 - 0.5) + 1
        else:
            speed = self.vehicle_speed

        action = np.array([steering_angle, speed])

        return action

