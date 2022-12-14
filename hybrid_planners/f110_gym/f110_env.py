# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
from PIL import Image
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from hybrid_planners.f110_gym.base_classes import Simulator

# others
import numpy as np
import os
import csv
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl
import matplotlib.pyplot as plt
# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class LinkyLogger:
    def __init__(self):
        self.path = "Data/Logs"  
        self.env_log = "/env_log.txt"
        self.lap = 0

        with open(self.path + self.env_log, "w") as f:
            f.write("Official F110 Env\n")

    def write_env_log(self, data):
        with open(self.path + self.env_log, "a") as f:
            f.write(data)


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH
    
    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            
            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
        
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
        
            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):        
        # kwargs extraction
        # print("Env has been made")
        self.logger = LinkyLogger()
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            self.map_path = "maps/" + self.map_name + ".yaml"
        except:
            raise RuntimeError("No map given")

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 1

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)
        self.empty_map_img = np.copy(self.sim.agents[0].scan_simulator.map_img)

        # stateful observations for rendering
        self.render_obs = None

        self.poses = []
        self.track_pts, self.track_widths = self.load_centerline()
        self.obs_rng = np.random.default_rng(self.seed)
        self.counter = 0

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done
        
        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2
        min_lap_time = 5
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        # closes = dist2 <= 0.02
        closes = np.sqrt(dist2) <= 0.2
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i] and self.current_time > min_lap_time:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        done = (self.collisions[self.ego_idx]) or (np.all(self.toggle_list >= 2) and self.current_time > min_lap_time)
        # This number (2) is 2x the number of laps desired
        
        # done = done and self.current_time > 10 #! this is a temporary hack for the porto map
        if self.current_time < min_lap_time:
            self.lap_counts[0] = 0

        # if done:
        #     print(f"d2: {dist2[0]} -> Distance: {np.sqrt(dist2[self.ego_idx])}")
        #     print(f"Collisions: {self.collisions[0]}")

        return done, self.toggle_list >= 4

    def check_location(self):
        location = np.array([self.poses_x[0], self.poses_y[0]])
        p_done = self.sim.agents[0].scan_simulator.check_location(location)
        if not p_done:
            return False
        print(f"Personl done called: {location}")
        return True

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        
        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }
        self.poses.append([obs['poses_x'][0], obs['poses_y'][0]])
        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep
        
        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        if self.check_location():
            obs['collisions'][0] = True
            done = True

        info = {'checkpoint_done': toggle_list}
        if done:
            self.log_data(obs)

        return obs, reward, done, info

    def log_data(self, obs):
        string = f"{self.logger.lap}: Time: {self.current_time:.2f}s Collisions: {obs['collisions'][0]} \n"
        self.logger.write_env_log(string)
        self.logger.lap += 1

    def reset(self, poses=None):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        if poses is None:
            self.start_xs = np.zeros(1)
            self.start_ys = np.zeros(1)
            self.start_thetas = np.zeros(1)
        else:
            self.start_xs = poses[:, 0]
            self.start_ys = poses[:, 1]
            self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        self.poses = []
        self.poses.append([obs['poses_x'][0], obs['poses_y'][0]])
        
        return obs, reward, done, info

    def data_reset(self):
        """
        Reset the data in the environment so the lap counting works
        """
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

    def load_centerline(self, file_name=None):
        """
        Loads a centerline from a csv file. 
        Note: the file must be in the same folder as the map which the simulator loads.

        Args:
            file_name (string): the name of a csv file with the centerline of the track in the form [x_i, y_i, w_l_i, w_r_i], location and width

        Returns:
            center_pts (np.ndarray): the loaded center point location
            widths (np.ndarray): the widths of the track
        """
        if file_name is None:
            map_path = os.path.splitext(self.map_path)[0]
            file_name = map_path + '_centerline.csv'
        else:
            file_name = self.map_path + file_name

        track = []
        with open(file_name, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            for lines in csvFile:  
                track.append(lines)
        track = np.array(track)

        center_pts = track[:, 0:2]
        widths = track[:, 2:4]

        return center_pts, widths

    def add_obstacles(self, n_obstacles, obstacle_size=[0.3, 0.3]):
        # def add_obstacles(self, n_obstacles, obstacle_size=[0.5, 0.5]):
        """
        Adds a set number of obstacles to the envioronment using the track centerline. 
        Note: this function requires a csv file with the centerline points in it which can be loaded. 
        Updates the renderer and the map kept by the laser scaner for each vehicle in the simulator

        Args:
            n_obstacles (int): number of obstacles to add
            obstacle_size (list(2)): rectangular size of obstacles
            
        Returns:
            None
        """
        ss = self.sim.agents[0].scan_simulator
        radius = 1

        obs_size_m = np.array(obstacle_size)
        obs_size_px = np.array(obs_size_m / ss.map_resolution, dtype=int)

        min_idx = int(len(self.track_pts) //10)
        # print(min_idx, len(self.track_pts))

        rand_idxs = self.obs_rng.integers(min_idx, len(self.track_pts)-min_idx, size=n_obstacles)
        rand_radii = self.obs_rng.random(size=(n_obstacles, 2)) -0.5
        rand_radii *= radius 

        obs_locations = self.track_pts[rand_idxs, :] + rand_radii
        # print(f"Obs locations: {obs_locations}")
        # print(f"Obs size: {obs_size_px}")
        # print(f"Map res: {ss.map_resolution}")
        # print(f"Map x: {ss.orig_x}")
        new_img = generate_obs_map_img(self.empty_map_img.copy(), obs_locations, ss.orig_x, ss.orig_y, obs_size_px, ss.map_resolution)

        # plt.figure(1)
        # plt.clf()
        # plt.imshow(new_img, origin='lower')
        # plt.title(f"{obs_locations}")
        # plt.pause(0.001)
        # plt.savefig(f"Data/sim_imgs/obs_map_{self.counter}.png")
        # self.counter += 1

        self.sim.update_map_img(new_img)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), num=1)
        # ax1.imshow(self.empty_map_img, origin='lower')
        # ax2.imshow(new_img, origin='lower')
        # ax3.imshow(new_img - self.empty_map_img, origin='lower')
        # # plt.show()
        # plt.pause(0.01)

        if self.renderer is not None:
            self.renderer.add_obstacles(obs_locations, obs_size_m)

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        
        if self.renderer is None:
            # first call, initialize everything
            from SuperSafety.f110_gym.rendering import EnvRenderer
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            self.renderer.update_map(self.map_name, self.map_ext)
            
        self.renderer.update_obs(self.render_obs)

        for render_callback in self.render_callbacks:
            render_callback(self.renderer)
        
        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass

    def close_rendering(self):
        if self.renderer is not None:
            self.renderer.close()        

    def save_traj(self, name="Traj"):
        poses = np.array(self.poses)
        plt.figure(1)
        plt.clf()
        plt.plot(poses[:,0], poses[:,1], 'b-')

        plt.gca().set_aspect('equal', adjustable='box')

        plt.pause(0.001)

        plt.savefig("Data/Trajectories/" + name + ".png")

        plt.show()


#TODO: njit this function
def generate_obs_map_img(map_img, obs_locations, orig_x, orig_y, obs_size_px, map_resolution):
    """Adds obstacles of the defined size to the map image.

    Args:
        map_img (nd array): image of the map image
        obs_locations (ndarray): set of coords in pixels
        orig_x (float): x offset to map original
        orig_y (float): y offset to map original
        obs_size_px (ndarray): 2x1 set of obs size in px
        map_resolution (float): meters to pixel resolution  

    Returns:
        map_img (ndarray): image with obstacles added
    """
    map_img = map_img.copy()
    for location in obs_locations:
        # convert the location to the pixel coordinates
        x = int((location[0] - orig_x) / map_resolution)
        y = int((location[1] - orig_y) / map_resolution)
        # print(f"x: {x}, y: {y}")
        map_img[y:y+obs_size_px[0], x:x+obs_size_px[1]] = 0
        # map_img != 

    return map_img
