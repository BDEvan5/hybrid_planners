from hybrid_planners.f110_gym.f110_env import F110Env
from hybrid_planners.Utils.utils import *
from hybrid_planners.Planners.PurePursuit import PurePursuit
from hybrid_planners.Planners.follow_the_gap import FollowTheGap
from hybrid_planners.Planners.AgentPlanner import TestVehicle

import numpy as np
import time

class TestLogger:
    def __init__(self):
        self.path = "Data/Logs"  
        self.env_log = "/test_log.csv"
        self.lap = 0

        with open(self.path + self.env_log, "w") as f:
            print(f"file cleaned")
            # f.write("act_steer, act_vel, pos_x, pos_y, theta, v\n")

    def write_env_log(self, data):
        with open(self.path + self.env_log, "a") as f:
            f.write(data)

class VehicleStateHistory:
    def __init__(self, path, vehicle_name):
        self.vehicle_name = vehicle_name
        self.path = "Data/Vehicles/" + path + vehicle_name+ "/"
        self.states = []
        self.actions = []
    

    def add_state(self, state):
        self.states.append(state)
    
    def add_action(self, action):
        self.actions.append(action)
    
    def save_history(self, lap_n=0):
        states = np.array(self.states)
        self.actions.append(np.array([0, 0])) # last action to equal lengths
        actions = np.array(self.actions)

        lap_history = np.concatenate((states, actions), axis=1)

        np.save(self.path + f"Lap_{lap_n}_history_{self.vehicle_name}.npy", lap_history)

        self.states = []
        self.actions = []


class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data, self.test_params = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.prev_obs = None

        self.race_track = None
        self.map_name = None
        self.reward = None

        # flags 
        self.show_test = self.test_params.show_test
        self.show_train = self.test_params.show_train
        self.verbose = self.test_params.verbose
        self.logging = self.test_params.logging

        self.logger = TestLogger()
        self.vehicle_state_history = None

    def run_testing_evaluation(self):
        for run in self.run_data:
            self.env = F110Env(map=run.map_name)
            self.map_name = run.map_name

            if run.planner == "PP": self.planner = PurePursuit(self.conf, run)
            elif run.planner == "FGM": self.planner = FollowTheGap(self.conf, run)
            else: self.planner = TestVehicle(run, self.conf)
            self.vehicle_state_history = VehicleStateHistory(run.path, run.run_name)

            self.n_test_laps = self.test_params.n_test_laps
            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)

            self.env.close_rendering()

    def run_testing(self):
        assert self.env != None, "No environment created"
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.planner.plan(observation)
                observation = self.run_step(action)
                if self.show_test: self.env.render('human_fast')

            if observation['lap_done']:
                if self.verbose: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if self.verbose: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    

            self.vehicle_state_history.save_history(i)

        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)

        return eval_dict

    # this is an overide
    def run_step(self, action):
        sim_steps = self.conf.sim_steps
        if self.vehicle_state_history: 
            self.vehicle_state_history.add_action(action)

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)

        if self.logging:
            data = f"{action[0]:.2f}, {action[1]:.2f}, {obs['poses_x'][0]:.2f}, {obs['poses_y'][0]:.2f}, {obs['poses_theta'][0]}, {obs['linear_vels_x'][0]}\n"
            self.logger.write_env_log(data)
        
        return observation

    def build_observation(self, obs, done):
        """Build observation

        Returns 
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams 
            
        """
        observation = {}
        observation['current_laptime'] = obs['lap_times'][0]
        observation['scan'] = obs['scans'][0] #TODO: introduce slicing here
        
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        observation['reward'] = 0.0
        if done and obs['lap_counts'][0] == 0: 
            observation['colision_done'] = True
        if self.race_track is not None:
            if self.race_track.check_done(observation) and obs['lap_counts'][0] == 0:
                observation['colision_done'] = True

            if self.prev_obs is None: observation['progress'] = 0
            elif self.prev_obs['lap_done'] == True: observation['progress'] = 0
            else: observation['progress'] = max(self.race_track.find_progress_percent(state[0:2]), self.prev_obs['progress'])
            # self.race_track.plot_vehicle(state[0:2], state[2])
            # taking the max progress
            

        if obs['lap_counts'][0] == 1:
            observation['lap_done'] = True

        if self.reward:
            observation['reward'] = self.reward(observation, self.prev_obs)

        if self.vehicle_state_history:
            self.vehicle_state_history.add_state(obs['full_states'][0])

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]

        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if self.show_train: self.env.render('human_fast')

        self.prev_obs = None
        observation = self.build_observation(obs, done)
        if self.race_track is not None:
            self.race_track.max_distance = 0.0

        return observation




def main():
    sim = TestSimulation("BenchmarkRuns")
    sim.run_testing_evaluation()

    # sim = TrainSimulation("BenchmarkRuns")
    # sim.run_training_evaluation()

if __name__ == '__main__':
    main()



