from hybrid_planners.f110_gym.f110_env import F110Env
from hybrid_planners.Utils.utils import *
from hybrid_planners.Planners.PurePursuit import PurePursuit
from hybrid_planners.Planners.follow_the_gap import FollowTheGap
from hybrid_planners.Planners.AgentPlanners import AgentTrainer, AgentTester

import numpy as np
import time, torch
from hybrid_planners.Utils.HistoryStructs import VehicleStateHistory

# settings
SHOW_TRAIN = False
SHOW_TEST = False
# SHOW_TEST = True
VERBOSE = True


class TestSimulation():
    def __init__(self, run_file: str =None):
        if run_file is not None:
            self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.prev_obs = None
        self.n_obstacles = None
        self.prev_action = None

        self.race_track = None
        self.map_name = None
        self.reward = None

        self.vehicle_state_history = None

    def run_testing_evaluation(self):
        for run in self.run_data:
            self.evaluate_run(run)

    def evaluate_run(self, run):
        seed = run.random_seed + 10*run.n
        # seed = 100
        np.random.seed(seed) # repetition seed
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)

        self.env = F110Env(map=run.map_name, seed=seed)
        self.map_name = run.map_name

        if run.architecture == "PP": self.planner = PurePursuit(self.conf, run)
        elif run.architecture == "FTG": self.planner = FollowTheGap(self.conf, run)
        else: self.planner = AgentTester(run, self.conf)
        # save_path = run.path + run.run_name + f"/TestData_{run.n_obstacles}/"
        save_path = run.path + run.run_name + f"/TestData_{run.n_obstacles}/"
        init_file_struct("Data/Vehicles/" +save_path)
        self.vehicle_state_history = VehicleStateHistory(save_path, run.run_name)

        self.n_test_laps = run.n_test_laps
        self.n_obstacles = run.n_obstacles
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
                if  SHOW_TEST: self.env.render('human_fast')

            # self.planner.architecture.history.save(self.planner.arch_path)

            if observation['lap_done']:
                if VERBOSE: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if VERBOSE: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    

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
        self.prev_action = action
        if self.vehicle_state_history: 
            self.vehicle_state_history.add_action(action)

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)

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
            observation['reward'] = self.reward(observation, self.prev_obs, self.prev_action)

        if self.vehicle_state_history:
            self.vehicle_state_history.add_state(obs['full_states'][0])

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]

        obs, step_reward, done, _ = self.env.reset(reset_pose)
        self.env.add_obstacles(self.n_obstacles)

        if SHOW_TRAIN: self.env.render('human_fast')

        self.prev_obs = None
        observation = self.build_observation(obs, done)
        if self.race_track is not None:
            self.race_track.max_distance = 0.0

        return observation




def main():
    sim = TestSimulation("RunPP")
    # sim = TestSimulation("BenchmarkRuns")
    sim.run_testing_evaluation()


def test_individual_run():
    with open("Data/Vehicles/FastTests2/Mod_columbia_small_2_1/Mod_columbia_small_2_1_record.yaml") as file:
        run_dict = yaml.load(file, Loader=yaml.FullLoader)

    run = Namespace(**run_dict)
    run.n_test_laps = 10

    sim = TestSimulation()
    sim.evaluate_run(run)


if __name__ == '__main__':
    main()
    # test_individual_run()



