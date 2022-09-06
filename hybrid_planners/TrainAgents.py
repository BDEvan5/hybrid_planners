
from hybrid_planners.f110_gym.f110_env import F110Env
from hybrid_planners.Utils.utils import *
from hybrid_planners.TestPlanner import TestSimulation, VehicleStateHistory
from hybrid_planners.Planners.AgentPlanners import AgentTrainer, AgentTester

import numpy as np
import time, torch
from hybrid_planners.Utils.Reward import *


# settings
# SHOW_TRAIN = True
SHOW_TRAIN = False
SHOW_TEST = False
VERBOSE = True

class TrainSimulation(TestSimulation):
    def __init__(self, run_file):
        super().__init__(run_file)

        self.reward = None
        self.n_train_steps = None
        self.previous_observation = None

    def run_training_evaluation(self):
        for run in self.run_data:
            seed = self.conf.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

            self.env = F110Env(map=run.map_name, seed=seed)
            self.map_name = run.map_name

            self.race_track = RaceTrack(run.map_name)
            self.race_track.load_centerline()
            self.reward = CrossTrackHeadReward(self.race_track, self.conf)
            self.n_obstacles = run.n_obstacles
            self.n_train_steps = run.n_train_steps

            self.planner = AgentTrainer(run, self.conf)
            self.completed_laps = 0

            self.run_training()

            #Test
            self.planner = AgentTester(run, self.conf)

            self.vehicle_state_history = VehicleStateHistory(run.path, run.run_name)

            self.n_test_laps = run.n_test_laps

            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)

            self.env.close_rendering()

    def run_training(self):
        assert self.env != None, "No environment created"
        start_time = time.time()
        print(f"Starting Baseline Training: {self.planner.name}")

        lap_counter, crash_counter = 0, 0
        observation = self.reset_simulation()

        for i in range(self.n_train_steps):
            self.prev_obs = observation
            action = self.planner.plan(observation)
            observation = self.run_step(action)

            self.planner.agent.train()

            if SHOW_TRAIN: self.env.render('human_fast')

            if observation['lap_done'] or observation['colision_done'] or observation['current_laptime'] > self.conf.max_laptime:
                self.planner.done_entry(observation)

                if observation['lap_done']:
                    if VERBOSE: print(f"{i}::Lap Complete {lap_counter} -> FinalR: {observation['reward']:.2f} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f} -> Progress: {observation['progress']:.2f}")

                    lap_counter += 1
                    self.completed_laps += 1

                elif observation['colision_done'] or self.race_track.check_done(observation):

                    if VERBOSE: print(f"{i}::Lap Crashed -> FinalR: {observation['reward']:.2f} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f} -> Progress: {observation['progress']:.2f}")
                    crash_counter += 1

                observation = self.reset_simulation()
                
            
        self.planner.t_his.print_update(True)
        self.planner.t_his.save_csv_data()
        self.planner.agent.save(self.planner.path)

        train_time = time.time() - start_time
        print(f"Finished Training: {self.planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")







def main():
    sim = TrainSimulation("BenchmarkRuns")
    sim.run_training_evaluation()


if __name__ == '__main__':
    main()



