import numpy as np 
from hybrid_planners.Utils.TD3 import TD3
from hybrid_planners.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from hybrid_planners.Utils.utils import init_file_struct, calculate_speed
from hybrid_planners.Planners.Architectures import E2eArchitecture

from matplotlib import pyplot as plt



class AgentTrainer: 
    def __init__(self, run, conf):
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        if run.architecture == "E2e": self.architecture = E2eArchitecture(run, conf)
        # elif run.architecture == "Serial": self.architecture = SerialArchitecutre()
        else: raise Exception("Unknown architecture")

        self.agent = TD3(self.architecture.state_space, self.architecture.action_space, 1, run.run_name)
        self.agent.create_agent(conf.h_size)

        self.t_his = TrainHistory(run, conf)

    def plan(self, obs, add_mem_entry=True):
        nn_state = self.architecture.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_state)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        self.nn_state = nn_state
        self.nn_act = self.agent.act(self.nn_state)

        self.action = self.architecture.transform_action(self.nn_act)

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            self.t_his.add_step_data(s_prime['reward'])

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)

        self.t_his.lap_done(s_prime['reward'], s_prime['progress'], False)
        self.agent.save(self.path)

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)
        self.nn_state = None


class AgentTester:
    def __init__(self, run, conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        self.path = conf.vehicle_path + run.path + run.run_name 

        self.actor = torch.load(self.path + '/' + run.run_name + "_actor.pth")

        if run.architecture == "E2e": self.architecture = E2eArchitecture(run, conf)
        # elif run.architecture == "Serial": self.architecture = SerialArchitecutre()
        else: raise Exception("Unknown architecture")

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs):
        nn_obs = self.architecture.transform_obs(obs)

        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.architecture.transform_action(nn_action)

        return self.action 


