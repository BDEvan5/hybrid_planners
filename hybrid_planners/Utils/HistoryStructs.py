import os, shutil
import csv
import numpy as np
from matplotlib import pyplot as plt
from hybrid_planners.Utils.utils import *
from matplotlib.ticker import MultipleLocator

SIZE = 20000


def plot_data(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = moving_average(values, moving_avg_period)
    plt.plot(moving_avg)    
    moving_avg = moving_average(values, moving_avg_period * 5)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period

class TrainHistory():
    def __init__(self, run, conf) -> None:
        self.path = conf.vehicle_path + run.path +  run.run_name 

        # training data
        self.ptr = 0
        self.lengths = np.zeros(SIZE)
        self.rewards = np.zeros(SIZE) 
        self.progresses = np.zeros(SIZE) 
        self.t_counter = 0 # total steps
        self.step_rewards = []
        
        # espisode data
        self.ep_counter = 0 # ep steps
        self.ep_reward = 0
        self.ep_rewards = []


    def add_step_data(self, new_r):
        self.ep_reward += new_r
        self.ep_rewards.append(new_r)
        self.ep_counter += 1
        self.t_counter += 1 
        self.step_rewards.append(new_r)

    def lap_done(self, reward, progress, show_reward=False):
        self.add_step_data(reward)
        self.lengths[self.ptr] = self.ep_counter
        self.rewards[self.ptr] = self.ep_reward
        self.progresses[self.ptr] = progress
        self.ptr += 1

        if show_reward:
            plt.figure(8)
            plt.clf()
            plt.plot(self.ep_rewards)
            plt.plot(self.ep_rewards, 'x', markersize=10)
            plt.title(f"Ep rewards: total: {self.ep_reward:.4f}")
            plt.ylim([-1.1, 1.5])
            plt.pause(0.0001)

        self.ep_counter = 0
        self.ep_reward = 0
        self.ep_rewards = []


    def print_update(self, plot_reward=True):
        if self.ptr < 10:
            return
        
        mean10 = np.mean(self.rewards[self.ptr-10:self.ptr])
        mean100 = np.mean(self.rewards[max(0, self.ptr-100):self.ptr])
        # score = moving_average(self.rewards[self.ptr-100:self.ptr], 10)
        print(f"Run: {self.t_counter} --> Moving10: {mean10:.2f} --> Moving100: {mean100:.2f}  ")
        
        if plot_reward:
            # raise NotImplementedError
            plot_data(self.rewards[0:self.ptr], figure_n=2)

    def save_csv_data(self):
        data = []
        for i in range(self.ptr):
            data.append([i, self.rewards[i], self.lengths[i], self.progresses[i]])
        save_csv_array(data, self.path + "/training_data_episodes.csv")

        data = []
        for i in range(len(self.step_rewards)):
            data.append([i, self.step_rewards[i]])
        save_csv_array(data, self.path + "/step_reward_data.csv")

        plot_data(self.rewards[0:self.ptr], figure_n=2)
        plt.figure(2)
        plt.savefig(self.path + "/training_rewards_episodes.png")
        plt.close()

        t_steps = np.cumsum(self.lengths[0:self.ptr])/100
        plt.figure(3)
        plt.clf()

        plt.plot(t_steps, self.rewards[0:self.ptr], '.', color='darkblue', markersize=4)
        if self.ptr > 20:
            plt.plot(t_steps, moving_average(self.rewards[0:self.ptr], 20), linewidth='4', color='r')
        # plt.gca().get_yaxis().set_major_locator(MultipleLocator(10))

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Reward per Episode")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_rewards_steps.png")

        plt.figure(4)
        plt.clf()
        plt.plot(t_steps, self.progresses[0:self.ptr], '.', color='darkblue', markersize=4)
        plt.plot(t_steps, true_moving_average(self.progresses[0:self.ptr], 20), linewidth='4', color='r')

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Progress")

        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path + "/training_progress_steps.png")

        plt.close()


class VehicleStateHistory:
    def __init__(self, full_path, vehicle_name):
        self.vehicle_name = vehicle_name
        self.path = "Data/Vehicles/" + full_path # path + vehicle_name+ "/"
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


