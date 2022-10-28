import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import csv
import glob, os 
from matplotlib.ticker import MultipleLocator

from RacingRewards.Utils.utils import *


def load_csv_data(path):
    """loads data from a csv training file

    Args:   
        path (file_path): path to the agent

    Returns:
        rewards: ndarray of rewards
        lengths: ndarray of episode lengths
        progresses: ndarray of track progresses
        laptimes: ndarray of laptimes
    """
    rewards, lengths, progresses, laptimes = [], [], [], []
    with open(f"{path}training_data_episodes.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
                progresses.append(float(row[3]))
                # laptimes.append(float(row[4]))

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    progresses = np.array(progresses)
    laptimes = np.array(laptimes)
    
    return rewards, lengths, progresses, laptimes

def plot_lap_times(path):
    name = path.split("/")[-2]
    rewards, lengths, progresses, laptimes = load_csv_data(path)
    steps = np.cumsum(lengths) / 100

    laptimes_success = laptimes[progresses>0.98]
    avg_lap_times = true_moving_average(laptimes_success, 20)
    steps_success = steps[progresses>0.98]

    laptimes_crash = laptimes[progresses<0.98]
    steps_crash = steps[progresses<0.98]

    plt.figure(1, figsize=(3.2, 2))
    plt.clf()

    plt.plot(steps_success, laptimes_success, '.', color='darkblue', markersize=4)
    plt.plot(steps_success, avg_lap_times, '-', color='red')
    plt.plot(steps_crash, laptimes_crash, '.', color='green', markersize=4)
    # plt.plot(steps_success, laptimes_success, '-')

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Laptimes (s)")
    plt.tight_layout()
    plt.grid()
    plt.ylim(0, 60)

    plt.savefig("Data/HighSpeedEval/" + f"Laptimes_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_reward_steps(path):
    rewards, lengths, progresses, _ = load_csv_data(path)
    steps = np.cumsum(lengths) / 100

    rewards_success = rewards[progresses>0.98]
    steps_success = steps[progresses>0.98]

    rewards_crash = rewards[progresses<0.98]
    steps_crash = steps[progresses<0.98]

    plt.figure(1, figsize=(3.2, 2))
    plt.plot(steps_success, rewards_success, '.', color='darkblue', markersize=3)
    plt.plot(steps_crash, rewards_crash, '.', color='green', markersize=3)
    ys = true_moving_average(rewards, 50)
    xs = np.linspace(steps[0], steps[-1], 200)
    ys = np.interp(xs, steps, ys)
    plt.plot(xs, ys, linewidth=3, color='r')
    plt.gca().get_xaxis().set_major_locator(MultipleLocator(250))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Reward per Ep.")

    plt.tight_layout()
    plt.grid()

    name = path.split("/")[-2]
    plt.savefig("Data/HighSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/LowSpeedEval/" + f"reward_steps_{name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

# thesis tests
def obstacle_training_progress_comparision(p):
    # map_name = "f1_aut"
    map_name = "columbia_small"
    set_n = 1
    n_repeats = 5

    agents = ["E2e", "Serial", "Mod"]

    steps_list, progress_list =[[] for i in range(len(agents))], [[] for i in range(len(agents))]
    for a, agent in enumerate(agents):
        for i in range(n_repeats):
            path = p + f"{agent}_{map_name}_{set_n}_{i}/"

            rewards, lengths, progresses, _ = load_csv_data(path)
            lengths = lengths[:-2] # removed end badness
            progresses = progresses[:-2]
            steps = np.cumsum(lengths) / 100
            avg_progress = true_moving_average(progresses, 20)* 100

            steps_list[a].append(steps)
            progress_list[a].append(avg_progress)


    plt.figure(1, figsize=(6, 2.5))
    plt.clf()
    colors = ["darkblue", "green", "red"]
    pp = ["#CB4335", "#2874A6", "#229954", "#D4AC0D", "#884EA0", "#BA4A00", "#17A589"]

    for i, agent in enumerate(agents):
        xs = np.linspace(0, 500, 300)
        min, max, mean = convert_to_min_max_avg(steps_list[i], progress_list[i])

        plt.plot(xs, mean, '-', color=pp[i], linewidth=2, label=agent)
        plt.gca().fill_between(xs, min, max, color=colors[i], alpha=0.2)

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.legend(loc='lower right', ncol=3)
    plt.tight_layout()
    plt.grid()

    plt.savefig(p+ f"obstacle_training_progress_{map_name}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(p+ f"obstacle_training_progress_{map_name}.svg", bbox_inches='tight', pad_inches=0)
    # plt.savefig("Data/ThesisEval/" + f"obstacle_training_progress_{map_name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def fast_repeatability(p):



    map_name = "columbia_small"
    set_n = 1

    agents = ["E2e", "Serial", "Mod"]
    # colors = ["darkblue", "green", "red"]
    pp = ["#CB4335", "#2874A6", "#229954", "#D4AC0D", "#884EA0", "#BA4A00", "#17A589"]
    

    n_repeats = 10
    for a, agent in enumerate(agents):
        steps_list = []
        progresses_list = []
        for i in range(n_repeats):
            path = p + f"{agent}_{map_name}_{set_n}_{i}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            lengths = lengths[:-2]
            progresses = progresses[:-2]
            steps = np.cumsum(lengths) / 100
            avg_progress = true_moving_average(progresses, 20)* 100
            steps_list.append(steps)
            progresses_list.append(avg_progress)

        plt.figure(a, figsize=(6, 2.5))
        plt.clf()

        xs = np.linspace(0, 1000, 300)
        for i in range(len(steps_list)):
            xs = steps_list[i]
            ys = true_moving_average(progresses_list[i], 50)
            plt.plot(xs, ys, '-', color=pp[a], linewidth=1.5)

        plt.title(agent)

        plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

        plt.xlabel("Training Steps (x100)")
        plt.ylabel("Track Progress %")
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.grid()

        plt.savefig(p + f"TrainingRepeatability_{agent}.svg", bbox_inches='tight', pad_inches=0.01)
        plt.savefig(p + f"TrainingRepeatability_{agent}.pdf", bbox_inches='tight', pad_inches=0.01)
        # plt.savefig("Data/EvalAnalysis/" + f"TrainingRepeatability_{agent}.pdf", bbox_inches='tight', pad_inches=0.01)

    plt.show()

def convert_to_min_max_avg(step_list, progress_list, length_xs=300):
    """Returns the 3 lines 
        - Minimum line
        - maximum line 
        - average line 
    """ 
    n = len(step_list)

    xs = np.arange(length_xs)
    ys = np.zeros((n, length_xs))
    for i in range(n):
        ys[i] = np.interp(xs, step_list[i], progress_list[i])

    min_line = np.min(ys, axis=0)
    max_line = np.max(ys, axis=0)
    avg_line = np.mean(ys, axis=0)

    return min_line, max_line, avg_line


p = "Data/Vehicles/BigObs5/"
# p = "Data/Vehicles/FFT2/"

obstacle_training_progress_comparision(p)

# fast_repeatability(p)


