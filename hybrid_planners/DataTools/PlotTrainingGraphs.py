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

# Slow tests
def slow_progress_training_comparision():
    p = "Data/Vehicles/SlowTests/"

    e2e_steps = []
    e2e_progresses = []
    serial_steps = []
    serial_progresses = []
    mod_steps = []
    mod_progresses = []

    map_name = "columbia_small"
    # map_name = "f1_aut"

    for i in range(5):
        path_e2e = p + f"E2e_{map_name}_1_{i}/"
        path_serial = p + f"Serial_{map_name}_1_{i}/"
        path_mod = p + f"Mod_{map_name}_1_{i}/"

        rewards_e2e, lengths_e2e, progresses_e2e, _ = load_csv_data(path_e2e)
        rewards_serial, lengths_serial, progresses_serial, _ = load_csv_data(path_serial)
        rewards_mod, lengths_mod, progresses_mod, _ = load_csv_data(path_mod)

        steps_e2e = np.cumsum(lengths_e2e) / 100
        avg_progress_e2e = true_moving_average(progresses_e2e, 20)* 100
        steps_serial = np.cumsum(lengths_serial) / 100
        avg_progress_serial = true_moving_average(progresses_serial, 20)* 100
        steps_mod = np.cumsum(lengths_mod) / 100
        avg_progress_mod = true_moving_average(progresses_mod, 20) * 100

        e2e_steps.append(steps_e2e)
        e2e_progresses.append(avg_progress_e2e)
        serial_steps.append(steps_serial)
        serial_progresses.append(avg_progress_serial)
        mod_steps.append(steps_mod)
        mod_progresses.append(avg_progress_mod)


    plt.figure(1, figsize=(6, 2.5))
    plt.clf()

    xs = np.linspace(0, 500, 300)
    min_e2e, max_e2e, mean_e2e = convert_to_min_max_avg(e2e_steps, e2e_progresses)
    min_serial, max_serial, mean_serial = convert_to_min_max_avg(serial_steps, serial_progresses)
    min_mod, max_mod, mean_mod = convert_to_min_max_avg(mod_steps, mod_progresses)

    plt.plot(xs, mean_e2e, '-', color='darkblue', linewidth=2, label='E2e')
    plt.gca().fill_between(xs, min_e2e, max_e2e, color='darkblue', alpha=0.2)
    plt.plot(xs, mean_serial, '-', color='red', linewidth=2, label='Serial')
    plt.gca().fill_between(xs, min_serial, max_serial, color='red', alpha=0.2)
    plt.plot(xs, mean_mod, '-', color='green', linewidth=2, label='Mod')
    plt.gca().fill_between(xs, min_mod, max_mod, color='green', alpha=0.2)

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.legend(loc='lower right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/LowSpeedEval/" + f"slow_progress_{map_name}.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def compare_ep_vs_step_rewards():
    n = 1
    name = "Agent_Progress_f1_esp_1_0"
    path = f"/home/benjy/Documents/AutonomousRacing/RacingRewards/Data/Vehicles/SlowTests/{name}/"
    save_path = "Data/LowSpeedEval/"
    rewards, lengths, progresses, _ = load_csv_data(path)

    plt.figure(1, figsize=(3.5,1.8))
    plt.clf()
    steps = np.cumsum(lengths) / 100
    plt.plot(steps, true_moving_average(rewards, 25), linewidth=1.5, color='darkgreen')
    plt.plot(steps, rewards, '.', color='darkblue', markersize=6)
    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Ep. Rewards")

    plt.tight_layout()
    plt.grid()

    plt.savefig(save_path + f"step_rewards_{name}.pdf", bbox_inches='tight')

    plt.pause(0.01)

    plt.figure(2, figsize=(3.5,1.8))
    plt.clf()
    steps = np.cumsum(lengths) / 100

    plt.plot(true_moving_average(rewards, 25), linewidth=1.5, color='darkgreen')
    plt.plot(rewards, '.', color='darkblue', markersize=6)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Episode Number")
    plt.ylabel("Ep. Rewards")

    plt.grid()
    plt.tight_layout()

    plt.savefig(save_path + f"ep_rewards_{name}.pdf", bbox_inches='tight')

    plt.show()

# Fast tests
def fast_reward_comparision():
    p = "Data/Vehicles/FastTests/"

    progress_steps = []
    progress_progresses = []
    cth_steps = []
    cth_progresses = []

    for i in range(4):
        path_progress = p + f"Agent_Progress_f1_esp_3_{i}/"
        path_cth = p + f"Agent_Cth_f1_esp_3_{i}/"

        rewards_progress, lengths_progress, progresses_progress, _ = load_csv_data(path_progress)
        rewards_cth, lengths_cth, progresses_cth, _ = load_csv_data(path_cth)

        steps_progress = np.cumsum(lengths_progress) / 100
        avg_progress_progress = true_moving_average(progresses_progress, 20)* 100
        steps_cth = np.cumsum(lengths_cth) / 100
        avg_progress_cth = true_moving_average(progresses_cth, 20) * 100

        progress_steps.append(steps_progress)
        progress_progresses.append(avg_progress_progress)
        cth_steps.append(steps_cth)
        cth_progresses.append(avg_progress_cth)


    plt.figure(1, figsize=(6, 2.5))
    plt.clf()

    xs = np.linspace(0, 1000, 300)
    min_progress, max_progress, mean_progress = convert_to_min_max_avg(progress_steps, progress_progresses, len(xs))
    min_cth, max_cth, mean_cth = convert_to_min_max_avg(cth_steps, cth_progresses, len(xs))

    plt.plot(xs, mean_progress, '-', color='red', linewidth=2, label='Progress')
    plt.gca().fill_between(xs, min_progress, max_progress, color='red', alpha=0.2)
    plt.plot(xs, mean_cth, '-', color='green', linewidth=2, label='Cth')
    plt.gca().fill_between(xs, min_cth, max_cth, color='green', alpha=0.2)

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.legend(loc='lower right', ncol=2)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3)
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "fast_reward_comparision.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def fast_progress_training_comparision():
    p = "Data/Vehicles/MaxSpeedTests/"

    steps_list = []
    progresses_list = []

    n_repeats = 1
    for i, v in enumerate(range(4, 9)): 
        steps_list.append([])
        progresses_list.append([])
        for j in range(n_repeats):
            path = p + f"Agent_Cth_f1_gbr_{v}_{j}/"
            rewards, lengths, progresses, _ = load_csv_data(path)
            steps = np.cumsum(lengths) / 100
            avg_progress = true_moving_average(progresses, 20)* 100
            steps_list[i].append(steps)
            progresses_list[i].append(avg_progress)

    plt.figure(1, figsize=(6, 3))

    colors = ['red', 'darkblue', 'green', 'orange', 'purple']
    labels = ['4 m/s', '5 m/s', '6 m/s', '7 m/s', '8 m/s']

    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)-1, -1, -1):
        min, max, mean = convert_to_min_max_avg(steps_list[i], progresses_list[i], len(xs))
        plt.plot(xs, mean, '-', color=colors[i], linewidth=2, label=labels[i])
        plt.gca().fill_between(xs, min, max, color=colors[i], alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "fast_progress_training_comparision.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def fast_repeatability():
    p = "Data/Vehicles/FastTests/"

    steps_list = []
    progresses_list = []

    n_repeats = 5
    for i in range(n_repeats):
        path = p + f"Agent_Cth_f1_mco_5_{i}/"
        rewards, lengths, progresses, _ = load_csv_data(path)
        steps = np.cumsum(lengths) / 100
        avg_progress = true_moving_average(progresses, 20)* 100
        steps_list.append(steps)
        progresses_list.append(avg_progress)

    plt.figure(1, figsize=(6, 2.5))

    color = 'blue'
    xs = np.linspace(0, 1000, 300)
    for i in range(len(steps_list)):
        xs = steps_list[i]
        ys = true_moving_average(progresses_list[i], 50)
        plt.plot(xs, ys, '-', color=color, linewidth=1.5)
        # plt.gca().fill_between(xs, min, max, color=color, alpha=0.2)

    plt.gca().get_yaxis().set_major_locator(MultipleLocator(25))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Track Progress %")
    plt.ylim(0, 100)
    # plt.legend(loc='lower right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(steps_list))
    plt.tight_layout()
    plt.grid()

    plt.savefig("Data/HighSpeedEval/" + "fast_repeatability.pdf", bbox_inches='tight', pad_inches=0)

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

### -------------------------------------

def plot_ep_lengths(path):
    rewards, lengths, progresses, _ = load_csv_data(path)

    plt.figure(1, (4,2.5))
    plt.clf()
    steps = np.cumsum(lengths) / 100
    plt.plot(steps, progresses, '.', color='darkblue', markersize=4)
    xs, ys =  normalised_true_moving_average(steps, progresses, 20)
    plt.plot(xs, ys, linewidth='4', color='r')
    # plt.gca().get_yaxis().set_major_locator(MultipleLocator(2))

    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Max Progess")

    plt.tight_layout()
    plt.grid()

    # if save_tex:
    #     tikzplotlib.save(path + "baseline_reward_plot.tex", strict=True, extra_axis_parameters=['height=4cm', 'width=0.5\\textwidth', 'clip mode=individual'])

    name = path.split("/")[-2]
    plt.savefig(path + f"training_progress_steps_{name}.pdf")
    plt.show()
    # plt.pause(0.001)



def generate_racing_training_graphs():
    p = "Data/Vehicles/FastTests/"

    rewards = ["Cth", "Progress"]
    for r in rewards:
        name = f"Agent_{r}_f1_esp_3_0/"

        path = p + name
        plot_reward_steps(path)
        plot_lap_times(path)


slow_progress_training_comparision()
# compare_ep_vs_step_rewards()

# fast_progress_training_comparision()
# fast_reward_comparision()
# generate_racing_training_graphs()
# fast_repeatability()


