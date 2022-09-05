import yaml 
import csv 
import os 
from argparse import Namespace
import shutil
import numpy as np
from numba import njit
from matplotlib import pyplot as plt



# Admin functions
def save_conf_dict(dictionary, save_name=None):
    if save_name is None:
        save_name  = dictionary["run_name"]
    path = "Data/Vehicles/" + dictionary["path"] + dictionary["run_name"] + f"/{save_name}_record.yaml"
    with open(path, 'w') as file:
        yaml.dump(dictionary, file)

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    np.random.seed(conf.random_seed)

    return conf





def init_file_struct(path):
    if os.path.exists(path):
        try:
            os.rmdir(path)
        except:
            shutil.rmtree(path)
    os.mkdir(path)

def init_reward_struct(path):
    if os.path.exists(path):
        return 
    os.mkdir(path)

@njit(cache=True)
def limit_phi(phi):
    while phi > np.pi:
        phi = phi - 2*np.pi
    while phi < -np.pi:
        phi = phi + 2*np.pi
    return phi

import matplotlib


def save_csv_array(data, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period


def true_moving_average(data, period):
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period):
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
        ret[-i-1] = t[-1]
    return ret

def setup_run_list(run_file):
    full_path =  "config/" + run_file + '.yaml'
    with open(full_path) as file:
        run_dict = yaml.load(file, Loader=yaml.FullLoader)


    run_list = []
    set_n = run_dict['set_n']
    for rep in range(run_dict['n']):
        for run in run_dict['runs']:
            run["n"] = rep
            run["set_n"] = set_n
            run['run_name'] = f"{run_dict['architecture']}_{run['map_name']}_{set_n}_{rep}"
            run['architecture'] = f"{run_dict['architecture']}"
            run['test_name'] = f"{run_dict['test_name']}"
            run['path'] = f"{run_dict['test_name']}/"
            run['test_name'] = f"{run_dict['test_name']}"
            run['racing'] = run_dict['racing']

            for key in run_dict.keys():
                if key not in run.keys() and key != "runs":
                    run[key] = run_dict[key]
            run_list.append(Namespace(**run))


    init_reward_struct("Data/Vehicles/" + run_list[0].path)

    return run_list

@njit(cache=True)
def calculate_speed(delta, f_s=0.9):
    b = 0.523
    g = 9.81
    l_d = 0.329
    # f_s = 0.7
    max_v = 6

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 0

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V


if __name__ == '__main__':


    plot_speed_profile()
