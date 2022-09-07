from matplotlib import pyplot as plt
import numpy as np
import glob
import os

from PIL import Image
import glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from hybrid_planners.DataTools.MapData import MapData
from hybrid_planners.Utils.Reward import RaceTrack 
from hybrid_planners.Utils.utils import *


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


class AnalyseTestLapData:
    def __init__(self):
        self.path = None
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.race_track = None
        self.summary_path = None
        self.lap_n = 0

    def explore_folder(self, path):
        vehicle_folders = glob.glob(f"{path}*/")
        print(f"{len(vehicle_folders)} folders found")

        for j, self.path in enumerate(vehicle_folders):
            if j >5: break
            print(f"Vehicle folder being opened: {self.path}")
            # init_file_struct(self.summary_path + "SteeringDists/")
            init_file_struct(self.path + "Trajectories/")
            # init_file_struct(self.path + "Curvatures/")
            # init_file_struct(self.path + "Hists/")
            # init_file_struct(self.path + "Velocities/")
            # init_file_struct(self.path + "FrictionPlots/")

            with open(self.path + "Statistics.txt", "w") as file:
                file.write(f"Name: {self.path}\n")
                file.write("Lap" + "Steering".rjust(16) + "Total Distance".rjust(16) + "Mean Curvature".rjust(16) + "Total Curvature".rjust(16) + "Mean Deviation".rjust(16) + "Total Deviation".rjust(16) + "Progress".rjust(16) + "Time".rjust(16) + "Avg Velocity".rjust(16) + "\n")

            self.vehicle_name = self.path.split("/")[-2]
            self.map_name = self.vehicle_name.split("_")[1]
            if self.map_name == "f1":
                self.map_name += "_" + self.vehicle_name.split("_")[2]
            elif self.map_name == "columbia": self.map_name += "_" + self.vehicle_name.split("_")[2]
            self.map_data = MapData(self.map_name)
            self.race_track = RaceTrack(self.map_name)
            self.race_track.load_centerline()

            _, _, p, _ = load_csv_data(self.path)
            print(f"{len(p)}")

            n = self.vehicle_name.split("_")[-2]
            i = self.vehicle_name.split("_")[-1].split(".")[0]
            seed =  100 + int(i) * 10
            # seed = 100
            self.obs_rng = np.random.default_rng(seed)
            for _ in range(len(p)+1):
                _b = self.obs_rng.integers(13, 120, size=6)
                _a = self.obs_rng.random(size=(6, 2)) 


            # for self.lap_n in range(2):
            for self.lap_n in range(10):
                if not self.load_lap_data(): break # no more laps
                # self.calculate_lap_statistics()
                # self.generate_steering_graphs()
                # self.plot_curvature_heat_map()

                # self.plot_velocity_heat_map()
                # self.plot_friction_graphs()
                self.plot_obs_graphs()

            self.generate_summary_stats()

    def load_lap_data(self):
        try:
            data = np.load(self.path + "TestData/" + f"Lap_{self.lap_n}_history_{self.vehicle_name}.npy")
            # data = np.load(self.path + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]



        return 1 # to say success

    def calculate_lap_statistics(self):
        if not self.load_lap_data(): return

        steering = np.abs(self.actions[:, 0])
        rms_steering = np.mean(np.abs(steering))

        pts = self.states[:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_distance = np.sum(ss)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts, ss, False)
        mean_curvature = np.mean(np.abs(ks))
        total_curvature = np.sum(np.abs(ks))

        hs = []
        for point in pts:
            idx, dists = self.race_track.get_trackline_segment(point)
            x, h = self.race_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        mean_deviation = np.mean(hs)
        total_deviation = np.sum(hs)

        time = len(pts) /10
        vs = self.states[:, 3]
        avg_velocity = np.mean(vs)

        progress = self.race_track.find_s(pts[-1])/self.race_track.total_s
        if progress < 0.01 or progress > 0.99:
            progress = 1 # it is finished

        with open(self.path + "Statistics.txt", "a") as file:
            file.write(f"{self.lap_n}, {rms_steering:14.4f}, {total_distance:14.4f}, {mean_curvature:14.4f}, {total_curvature:14.4f}, {mean_deviation:14.4f}, {total_deviation:14.4f}, {progress:14.4f}, {time:14.2f}, {avg_velocity:14.4f}\n")

    def generate_summary_stats(self):
        progress_ind = 7
        n_values = 10
        data = []
        for i in range(n_values): 
            data.append([])

        n_success, n_total = 0, 0
        progresses = []
        with open(self.path + "Statistics.txt", 'r') as file:
            lines = file.readlines()
            if len(lines) < 3: return
            
            for lap_n in range(len(lines)-2):
                line = lines[lap_n+2] # first lap is heading
                line = line.split(',')
                progress = float(line[progress_ind])
                n_total += 1
                progresses.append(progress)
                if progress < 0.01 or progress > 0.99:
                    n_success += 1
                    for i in range(n_values):
                        data[i].append(float(line[i]))
                else:
                    continue
        
        progresses = np.array(progresses)
        data = np.array(data)
        with open(self.path + "SummaryStatistics.txt", "w") as file:
            file.write(lines[0])
            file.write(lines[1])
            file.write("0")
            for i in range(1, n_values):
                if i == progress_ind:
                    file.write(f", {np.mean(progresses*100):14.4f}")
                else:
                    file.write(f", {np.mean(data[i]):14.4f}")
            file.write(f", {n_success/n_total * 100}")
            file.write("\n")



    def generate_steering_graphs(self):
        # if self.lap_n != 0: return # only do it for one lap.
        steering = self.actions[:, 0]

        name = self.vehicle_name.split("_")[1]
        if name == "C2": name = "PP"
        
        plt.figure(1)
        plt.clf()
        color = "orange"
        plt.gca().hist(steering, bins=9, density=False, weights=np.ones(len(steering)) / len(steering), color=color)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.xlim(-0.45, 0.45)
        plt.ylim(0, 0.35)

        plt.xticks([])
        plt.yticks([])
        plt.title(f"{name}", fontsize=24)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{self.path}/Hists/{self.vehicle_name}_steering_hist_{self.lap_n}.png", bbox_inches='tight')

    def plot_curvature_heat_map(self): 
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        diffs = np.diff(self.states[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(self.states[:, :2], seg_lengths, False)

        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, ks.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(abs(ks))
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line,fraction=0.046, pad=0.04)
        plt.gca().set_aspect('equal', adjustable='box')

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.savefig(self.path + f"Curvatures/{self.vehicle_name}_curve_map_{self.lap_n}.svg", bbox_inches='tight')
        plt.savefig(self.path + f"Curvatures/Curvature_{self.lap_n}_{self.vehicle_name}.pdf", bbox_inches='tight', pad_inches=0)
        # plt.figure(2)
        # plt.clf()
        # plt.plot(ks, '-')
        # plt.savefig(self.path + f"Curvatures/{self.vehicle_name}_curve_graph_{self.lap_n}.svg", bbox_inches='tight')

    def plot_velocity_heat_map(self): 
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        # N = len(points)
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(vs.min(), vs.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line,fraction=0.046, pad=0.04)
        # plt.xlim(points[:, 0, 0].min()-10, points[:, 0, 0].max()+10)
        # plt.ylim(points[:, 0, 1].min()-10, points[:, 0, 1].max()+10)
        plt.gca().set_aspect('equal', adjustable='box')

        
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # plt.show()

        plt.savefig(self.path + f"Velocities/{self.vehicle_name}_velocity_map_{self.lap_n}.svg", bbox_inches='tight')
        plt.savefig(self.path + f"Velocities/{self.vehicle_name}_traj_{self.lap_n}.pdf", bbox_inches='tight')

    def plot_friction_graphs(self):
        fig, axs = plt.subplots(2, 2)
        vs = self.states[:, 3]
        ds = self.states[:, 2]
        slips = self.states[:,6]

        L = 0.329
        fs = vs ** 2 / L * np.tan(ds)

        axs[0,0].plot(fs)
        axs[1,0].plot(slips)
        axs[0, 1].plot(vs)
        axs[1, 1].plot(ds)

        axs[0, 0].set_title("Friction")
        axs[0, 1].set_title("Velocity")
        axs[1, 1].set_title("Delta")
        axs[1, 0].set_title("Slip")
        axs[0, 0].set_ylim(-40, 40)

        # plt.show()
        plt.title(f"{self.vehicle_name}")
        plt.savefig(self.path + f"FrictionPlots/{self.vehicle_name}_friction_graph_{self.lap_n}.svg", bbox_inches='tight')

        plt.close()

    def plot_curvature_graphs(self, lap_n):
        if not self.load_lap_data(lap_n): return

        diffs = np.diff(self.states[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(self.states[:, :2], seg_lengths, False)

        plt.figure(1)
        for i in range(len(ths)):
            plt.arrow(self.states[i, 0], self.states[i, 1], np.cos(ths[i]), np.sin(ths[i]), color='r', width=0.01)
            plt.plot(self.states[i, 0], self.states[i, 1], 'go', markersize=1)

        # plt.figure(2)
        # plt.plot(ks)

        plt.pause(0.0001)


    def plot_obs_graphs(self):
        plt.figure(1)
        plt.clf()
        self.map_data.plot_map_img_obs(self.obs_rng)
        points = self.states[:, 0:2]
        
        xs, ys = self.map_data.pts2rc(points)
        plt.plot(xs, ys, 'b-')

        plt.show()

        # plt.savefig(self.path + f"Trajectories/{self.vehicle_name}_curve_map_{self.lap_n}.svg", bbox_inches='tight')
        # plt.savefig(self.path + f"Curvatures/Curvature_{self.lap_n}_{self.vehicle_name}.pdf", bbox_inches='tight', pad_inches=0)



def analyse_folder():
    path = "Data/Vehicles/BigSlowTests/"
    # path = "Data/Vehicles/FastTests/"
    # path = "Data/Vehicles/SlowTests/"

    TestData = AnalyseTestLapData()
    TestData.explore_folder(path)

if __name__ == '__main__':
    analyse_folder()
