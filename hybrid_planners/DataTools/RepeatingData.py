import numpy as np
import matplotlib.pyplot as plt
import os, glob
import trajectory_planning_helpers as tph
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib.collections import LineCollection

from hybrid_planners.DataTools.MapData import MapData
from hybrid_planners.Utils.Reward import RaceTrack 
from hybrid_planners.Utils.utils import *

class LapData:
    """Holds the data for a single lap of a single test
    """
    def __init__(self, path, lap_n):
        self.states = None
        self.actions = None
        self.vehicle_name = path.split('/')[-3]
        self.lap_n = lap_n

        try:
            data = np.load(path + f"/Lap_{lap_n}_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            print(f"No data for: " + f"/Lap_{lap_n}_history_{self.vehicle_name}.npy")
        self.states = data[:, :7]
        self.actions = data[:, 7:]

        self.avg_steering = None
        self.total_distance = None
        self.mean_curvature = None
        self.total_curvature = None
        self.mean_deviation = None
        self.total_deviation = None
        self.time = None
        self.avg_velocity = None
        self.progress = None

    def calculate_lap_statistics(self, race_track: RaceTrack):
        steering = np.abs(self.actions[:, 0])
        self.avg_steering = np.mean(np.abs(steering))

        pts = self.states[:, 0:2]
        ss = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.total_distance = np.sum(ss)

        ths, ks = tph.calc_head_curv_num.calc_head_curv_num(pts, ss, False)
        self.mean_curvature = np.mean(np.abs(ks))
        self.total_curvature = np.sum(np.abs(ks))

        hs = []
        for point in pts:
            idx, dists = race_track.get_trackline_segment(point)
            x, h = race_track.interp_pts(idx, dists)
            hs.append(h)

        hs = np.array(hs)
        self.mean_deviation = np.mean(hs)
        self.total_deviation = np.sum(hs)

        self.time = len(pts) /10
        vs = self.states[:, 3]
        self.avg_velocity = np.mean(vs)

        self.progress = race_track.find_s(pts[-1])/race_track.total_s
        if self.progress < 0.01 or self.progress > 0.99:
            self.progress = 1 # it is finished


    # def format_stats(self):
    #     with open(self.path + "Statistics.txt", "a") as file:
    #         file.write(f"{self.lap_n}, {rms_steering:14.4f}, {total_distance:14.4f}, {mean_curvature:14.4f}, {total_curvature:14.4f}, {mean_deviation:14.4f}, {total_deviation:14.4f}, {progress:14.4f}, {time:14.2f}, {avg_velocity:14.4f}\n")

        

class TestData:
    """Holds the test data for multiple laps
    """
    def __init__(self, path):
        self.path = path
        self.laps = []

        self.test_name = path.split('/')[-2]
        self.vehicle_name = self.path.split("/")[-3]
        self.map_name = self.vehicle_name.split("_")[1]
        if self.map_name == "f1":
            self.map_name += "_" + self.vehicle_name.split("_")[2]
        elif self.map_name == "columbia": self.map_name += "_" + self.vehicle_name.split("_")[2]

        self.map_data = MapData(self.map_name)
        self.race_track = RaceTrack(self.map_name)
        self.race_track.load_centerline()

        self.load_test_data()

    def load_test_data(self):
        n_laps = len(glob.glob(self.path + "Lap_*_history_*.npy"))
        for i in range(n_laps):
            d = LapData(self.path, i)
            d.calculate_lap_statistics(self.race_track)
            self.laps.append(d)

    def make_detailed_file(self):
        with open(self.path + "StatsDetail.txt", "w") as f:
            f.write(f"Name: {self.path}\n")
            f.write("Lap" + "Steering".rjust(16) + "Total Distance".rjust(16) + "Mean Curvature".rjust(16) + "Total Curvature".rjust(16) + "Mean Deviation".rjust(16) + "Total Deviation".rjust(16) + "Progress".rjust(16) + "Time".rjust(16) + "Avg Velocity".rjust(16) + "\n")
            
            for i, l in enumerate(self.laps):
                f.write(f"{i}, {l.avg_steering:14.4f}, {l.total_distance:14.2f}, {l.mean_curvature:14.4f}, {l.total_curvature:14.2f}, {l.mean_deviation:14.4f}, {l.total_deviation:14.2f}, {l.progress:14.2f}, {l.time:14.2f}, {l.avg_velocity:14.2f}\n")

    def make_summary_file(self):
        with open(self.path + "StatsSummary.txt", "w") as f:
            f.write(f"Name: {self.path}\n")
            f.write("Lap" + "Steering".rjust(16) + "Total Distance".rjust(16) + "Mean Curvature".rjust(16) + "Total Curvature".rjust(16) + "Mean Deviation".rjust(16) + "Total Deviation".rjust(16) + "Progress".rjust(16) + "Time".rjust(16) + "Avg Velocity".rjust(16) + "\n")
            f.write(f"Mean:, {np.mean([l.avg_steering for l in self.laps]):14.4f}, ")
            f.write(f"{np.mean([l.total_distance for l in self.laps]):14.2f}, ")
            f.write(f"{np.mean([l.mean_curvature for l in self.laps]):14.4f}, ")
            f.write(f"{np.mean([l.total_curvature for l in self.laps]):14.2f}, ")
            f.write(f"{np.mean([l.mean_deviation for l in self.laps]):14.4f}, ")
            f.write(f"{np.mean([l.total_deviation for l in self.laps]):14.2f}, ")
            f.write(f"{np.mean([l.progress for l in self.laps]):14.2f}, ")
            f.write(f"{np.mean([l.time for l in self.laps]):14.2f}, ")
            f.write(f"{np.mean([l.avg_velocity for l in self.laps]):14.2f}\n")

            f.write(f"Std: , {np.std([l.avg_steering for l in self.laps]):14.4f}, ")
            f.write(f"{np.std([l.total_distance for l in self.laps]):14.2f}, ")
            f.write(f"{np.std([l.mean_curvature for l in self.laps]):14.4f}, ")
            f.write(f"{np.std([l.total_curvature for l in self.laps]):14.2f}, ")
            f.write(f"{np.std([l.mean_deviation for l in self.laps]):14.4f}, ")
            f.write(f"{np.std([l.total_deviation for l in self.laps]):14.2f}, ")
            f.write(f"{np.std([l.progress for l in self.laps]):14.2f}, ")
            f.write(f"{np.std([l.time for l in self.laps]):14.2f}, ")
            f.write(f"{np.std([l.avg_velocity for l in self.laps]):14.2f}\n")

            f.write(f"Q1:  , {np.percentile([l.avg_steering for l in self.laps], 25):14.4f}, ")
            f.write(f"{np.percentile([l.total_distance for l in self.laps], 25):14.2f}, ")
            f.write(f"{np.percentile([l.mean_curvature for l in self.laps], 25):14.4f}, ")
            f.write(f"{np.percentile([l.total_curvature for l in self.laps], 25):14.2f}, ")
            f.write(f"{np.percentile([l.mean_deviation for l in self.laps], 25):14.4f}, ")
            f.write(f"{np.percentile([l.total_deviation for l in self.laps], 25):14.2f}, ")
            f.write(f"{np.percentile([l.progress for l in self.laps], 25):14.2f}, ")
            f.write(f"{np.percentile([l.time for l in self.laps], 25):14.2f}, ")
            f.write(f"{np.percentile([l.avg_velocity for l in self.laps], 25):14.2f}\n")

            f.write(f"Q2:  , {np.percentile([l.avg_steering for l in self.laps], 50):14.4f}, ")
            f.write(f"{np.percentile([l.total_distance for l in self.laps], 50):14.2f}, ")
            f.write(f"{np.percentile([l.mean_curvature for l in self.laps], 50):14.4f}, ")
            f.write(f"{np.percentile([l.total_curvature for l in self.laps], 50):14.2f}, ")
            f.write(f"{np.percentile([l.mean_deviation for l in self.laps], 50):14.4f}, ")
            f.write(f"{np.percentile([l.total_deviation for l in self.laps], 50):14.2f}, ")
            f.write(f"{np.percentile([l.progress for l in self.laps], 50):14.2f}, ")
            f.write(f"{np.percentile([l.time for l in self.laps], 50):14.2f}, ")
            f.write(f"{np.percentile([l.avg_velocity for l in self.laps], 50):14.2f}\n")

            f.write(f"Q3:  , {np.percentile([l.avg_steering for l in self.laps], 50):14.4f}, ")
            f.write(f"{np.percentile([l.total_distance for l in self.laps], 75):14.2f}, ")
            f.write(f"{np.percentile([l.mean_curvature for l in self.laps], 75):14.4f}, ")
            f.write(f"{np.percentile([l.total_curvature for l in self.laps], 75):14.2f}, ")
            f.write(f"{np.percentile([l.mean_deviation for l in self.laps], 75):14.4f}, ")
            f.write(f"{np.percentile([l.total_deviation for l in self.laps], 75):14.2f}, ")
            f.write(f"{np.percentile([l.progress for l in self.laps], 75):14.2f}, ")
            f.write(f"{np.percentile([l.time for l in self.laps], 75):14.2f}, ")
            f.write(f"{np.percentile([l.avg_velocity for l in self.laps], 75):14.2f}\n")

    def generate_summary_line(self):
        ret_string =  f"{self.test_name}".ljust(10) + f"{np.mean([l.total_distance for l in self.laps]):14.2f}, {np.mean([l.progress for l in self.laps]):14.2f}, {np.mean([l.time for l in self.laps]):14.2f}, {np.mean([l.avg_velocity for l in self.laps]):14.2f}, {np.mean([l.avg_steering for l in self.laps]):14.4f}, {np.mean([l.mean_curvature for l in self.laps]):14.4f}, {np.mean([l.total_curvature for l in self.laps]):14.2f}, {np.mean([l.mean_deviation for l in self.laps]):14.4f}, {np.mean([l.total_deviation for l in self.laps]):14.2f}"

        return ret_string

    def generate_time_distribution(self):
        time_list = [l.time for l in self.laps if l.progress > 0.99]

        plt.figure(1)
        plt.clf()
        plt.hist(time_list, bins=15)

        # plt.title(f"Time distribution for {self.vehicle_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.xlim(38, 41)
        plt.ylim(0, 20)

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"Data/TestImgs/{self.vehicle_name}_times.svg")
        plt.savefig(f"Data/TestImgs/{self.vehicle_name}_times.pdf", bbox_inches='tight', pad_inches=0)

        # plt.figure(2)
        # plt.clf()
        # plt.boxplot(time_list, vert=False)
        # plt.xlabel("Time (s)")

        # plt.savefig(f"Data/TestImgs/{self.vehicle_name}_times_box.svg")

        # plt.show()

    def generate_progress_distribution(self):
        time_list = [l.progress for l in self.laps]

        plt.figure(1)
        plt.clf()
        plt.hist(time_list, bins=15)

        plt.title(f"Time distribution for {self.vehicle_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")

        plt.savefig(f"Data/TestImgs/{self.vehicle_name}_progress.svg")

        plt.figure(2)
        plt.clf()
        plt.boxplot(time_list, vert=False)
        plt.xlabel("Time (s)")
        plt.savefig(f"Data/TestImgs/{self.vehicle_name}_progress_box.svg")


        # plt.show()


def make_thesis_dist():
    map_name = "columbia_small"
    planners = ["E2e", "Mod", "Serial"]
    # planner = "E2e"
    # planner = "Mod"
    # planner = "Serial"
    for planner in planners:
        path = f"Data/Vehicles/SlowTests/{planner}_{map_name}_1_0/TestData/"

        td = TestData(path)
        # td.load_test_data()
        # td.make_detailed_file()
        td.make_summary_file()
        td.generate_time_distribution()
        # td.generate_progress_distribution()


def make_train_repeat_plot():
    map_name = "columbia_small"
    planners = ["E2e", "Mod", "Serial"]
    # planner = "E2e"
    # planner = "Mod"
    # planner = "Serial"
    for planner in planners:
        for i in range(5):
            path = f"Data/Vehicles/SlowTests/{planner}_{map_name}_1_{i}/"

            td = TestData(path)
            td.load_test_data()
            




make_thesis_dist()

