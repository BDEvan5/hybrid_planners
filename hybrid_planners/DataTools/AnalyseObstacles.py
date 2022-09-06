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

    def plot_avg_progress(self, x, color):
        progresses = np.array([l.progress for l in self.laps]) * 100
        avg_progress = np.mean(progresses)
        q1 = np.percentile(progresses, 25)
        q3 = np.percentile(progresses, 75)
        std_err = np.array([avg_progress - q1, q3 - avg_progress])[:, None]

        # give it [lower_err, upper_err]
        plt.plot(x, avg_progress, '.', color=color, markersize=4)
        plt.errorbar(x, avg_progress, yerr=std_err, color=color, capsize=1, alpha=0.6, linewidth=2)
        
        return avg_progress

class TestSet:
    """Holds all the tests for a comparative assessment
    """
    def __init__(self, path):
        self.path = path
        self.test_sets = []

    def load_test_set(self):
        folders = glob.glob(os.path.join(self.path, "*/"))
        folders.sort()
        for i, f in enumerate(folders):
            print(f"Load: {f}")
            d = TestData(f)
            self.test_sets.append(d)

    def print_big_data(self):
        for test_set in self.test_sets:
            test_set.make_detailed_file()
            test_set.make_summary_file()
            print(test_set.path)

    def write_test_summary(self):
        with open(self.path + "test_summary.txt", "w") as f:
            f.write("Test Name".ljust(10) + "Total Distance".ljust(14) + "Progress".ljust(14) + "Time".ljust(14) + "Avg Velocity".ljust(14) + "Avg Steering".ljust(14) + "Mean Curvature".ljust(14) + "Total Curvature".ljust(14) + "Mean Deviation".ljust(14) + "Total Deviation".ljust(14) + "\n" )
            for test_set in self.test_sets:
                f.write(test_set.generate_summary_line() + "\n")

    def plot_avg_progress(self, color, offset):
        plt.figure(1, figsize=(6, 2.5))

        xs = []
        avgs = []
        for i, test_set in enumerate(self.test_sets):
            avg = test_set.plot_avg_progress(i+offset, color)
            avgs.append(avg)
            xs.append(i+offset)

        plt.plot(xs, avgs, '-', color=color, linewidth=2)

        plt.xlabel("Number of Obstacles")
        plt.ylabel("Average Progress")
        plt.grid(True)
        plt.tight_layout()
        plt.ylim(0, 105)
        plt.gca().get_xaxis().set_major_locator(MultipleLocator(1))


        # plt.show()


def generate_pp_data_set():
    path = "Data/Vehicles/RunPP/PP_f1_aut_2_0/"
    # path = "Data/Vehicles/RunPP/PP_f1_aut_1_0/"

    test_set = TestSet(path)
    test_set.load_test_set()
    # test_set.print_big_data()
    test_set.write_test_summary()
    test_set.plot_avg_progress()

def generate_pp_obs_graph():
    path = "Data/Vehicles/RunPP/PP_f1_aut_2_0/"

    test_set = TestSet(path)
    test_set.load_test_set()
    test_set.plot_avg_progress('blue', 0.05)

    path = "Data/Vehicles/RunPP/PP_f1_aut_4_0/"

    test_set = TestSet(path)
    test_set.load_test_set()
    test_set.plot_avg_progress('darkgreen', -0.05)

    plt.savefig("Data/LowSpeedEval/PP_Obs_progress.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


# generate_pp_data_set()
generate_pp_obs_graph()
