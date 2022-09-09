from matplotlib import pyplot as plt
import csv
import numpy as np

# Track base
class RaceTrack:
    def __init__(self, map_name) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = map_name
        self.total_s = None

        self.max_distance = 0
        self.distance_allowance = 0.4

    def plot_wpts(self):
        plt.figure(1)
        plt.plot(self.wpts[:, 0], self.wpts[:, 1], 'b-')
        for i, pt in enumerate(self.wpts):
            # plt.plot(pt[0], pt[1], )
            plt.text(pt[0], pt[1], f"{i}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def load_centerline(self):
        # filename = 'map_data/' + self.map_name + '_std.csv'
        filename = 'maps/' + self.map_name + '_centerline.csv'
        xs, ys, w_rs, w_ls = [0], [0], [], []
        with open(filename, 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))
                w_rs.append(float(lines[2]))
                w_ls.append(float(lines[3]))
        xs[-1] = 0
        ys[-1] = 0
        self.xs = np.array(xs)[:, None]
        self.ys = np.array(ys)[:, None]
        self.centre_length = len(xs)

        self.wpts = np.vstack((xs, ys)).T

        # self.wpts = self.wpts[::-1, :] #? Reverse if wanted

        diffs = np.diff(self.wpts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.ss = np.cumsum(seg_lengths)
        self.ss = np.insert(self.ss, 0, 0)

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def find_s(self, point):
        idx, dists = self.get_trackline_segment(point)

        x, h = self.interp_pts(idx, dists)

        s = self.ss[idx] + x

        return s

    def find_progress_percent(self, point):
        s = self.find_s(point)
        return s/self.total_s

    def interp_pts(self, idx, dists):
        """
        
        """
        # finds the reflected distance along the line joining wpt1 and wpt2
        # uses Herons formula for the area of a triangle
        d_ss = self.ss[idx+1] - self.ss[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            Area = Area_square**0.5
            h = Area * 2/d_ss
            if np.isnan(h):
                h = 0
            x = (d1**2 - h**2)**0.5

        return x, h

    def get_trackline_segment(self, point):
        """Returns the first index representing the line segment that is closest to the point.

        wpt1 = pts[idx]
        wpt2 = pts[idx+1]

        dists: the distance from the point to each of the wpts.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)
        if min_dist_segment == 0:
            return 0, dists
        elif min_dist_segment == len(dists)-1:
            return len(dists)-2, dists 

        if dists[min_dist_segment+1] < dists[min_dist_segment-1]:
            return min_dist_segment, dists
        else: 
            return min_dist_segment - 1, dists

    def get_cross_track_heading(self, point):
        idx, dists = self.get_trackline_segment(point)
        point_diff = self.wpts[idx+1, :] - self.wpts[idx, :]
        trackline_heading = np.arctan2(point_diff[1], point_diff[0])

        x, h = self.interp_pts(idx, dists)

        return trackline_heading, h

    def plot_vehicle(self, point, theta):
        idx, dists = self.get_trackline_segment(point)
        point_diff = self.wpts[idx+1, :] - self.wpts[idx, :]
        trackline_heading = np.arctan2(point_diff[1], point_diff[0])

        x, h = self.interp_pts(idx, dists)

        track_pt = self.wpts[idx] + x * np.array([np.cos(trackline_heading), np.sin(trackline_heading)])

        plt.figure(1)
        plt.clf()
        size = 1.2
        plt.xlim([point[0]-size, point[0]+size])
        plt.ylim([point[1]-size, point[1]+size])
        plt.plot(self.wpts[:,0], self.wpts[:,1], 'b-x', linewidth=2)
        plt.plot(self.wpts[idx:idx+2, 0], self.wpts[idx:idx+2, 1], 'r-', linewidth=2)
        plt.plot([point[0], track_pt[0]], [point[1], track_pt[1]], 'orange', linewidth=2)
        plt.plot(track_pt[0], track_pt[1],'o', color='orange', markersize=6)

        plt.plot(point[0], point[1], 'go', markersize=6)
        plt.arrow(point[0], point[1], np.cos(theta), np.sin(theta), color='g', head_width=0.1, head_length=0.1, linewidth=2)

        plt.pause(0.0001)

    def check_done(self, observation):
        position = observation['state'][0:2]
        s = self.find_s(position)

        if s <= (self.max_distance - self.distance_allowance) and self.max_distance < 0.8*self.total_s and s > 0.01:
            # check if I went backwards, unless the max distance is almost finished and that it isn't starting
            return True # made negative progress
        self.max_distance = max(self.max_distance, s)

        return False

from hybrid_planners.Planners.PurePursuit import PurePursuit

class DeviationPP:
    def __init__(self, conf, run):
        self.pp = PurePursuit(conf, run)
        

    def __call__(self, observation, prev_obs, action):
            if prev_obs is None: return 0

            if observation['lap_done']:
                return 1  # complete
            if observation['colision_done']:
                return -1 # crash
            
            pp_steering = self.pp.plan(prev_obs)[0]

            reward = 0.2 - abs(pp_steering - action[0]) * 0.5
            reward = max(reward, 0) # limit at 0

            return reward


class DistanceReward():
    def __init__(self, race_track: RaceTrack) -> None:
        self.race_track = race_track

    def __call__(self, observation, prev_obs, n=None):
        if prev_obs is None: return 0

        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        
        
        position = observation['state'][0:2]
        prev_position = prev_obs['state'][0:2]
        theta = observation['state'][2]

        s = self.race_track.find_s(prev_position)
        ss = self.race_track.find_s(position)
        reward = (ss - s) / self.race_track.total_s
        if abs(reward) > 0.5: # happens at end of eps
            return 0.001 # assume positive progress near end

        # self.race_track.plot_vehicle(position, theta)


        # reward *= 0 # remove all reward
        return reward 

class CrossTrackHeadReward:
    def __init__(self, race_track: RaceTrack, conf):
        self.race_track = race_track
        self.r_veloctiy = conf.r_velocity
        self.r_distance = conf.r_distance
        self.max_v = conf.max_v

    def __call__(self, observation, prev_obs):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash

        position = observation['state'][0:2]
        theta = observation['state'][2]
        heading, distance = self.race_track.get_cross_track_heading(position)
        # self.race_track.plot_vehicle(position, theta)

        d_heading = abs(robust_angle_difference_rad(heading, theta))
        r_heading  = np.cos(d_heading)  * self.r_veloctiy # velocity
        r_heading *= (observation['state'][3] / self.max_v)

        r_distance = distance * self.r_distance

        reward = r_heading - r_distance
        return reward

def robust_angle_difference_degree(x, y):
    """Returns the difference between two angles in DEGREES
    r = x - y"""
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    r = np.arctan2(np.sin(x-y), np.cos(x-y))
    return np.rad2deg(r)

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))


class StdReward:
    def __call__(self, observation, prev_obs):
        if observation['lap_done']:
            return 1  # complete
        if observation['colision_done']:
            return -1 # crash
        return 0

def test_angle_diff():
    print(f"x: -10, y: 10 --> angle: {robust_angle_difference_degree(-10, 10)}")
    print(f"x: -10, y: 20 --> angle: {robust_angle_difference_degree(-10, 20)}")
    print(f"x: -30, y: 10 --> angle: {robust_angle_difference_degree(-30, 10)}")

    print(f"x: 10, y: -10 --> angle: {robust_angle_difference_degree(10, -10)}")
    print(f"x: 10, y: -20 --> angle: {robust_angle_difference_degree(10, -20)}")
    print(f"x: 30, y: -10 --> angle: {robust_angle_difference_degree(30, -10)}")

    print(f"x: 180, y: -180 --> angle: {robust_angle_difference_degree(180, -180)}")
    print(f"x: 170, y: -180 --> angle: {robust_angle_difference_degree(170, -180)}")
    print(f"x: 180, y: -170 --> angle: {robust_angle_difference_degree(180, -170)}")

if __name__ == '__main__':
    test_angle_diff()
    pass
