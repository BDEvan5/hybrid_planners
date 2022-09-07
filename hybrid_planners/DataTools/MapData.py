import numpy as np
from matplotlib import pyplot as plt
import csv, yaml
from PIL import Image
from matplotlib.collections import LineCollection

class MapData:
    def __init__(self, map_name):
        self.path = "maps/"
        # self.path = "map_data/"
        self.map_name = map_name

        self.xs, ys = None, None
        self.t_ss, self.t_xs, self.t_ys, self.t_ths, self.t_ks, self.t_vs, self.t_accs = None, None, None, None, None, None, None

        self.N = 0
        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.load_map_img()
        self.load_centerline()
        try:
            self.load_raceline()
        except: pass

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def load_centerline(self):
        xs, ys = [], []
        # with open(self.path + self.map_name + "_std.csv", 'r') as file:
        with open(self.path + self.map_name + "_centerline.csv", 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))

        self.xs = np.array(xs)
        self.ys = np.array(ys)

        self.N = len(xs)

    def load_raceline(self):
        ss, xs, ys, thetas, ks, vs, accs = [], [], [], [], [], [], []

        waypoints = np.loadtxt(self.path + self.map_name + '_raceline.csv', delimiter=',', skiprows=0)

        for i in range(len(waypoints)):
            if i ==0 or i ==1 or i ==2:
                continue
            lines = waypoints[i]
            ss.append(float(lines[0]))
            xs.append(float(lines[1]))
            ys.append(float(lines[2]))
            thetas.append(float(lines[3]))
            ks.append(float(lines[4]))
            vs.append(float(lines[5]))
            # accs.append(float(lines[6]))

        self.t_ss = np.array(ss)
        self.t_xs = np.array(xs)
        self.t_ys = np.array(ys)
        self.t_ths = np.array(thetas)
        self.t_ks = np.array(ks)
        self.t_vs = np.array(vs)
        # self.t_accs = np.array(accs)

    def xy2rc(self, xs, ys):
        xs = (xs - self.map_origin[0]) / self.map_resolution
        ys = (ys - self.map_origin[1]) /self.map_resolution
        return xs, ys

    def pts2rc(self, pts):
        return self.xy2rc(pts[:,0], pts[:,1])
    
    def plot_centre_line(self):
        xs, ys = self.xy2rc(self.xs, self.ys)
        plt.plot(xs, ys, '-', color='black')

    def plot_race_line(self):
        xs, ys = self.xy2rc(self.t_xs, self.t_ys)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(self.t_vs.min(), self.t_vs.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.t_vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)

    def plot_map_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

    def plot_map_data(self):
        self.plot_map_img()

        self.plot_centre_line()
        
        self.plot_race_line()

        plt.show()

    def plot_map_img_obs(self, rng):
        obstacle_size=[0.7, 0.7]
        n_obstacles = 6
        track_pts = np.array([self.xs, self.ys]).T
        radius = 1

        obs_size_m = np.array(obstacle_size)
        obs_size_px = np.array(obs_size_m / self.map_resolution, dtype=int)

        min_idx = int(len(track_pts) //10)
        rand_idxs = rng.integers(min_idx, len(track_pts)-min_idx, size=n_obstacles)

        rand_radii = rng.random(size=(n_obstacles, 2)) * radius

        obs_locations = track_pts[rand_idxs, :] + rand_radii

        new_img = self.map_img.copy()
        new_img[new_img == 1] = 180
        new_img[new_img == 0 ] = 230
        new_img[0, 1] = 255
        new_img[0, 0] = 0
        new_img = generate_obs_map_img(new_img, obs_locations, self.map_origin[0], self.map_origin[1], obs_size_px, self.map_resolution)

        plt.imshow(new_img, origin='lower', cmap='gray')

def generate_obs_map_img(map_img, obs_locations, orig_x, orig_y, obs_size_px, map_resolution):
    """Adds obstacles of the defined size to the map image.

    Args:
        map_img (nd array): image of the map image
        obs_locations (ndarray): set of coords in pixels
        orig_x (float): x offset to map original
        orig_y (float): y offset to map original
        obs_size_px (ndarray): 2x1 set of obs size in px
        map_resolution (float): meters to pixel resolution  

    Returns:
        map_img (ndarray): image with obstacles added
    """
    for location in obs_locations:
        # convert the location to the pixel coordinates
        x = int((location[0] - orig_x) / map_resolution)
        y = int((location[1] - orig_y) / map_resolution)
        map_img[y:y+obs_size_px[0], x:x+obs_size_px[1]] = 0

    return map_img


def main():
    map_name = "f1_gbr"

    map_data = MapData(map_name)
    map_data.plot_map_data()

if __name__ == '__main__':

    main()