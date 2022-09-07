import numpy as np
import glob, os, shutil

def reorder_files():
    folder = "Data/Vehicles/BigSlowTests/"
    vehicles = glob.glob(folder + "*/")
    
    agents = [vehicle.split("/")[-2] for vehicle in vehicles]

    # make dst folders
    # for a in range(len(agents)):
    #     path = vehicles[a] + "TestData/"
    #     print(f"Path to make: {path}")
    #     os.mkdir(path)

    laps = glob.glob(folder + "Lap*.npy")
    for i in range(len(laps)):
        lap_name = laps[i].split("/")[-1]
        # print(f"Lap name: {lap_name}")
        vehicle_segments = lap_name.split('_')
        # print(f"Vehicle segments: {vehicle_segments}")
        vehicle_name = "_".join(vehicle_segments[3:7])
        vehicle_name += "_" + vehicle_segments[7].split('.')[0]
        # print(f"Vehicle name: {vehicle_name}")

        # print(f"Src: {laps[i]}")
        dst = folder + vehicle_name + "/TestData/" + lap_name
        print(f"Dst: {dst}")
        shutil.move(laps[i], dst)

reorder_files()
