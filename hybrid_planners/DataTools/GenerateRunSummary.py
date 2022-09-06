import numpy as np
import matplotlib.pyplot as plt
import glob


def generate_summary_table():
    folder = "Data/Vehicles/SlowTests/"

    map_name = "columbia_small"
    # map_name = "f1_aut"
    folders = glob.glob(folder + f"*_{map_name}*/")
    agents = [f.split("/")[-2] for f in folders]
    agents.sort()
    print(agents)
    
    for i, agent in enumerate(agents):
        with open(folder + f"{agent}/SummaryStatistics.txt", 'r') as agent_file:
            lines = agent_file.readlines()
            if i == 0:
                with open(folder + f"RunSummaryTable_{map_name}.txt", 'w') as summary_file:
                    summary_file.write(f"Agent ".ljust(30)  + lines[1])
        
            with open(folder + f"RunSummaryTable_{map_name}.txt", 'a') as summary_file:
                summary_file.write(f"{agent} ".ljust(30)  + lines[2])



generate_summary_table()

